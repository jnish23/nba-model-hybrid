"""
Point-in-time injury features from official NBA injury report PDFs.

Pipeline:
  1. parse_all_reports(): parse every downloaded PDF (cached incrementally to a CSV)
  2. link_reports(): attach teamId, gameId (via date + team), personId (via normalized name)
  3. snapshot(): for each game, keep the latest report published at or before a cutoff
     (e.g. 8 PM ET the night before, or 30 min before tip) -> what was known at that time
  4. reported_features(): value the reported absences with each player's pre-game value from
     core.availability_features, weighted by the empirical probability a player with that status
     does not play.

Report times and game times are US/Eastern as printed in the PDFs; game dates in team_game_full
are local NBA game dates (same calendar day as the ET date for all regular tip times).
"""
import logging
import re
import sqlite3
import unicodedata
import difflib
from pathlib import Path

import numpy as np
import pandas as pd

from core.injury_report_parser import parse_report, PARSER_VERSION

REPORT_DIR = Path('data/raw/injury_reports')
PARSED_CACHE = Path('data/processed/injury_reports_parsed.csv')

TEAM_NAME_ALIASES = {'laclippers': 'losangelesclippers', 'charlottebobcats': 'charlottehornets'}


def _norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z]', '', s.lower())


def map_unique(series, func):
    """series.map(func), but func runs once per distinct value (the 750k injury-report rows carry only a
    few thousand distinct names / files). NaN maps to NaN, as func(NaN) does for every use here."""
    codes, uniques = pd.factorize(series)
    mapped = pd.Series(uniques).map(func)
    out = mapped.reindex(codes)               # code -1 (NaN input) is not in the index -> NaN
    out.index = series.index
    return out


def report_name_key(name):
    """'Williams III, Robert' / 'SamuelsJr.,Jermaine' -> 'robertwilliamsiii' / 'jermainesamuelsjr'."""
    if not isinstance(name, str):
        return None
    if ',' in name:
        last, first = name.split(',', 1)
        return _norm(first) + _norm(last)
    return _norm(name)


def db_name_key(name):
    """'Robert Williams III' -> 'robertwilliamsiii'."""
    return _norm(name) if isinstance(name, str) else None


# ---------------------------------------------------------------- 1. parse

def parse_all_reports(report_dir=REPORT_DIR, cache=PARSED_CACHE):
    """Parse all PDFs under report_dir, reusing the cache for files already parsed."""
    pdfs = sorted(Path(report_dir).rglob('*.pdf'))
    cached = pd.read_csv(cache, dtype={'source_file': str}) if Path(cache).exists() else pd.DataFrame()
    if not cached.empty:
        current = cached.get('parser_version', pd.Series(0, index=cached.index)).fillna(0) == PARSER_VERSION
        stale = (~current).sum()
        if stale:
            logging.info(f"Dropping {stale} cached rows from older parser versions")
        cached = cached[current]
    done = set(cached['source_file']) if not cached.empty else set()
    todo = [p for p in pdfs if p.name not in done]
    logging.info(f"{len(pdfs)} PDFs, {len(todo)} to parse")
    frames, failed = [], []
    for i, p in enumerate(todo, 1):
        try:
            df = parse_report(p)
            if df.empty:
                df = pd.DataFrame({'report_ts': [pd.NaT]})  # remember that it parsed to nothing
            df['source_file'] = p.name
            df['parser_version'] = PARSER_VERSION
            frames.append(df)
        except Exception as e:  # corrupt / odd PDFs are logged, not fatal
            failed.append((p.name, repr(e)))
        if i % 250 == 0:
            logging.info(f"parsed {i}/{len(todo)}")
    if failed:
        logging.warning(f"{len(failed)} PDFs failed to parse, e.g. {failed[:3]}")
    if frames:
        cached = pd.concat([cached] + frames, ignore_index=True)
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache, index=False)
    out = cached.dropna(subset=['player', 'not_submitted'], how='all').copy()
    out['report_ts'] = pd.to_datetime(out['report_ts'])
    out['game_date'] = pd.to_datetime(out['game_date'])
    return out


MAX_REPORT_AGE_HOURS = 6.0


def check_report_freshness(reports, game_date, cutoff, max_age_hours=MAX_REPORT_AGE_HOURS):
    """
    How stale is the newest injury report that covers `game_date`, as of `cutoff` (US/Eastern)?
    Returns (latest report_ts or None, age in hours or None, stale: bool). stale = no report at all
    for that date, or the newest one was published more than `max_age_hours` before the cutoff.
    parse_all_reports() only sees PDFs already on disk, so when the live download fails (network, site
    change) the features silently come from an older report or from none; this makes that visible.
    """
    day = reports[pd.to_datetime(reports['game_date']).dt.normalize() == pd.Timestamp(game_date).normalize()]
    ts = pd.to_datetime(day['report_ts'])
    ts = ts[ts <= pd.Timestamp(cutoff)]
    if ts.empty:
        return None, None, True
    latest = ts.max()
    age = (pd.Timestamp(cutoff) - latest) / pd.Timedelta(hours=1)
    return latest, float(age), bool(age > max_age_hours)


# ---------------------------------------------------------------- 2. link

def _team_lookup(conn):
    names = pd.concat([
        pd.read_sql("SELECT DISTINCT home_team AS name, teamId_home AS teamId FROM nba_odds WHERE teamId_home IS NOT NULL", conn),
        pd.read_sql("SELECT DISTINCT away_team AS name, teamId_away AS teamId FROM nba_odds WHERE teamId_away IS NOT NULL", conn),
    ]).drop_duplicates()
    lookup = {_norm(n): int(t) for n, t in zip(names['name'], names['teamId'])}
    for alias, canonical in TEAM_NAME_ALIASES.items():
        if canonical in lookup:
            lookup.setdefault(alias, lookup[canonical])
    return lookup


FUZZY_MATCHES = []  # audit trail of the last link_reports() call (see scripts/audit_training_warnings.py)


def link_reports(reports, db_path, upcoming=None):
    """
    Add teamId, gameId, personId. Rows that cannot be linked keep NaN ids (counts are logged).
    `upcoming`: optional (gameId, teamId, gameDate, seasonYear) for games not yet in team_game_full.
    """
    with sqlite3.connect(db_path) as conn:
        team_lookup = _team_lookup(conn)
        games = pd.read_sql("SELECT gameId, teamId, gameDate, seasonYear FROM team_game_full", conn)
        players = pd.read_sql("""SELECT personId, playerName, teamId, seasonYear, MIN(gameDate) AS first_date
                                 FROM player_basic_stats GROUP BY personId, playerName, teamId, seasonYear""", conn)
    if upcoming is not None and len(upcoming):
        games = pd.concat([games, upcoming[['gameId', 'teamId', 'gameDate', 'seasonYear']]], ignore_index=True)
        games = games.drop_duplicates(['gameId', 'teamId'])
    games['gameDate'] = pd.to_datetime(games['gameDate']).dt.strftime('%Y-%m-%d')

    r = reports.copy()
    r['teamId'] = map_unique(r['team'], lambda t: team_lookup.get(_norm(t)))

    # Team cells sometimes wrap ("Minnesota" / "Timberwolves" on separate lines). Resolve leftovers
    # against the two teams in the row's matchup: the one whose full name contains the fragment.
    id_to_names = {}
    for name, tid in team_lookup.items():
        id_to_names.setdefault(tid, []).append(name)
    with sqlite3.connect(db_path) as conn:
        tri = pd.read_sql("SELECT DISTINCT teamTricode, teamId FROM team_game_full", conn)
    tri_to_id = dict(zip(tri['teamTricode'], tri['teamId'].astype(int)))
    tri_to_id.update({'PHO': tri_to_id.get('PHX'), 'GS': tri_to_id.get('GSW'), 'NO': tri_to_id.get('NOP'),
                      'NY': tri_to_id.get('NYK'), 'SA': tri_to_id.get('SAS'), 'UTAH': tri_to_id.get('UTA')})

    def _resolve(row):
        frag = _norm(row['team'])
        if not frag or not isinstance(row['matchup'], str):
            return np.nan
        hits = [tid for code in row['matchup'].split('@') if (tid := tri_to_id.get(code))
                and any(frag in n for n in id_to_names.get(tid, []))]
        return hits[0] if len(hits) == 1 else np.nan

    todo = r['teamId'].isna() & r['team'].notna()
    if todo.any():
        r.loc[todo, 'teamId'] = r[todo].apply(_resolve, axis=1)
    unmatched_teams = r.loc[r['teamId'].isna(), 'team'].dropna().unique()
    if len(unmatched_teams):
        logging.warning(f"Unmatched team names: {unmatched_teams[:10]}")

    games['game_date'] = pd.to_datetime(games['gameDate'])
    r = r.merge(games[['gameId', 'teamId', 'game_date', 'seasonYear']], on=['teamId', 'game_date'], how='left')

    # Player matching is point-in-time, so live inference links exactly like training:
    #  1. exact normalized key among players who appeared for this team this season BEFORE the game
    #  2. exact key anywhere in the database (player identity only, no outcome information)
    #  3. fuzzy match among the team's earlier appearances this season, only if the key is unknown
    #     everywhere (fuzzy matching a known name once mapped Jalen Williams to Jaylin Williams)
    players['key'] = map_unique(players['playerName'], db_name_key)
    players['first_date'] = pd.to_datetime(players['first_date'])
    r['key'] = map_unique(r['player'], report_name_key)
    first_app = players.sort_values('first_date').groupby(['key', 'teamId', 'seasonYear'])[['personId', 'first_date']].first()
    hit = first_app.reindex(pd.MultiIndex.from_frame(r[['key', 'teamId', 'seasonYear']]))
    prior = hit['first_date'].values < r['game_date'].values
    r['personId'] = np.where(prior, hit['personId'].values, np.nan)
    any_key = players.sort_values('first_date').groupby('key')['personId'].last()
    miss = r['personId'].isna() & r['key'].notna()
    r.loc[miss, 'personId'] = r.loc[miss, 'key'].map(any_key)

    miss = r['personId'].isna() & r['key'].notna() & r['teamId'].notna()
    roster_keys = players.groupby(['teamId', 'seasonYear'])
    fuzzy_hits = 0
    FUZZY_MATCHES.clear()
    for (tid, season), grp in r[miss].groupby(['teamId', 'seasonYear']):
        try:
            roster = roster_keys.get_group((tid, season))
        except KeyError:
            continue
        for (key, gdate), sub in grp.groupby(['key', 'game_date']):
            before = roster[roster['first_date'] < gdate]
            m = difflib.get_close_matches(key, before['key'].tolist(), n=1, cutoff=0.85)
            if m:
                r.loc[sub.index, 'personId'] = before.loc[before['key'] == m[0], 'personId'].iloc[0]
                fuzzy_hits += len(sub)
                FUZZY_MATCHES.append({'teamId': tid, 'seasonYear': season, 'game_date': gdate,
                                      'report_name': sub['player'].iloc[0], 'report_key': key, 'matched_key': m[0],
                                      'personId': r.loc[sub.index[0], 'personId'], 'n_rows': len(sub)})

    player_rows = r['player'].notna()
    logging.info(f"Linked: team {r['teamId'].notna().mean():.1%}, game {r['gameId'].notna().mean():.1%}, "
                 f"player {r.loc[player_rows, 'personId'].notna().mean():.1%} of player rows "
                 f"({fuzzy_hits} via fuzzy match)")
    return r


# ---------------------------------------------------------------- 3. snapshot

# Home teams that host 10:00-11:30 PM ET tip-offs; elsewhere a 10:xx/11:xx tip is a morning
# (international) game. Without this, a 10:00 AM ET game would get a 10 PM cutoff (leakage).
WESTERN_HOME = {'LAL', 'LAC', 'GSW', 'GS', 'SAC', 'POR', 'PHX', 'PHO', 'UTA', 'UTAH', 'DEN', 'SEA'}


def game_tip_times(reports):
    """ET tip time per gameId from the reports' game_time (HH:MM, AM/PM not printed)."""
    t = reports.dropna(subset=['gameId', 'game_time', 'matchup']).drop_duplicates('gameId')[['gameId', 'game_date', 'game_time', 'matchup']]
    hm = t['game_time'].str.extract(r'(\d\d):(\d\d)').astype(float)
    hour = hm[0].copy()
    home = t['matchup'].str.split('@').str[1]
    late = hour.isin([10, 11]) & home.isin(WESTERN_HOME)
    morning = hour.isin([10, 11]) & ~late
    hour = np.where(morning, hour, hour % 12 + 12)
    t['tip_ts'] = t['game_date'] + pd.to_timedelta(hour, unit='h') + pd.to_timedelta(hm[1], unit='m')
    n_morning = int(morning.sum())
    if n_morning:
        logging.info(f"{n_morning} games with 10:xx/11:xx tips at non-western home teams treated as AM")
    return t[['gameId', 'tip_ts']]


def snapshot(reports, cutoff, schedule):
    """
    For each game, use the latest report (file) published at or before the cutoff that lists the
    game (any row for its matchup, including 'not yet submitted'). Within that report:
      - a team with rows -> its listed players
      - a team marked NOT YET SUBMITTED -> missing
      - a team not listed at all while its opponent is -> nothing to report (0 injuries)
    Games no report lists before the cutoff are missing for both teams.
    `cutoff` is a function(DataFrame with game_date, tip_ts) -> Series of timestamps, e.g.
      night_before = lambda g: g['game_date'] - pd.Timedelta(hours=4)          # 8 PM ET day before
      pre_tip      = lambda g: g['tip_ts'] - pd.Timedelta(minutes=30)
    `schedule` has (gameId, teamId) for both teams of every game (see load_schedule).
    Returns (rows, coverage): rows of the chosen report per game, and one coverage row per
    (gameId, teamId) for both teams of every game that had a report.
    """
    r = reports.dropna(subset=['gameId', 'teamId']).merge(game_tip_times(reports), on='gameId', how='left')
    r['cutoff'] = cutoff(r)
    r = r[r['report_ts'] <= r['cutoff']]
    chosen = r.groupby('gameId')['report_ts'].transform('max')
    r = r[r['report_ts'] == chosen]

    games = r.groupby('gameId').agg(report_ts=('report_ts', 'first'), cutoff=('cutoff', 'first')).reset_index()
    nys = r[r['not_submitted'].astype(bool)].groupby(['gameId', 'teamId']).size().rename('nys').reset_index()
    cov = games.merge(schedule[['gameId', 'teamId']], on='gameId', how='inner')
    cov = cov.merge(nys, on=['gameId', 'teamId'], how='left')
    cov['not_submitted'] = cov['nys'].notna()
    return r, cov.drop(columns='nys')


def load_schedule(db_path, upcoming=None):
    """(gameId, teamId) for both teams of every game in team_game_full (+ optional upcoming games)."""
    with sqlite3.connect(db_path) as conn:
        sched = pd.read_sql("SELECT DISTINCT gameId, teamId FROM team_game_full", conn)
    if upcoming is not None and len(upcoming):
        sched = pd.concat([sched, upcoming[['gameId', 'teamId']]], ignore_index=True).drop_duplicates()
    return sched


# ---------------------------------------------------------------- 4. features

def status_play_rates(rows, cand):
    """Empirical P(played | status) for rotation players, from linked report rows."""
    m = rows.dropna(subset=['personId']).merge(
        cand[['gameId', 'teamId', 'personId', 'played', 'rotation']], on=['gameId', 'teamId', 'personId'], how='inner')
    m = m[m['rotation']]
    return m.groupby('status')['played'].agg(['mean', 'size'])


def reported_features(rows, coverage, cand, play_rates):
    """
    Per (gameId, teamId): value of rotation players expected to miss the game according to the
    snapshot, weighted by (1 - P(play | status)).
      rep_lost       all reported rotation players
      rep_lost_new   ...who played the team's previous game (fresh news)
      rep_n_out      rotation players listed Out
      rep_missing    team had no report yet (no snapshot or 'not yet submitted') -> features are 0
    """
    p_play = play_rates['mean'].to_dict()
    r = rows.dropna(subset=['personId', 'status']).copy()
    r['p_miss'] = 1 - r['status'].map(p_play).fillna(1.0)
    r = r.drop_duplicates(['gameId', 'teamId', 'personId'])
    m = cand.merge(r[['gameId', 'teamId', 'personId', 'status', 'p_miss']], on=['gameId', 'teamId', 'personId'], how='inner')
    m = m[m['rotation']]
    m['lost'] = m['value'] * m['p_miss']
    m['lost_new'] = np.where(m['gap'] == 1, m['lost'], 0.0)
    m['is_out'] = m['status'] == 'Out'
    feats = m.groupby(['gameId', 'teamId']).agg(rep_lost=('lost', 'sum'), rep_lost_new=('lost_new', 'sum'),
                                                rep_n_out=('is_out', 'sum')).reset_index()

    base = cand[['gameId', 'teamId']].drop_duplicates().merge(coverage, on=['gameId', 'teamId'], how='left')
    base = base.merge(feats, on=['gameId', 'teamId'], how='left')
    base['rep_missing'] = base['report_ts'].isna() | base['not_submitted'].eq(True)
    for c in ('rep_lost', 'rep_lost_new', 'rep_n_out'):
        base[c] = base[c].fillna(0.0)
    return base


# ---------------------------------------------------------------- 5. model features (training + live)

# Clock cutoffs (ET) relative to the game date, each capped at 30 min before tip so a report
# published after tip-off is never used (early-afternoon tips would otherwise get a 6 PM report).
SNAPSHOT_CLOCK = {
    'night_before_8pm': pd.Timedelta(hours=-4),
    'game_day_noon': pd.Timedelta(hours=12),
    'game_day_6pm': pd.Timedelta(hours=18),
    'tip_minus_30m': None,
}


def _capped(offset):
    def cutoff(g):
        pre_tip = g['tip_ts'] - pd.Timedelta(minutes=30)
        if offset is None:
            return pre_tip
        clock = g['game_date'] + offset
        return clock.where(pre_tip.isna() | (clock <= pre_tip), pre_tip)
    return cutoff


SNAPSHOT_CUTOFFS = {name: _capped(off) for name, off in SNAPSHOT_CLOCK.items()}
INJ_COLS = ['inj_lost', 'inj_lost_new', 'inj_n_out', 'inj_missing']
HISTORY_START_SEASON = '2019-20'  # first season with official reports; fixed so values are reproducible

# Report slots on the canonical download grid (scripts/scrape_injury_reports.SLOTS), as
# (hour24, minute) of the filename stamp. Training uses only these files.
GRID_SLOTS = {(11, 0), (12, 0), (13, 0), (17, 0), (18, 0), (19, 0), (20, 0)}


def report_slot(source_file):
    """'Injury-Report_2024-01-15_05PM.pdf' / '..._05_30PM.pdf' -> (17, 0) / (17, 30)."""
    m = re.search(r'_(\d{2})(?:_(\d{2}))?(AM|PM)\.pdf$', str(source_file))
    if not m:
        return None
    hour = int(m.group(1)) % 12 + (12 if m.group(3) == 'PM' else 0)
    return (hour, int(m.group(2) or 0))


def build_injury_features(db_path, seasons, snapshot_name='game_day_noon', as_of=None, upcoming=None,
                          reports=None):
    """
    Per (gameId, teamId) model features from the official injury report, as known at a snapshot.
    Used identically for training (historical games) and live inference (pass `upcoming` games
    and `as_of` = now), so the two cannot drift apart.

      inj_lost       value of rotation players expected to miss (value x P(miss | status))
      inj_lost_new   ...of those, players who played the team's previous game
      inj_n_out      rotation players listed Out
      inj_missing    1 if no report covered the team at the snapshot (features are then 0)

    snapshot_name: key of SNAPSHOT_CUTOFFS. as_of: optional timestamp; the cutoff becomes
    min(snapshot cutoff, as_of) - for live use, as_of = now.
    P(play | status) for a season is estimated from report rows of earlier seasons only
    (the earliest season, which has none, uses STATUS_PRIOR - never later data).
    """
    reports = parse_all_reports(report_dir=Path('nonexistent')) if reports is None else reports
    if as_of is None:
        # Training: only reports on the fixed download grid, so the result does not depend on
        # which extra files live runs happened to save (reproducible retraining).
        on_grid = {f: report_slot(f) in GRID_SLOTS for f in reports['source_file'].unique()}
        reports = reports[reports['source_file'].map(on_grid).astype(bool)]
    linked = link_reports(reports, db_path, upcoming=upcoming)
    schedule = load_schedule(db_path, upcoming=upcoming)
    from core.availability_features import build_player_game_candidates
    # Always load player history from the same start season, whatever `seasons` is requested:
    # player ratings are EWMAs carried across seasons and centered on the prior league mean, so a
    # different start would change values (training vs live mismatch).
    with sqlite3.connect(db_path) as conn:
        db_seasons = pd.read_sql("SELECT DISTINCT seasonYear FROM team_game_full", conn)['seasonYear'].tolist()
    last = max(seasons)
    history = sorted(s for s in set(db_seasons) | set(seasons) if HISTORY_START_SEASON <= s <= last)
    cand, _ = build_player_game_candidates(db_path, history, upcoming=upcoming)
    cand = cand[cand['seasonYear'].isin(seasons)]

    base_cutoff = SNAPSHOT_CUTOFFS[snapshot_name]
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        cutoff = lambda g: base_cutoff(g).clip(upper=as_of)
    else:
        cutoff = base_cutoff
    rows, cov = snapshot(linked, cutoff, schedule)

    rate_table = load_status_play_rates(db_path, reports)
    out = []
    for season in seasons:
        rates = rates_for_season(rate_table, season)
        s_rows = rows[rows['seasonYear'] == season]
        s_cov = cov[cov['gameId'].isin(cand.loc[cand['seasonYear'] == season, 'gameId'])]
        out.append(reported_features(s_rows, s_cov, cand[cand['seasonYear'] == season], rates))
    f = pd.concat(out, ignore_index=True)
    f = f.rename(columns={'rep_lost': 'inj_lost', 'rep_lost_new': 'inj_lost_new', 'rep_n_out': 'inj_n_out',
                          'rep_missing': 'inj_missing'})
    # Every scheduled team-game of the requested seasons gets a row. Teams with no player history
    # yet (their first game of the season) have no candidates, so they are marked missing - in
    # training and live alike.
    with sqlite3.connect(db_path) as conn:
        sched = pd.read_sql("SELECT DISTINCT gameId, teamId, seasonYear FROM team_game_full", conn)
    if upcoming is not None and len(upcoming):
        sched = pd.concat([sched, upcoming[['gameId', 'teamId', 'seasonYear']]]).drop_duplicates(['gameId', 'teamId'])
    sched = sched[sched['seasonYear'].isin(seasons)][['gameId', 'teamId']]
    sched['teamId'] = sched['teamId'].astype('int64')
    f['teamId'] = f['teamId'].astype('int64')
    f = sched.merge(f, on=['gameId', 'teamId'], how='left')
    f['inj_missing'] = f['inj_missing'].ne(False).astype(int)
    for c in ('inj_lost', 'inj_lost_new', 'inj_n_out'):
        f[c] = f[c].fillna(0.0)
    return f[['gameId', 'teamId'] + INJ_COLS]


STATUS_RATES_PATH = Path('models/saved/injury_status_play_rates.json')


# P(play | status) used for the first season, which has no earlier data to estimate from (it used to
# borrow the pooled rates of all LATER seasons, i.e. the future). These are the conventional meanings
# of the NBA report categories (Out ~0, Doubtful ~25%, Questionable ~50%, Probable ~75%, Available ~100%);
# n=0 marks them as a prior, not an estimate.
STATUS_PRIOR = {'Out': 0.0, 'Doubtful': 0.25, 'Questionable': 0.50, 'Probable': 0.75, 'Available': 1.0}


def compute_status_play_rates(db_path, reports=None, snapshot_name='tip_minus_30m'):
    """
    P(played | status) per season, estimated from EARLIER seasons only (the first season, which
    has no earlier data, uses STATUS_PRIOR), plus 'all' = every season pooled (used for seasons
    after the data, i.e. live). Saved to STATUS_RATES_PATH. Rates come from the latest report
    before tip, where a status is closest to the final decision.
    """
    import json
    reports = parse_all_reports(report_dir=Path('nonexistent')) if reports is None else reports
    linked = link_reports(reports, db_path)
    seasons = sorted(linked['seasonYear'].dropna().unique())
    from core.availability_features import build_player_game_candidates
    cand, _ = build_player_game_candidates(db_path, seasons)
    rows, _ = snapshot(linked, SNAPSHOT_CUTOFFS[snapshot_name], load_schedule(db_path))
    per_season = {s: status_play_rates(rows[rows['seasonYear'] == s], cand) for s in seasons}

    def pooled(keys):
        df = pd.concat([per_season[k].assign(played=lambda d: d['mean'] * d['size']) for k in keys])
        g = df.groupby(level=0)[['played', 'size']].sum()
        return {st: {'p_play': float(r.played / r['size']), 'n': int(r['size'])} for st, r in g.iterrows()}

    table = {s: pooled([k for k in seasons if k < s]) if any(k < s for k in seasons)
             else {st: {'p_play': p, 'n': 0} for st, p in STATUS_PRIOR.items()}
             for s in seasons}
    table['all'] = pooled(seasons)
    STATUS_RATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_RATES_PATH.write_text(json.dumps(table, indent=2))
    logging.info(f"Saved status play rates for {len(seasons)} seasons to {STATUS_RATES_PATH}")
    return table


def load_status_play_rates(db_path, reports=None):
    import json
    if STATUS_RATES_PATH.exists():
        return json.loads(STATUS_RATES_PATH.read_text())
    return compute_status_play_rates(db_path, reports)


def rates_for_season(table, season):
    """Rates for a season as a DataFrame indexed by status (column 'mean'); fails loudly if empty."""
    entry = table.get(season, table['all'])
    rates = pd.DataFrame({'mean': {k: v['p_play'] for k, v in entry.items()},
                          'size': {k: v['n'] for k, v in entry.items()}})
    if rates.empty or 'Out' not in rates.index:
        raise ValueError(f"No status play rates available for season {season}")
    return rates


def add_injury_features(game_df, inj):
    """Attach *_home / *_away injury columns to a one-row-per-game frame."""
    g = game_df.copy()
    g['_gid'] = g['gameId'].astype(str).str.zfill(10)
    inj = inj.copy()
    inj['_gid'] = inj['gameId'].astype(str).str.zfill(10)
    inj = inj.drop(columns='gameId')
    for side in ('home', 'away'):
        tid = f'teamId_{side}'
        side_df = inj.rename(columns={'teamId': tid, **{c: f'{c}_{side}' for c in INJ_COLS}})
        side_df[tid] = side_df[tid].astype('int64')
        g[tid] = g[tid].astype('int64')
        n = len(g)
        g = g.merge(side_df, on=['_gid', tid], how='left', validate='many_to_one')     # one report row per (game, team)
        if len(g) != n:
            raise ValueError(f"joining {side} injury features changed the row count ({n} -> {len(g)})")
    # diff_inj_* are created with every other home/away difference by
    # GameFeatureBuilder.add_derived_features (call this BEFORE that step).
    return g.drop(columns='_gid')
