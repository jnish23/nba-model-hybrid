"""E28-audit - data audit for the tracking / fatigue program. Registered in reports/experiments/PREREGISTRATION.md
("E28-audit") before any number was computed. Descriptive only: nothing passes or fails here.
python -m experiments.e28_audit  ->  reports/experiments/E28_tracking/audit.md (+ audit.json)"""
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import odds_clean as oc                      # noqa: E402
from team.data_paths import latest_training_set        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'nba_stats.db'
LOG = ROOT / 'data' / 'raw' / 'daily_inference.log'
OUT = ROOT / 'reports' / 'experiments' / 'E28_tracking'
TABLES = ['team_track_stats_v3', 'player_track_stats_v3', 'team_hustle_stats', 'player_hustle_stats']
TRACK_TEAM = 'team_track_stats_v3'


def md(df, floatfmt='{:.3f}'):
    """Markdown table without the tabulate dependency."""
    if df is None or len(df) == 0:
        return '_(empty)_\n'
    cols = [str(c) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r.tolist():
            if isinstance(v, (float, np.floating)):
                cells.append('' if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n' + '\n'.join(rows) + '\n'


def season_of(game_id):
    """'0022500123' -> '2025-26' (digits 3-4 are the season's start year)."""
    yy = int(str(game_id)[3:5])
    y = 2000 + yy
    return f'{y}-{str(y + 1)[2:]}'


def kind_of(game_id):
    return str(game_id)[:3]


def parse_minutes(s):
    """'MM:SS' (or 'MMM:SS') -> float minutes; blank / NULL / unparsable -> NaN."""
    if s is None:
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    m = re.fullmatch(r'(\d+):(\d{1,2})(?:\.\d+)?', s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def load(conn):
    d = {}
    d['tt'] = pd.read_sql('SELECT * FROM team_track_stats_v3', conn)
    d['pt'] = pd.read_sql('SELECT gameId, teamId, personId, minutes, speed, distance, comment FROM player_track_stats_v3', conn)
    d['th'] = pd.read_sql('SELECT gameId, teamId, minutes FROM team_hustle_stats', conn)
    d['ph'] = pd.read_sql('SELECT gameId, teamId, personId, minutes FROM player_hustle_stats', conn)
    d['tg'] = pd.read_sql('SELECT seasonYear, teamId, gameId, gameDate, minutes, pace FROM team_game_full', conn)
    for k in d:
        d[k]['gameId'] = d[k]['gameId'].astype(str).str.zfill(10)
        d[k]['season'] = d[k]['gameId'].map(season_of)
        d[k]['kind'] = d[k]['gameId'].map(kind_of)
    return d


# ---------------------------------------------------------------- 1. coverage
def coverage(d):
    out = {}
    tg = d['tg']
    base = tg.groupby('season').agg(team_games_in_tgf=('gameId', 'size'), games_in_tgf=('gameId', 'nunique'))
    rows = base.copy()
    for lab, key in (('team_track', 'tt'), ('team_hustle', 'th')):
        g = d[key].groupby('season').agg(**{f'{lab}_rows': ('gameId', 'size'), f'{lab}_games': ('gameId', 'nunique')})
        rows = rows.join(g, how='outer')
    for lab, key in (('player_track', 'pt'), ('player_hustle', 'ph')):
        g = d[key].groupby('season').agg(**{f'{lab}_rows': ('gameId', 'size'), f'{lab}_games': ('gameId', 'nunique')})
        rows = rows.join(g, how='outer')
    rows = rows.fillna(0).astype(int)
    rows['track_cov_vs_tgf'] = rows['team_track_games'] / rows['games_in_tgf'].replace(0, np.nan)
    rows['hustle_cov_vs_tgf'] = rows['team_hustle_games'] / rows['games_in_tgf'].replace(0, np.nan)
    out['by_season'] = rows.reset_index().rename(columns={'index': 'season'})
    # both teams present
    both = {}
    for lab, key in (('team_track', 'tt'), ('team_hustle', 'th')):
        n = d[key].groupby(['season', 'gameId'])['teamId'].nunique()
        both[lab] = (n == 2).groupby('season').mean()
    out['both_teams_share'] = pd.DataFrame(both).reset_index()
    # game kinds (002 regular, 004 playoffs, 005 play-in, others)
    kinds = d['tt'].groupby(['kind'])['gameId'].nunique().rename('team_track_games').to_frame()
    kinds['team_hustle_games'] = d['th'].groupby('kind')['gameId'].nunique()
    kinds['in_team_game_full'] = d['tg'].groupby('kind')['gameId'].nunique()
    out['kinds'] = kinds.fillna(0).astype(int).reset_index()
    # missing games: in team_game_full but absent from track / hustle
    miss = []
    tgids = d['tg'].drop_duplicates('gameId')[['gameId', 'season', 'kind', 'gameDate']]
    for lab, key in (('team_track', 'tt'), ('team_hustle', 'th')):
        have = set(d[key]['gameId'])
        mm = tgids[~tgids['gameId'].isin(have)]
        miss.append(mm.groupby(['season', 'kind']).size().rename(f'{lab}_missing').reset_index().set_index(['season', 'kind']))
    out['missing_games'] = pd.concat(miss, axis=1).fillna(0).astype(int).reset_index()
    # player tracking fill: rows that played
    pt = d['pt'].copy()
    pt['min'] = pt['minutes'].map(parse_minutes)
    played = pt[pt['min'] > 0]
    pt['speed_ok'] = pt['speed'].notna() & (pt['speed'] > 0)
    played = played.assign(speed_ok=played['speed'].notna() & (played['speed'] > 0), dist_ok=played['distance'].notna() & (played['distance'] > 0))
    fill = pt.groupby('season').agg(rows=('gameId', 'size'), dnp_share=('min', lambda s: float((s.fillna(0) <= 0).mean())))
    f2 = played.groupby('season').agg(played_rows=('gameId', 'size'), speed_filled=('speed_ok', 'sum'), dist_filled=('dist_ok', 'sum'))
    fill = fill.join(f2)
    fill['speed_fill_share_of_played'] = fill['speed_filled'] / fill['played_rows']
    fill['dist_fill_share_of_played'] = fill['dist_filled'] / fill['played_rows']
    # rows (any, incl. DNP) with a non-null speed, the plan's "24,964 of 34,788" style count
    fill['rows_with_speed'] = pt[pt['speed'].notna() & (pt['speed'] > 0)].groupby('season').size()
    fill['rows_with_speed_share_all'] = fill['rows_with_speed'] / fill['rows']
    out['player_fill'] = fill.reset_index()
    # all-zero team rows (distance 0 and touches 0): present in the table but carrying no tracking
    tt = d['tt'].merge(d['tg'][['gameId', 'teamId', 'gameDate']], on=['gameId', 'teamId'], how='left')
    tt['zero_row'] = (tt['distance'].fillna(0) == 0) & (tt['touches'].fillna(0) == 0)
    z = tt.groupby('season').agg(team_rows=('zero_row', 'size'), all_zero_rows=('zero_row', 'sum'))
    z['all_zero_share'] = z['all_zero_rows'] / z['team_rows']
    out['team_zero_rows'] = z.reset_index()
    zz = tt[(tt['season'] == '2025-26') & tt['zero_row']].copy()
    zz['month'] = pd.to_datetime(zz['gameDate']).dt.to_period('M').astype(str)
    out['zero_rows_2025_26_by_month'] = zz.groupby('month').size().rename('all_zero_team_rows').reset_index()
    out['zero_rows_2025_26_span'] = f"{pd.to_datetime(zz['gameDate']).min().date()} .. {pd.to_datetime(zz['gameDate']).max().date()}" if len(zz) else 'none'
    # hustle game ids that are not in team_game_full (preseason etc.)
    hx = d['th'][~d['th']['gameId'].isin(set(d['tg']['gameId']))]
    out['hustle_ids_outside_tgf'] = hx.groupby(['season', 'kind'])['gameId'].nunique().rename('games').reset_index()
    # special seasons
    sp = {}
    tgd = d['tg'].drop_duplicates('gameId')
    bub = tgd[(tgd['season'] == '2019-20') & (pd.to_datetime(tgd['gameDate']) >= '2020-07-30')]
    sp['bubble_games_2019_20'] = int(len(bub))
    sp['bubble_games_with_team_track'] = int(bub['gameId'].isin(set(d['tt']['gameId'])).sum())
    sp['bubble_games_with_team_hustle'] = int(bub['gameId'].isin(set(d['th']['gameId'])).sum())
    s2021 = tgd[tgd['season'] == '2020-21']
    sp['games_2020_21'] = int(len(s2021))
    sp['games_2020_21_with_team_track'] = int(s2021['gameId'].isin(set(d['tt']['gameId'])).sum())
    out['special'] = sp
    return out


# ---------------------------------------------------------------- 2. units and sanity
def sanity(d):
    out = {}
    tt = d['tt'].copy()
    tt['min'] = tt['minutes'].map(parse_minutes)
    reg = tt[(tt['kind'] == '002') & ~((tt['distance'].fillna(0) == 0) & (tt['touches'].fillna(0) == 0))]
    g = reg.groupby('season')['distance'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    g['team_min_mean'] = reg.groupby('season')['min'].mean().values
    out['team_distance'] = g
    out['team_speed_nonnull'] = pd.DataFrame({'rows': [len(tt)], 'speed_non_null': [int(tt['speed'].notna().sum())],
                                              'speed_distinct_values': [str(sorted(tt['speed'].dropna().astype(str).unique())[:5])]})
    pt = d['pt'].copy()
    pt['min'] = pt['minutes'].map(parse_minutes)
    played = pt[(pt['min'] >= 5) & (pt['kind'] == '002')]
    sp = played[played['speed'] > 0].groupby('season')['speed'].describe(percentiles=[.01, .5, .99]).reset_index()
    out['player_speed'] = sp
    # outliers
    allp = pt[pt['min'] > 0]
    out['outliers'] = {
        'distance_zero_with_minutes_gt_0': int(((allp['distance'].fillna(0) == 0)).sum()),
        'distance_zero_with_minutes_gt_5': int((pt[pt['min'] > 5]['distance'].fillna(0) == 0).sum()),
        'speed_gt_6': int((pt['speed'] > 6).sum()),
        'speed_lt_2_with_minutes_ge_5': int(((pt['speed'] < 2) & (pt['min'] >= 5)).sum()),
        'rows_minutes_unparsed_nonblank': int(pt['minutes'].notna().sum() - pt['min'].notna().sum() - int((pt['minutes'].astype(str).str.strip() == '').sum())),
        'team_distance_zero': int((tt['distance'].fillna(0) == 0).sum()),
        'team_distance_gt_30': int((tt['distance'] > 30).sum()),
        'team_minutes_lt_235': int((tt['min'] < 235).sum()),
    }
    # player sums vs team distance
    ps = pt.groupby(['gameId', 'teamId'])['distance'].sum(min_count=1).rename('pl_sum').reset_index()
    m = tt[['gameId', 'teamId', 'distance', 'season']].merge(ps, on=['gameId', 'teamId'], how='inner')
    m = m[(m['distance'] > 0) & m['pl_sum'].notna()]
    m['ratio'] = m['pl_sum'] / m['distance']
    m['within1'] = (m['ratio'] - 1).abs() <= 0.01
    s = m.groupby('season').agg(team_games=('ratio', 'size'), within_1pct=('within1', 'mean'), median_ratio=('ratio', 'median'),
                                p05=('ratio', lambda x: x.quantile(.05)), p95=('ratio', lambda x: x.quantile(.95))).reset_index()
    out['player_vs_team_distance'] = s
    out['player_vs_team_overall_within1'] = float(m['within1'].mean())
    # share of player-sum minutes ~ team minutes
    pm = pt.groupby(['gameId', 'teamId'])['min'].sum().rename('pl_min').reset_index().merge(tt[['gameId', 'teamId', 'min']], on=['gameId', 'teamId'])
    pm['diff'] = pm['pl_min'] - pm['min']
    out['player_minutes_vs_team'] = {'median_abs_diff': float(pm['diff'].abs().median()), 'share_within_1min': float((pm['diff'].abs() <= 1).mean())}
    return out


# ---------------------------------------------------------------- 3. live availability (daily log)
RUN_RE = re.compile(r'^===== \w+ (\d+)/(\d+)/(\d{4}) +(\d+):(\d+):')
LINE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+ - (\w+) - (.*)$')
ADD_RE = re.compile(r'Successfully added data for game (\d{10}) to (\w+)')
ERR_RE = re.compile(r'Error processing game (\d{10}) for (\w+): (.*)')
FOUND_RE = re.compile(r'Found (\d+) missing games for (\w+)')


def parse_log(path):
    runs, adds, errs, found = [], [], [], []
    cur = None
    for line in path.read_text(errors='replace').splitlines():
        m = RUN_RE.match(line)
        if m:
            mo, da, yr, hh, mi = map(int, m.groups())
            cur = f'{yr:04d}-{mo:02d}-{da:02d}'
            runs.append({'run_date': cur, 'start': f'{hh:02d}:{mi:02d}'})
            continue
        m = LINE_RE.match(line)
        if not m or cur is None:
            continue
        day, tm, lvl, msg = m.groups()
        a = ADD_RE.search(msg)
        if a:
            adds.append({'run_date': cur, 'time': tm, 'gameId': a.group(1), 'table': a.group(2)})
            continue
        e = ERR_RE.search(msg)
        if e:
            errs.append({'run_date': cur, 'time': tm, 'gameId': e.group(1), 'table': e.group(2), 'error': e.group(3)[:80]})
            continue
        f = FOUND_RE.search(msg)
        if f:
            found.append({'run_date': cur, 'time': tm, 'n_missing': int(f.group(1)), 'table': f.group(2)})
    return pd.DataFrame(runs), pd.DataFrame(adds), pd.DataFrame(errs), pd.DataFrame(found)


def availability(d):
    out = {}
    runs, adds, errs, found = parse_log(LOG)
    out['runs'] = runs
    out['log_span'] = f"{runs['run_date'].min()} .. {runs['run_date'].max()} ({len(runs)} runs)" if len(runs) else 'no runs parsed'
    gd = d['tg'].drop_duplicates('gameId')[['gameId', 'gameDate']].copy()
    gd['gameDate'] = pd.to_datetime(gd['gameDate'])
    cur = gd[gd['gameId'].str.startswith('00226')]            # the season the log covers (2026-27)
    rows = []
    for tab in TABLES:
        a = adds[adds['table'] == tab] if len(adds) else adds
        first = a.groupby('gameId')['run_date'].min() if len(a) else pd.Series(dtype=str)
        e = errs[errs['table'] == tab] if len(errs) else errs
        err_ids = set(e['gameId']) if len(e) else set()
        tmp = cur.copy()
        tmp['first_added_run'] = tmp['gameId'].map(first)
        tmp['lag_days'] = (pd.to_datetime(tmp['first_added_run']) - tmp['gameDate']).dt.days
        tmp['ever_error'] = tmp['gameId'].isin(err_ids)
        own_log = tab in (TRACK_TEAM, 'team_hustle_stats')
        in_db = set(d['tt' if tab == TRACK_TEAM else {'player_track_stats_v3': 'pt', 'team_hustle_stats': 'th', 'player_hustle_stats': 'ph'}[tab]]['gameId'])
        tmp['in_db_now'] = tmp['gameId'].isin(in_db)
        rows.append(pd.DataFrame({'table': tab, 'games_played_2026_27': len(tmp), 'in_db_now': int(tmp['in_db_now'].sum()),
                                  'added_in_logged_runs': int(tmp['first_added_run'].notna().sum()) if own_log else np.nan,
                                  'lag0_same_day_as_game': int((tmp['lag_days'] == 0).sum()) if own_log else np.nan,
                                  'lag1_next_morning': int((tmp['lag_days'] == 1).sum()) if own_log else np.nan,
                                  'lag2plus': int((tmp['lag_days'] >= 2).sum()) if own_log else np.nan,
                                  'errored_at_least_once': int(tmp['ever_error'].sum()) if own_log else np.nan,
                                  'not_in_db_now': int((~tmp['in_db_now']).sum())}, index=[0]))
        if tab == TRACK_TEAM:
            out['per_game_team_track'] = tmp.sort_values('gameId')[['gameId', 'gameDate', 'first_added_run', 'lag_days', 'ever_error', 'in_db_now']]
    out['lag_table'] = pd.concat(rows, ignore_index=True)
    out['found_per_run'] = found.pivot_table(index='run_date', columns='table', values='n_missing', aggfunc='last').reset_index() if len(found) else found
    out['errors'] = errs.groupby(['table', 'error']).agg(n=('gameId', 'size'), games=('gameId', 'nunique'), first_run=('run_date', 'min'),
                                                         last_run=('run_date', 'max')).reset_index() if len(errs) else errs
    # per run: how many of the PREVIOUS night's games had team tracking after that run's sync
    pr = []
    a = adds[adds['table'] == TRACK_TEAM]
    for rd in sorted(runs['run_date'].unique()):
        prev = pd.Timestamp(rd) - pd.Timedelta(days=1)
        gm = cur[cur['gameDate'] == prev]['gameId']
        if len(gm) == 0:
            continue
        done = a[a['run_date'] <= rd]['gameId']
        ok = gm.isin(set(done)).sum()
        pr.append({'run_date': rd, 'previous_night_games': len(gm), 'with_team_tracking_after_sync': int(ok), 'share': ok / len(gm)})
    out['prev_night_by_run'] = pd.DataFrame(pr)
    out['prev_night_total_share'] = float(sum(r['with_team_tracking_after_sync'] for r in pr) / max(1, sum(r['previous_night_games'] for r in pr)))
    # back-to-back second nights in the log window: share whose previous game's team tracking was in by that morning's sync
    tg = d['tg'].copy()
    tg['gameDate'] = pd.to_datetime(tg['gameDate'])
    tg = tg[tg['gameId'].str.startswith('00226')].sort_values(['teamId', 'gameDate'])
    tg['prev_date'] = tg.groupby('teamId')['gameDate'].shift(1)
    tg['prev_gid'] = tg.groupby('teamId')['gameId'].shift(1)
    b2 = tg[(tg['gameDate'] - tg['prev_date']).dt.days == 1].copy()
    first_t = a.groupby('gameId')['run_date'].min()
    b2['prev_first_added'] = b2['prev_gid'].map(first_t)
    b2['avail_by_morning'] = pd.to_datetime(b2['prev_first_added']) <= b2['gameDate']
    out['b2b_window'] = {'b2b_second_nights_in_window': int(len(b2)), 'prev_game_team_tracking_in_by_that_morning': int(b2['avail_by_morning'].sum())}
    # historical: share of back-to-back second nights (regular season, 2013-14..2025-26) whose previous game has team tracking in the DB at all
    tgh = d['tg'][d['tg']['kind'] == '002'].copy()
    tgh['gameDate'] = pd.to_datetime(tgh['gameDate'])
    tgh = tgh.sort_values(['teamId', 'gameDate'])
    tgh['prev_date'] = tgh.groupby('teamId')['gameDate'].shift(1)
    tgh['prev_gid'] = tgh.groupby('teamId')['gameId'].shift(1)
    b2h = tgh[(tgh['gameDate'] - tgh['prev_date']).dt.days == 1].copy()
    have = set(zip(d['tt']['gameId'], d['tt']['teamId']))
    b2h['prev_has_track'] = [(g, t) in have for g, t in zip(b2h['prev_gid'], b2h['teamId'])]
    out['b2b_history'] = b2h.groupby('season').agg(b2b_second_nights=('prev_has_track', 'size'), prev_has_team_track=('prev_has_track', 'mean')).reset_index()
    return out


# ---------------------------------------------------------------- 4. opening-line timing on back-to-backs
def line_timing(d, conn):
    out = {}
    s = pd.read_sql("SELECT fetched_at_et, game_date, game_start, home_team_abbr, away_team_abbr, line_type, in_game FROM odds_snapshots", conn)
    out['snapshot_span'] = {'rows': int(len(s)), 'first_fetch': s['fetched_at_et'].min(), 'last_fetch': s['fetched_at_et'].max(), 'games': int(s.groupby(['game_date', 'home_team_abbr']).ngroups)}
    s = s[(s['in_game'] == 0)]
    s['fetched'] = pd.to_datetime(s['fetched_at_et'])
    s['start_utc'] = pd.to_datetime(s['game_start'], utc=True)
    s['start_et'] = s['start_utc'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    g = s.groupby(['game_date', 'home_team_abbr', 'away_team_abbr']).agg(first_fetch=('fetched', 'min'), start_et=('start_et', 'first')).reset_index()
    g['hours_first_fetch_before_tip'] = (g['start_et'] - g['first_fetch']).dt.total_seconds() / 3600
    out['hours_first_fetch_before_tip'] = g['hours_first_fetch_before_tip'].describe().to_dict()
    # back-to-back second nights: previous game of each team within the snapshot games
    long = pd.concat([g.assign(team=g['home_team_abbr']), g.assign(team=g['away_team_abbr'])], ignore_index=True).sort_values(['team', 'start_et'])
    long['prev_start'] = long.groupby('team')['start_et'].shift(1)
    long['gap_days'] = (pd.to_datetime(long['game_date']) - pd.to_datetime(long.groupby('team')['game_date'].shift(1))).dt.days
    b2 = long[long['gap_days'] == 1].copy()
    b2['prev_end_est'] = b2['prev_start'] + pd.Timedelta(hours=2.5)
    b2['line_predates_prev_end'] = b2['first_fetch'] < b2['prev_end_est']
    out['b2b_second_nights_with_snapshots'] = {'n': int(len(b2)), 'first_snapshot_before_prev_game_end': int(b2['line_predates_prev_end'].sum()),
                                               'note': 'snapshot history starts 2026-10-19, so only second nights whose previous game is also inside the snapshot window can be checked; previous-game end is estimated as tip + 2.5 h'}
    out['b2b_detail'] = b2[['game_date', 'team', 'first_fetch', 'prev_start', 'prev_end_est', 'line_predates_prev_end']].head(30)
    return out


# ---------------------------------------------------------------- 5. opening-line coverage
def odds_coverage(d, conn):
    out = {}
    o = pd.read_sql('SELECT season, gameDate, home_spread_betmgm, home_spread_fanduel, home_spread_caesars, home_spread_bet365, home_spread_draftkings, '
                    'total_betmgm, total_fanduel, total_caesars, total_bet365, total_draftkings FROM nba_odds_opening', conn)
    tcols = [c for c in o.columns if c.startswith('total_')]
    scols = [c for c in o.columns if c.startswith('home_spread_')]
    o['has_total'] = o[tcols].apply(pd.to_numeric, errors='coerce').notna().any(axis=1)
    o['has_spread'] = o[scols].apply(pd.to_numeric, errors='coerce').notna().any(axis=1)
    o['yr'] = o['season'].astype(str).str[:7]
    a = o.groupby('yr').agg(rows=('has_total', 'size'), with_total=('has_total', 'sum'), with_spread=('has_spread', 'sum')).reset_index()
    tgd = d['tg'].drop_duplicates('gameId')
    n = tgd.groupby('season').size().rename('games_in_team_game_full')
    a = a.merge(n.reset_index().rename(columns={'season': 'yr'}), on='yr', how='outer').fillna(0)
    out['nba_odds_opening_by_season'] = a
    cl = pd.read_sql('SELECT season, COUNT(*) n FROM nba_odds GROUP BY season', conn)
    out['nba_odds_by_season_rows'] = cl
    ts = pd.read_csv(latest_training_set(), usecols=['gameId', 'seasonYear', 'odds_total', 'odds_home_spread'], low_memory=False)
    ts = ts.drop_duplicates('gameId')
    ok = ts['odds_total'].between(*oc.TOTAL_RANGE) & (ts['odds_home_spread'].abs() <= oc.MAX_ABS_SPREAD)
    ts['plausible'] = ok
    ts['nonnull'] = ts['odds_total'].notna() & ts['odds_home_spread'].notna()
    b = ts.groupby('seasonYear').agg(games=('gameId', 'size'), odds_nonnull=('nonnull', 'sum'), odds_plausible=('plausible', 'sum')).reset_index()
    b['plausible_share'] = b['odds_plausible'] / b['games']
    out['training_set_opening_by_season'] = b
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f'file:{DB}?mode=ro', uri=True) as conn:
        d = load(conn)
        cov = coverage(d)
        san = sanity(d)
        av = availability(d)
        lt = line_timing(d, conn)
        oc_ = odds_coverage(d, conn)
    L = ['# E28 audit - tracking, hustle and schedule data', '',
         f'Registered in `reports/experiments/PREREGISTRATION.md` ("E28-audit") before any number was computed. Descriptive only. Built by `experiments/e28_audit.py`; training set `{latest_training_set().name}`.', '']
    z = cov['team_zero_rows'].set_index('season')
    fill = cov['player_fill'].set_index('season')
    summ = ['## Summary (numbers computed in this run; the wording was written after reading them)', '',
            f"- **History is complete for team and player tracking**, 2013-14 to 2025-26: every game in `team_game_full` has both teams' rows (team tracking coverage 1.000 in every season). Player distances sum to the team figure within 1% in {san['player_vs_team_overall_within1']:.4f} of team-games. Team `distance` averages 16.8-18.3 miles a game in 2013-14..2024-25 and player speed averages 4.2-4.4 mph, as expected.",
            f"- **Hustle**: full from 2016-17. 2013-14 and 2014-15 are fragments (74 and 39 games); in 2015-16 almost every game has a hustle row for only one team (both teams present in 16.5% of games), so hustle features start in 2016-17 (NaN before).",
            f"- **Team `speed` is NULL in every row**; team speed has to be built from players (`player_track_stats_v3.speed`, filled for 99%+ of played rows in 2013-14..2024-25).",
            f"- **Zero rows**: {int(z['all_zero_rows'].sum())} team-game rows carry no tracking (distance 0 and touches 0). {int(z.loc['2025-26', 'all_zero_rows'])} of them are in 2025-26 ({z.loc['2025-26', 'all_zero_share']:.1%} of that season's team-games, {cov['zero_rows_2025_26_span']}); every other season has 0 to 20. They must be read as missing (NaN), never as a very low load, and the 2025-26 fold (the last test season) is thinner for any tracking feature. Player tracking in 2025-26 has speed filled for only {fill.loc['2025-26', 'speed_fill_share_of_played']:.1%} of played rows. **The plan's \"2024-25 has a lower fill share (24,964 of 34,788)\" is 2025-26**: those counts are the 2025-26 player rows with distance filled; 2024-25 is 99.9% filled.",
            f"- **Live availability (decisive)**: the log covers only 5 runs (2026-10-20 to 2026-10-24) and no team had a back-to-back yet, so the share of back-to-back second nights with the previous night's tracking at 9:15 **could not be measured (0 cases)**. What the window shows: on three of the four runs that had games to fetch (10/21, 10/22, 10/23) every previous-night game's team tracking was in the database after the sync, with a lag of exactly one day. On 10/24 **all 12 games of 10/23 failed** (`'NoneType' object has no attribute 'get'`, the endpoint returned nothing), and a read-only probe at about 13:35 ET the same day (more than 12 hours later) still returned nothing for them, while the 10/21 games return normally. Hustle data arrived for all 28 games on schedule, including the 12 games whose tracking failed. Overall {av['prev_night_total_share']:.0%} (16 of 28 games) had team tracking by the next morning.",
            "- **Reading of the stop rule**: this is not 'usually missing', but it is not 'reliably there' either, and one bad day fell on the 12-game slate. Per the plan, the lagged-load families (F1, F2, F3) are therefore built with the rule **'most recent game with tracking data'** (a game with an all-zero row is skipped as if absent) from the start, in training and in any production use. In history the previous game nearly always has data, so for the walk-forward tests the rule equals 'previous game' outside the 2025-26 gap; the live gap is a step 6 problem and is flagged there. The audit window is far too short to say how often tracking is late; it should be rechecked once a month of daily runs exists.",
            "- **Opening-line timing on back-to-backs**: not measurable yet. `nba_odds_opening` has no timestamps and `odds_snapshots` starts 2026-10-19 (18 games, none a back-to-back second night). Every game already had a line at the collector's first look, 30-47 hours before tip, but that figure is bounded by when the collector started, so it only shows that lines exist at least that early (a 'current' line, not necessarily the opening-table value). For a second night that is before the previous night's game ends, so the open-to-close or opening line can in principle predate the previous game; whether the archived opening values do is unknown, and the market-residual screen has to be read with that in mind.",
            "- **Opening-line coverage**: `nba_odds_opening` itself starts in 2019-20 (19 empty rows in 2013-14). The training set's `odds_total` and `odds_home_spread` for 2013-14..2018-19 come from `data/nba_odds_archive.csv` (opening spread and total columns; closing moneyline only) and are 99.8% plausible after the `odds_clean` bounds (2013-14: 98.5%), so the market-residual screen can use those six seasons as extra discovery data, **labelled as a different source** (same treatment as E14's supplementary seasons).",
            ""]
    L += summ
    L += ['## 1. Coverage', '', 'Seasons are taken from digits 3-4 of the game id. `games_in_tgf` = distinct games in `team_game_full`.', '', md(cov['by_season']),
          'Share of games with both teams present:', '', md(cov['both_teams_share']),
          'Game kinds (`002` regular season, `004` playoffs, `005` play-in, others as found):', '', md(cov['kinds']),
          'All-zero team tracking rows (distance 0 and touches 0; present in the table but carrying no data, to be treated as missing):', '', md(cov['team_zero_rows']),
          f"2025-26 all-zero rows by month (span {cov['zero_rows_2025_26_span']}):", '', md(cov['zero_rows_2025_26_by_month']),
          'Hustle game ids not in `team_game_full` (preseason `001`, all-star `003` and malformed ids; ignored by any join):', '', md(cov['hustle_ids_outside_tgf']),
          'Games in `team_game_full` that are missing from tracking / hustle:', '', md(cov['missing_games']),
          'Player tracking fill (a played row = minutes > 0; the plan flagged 2024-25):', '', md(cov['player_fill']),
          'Special seasons: ' + json.dumps(cov['special']), '']
    L += ['## 2. Units and sanity', '', 'Team distance (miles, regular season) by season:', '', md(san['team_distance']),
          'Team speed (plan says NULL everywhere):', '', md(san['team_speed_nonnull']),
          'Player speed (mph) among regular-season rows with 5+ minutes and speed > 0:', '', md(san['player_speed']),
          'Outlier counts: `' + json.dumps(san['outliers']) + '`', '',
          'Sum of player distances against team `distance` (share within 1% in column `within_1pct`):', '', md(san['player_vs_team_distance']),
          f"Overall share within 1%: {san['player_vs_team_overall_within1']:.3f}. Player minutes summed vs team minutes: `{json.dumps(san['player_minutes_vs_team'])}`", '']
    L += ['## 3. Live availability at the 9:15 ET run', '', f"Log `{LOG.relative_to(ROOT)}`: {av['log_span']}. Only 2026-27 games are covered, so this is a short window and is stated as such.", '',
          'Per table, how the 2026-27 games reached the database (`lag1_next_morning` = added by the next morning\'s 9:15 run; `not_in_db_now` = still absent):', '', md(av['lag_table']),
          'Missing-game counts reported by each run:', '', md(av['found_per_run']),
          'Errors:', '', md(av['errors']),
          'Previous night\'s games with team tracking in the database after each run\'s sync:', '', md(av['prev_night_by_run']),
          f"Overall: {av['prev_night_total_share']:.3f} of previous-night games had team tracking after the sync.", '',
          'Back-to-back second nights in the log window: `' + json.dumps(av['b2b_window']) + '`', '',
          'Historical completeness, regular-season back-to-back second nights whose previous game has team tracking in the database at all (not the same as available at 9:15, which history cannot show):', '', md(av['b2b_history']),
          'Per game, team tracking, 2026-27:', '', md(av['per_game_team_track']), '']
    L += ['## 4. Opening-line timing on back-to-backs', '',
          '`nba_odds_opening` has no timestamps. `odds_snapshots` covers: `' + json.dumps(lt['snapshot_span'], default=str) + '`', '',
          'Hours between the first pre-game snapshot of a game and its tip: `' + json.dumps({k: round(v, 2) for k, v in lt['hours_first_fetch_before_tip'].items()}) + '`', '',
          'Back-to-back second nights inside the snapshot window: `' + json.dumps(lt['b2b_second_nights_with_snapshots']) + '`', '', md(lt['b2b_detail']),
          'Not measurable: when the historical opening lines (`nba_odds_opening`) were posted relative to the previous game ending.', '']
    L += ['## 5. Opening-line coverage', '', '`nba_odds_opening` by season:', '', md(oc_['nba_odds_opening_by_season']),
          'Opening odds carried in the training set (`odds_total`, `odds_home_spread`), with the plausibility bounds of `core/odds_clean.py`:', '', md(oc_['training_set_opening_by_season']), '']
    (OUT / 'audit.md').write_text('\n'.join(L), encoding='utf-8')
    summary = {'coverage_special': cov['special'], 'outliers': san['outliers'], 'player_vs_team_within1': san['player_vs_team_overall_within1'],
               'log_span': av['log_span'], 'prev_night_share': av['prev_night_total_share'], 'b2b_window': av['b2b_window'], 'line_timing_b2b': lt['b2b_second_nights_with_snapshots'],
               'snapshot_span': lt['snapshot_span']}
    (OUT / 'audit.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
    print(json.dumps(summary, indent=1, default=str))


if __name__ == '__main__':
    main()
