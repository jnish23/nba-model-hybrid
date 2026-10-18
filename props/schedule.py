"""
Authoritative tip times. The vendor's tip time is not reliable in history: in the density pilot one game was listed 70 minutes late
until it started (so its "T-45 .. T-5" snapshots were in-game), and three more were revised at or after the start. The NBA's own
schedule (nba_api ScheduleLeagueV2: one call per season, cached to disk) is stable.

Reference clock T for every minutes-to-tip number: tip_ref = NBA scheduled tip + an offset of 0 to 10 minutes, taken from the vendor's
own earliest tip estimate when that is within 15 minutes of the NBA time (ref_offset_min), else 10. The vendor's live feed reports the
scheduled time plus 10 minutes for most games (35 of 46 upcoming games on 2026-10-18) and the scheduled time itself for the rest
(11 of 46, probably nationally televised games), and games start about when the vendor's estimate says, so T = 0 is about when books
pull their lines. A snapshot with a negative minutes-to-tip under this clock is treated as in-game and excluded from analysis.
"""
import logging
from pathlib import Path

import pandas as pd

from core.timeutil import et_game_date, to_utc_iso

TIP_OFFSET_MIN = 10
CACHE_DIR = Path('data/raw/nba_schedule')
COLS = ['game_id', 'nba_tip_utc', 'et_game_date', 'home_team_id', 'away_team_id', 'home_team', 'away_team', 'game_label']


def _normalize(df):
    out = pd.DataFrame({
        'game_id': df['gameId'].astype(str).str.zfill(10),
        'nba_tip_utc': pd.to_datetime(df['gameDateTimeUTC'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'home_team_id': df['homeTeam_teamId'].astype('int64'), 'away_team_id': df['awayTeam_teamId'].astype('int64'),
        'home_team': (df['homeTeam_teamCity'].fillna('') + ' ' + df['homeTeam_teamName'].fillna('')).str.strip(),
        'away_team': (df['awayTeam_teamCity'].fillna('') + ' ' + df['awayTeam_teamName'].fillna('')).str.strip(),
        'game_label': df.get('gameLabel', pd.Series('', index=df.index)).fillna('')})
    out['et_game_date'] = [et_game_date(t) for t in out['nba_tip_utc']]
    return out[COLS]


def load_schedule(season, cache_dir=CACHE_DIR, refresh=False):
    """Season schedule (regular season, playoffs, cup) with scheduled tip times in UTC. Cached; one nba_api call per season."""
    cache = Path(cache_dir) / f'schedule_{season}.csv'
    if cache.exists() and not refresh:
        return pd.read_csv(cache, dtype={'game_id': str})
    from nba_api.stats.endpoints import scheduleleaguev2
    raw = scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=60).get_data_frames()[0]
    sch = _normalize(raw.dropna(subset=['gameDateTimeUTC']))
    cache.parent.mkdir(parents=True, exist_ok=True)
    sch.to_csv(cache, index=False)
    logging.info(f'fetched {len(sch)} scheduled games for {season}')
    return sch


def tip_ref_utc(nba_tip_utc, offset_min=TIP_OFFSET_MIN):
    return to_utc_iso(pd.Timestamp(nba_tip_utc) + pd.Timedelta(minutes=offset_min))


def ref_offset_min(vendor_tip_utc, nba_tip_utc, trust_within_min=15.0, cap_min=TIP_OFFSET_MIN):
    """
    Minutes after the NBA scheduled tip used as T = 0. The vendor's own tip estimate is used when it is plausible (within
    trust_within_min of the NBA time), clipped to [0, cap]: most games are listed at scheduled + 10, but some (11 of the 46 upcoming
    games on 2026-10-18, probably nationally televised ones) at the scheduled time itself, and then T = 0 must not be later than that.
    A vendor time that is far off (one pilot game was 70 minutes late) or missing falls back to the cap.
    """
    if vendor_tip_utc is None or (isinstance(vendor_tip_utc, float) and pd.isna(vendor_tip_utc)):
        return float(cap_min)
    d = (pd.Timestamp(vendor_tip_utc) - pd.Timestamp(nba_tip_utc)).total_seconds() / 60.0
    return float(min(max(d, 0.0), cap_min)) if abs(d) <= trust_within_min else float(cap_min)


def match_games(events, schedule, team_ids):
    """
    Attach the NBA game to vendor events. events: event_id_vendor, home_team, away_team, commence_time_utc.
    team_ids: callable name -> NBA team id (PlayerMatcher.team_id). Match on both team ids and the nearest scheduled tip within
    +-36 hours of the vendor's time (the vendor's time can be an hour or more off, never a day).
    Returns events plus game_id, nba_tip_utc, tip_ref_utc (NaN where unmatched).
    """
    sch = schedule.assign(_t=pd.to_datetime(schedule['nba_tip_utc'], utc=True))
    out = []
    for e in events.itertuples(index=False):
        h, a = team_ids(e.home_team), team_ids(e.away_team)
        cand = sch[(sch.home_team_id == h) & (sch.away_team_id == a)]
        vt = pd.Timestamp(e.commence_time_utc)
        vt = vt.tz_localize('UTC') if vt.tzinfo is None else vt.tz_convert('UTC')
        if len(cand):
            d = (cand['_t'] - vt).abs()
            best = cand.loc[d.idxmin()]
            if d.min() <= pd.Timedelta(hours=36):
                out.append((e.event_id_vendor, best['game_id'], best['nba_tip_utc'],
                            tip_ref_utc(best['nba_tip_utc'], ref_offset_min(vt, best['_t']))))
                continue
        out.append((e.event_id_vendor, None, None, None))
    return events.merge(pd.DataFrame(out, columns=['event_id_vendor', 'game_id', 'nba_tip_utc', 'tip_ref_utc']),
                        on='event_id_vendor', how='left', suffixes=('_old', ''))


REF_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_ref (
    event_id_vendor TEXT PRIMARY KEY, game_id TEXT, nba_tip_utc TEXT, tip_ref_utc TEXT, vendor_tip_first_utc TEXT,
    vendor_tip_last_utc TEXT, vendor_tip_revisions INTEGER
);
CREATE TABLE IF NOT EXISTS snapshot_ref (
    snapshot_ts_utc TEXT NOT NULL, event_id_vendor TEXT NOT NULL, minutes_to_ref REAL, in_game INTEGER,
    PRIMARY KEY (snapshot_ts_utc, event_id_vendor)
);
"""


def team_id_map(nba_db_path):
    """Vendor team name -> NBA team id (full names as in team_basic_stats, plus the Clippers' two spellings)."""
    import sqlite3
    with sqlite3.connect(f'file:{nba_db_path}?mode=ro', uri=True) as conn:
        t = pd.read_sql('SELECT DISTINCT teamId, teamName FROM team_basic_stats', conn)
    m = {n.lower(): int(i) for i, n in zip(t['teamId'], t['teamName'])}
    m.setdefault('los angeles clippers', m.get('la clippers'))
    return lambda name: m.get(str(name).lower())


def annotate(conn, schedules, team_ids):
    """
    Fill event_ref (NBA game id, scheduled tip, reference tip, vendor tip revisions) and snapshot_ref (minutes to the reference
    tip and an in_game flag for snapshots at or after it) for every event and snapshot in the store. Idempotent; additive tables
    only, so the live collector's statements are unaffected. schedules: DataFrame from load_schedule (concat several seasons).
    Returns a summary dict.
    """
    conn.executescript(REF_SCHEMA)
    ev = pd.read_sql('SELECT event_id_vendor, home_team, away_team, commence_time_utc FROM events', conn)
    if ev.empty:
        return {'events': 0, 'matched': 0, 'in_game_snapshots': 0, 'revised_events': 0}
    per_snap = pd.read_sql('SELECT DISTINCT event_id_vendor, snapshot_ts_utc, commence_time_utc FROM prop_snapshots', conn)
    per_snap = per_snap.sort_values(['event_id_vendor', 'snapshot_ts_utc'])
    g = per_snap.groupby('event_id_vendor')['commence_time_utc']
    tips = pd.DataFrame({'first_tip': g.first(), 'last_tip': g.last(), 'n': g.nunique()}).reset_index()
    # events.commence_time_utc is the LAST tip the vendor reported (often the actual start); the reference clock needs the vendor's
    # earliest estimate, which is what a pre-game bettor would have seen
    ev = ev.merge(tips[['event_id_vendor', 'first_tip']], on='event_id_vendor', how='left')
    ev['commence_time_utc'] = ev['first_tip'].fillna(ev['commence_time_utc'])
    m = match_games(ev.drop(columns='first_tip'), schedules, team_ids)
    m = m.merge(tips, on='event_id_vendor', how='left')
    rows = [(r.event_id_vendor, r.game_id if isinstance(r.game_id, str) else None,
             r.nba_tip_utc if isinstance(r.nba_tip_utc, str) else None,
             r.tip_ref_utc if isinstance(r.tip_ref_utc, str) else None, r.first_tip, r.last_tip,
             int(r.n - 1) if pd.notna(r.n) else 0) for r in m.itertuples(index=False)]
    # Matched events replace their old row; an event this call cannot match (its season's schedule was not passed in) keeps
    # whatever annotation it already has instead of being wiped.
    conn.executemany('INSERT OR REPLACE INTO event_ref VALUES (?,?,?,?,?,?,?)', [r for r in rows if r[1] is not None])
    conn.executemany('INSERT OR IGNORE INTO event_ref VALUES (?,?,?,?,?,?,?)', [r for r in rows if r[1] is None])
    log = pd.read_sql('SELECT l.snapshot_ts_utc, l.event_id_vendor, r.tip_ref_utc FROM snapshot_log l '
                      'JOIN event_ref r USING (event_id_vendor) WHERE r.tip_ref_utc IS NOT NULL', conn)
    if len(log):
        mins = (pd.to_datetime(log['tip_ref_utc'], utc=True) - pd.to_datetime(log['snapshot_ts_utc'], utc=True)
                ).dt.total_seconds() / 60.0
        conn.executemany('INSERT OR REPLACE INTO snapshot_ref VALUES (?,?,?,?)',
                         [(a, b, float(x), int(x < 0)) for a, b, x in zip(log['snapshot_ts_utc'], log['event_id_vendor'], mins)])
        n_in = int((mins < 0).sum())
    else:
        n_in = 0
    conn.commit()
    return {'events': len(ev), 'matched': int(m['game_id'].notna().sum()), 'in_game_snapshots': n_in,
            'revised_events': int((m['n'] > 1).sum())}
