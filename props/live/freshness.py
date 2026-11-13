"""Freshness checks of the live inputs."""
import sqlite3

import pandas as pd

from props.features import guards

HISTORY_LAG_HOURS = 5.0            # games that began more than this long ago must be in the box-score tables
GAME_PREFIXES = ('002', '004')
STAT_TABLES = ('player_advanced_stats_v3', 'player_basic_stats', 'player_track_stats_v3', 'player_usage_stats_v3')     # the four tables rows.load_history reads
STAT_GAME_PREFIXES = GAME_PREFIXES + ('005',)           # regular season, playoffs, play-in (rows.GAME_TYPES)


def missing_stat_tables(nba_db, season):
    """
    (game_id, table) gaps among the four box-score tables load_history reads, for one season such as '2026-27': a game that has rows in at least one of
    them but none in `table`. stale_history cannot see these (it only knows which games are in the merged history). Read-only, one DISTINCT scan per table.
    Typical gaps: the NBA tracking/usage sync gave up on a game (history features then drop that game from the tracking rates), or player_basic_stats lags
    the roster table (every player of that game would read as did-not-play). A game still being ingested shows up here briefly; that is expected.
    """
    yy = str(season)[2:4]
    marks = ','.join('?' * len(STAT_GAME_PREFIXES))
    by_table = {}
    with sqlite3.connect(f'file:{nba_db}?mode=ro', uri=True) as conn:
        for t in STAT_TABLES:
            rows = conn.execute(f'SELECT DISTINCT gameId FROM "{t}" WHERE substr(gameId, 4, 2) = ? AND substr(gameId, 1, 3) IN ({marks})',
                                (yy, *STAT_GAME_PREFIXES)).fetchall()
            by_table[t] = [str(r[0]).zfill(10) for r in rows]
    everything = set().union(*by_table.values())
    gaps = guards.table_coverage(everything, by_table)
    return sorted((g, t) for t, games in gaps.items() for g in games)


def stale_history(hist, sched, now, before_date):
    """Games of dates before `before_date` that began more than HISTORY_LAG_HOURS ago and are missing from the box-score history: the features of the slot's games would be wrong.
    (Games of the slot's own date are not history for it.)"""
    s = sched[sched['game_id'].str[:3].isin(GAME_PREFIXES) & (sched['et_game_date'] < before_date)]
    due = s[pd.to_datetime(s['nba_tip_utc'], utc=True) < now - pd.Timedelta(hours=HISTORY_LAG_HOURS)]
    return sorted(set(due['game_id']) - set(hist['game_id'].astype(str)))
