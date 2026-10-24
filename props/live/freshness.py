"""Freshness checks of the live inputs."""
import pandas as pd

HISTORY_LAG_HOURS = 5.0            # games that began more than this long ago must be in the box-score tables
GAME_PREFIXES = ('002', '004')


def stale_history(hist, sched, now, before_date):
    """Games of dates before `before_date` that began more than HISTORY_LAG_HOURS ago and are missing from the box-score history: the features of the slot's games would be wrong.
    (Games of the slot's own date are not history for it.)"""
    s = sched[sched['game_id'].str[:3].isin(GAME_PREFIXES) & (sched['et_game_date'] < before_date)]
    due = s[pd.to_datetime(s['nba_tip_utc'], utc=True) < now - pd.Timedelta(hours=HISTORY_LAG_HOURS)]
    return sorted(set(due['game_id']) - set(hist['game_id'].astype(str)))
