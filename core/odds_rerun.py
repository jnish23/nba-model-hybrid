"""
Re-running the daily inference for games whose betting lines were not posted yet.

The 9:15 AM run predicts every game on the slate, but the books post some lines later. This module decides what is still
unpriced, when the next re-run should happen, and writes the small status file the dashboard shows ("N games awaiting odds,
next check at ...").  scripts/schedule_odds_rerun.py is the command line around it; the pure decisions live here so they are tested.
"""
import datetime as dt
import json
import sqlite3
from pathlib import Path

import pandas as pd

RERUN_INTERVAL_MINUTES = 60
MAX_RERUNS = 6                       # 9:15 + 6 hourly re-runs reaches mid-afternoon; after that the label stays "No odds found"
TIPOFF_MARGIN_MINUTES = 15           # a re-run must start at least this long before the first unpriced game tips off
ODDS_COLUMNS = ('odds_home_ml', 'odds_home_spread', 'odds_total')
STATUS_FILE_NAME = 'status.json'


def unpriced_games(db_path, game_date, table='predictions_log_v2'):
    """Rows (game_id, matchup) of the PENDING games on `game_date` that lack the moneyline, the spread or the total line."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT game_id, matchup, {', '.join(ODDS_COLUMNS)} FROM {table} "
                               f"WHERE game_date = ? AND status = 'PENDING'", conn, params=(str(game_date),))
    missing = df[list(ODDS_COLUMNS)].isna().any(axis=1)
    return df.loc[missing, ['game_id', 'matchup']].sort_values('game_id').reset_index(drop=True)


def next_rerun_time(now, tipoffs, attempts, interval_minutes=RERUN_INTERVAL_MINUTES, max_reruns=MAX_RERUNS,
                    margin_minutes=TIPOFF_MARGIN_MINUTES):
    """
    When to re-run, or None when no further re-run should happen.
    now: timezone-aware datetime. tipoffs: tip-off datetimes (timezone-aware) of the games still unpriced and not started.
    attempts: re-runs already done today. The next run is `interval_minutes` from now; it is only worth running if the first
    unpriced game has not tipped off by then (a started game's lines are live lines, not the pre-game ones the model needs).
    """
    tipoffs = [t for t in tipoffs if t > now]          # a game that has tipped off no longer needs lines
    if attempts >= max_reruns or not tipoffs:
        return None
    when = now + dt.timedelta(minutes=interval_minutes)
    if when + dt.timedelta(minutes=margin_minutes) >= min(tipoffs):
        return None
    return when


def build_status(game_date, missing, next_rerun, attempts, now, reason):
    """The dict written to status.json (times are ISO strings in UTC)."""
    utc = lambda t: None if t is None else t.astimezone(dt.timezone.utc).isoformat(timespec='minutes')
    return {
        'game_date': str(game_date),
        'updated': utc(now),
        'attempts': int(attempts),
        'next_rerun': utc(next_rerun),
        'missing': [{'game_id': str(r.game_id), 'matchup': r.matchup} for r in missing.itertuples()],
        'reason': reason,
    }


def write_status(path, status):
    Path(path).write_text(json.dumps(status, indent=1), encoding='utf-8')


def read_status(path):
    """The status dict, or None when the file is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _pacific(t):
    try:
        from zoneinfo import ZoneInfo
        return t.astimezone(ZoneInfo('America/Los_Angeles'))
    except Exception:                                    # no tz database: Pacific standard time
        return t.astimezone(dt.timezone(dt.timedelta(hours=-8)))


def rerun_message(status, game_date, now=None):
    """
    The sentence the dashboard shows above Today's Games, or None when there is nothing to say (no status, another day's status, or
    every game has its lines).
    """
    if not status or str(status.get('game_date')) != str(game_date) or not status.get('missing'):
        return None
    now = now or dt.datetime.now(dt.timezone.utc)
    n = len(status['missing'])
    games = f"{n} game{'s' if n != 1 else ''}"
    if status.get('next_rerun'):
        when = dt.datetime.fromisoformat(status['next_rerun'])
        if when > now:
            local = _pacific(when)
            return (f"{games} still ha{'ve' if n != 1 else 's'} no odds. The model re-checks automatically at "
                    f"{local.hour % 12 or 12}:{local:%M} {'AM' if local.hour < 12 else 'PM'} Pacific and fills them in once lines are posted.")
        return f"{games} still ha{'ve' if n != 1 else 's'} no odds. A re-check is due now; refresh in a few minutes."
    return f"{games} ha{'ve' if n != 1 else 's'} no odds and no more automatic re-checks are planned today ({status.get('reason', '')})."
