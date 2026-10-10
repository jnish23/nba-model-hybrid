"""
Collect TIMESTAMPED odds snapshots (and the latest official injury report) going forward.

Why: nba_odds / nba_odds_opening have no timestamps, so a backtest cannot tell whether a price was
still available when a piece of news (e.g. an injury report) came out. Running this on a schedule
(e.g. every 30-60 min from 9 AM to tip on game days) builds a line history: for every book, the
current spread / moneyline / total with the time it was observed.

Rows are written to the `odds_snapshots` table in data/nba_stats.db (one row per game x book x
fetch), with:
  fetched_at_et     when the line was observed (US/Eastern)
  game_status       SBR's status text at fetch time
  in_game           1 if the game had started when the line was fetched (fetch time >= scheduled
                    start, or a status that shows a period/score/final). Exclude these from
                    pre-game analysis: in-game lines leaked into nba_odds before
                    (see models/odds_clean.py). The distinct statuses seen are logged.
  line_type         'current' (every run) or 'opening' (stored once per game and book: the first
                    time it is seen)

Usage:
    python scripts/snapshot_odds.py                 # today + tomorrow, odds + latest injury report
    python scripts/snapshot_odds.py --no-injuries
"""
import argparse
import datetime as dt
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DB_PATH = Path('data/nba_stats.db')
BOOKS = ['betmgm', 'fanduel', 'caesars', 'bet365', 'draftkings', 'bet_rivers_ny']
FIELDS = ['home_spread', 'home_spread_odds', 'away_spread', 'away_spread_odds', 'home_ml', 'away_ml',
          'total', 'over_odds', 'under_odds']
PREGAME_RE = re.compile(r'\b\d{1,2}:\d{2}\s*(am|pm)\b|scheduled|postponed', re.I)  # SBR shows a tip time pre-game

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    fetched_at_et TEXT, game_date TEXT, game_start TEXT, game_status TEXT, in_game INTEGER,
    home_team TEXT, away_team TEXT, home_team_abbr TEXT, away_team_abbr TEXT,
    line_type TEXT, book TEXT,
    home_spread REAL, home_spread_odds REAL, away_spread REAL, away_spread_odds REAL,
    home_ml REAL, away_ml REAL, total REAL, over_odds REAL, under_odds REAL
)"""


def now_et():
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo('America/New_York')).replace(tzinfo=None)
    except Exception:
        logging.warning("US/Eastern time zone unavailable (pip install tzdata); using local time")
        return dt.datetime.now()


IN_GAME_RE = re.compile(r'final|half|end|\b(1st|2nd|3rd|4th|ot)\b|\bq[1-4]\b|\d{1,2}:\d{2}\s+(1st|2nd|3rd|4th|ot)', re.I)


def started(game_start, fetched_at):
    """True if fetched_at (ET, naive) is at/after SBR's scheduled start (ISO, usually UTC)."""
    try:
        start = pd.Timestamp(game_start)
        if start.tzinfo is None:
            start = start.tz_localize('UTC')
        return pd.Timestamp(fetched_at).tz_localize('America/New_York') >= start
    except Exception:
        return False


def in_game_flag(status, game_start, fetched_at):
    return int(started(game_start, fetched_at) or bool(IN_GAME_RE.search(str(status or ''))))


def is_in_game(status):
    """SBR status is a tip time ('7:30 PM ET') before the game, else a period/score/'Final'."""
    s = str(status or '').strip()
    if not s:
        return 0
    return 0 if PREGAME_RE.search(s) else 1


def _num(v):
    if isinstance(v, str) and v.strip().upper() in ('PK', 'PICK', 'EVEN', 'EV'):
        return 0.0
    return pd.to_numeric(v, errors='coerce')


def fetch(date, current_line):
    from sbrscrape import Scoreboard
    sb = Scoreboard(sport='NBA', date=date, current_line=current_line)  # swallows its own errors
    if not hasattr(sb, 'games'):
        # sbrscrape returns without setting .games both when no games are listed and on some errors
        logging.info(f"SBR returned no games for {date} (current_line={current_line}); "
                     "expected on off days, otherwise check for an error printed above")
    return getattr(sb, 'games', None) or []


def to_rows(games, fetched_at, game_date, line_type):
    rows = []
    for g in games:
        base = {
            'fetched_at_et': fetched_at.strftime('%Y-%m-%d %H:%M:%S'), 'game_date': game_date,
            'game_start': g.get('date'), 'game_status': g.get('status'),
            'in_game': in_game_flag(g.get('status'), g.get('date'), fetched_at),
            'home_team': g.get('home_team'), 'away_team': g.get('away_team'),
            'home_team_abbr': g.get('home_team_abbr'), 'away_team_abbr': g.get('away_team_abbr'),
            'line_type': line_type,
        }
        for book in BOOKS:
            vals = {f: (g.get(f) or {}).get(book) if isinstance(g.get(f), dict) else None for f in FIELDS}
            if all(v is None for v in vals.values()):
                continue
            rows.append({**base, 'book': book, **{f: _num(v) for f, v in vals.items()}})
    return rows


def snapshot(dates, db_path=DB_PATH):
    fetched_at = now_et()
    rows = []
    for d in dates:
        for line_type, current in (('current', True), ('opening', False)):
            try:
                rows += to_rows(fetch(d, current), fetched_at, d, line_type)
            except Exception as e:
                logging.error(f"SBR fetch failed for {d} ({line_type}): {e}")
    if not rows:
        logging.info(f"No odds found for {dates} at {fetched_at:%Y-%m-%d %H:%M}")
        return 0
    df = pd.DataFrame(rows)
    logging.info(f"SBR statuses seen: {sorted(df['game_status'].astype(str).unique())}")
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)
        # Openers: keep only the first sighting per game and book
        key = ['game_date', 'home_team', 'away_team', 'book']
        seen = pd.read_sql("SELECT DISTINCT game_date, home_team, away_team, book FROM odds_snapshots "
                           "WHERE line_type = 'opening'", conn)
        opening = df[df['line_type'] == 'opening'].merge(seen.assign(_seen=1), on=key, how='left')
        df = pd.concat([df[df['line_type'] == 'current'], opening[opening['_seen'].isna()].drop(columns='_seen')])
        df.to_sql('odds_snapshots', conn, if_exists='append', index=False)
    logging.info(f"Saved {len(df)} rows ({df['home_team'].nunique()} games, "
                 f"{int(df['in_game'].sum())} in-game) at {fetched_at:%Y-%m-%d %H:%M} ET")
    return len(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-injuries', action='store_true', help='Skip fetching the latest injury report')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    today = now_et().date()
    snapshot([today.isoformat(), (today + dt.timedelta(days=1)).isoformat()])
    if not args.no_injuries:
        from scripts.scrape_injury_reports import live
        live()


if __name__ == "__main__":
    main()
