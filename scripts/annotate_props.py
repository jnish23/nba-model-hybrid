"""
Attach the NBA game id, scheduled tip and reference tip (scheduled + 10 min) to every event, and minutes-to-tip plus an in_game flag
to every snapshot (tables event_ref and snapshot_ref in data/props_odds.db). Idempotent; fetches each season's schedule once (cached in
data/raw/nba_schedule). Run before analysis scripts and reports; build_report does it too.
    python scripts/annotate_props.py [--seasons 2025-26 2026-27]
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.season_utils import season_for_date  # noqa: E402
from props import schedule  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--props-db', default='data/props_odds.db')
    ap.add_argument('--nba-db', default='data/nba_stats.db')
    ap.add_argument('--seasons', nargs='*')
    args = ap.parse_args()
    conn = sqlite3.connect(args.props_db)
    seasons = args.seasons or sorted({season_for_date(d) for (d,) in conn.execute(
        'SELECT DISTINCT substr(commence_time_utc, 1, 10) FROM events')})
    sch = pd.concat([schedule.load_schedule(s) for s in seasons], ignore_index=True)
    print(schedule.annotate(conn, sch, schedule.team_id_map(args.nba_db)))


if __name__ == '__main__':
    main()
