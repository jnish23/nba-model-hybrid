"""
Download the missing hourly NBA injury reports into data/raw/injury_reports_hourly (separate from the team pipeline's folder).
    python scripts/download_injury_reports_hourly.py --start 2021-10-01 --end 2026-06-30 --dry-run   # count only, no requests
    python scripts/download_injury_reports_hourly.py --start 2021-10-01 --end 2026-06-30
Resumable: rerun to continue; a manifest records every slot already requested.
"""
import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props import injury_archive  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2021-10-01')
    ap.add_argument('--end', default='2026-06-30')
    ap.add_argument('--db', default='data/nba_stats.db')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    days = injury_archive.game_days(args.db, dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))
    todo = injury_archive.plan(days)
    est_min = len(todo) * 1.3 * 0.4 / 60
    print(f'{len(days)} game days, {len(injury_archive.NEW_SLOTS)} new hourly slots each: {len(todo)} requests to make '
          f'(~{len(todo) * 85 / 1024:.0f} MB if all exist, about {est_min:.0f} minutes at 0.35 s between requests)')
    if args.dry_run:
        return 0
    print(injury_archive.download(days))
    return 0


if __name__ == '__main__':
    sys.exit(main())
