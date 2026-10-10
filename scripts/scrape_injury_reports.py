"""
Download official NBA injury report PDFs (point-in-time snapshots).

Archive URL pattern (ET timestamps in the filename; the report's own header carries the exact
publish time, e.g. file "..._05PM.pdf" says "05:30 PM"):
    https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HHAM|PM.pdf      (through 2024-25)
    https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH_MMAM|PM.pdf   (2025-26, half-hourly)
Coverage found by probing: from ~2019-01 on; ~2 per day in 2019-20 (01PM, 05PM), hourly from
~2021-22, half-hourly in 2025-26. Each report lists that day's games AND the next day's.

For each calendar date with games on that day or the next, the slots in SLOTS are requested;
a slot that 403s is recorded in the manifest so it is never requested again.

Usage:
    python scripts/scrape_injury_reports.py --start 2019-10-01 --end 2026-04-12     # backfill
    python scripts/scrape_injury_reports.py --live                                # today's latest slot
"""
import argparse
import datetime as dt
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
OUT_DIR = Path('data/raw/injury_reports')
MANIFEST = OUT_DIR / 'manifest.csv'
DB_PATH = Path('data/nba_stats.db')

# (hour, ampm) slots per date: morning game-day, early afternoon, the 5 PM day-before deadline,
# 6-7 PM (just before most tip-offs, hourly era only), and 8 PM (late tips; also a "night before"
# view of the next day's games). Files are typically stamped ~:30 past the slot hour.
SLOTS = [(11, 'AM'), (12, 'PM'), (1, 'PM'), (5, 'PM'), (6, 'PM'), (7, 'PM'), (8, 'PM')]
# Keep in sync with models/injury_report_features.GRID_SLOTS (training uses only these slots).
REQUEST_DELAY = 0.35  # seconds between requests


def slot_stamps(date: dt.date, hour: int, ampm: str):
    """Filename variants for a slot, preferred first (old hourly format, then half-hourly format)."""
    d = date.isoformat()
    return [f"{d}_{hour:02d}{ampm}", f"{d}_{hour:02d}_00{ampm}", f"{d}_{hour:02d}_30{ampm}"]


def game_dates(db_path, start, end):
    with sqlite3.connect(db_path) as conn:
        dates = pd.read_sql("SELECT DISTINCT gameDate FROM team_game_full", conn)['gameDate']
    dates = pd.to_datetime(dates).dt.date
    dates = set(d for d in dates if start <= d <= end)
    # A report on day D also covers D+1, so the day before each game date is useful too
    return sorted(dates | {d - dt.timedelta(days=1) for d in dates})


def load_manifest():
    if MANIFEST.exists():
        return pd.read_csv(MANIFEST, dtype=str)
    return pd.DataFrame(columns=['date', 'slot', 'stamp', 'status', 'path'])


def fetch(session, stamp):
    """Return (http_status, content or None)."""
    for attempt in range(3):
        try:
            r = session.get(BASE_URL.format(stamp=stamp), timeout=30)
            if r.status_code == 200 and r.content[:4] == b'%PDF':
                return 200, r.content
            if r.status_code in (403, 404):
                return r.status_code, None
            logging.warning(f"{stamp}: HTTP {r.status_code}, retrying")
        except requests.RequestException as e:
            logging.warning(f"{stamp}: {e}, retrying")
        time.sleep(2 * (attempt + 1))
    return -1, None


def download(dates, slots=SLOTS):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    done = set(zip(manifest['date'], manifest['slot']))
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (research; nba-model-hybrid)'

    new_rows = []
    todo = [(d, s) for d in dates for s in slots if (d.isoformat(), f"{s[0]:02d}{s[1]}") not in done]
    logging.info(f"{len(todo)} date-slots to request ({len(done)} already in manifest)")
    for i, (date, (hour, ampm)) in enumerate(todo, 1):
        slot = f"{hour:02d}{ampm}"
        row = {'date': date.isoformat(), 'slot': slot, 'stamp': '', 'status': 'missing', 'path': ''}
        for stamp in slot_stamps(date, hour, ampm):
            status, content = fetch(session, stamp)
            time.sleep(REQUEST_DELAY)
            if status == 200:
                path = OUT_DIR / str(date.year) / f"Injury-Report_{stamp}.pdf"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                row.update(stamp=stamp, status='ok', path=str(path))
                break
            if status == -1:
                row['status'] = 'error'  # transient; not recorded so it is retried next run
        # A slot for today or later may simply not be published yet: don't record it as missing
        if row['status'] == 'ok' or (row['status'] == 'missing' and date < dt.date.today()):
            new_rows.append(row)
        if i % 100 == 0 or i == len(todo):
            manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
            manifest.to_csv(MANIFEST, index=False)
            new_rows = []
            ok = (manifest['status'] == 'ok').sum()
            logging.info(f"{i}/{len(todo)} requested; manifest: {ok} ok, {len(manifest) - ok} missing")
    return manifest


def now_et():
    """Current US/Eastern wall-clock time (report filenames and headers are ET)."""
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo('America/New_York')).replace(tzinfo=None)
    except Exception:  # Windows without the tzdata package
        logging.warning("US/Eastern time zone unavailable (pip install tzdata); assuming UTC-4/-5 from UTC")
        utc = dt.datetime.utcnow()
        # US DST: second Sunday in March to first Sunday in November
        y = utc.year
        start = dt.datetime(y, 3, 8) + dt.timedelta(days=(6 - dt.datetime(y, 3, 8).weekday()) % 7, hours=7)
        end = dt.datetime(y, 11, 1) + dt.timedelta(days=(6 - dt.datetime(y, 11, 1).weekday()) % 7, hours=6)
        return utc - dt.timedelta(hours=4 if start <= utc < end else 5)


def live(as_of=None):
    """
    Download the most recent report published at or before `as_of` (US/Eastern; default now),
    trying half-hourly and hourly filename slots back from that time. Returns the saved path.
    Passing as_of = the training snapshot time (e.g. noon on game day) makes live features use the
    same report the training set would have used.
    """
    now = as_of or now_et()
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (research; nba-model-hybrid)'
    for minutes_back in range(0, 24 * 60, 30):
        t = now - dt.timedelta(minutes=minutes_back)
        hour12 = t.hour % 12 or 12
        ampm = 'AM' if t.hour < 12 else 'PM'
        minute = '00' if t.minute < 30 else '30'
        for stamp in (f"{t.date().isoformat()}_{hour12:02d}_{minute}{ampm}", f"{t.date().isoformat()}_{hour12:02d}{ampm}"):
            status, content = fetch(session, stamp)
            time.sleep(REQUEST_DELAY)
            if status == 200:
                path = OUT_DIR / str(t.year) / f"Injury-Report_{stamp}.pdf"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                # Hourly-era files are printed ~30 min after their slot: keep looking further back
                # if this one was published after as_of (training would not use it either).
                from models.injury_report_parser import parse_report
                try:
                    printed = parse_report(path)['report_ts'].iloc[0]
                except Exception:
                    printed = None
                if printed is not None and pd.notna(printed) and printed > now:
                    logging.info(f"{stamp} was published at {printed}, after {now}; looking further back")
                    continue
                logging.info(f"Saved latest report {stamp} (published {printed}) -> {path}")
                return path
    logging.warning("No report found in the last 24 hours")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', default='2019-10-01')
    parser.add_argument('--end', default=dt.date.today().isoformat())
    parser.add_argument('--live', action='store_true', help="Fetch today's latest report only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    if args.live:
        live()
        return
    dates = game_dates(DB_PATH, dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))
    logging.info(f"{len(dates)} dates from {dates[0]} to {dates[-1]}")
    download(dates)


if __name__ == "__main__":
    main()
