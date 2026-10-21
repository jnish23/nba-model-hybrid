"""
Props-side injury-report archive. The team pipeline's archive (data/raw/injury_reports) holds 7 reports per game day by design; the NBA
publishes hourly all day in every season from 2021-22 (and every 15 minutes in at least part of 2025-26). Decision times T-240, T-60 and
T-5 need the latest report at or before T, so staleness depends on how dense the archive is.

This module downloads the missing hourly slots into its OWN folder (data/raw/injury_reports_hourly) with its own manifest, so the team
pipeline's report folder, parsed cache and features are untouched. Same URL pattern, request delay and file validation as
core/scrape_injury_reports.py, which it reuses.
"""
import datetime as dt
import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests

from core.scrape_injury_reports import REQUEST_DELAY, fetch, slot_stamps

HOURLY_DIR = Path('data/raw/injury_reports_hourly')
MANIFEST = HOURLY_DIR / 'manifest.csv'
# ET hours that matter for T-240 .. T-5 (earliest tip noon ET -> 8 AM; latest tip 10:30 PM ET). The team archive already has
# 11 AM, 12 PM, 1 PM, 5 PM, 6 PM, 7 PM, 8 PM, so only these are new:
NEW_SLOTS = [(8, 'AM'), (9, 'AM'), (10, 'AM'), (2, 'PM'), (3, 'PM'), (4, 'PM'), (9, 'PM'), (10, 'PM')]


def game_days(db_path, start, end):
    """Game dates only (the day before is already covered by the evening slots of the team archive)."""
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
        d = pd.read_sql("SELECT DISTINCT substr(gameDate, 1, 10) AS d FROM team_game_full", conn)['d']
    days = sorted(dt.date.fromisoformat(x) for x in d if start <= dt.date.fromisoformat(x) <= end)
    return days


def load_manifest(path=MANIFEST):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=['date', 'slot', 'stamp', 'status', 'path'])


def plan(dates, slots=NEW_SLOTS, manifest=None):
    manifest = load_manifest() if manifest is None else manifest
    done = set(zip(manifest['date'], manifest['slot']))
    return [(d, s) for d in dates for s in slots if (d.isoformat(), f'{s[0]:02d}{s[1]}') not in done]


def download(dates, slots=NEW_SLOTS, out_dir=HOURLY_DIR, manifest_path=MANIFEST, session=None, delay=REQUEST_DELAY,
             save_every=100, fetch_fn=fetch, sleep=time.sleep):
    """Resumable: every (date, slot) is requested once and recorded; reruns skip what the manifest already holds."""
    out_dir, manifest_path = Path(out_dir), Path(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    todo = plan(dates, slots, manifest)
    session = session or requests.Session()
    if hasattr(session, 'headers'):
        session.headers['User-Agent'] = 'Mozilla/5.0 (research; nba-model-hybrid)'
    logging.info(f'{len(todo)} date-slots to request ({len(manifest)} already in the manifest)')
    new_rows, n_ok = [], 0
    for i, (date, (hour, ampm)) in enumerate(todo, 1):
        slot = f'{hour:02d}{ampm}'
        row = {'date': date.isoformat(), 'slot': slot, 'stamp': '', 'status': 'missing', 'path': ''}
        for stamp in slot_stamps(date, hour, ampm):
            status, content = fetch_fn(session, stamp)
            sleep(delay)
            if status == 200:
                path = out_dir / str(date.year) / f'Injury-Report_{stamp}.pdf'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                row.update(stamp=stamp, status='ok', path=str(path))
                n_ok += 1
                break
            if status == -1:
                row['status'] = 'error'                     # transient: not recorded, so the next run retries it
        if row['status'] == 'ok' or (row['status'] == 'missing' and date < dt.date.today()):
            new_rows.append(row)
        if i % save_every == 0 or i == len(todo):
            manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
            manifest.to_csv(manifest_path, index=False)
            new_rows = []
            logging.info(f'{i}/{len(todo)} requested; {n_ok} downloaded this run; manifest {len(manifest)} rows')
    return {'requested': len(todo), 'downloaded': n_ok}
