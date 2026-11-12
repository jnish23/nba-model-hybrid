"""How long after its slot time is an injury report on disk? Read-only: lists file names and modified times in the production
folder. mtime is the LAST write (core.scrape_injury_reports.live rewrites a file each time it fetches it), so for a slot the daily
run does not touch (12:00 ET, fetched at 11:15-11:22 ET) it is the half-hourly snapshot job's download time.
Run from the worktree root:  C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe reports/audit_pregame_2026-11/B2_cutoff_guard/report_lag_evidence.py
"""
import glob, os, re
import pandas as pd

rows = []
for f in glob.glob('C:/Users/jorda/nba-model-hybrid/data/raw/injury_reports/2026/*.pdf'):
    n = os.path.basename(f)
    m = re.search(r'_(\d{4}-\d\d-\d\d)_(\d\d)(?:_(\d\d))?(AM|PM)', n)
    h = int(m.group(2)) % 12 + (12 if m.group(4) == 'PM' else 0)
    stamp = pd.Timestamp(m.group(1)) + pd.Timedelta(hours=h, minutes=int(m.group(3) or 0))
    mtime = pd.Timestamp(os.path.getmtime(f), unit='s', tz='UTC').tz_convert('America/New_York').tz_localize(None)
    rows.append((n, stamp, mtime, (mtime - stamp).total_seconds() / 60))
d = pd.DataFrame(rows, columns=['file', 'stamp', 'mtime_et', 'lag_min'])
d = d[d.stamp >= '2026-10-20']
print(f"{len(d)} files stamped since 2026-10-20; {(d.lag_min < 2).sum()} written within 2 min of the slot; "
      f"within that group lag min/median/max = {d[d.lag_min < 2].lag_min.min():.2f} / {d[d.lag_min < 2].lag_min.median():.2f} / {d[d.lag_min < 2].lag_min.max():.2f} min")
noon = d[(d.stamp.dt.hour == 12) & (d.stamp.dt.minute == 0) & (d.stamp >= '2026-11-01')]
print(f"12:00 slot, {len(noon)} days since 2026-11-01: lag min/max = {noon.lag_min.min():.2f} / {noon.lag_min.max():.2f} min")
late = d[d.lag_min >= 2]
print(f"{len(late)} files written 2+ min after their slot (bulk restore of 2026-10-2x files; 11:00 slot rewritten by the daily run at ~11:17 ET)")
