@echo off
REM Props odds collector (Task Scheduler task "NBA Props Snapshots", every 10 minutes).
REM Lean setting: a snapshot every 30 min over the last 4 h before tip, no fast window, plus one closing snapshot in the last 10 min.
REM Reads ODDS_API_KEY from the gitignored .env. Live credit caps (live ledger only; the backfill has its own): 3,000 per UTC day, 300,000 in total.
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=C:\Users\jorda\nba-model-hybrid\.venv\Scripts\python.exe
"%PY%" scripts\snapshot_props.py --window-hours 4 --fast-window-min 0 --slow-cadence-min 30 --close-snapshot-min 10 --daily-cap 3000 --total-cap 300000 >> "data\raw\snapshot_props.log" 2>&1
