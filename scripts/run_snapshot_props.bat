@echo off
REM Props odds collector (Task Scheduler task "NBA Props Snapshots", every 10 minutes).
REM Lean prototype setting: a snapshot every 30 min over the last 4 h before tip, no fast window, plus one closing snapshot
REM in the last 10 min. Reads ODDS_API_KEY from the gitignored .env. Credit caps are enforced inside the script.
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=C:\Users\jorda\nba-model-hybrid\.venv\Scripts\python.exe
"%PY%" scripts\snapshot_props.py --window-hours 4 --fast-window-min 0 --slow-cadence-min 30 --close-snapshot-min 10 >> "data\raw\snapshot_props.log" 2>&1
