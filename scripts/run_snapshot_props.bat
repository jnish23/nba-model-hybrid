@echo off
REM Props odds collector (NOT scheduled yet). Needs ODDS_API_KEY as a user environment variable.
REM Intended Task Scheduler trigger: every 10 minutes on game days; per-event cadence is decided inside the script.
cd /d "%~dp0.."
".venv\Scripts\python.exe" scripts\snapshot_props.py >> "data\raw\snapshot_props.log" 2>&1
