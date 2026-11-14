@echo off
REM Early prop snapshots (T-480 and T-360) for the "bet before T-240" test. Task Scheduler task "NBA Props Early Snapshots", every 10 minutes.
REM Writes ONLY to data\props_odds_early.db and data\raw\props_api_early: the production store data\props_odds.db is never touched.
REM Hard caps inside the script: 4,620 credits in total, 330 per UTC day, 14 game days. It stops by itself; the task can then be deleted.
REM Reads ODDS_API_KEY from the gitignored .env.
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=C:\Users\jorda\nba-model-hybrid\.venv\Scripts\python.exe
"%PY%" scripts\snapshot_props_early.py >> "data\raw\snapshot_props_early.log" 2>&1
