@echo off
REM Props settlement + private dashboard publish (Task Scheduler "NBA Props Settle", daily 10:30 local, after the 09:15 box-score sync).
REM Settles finished bets, then refreshes the live export (odds now included) and pushes it to the PRIVATE dashboard repo. Logs: data\raw\props_settle.log, data\raw\props_publish.log
cd /d "%~dp0.."
set PY=C:\Users\jorda\nba-model-hybrid\.venv\Scripts\python.exe
echo ===== %DATE% %TIME% ===== >> "data\raw\props_settle.log"
"%PY%" scripts\props_settle_job.py >> "data\raw\props_settle.log" 2>&1
echo ===== %DATE% %TIME% ===== >> "data\raw\props_publish.log"
"%PY%" scripts\publish_props_dashboard.py >> "data\raw\props_publish.log" 2>&1
