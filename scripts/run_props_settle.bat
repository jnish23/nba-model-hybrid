@echo off
REM Props settlement + dashboard export (Task Scheduler "NBA Props Settle", daily after the box-score sync). Log: data\raw\props_settle.log
cd /d "%~dp0.."
set PY=C:\Users\jorda\nba-model-hybrid\.venv\Scripts\python.exe
echo ===== %DATE% %TIME% ===== >> "data\raw\props_settle.log"
"%PY%" scripts\props_settle_job.py >> "data\raw\props_settle.log" 2>&1
"%PY%" scripts\export_props_dashboard_data.py --live >> "data\raw\props_settle.log" 2>&1
