@echo off
REM Publish-only refresh of the private props dashboard (Task Scheduler "NBA Props Publish PM", daily 15:30 local): odds now for the pending bets, then push. No pause, so it can run unattended.
REM Log: data\raw\props_publish.log
cd /d "%~dp0.."
set PY=C:\Users\jorda\nba-model-hybrid\.venv\Scripts\python.exe
echo ===== %DATE% %TIME% (scheduled) ===== >> "data\raw\props_publish.log"
"%PY%" scripts\publish_props_dashboard.py >> "data\raw\props_publish.log" 2>&1
