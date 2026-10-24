@echo off
REM Refresh the props dashboard data (odds now, settled bets) and publish it to the private dashboard repo. Double-click any time to see newer odds; the deployed app updates within a minute or two.
cd /d "%~dp0.."
set PY=C:\Users\jorda\nba-model-hybrid\.venv\Scripts\python.exe
echo ===== %DATE% %TIME% ===== >> "data\raw\props_publish.log"
"%PY%" scripts\publish_props_dashboard.py >> "data\raw\props_publish.log" 2>&1
echo Done. Last lines of data\raw\props_publish.log:
powershell -NoProfile -Command "Get-Content 'data\raw\props_publish.log' -Tail 8"
pause
