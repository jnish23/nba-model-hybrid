@echo off
REM Scheduled by Windows Task Scheduler ("NBA Daily Inference"): syncs the current season into
REM data\nba_stats.db, predicts today's games with the game-day-noon (ET) injury report, then
REM publishes the dashboard data to the dashboard-data branch (refreshes the hosted dashboard).
cd /d "%~dp0.."
echo ===== %DATE% %TIME% ===== >> "data\raw\daily_inference.log"
".venv\Scripts\python.exe" scripts\run_inference_v2.py >> "data\raw\daily_inference.log" 2>&1
".venv\Scripts\python.exe" scripts\export_dashboard_data.py --publish >> "data\raw\daily_inference.log" 2>&1
