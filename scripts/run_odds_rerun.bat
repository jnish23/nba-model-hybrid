@echo off
REM One-shot re-run created by scripts\schedule_odds_rerun.py when some of today's games had no betting lines at the
REM morning run: predicts only the games that have not started and still lack a line, schedules the next re-run if one is
REM still needed, and publishes the dashboard data (predictions + status.json).
cd /d "%~dp0.."
echo ===== RERUN %DATE% %TIME% ===== >> "data\raw\daily_inference.log"
".venv\Scripts\python.exe" scripts\run_inference_v2.py --only-unpriced >> "data\raw\daily_inference.log" 2>&1
".venv\Scripts\python.exe" scripts\schedule_odds_rerun.py --after-rerun >> "data\raw\daily_inference.log" 2>&1
".venv\Scripts\python.exe" scripts\export_dashboard_data.py --publish >> "data\raw\daily_inference.log" 2>&1
