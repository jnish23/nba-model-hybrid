@echo off
REM Props signal job (Task Scheduler "NBA Props Signals", every 10 minutes): decision slots T-240 / T-60 / T-5 -> signals and first-qualification paper bets, then the dashboard refresh.
REM Refuses to run if the rule file, model bundle or features database differ from reports\props\live_config.json. Log: data\raw\props_signal.log
cd /d "%~dp0.."
set PY=C:\Users\jorda\nba-model-hybrid\.venv\Scripts\python.exe
echo ===== %DATE% %TIME% ===== >> "data\raw\props_signal.log"
"%PY%" scripts\props_signal_job.py >> "data\raw\props_signal.log" 2>&1
REM After the signal run: while bets are live (prices before the games, provisional results after) refresh the private dashboard; pushes are throttled to material changes or every 30 min.
echo ===== %DATE% %TIME% (if-active) ===== >> "data\raw\props_publish.log"
"%PY%" scripts\publish_props_dashboard.py --if-active >> "data\raw\props_publish.log" 2>&1
