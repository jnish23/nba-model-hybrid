@echo off
REM Props signal job (Task Scheduler "NBA Props Signals", every 10 minutes): decision slots T-240 / T-60 / T-5 -> signals and first-qualification paper bets.
REM Refuses to run if the rule file, model bundle or features database differ from reports\props\live_config.json. Log: data\raw\props_signal.log
cd /d "%~dp0.."
set PY=C:\Users\jorda\nba-model-hybrid\.venv\Scripts\python.exe
echo ===== %DATE% %TIME% ===== >> "data\raw\props_signal.log"
"%PY%" scripts\props_signal_job.py >> "data\raw\props_signal.log" 2>&1
