@echo off
REM Scheduled by Windows Task Scheduler ("NBA Daily Inference"): syncs the current season into
REM data\nba_stats.db, predicts today's games with the game-day-noon (ET) injury report, then
REM publishes the dashboard data to the dashboard-data branch (refreshes the hosted dashboard).
REM
REM Fire time: 09:15 Pacific = 12:15 ET, all year, as a LOCAL-time trigger. ET and Pacific change clocks on the same
REM night, so 09:15 local is 12:15 ET in both halves of the year; a trigger fixed to a UTC offset drifts by an hour at the
REM change, which is how the run slipped to 11:15 ET after 2026-11-01. The model is trained on the injury report as of noon ET
REM on game day, which can be downloaded about 5 minutes after noon. Any fire time from 09:05 Pacific (12:05 ET) on needs no wait.
REM run_inference_v2.py itself waits for that report if it is started earlier (up to 90 minutes, after the database sync; the
REM task's execution time limit is 2 hours) and exits with code 3, without predicting anything, if it is started even earlier.
REM See its --no-wait-for-injury-cutoff and --max-wait-minutes options.
REM
REM Exit code (Task Scheduler shows it as "Last Run Result"): the first non-zero code of the three steps, 0 only if all succeeded.
REM What each exit code of run_inference_v2.py means and what this file does for it:
REM   0 : normal run, an off day (the NBA lists no games) or a re-run with nothing to do. Both later steps run.
REM   1 : the run crashed (unexpected error, traceback in the log). Both later steps still run (re-run scheduler picks up unpriced
REM       games, the publish is safe: it republishes what is in the database).
REM   3 : TOO EARLY. Nothing was predicted, so the re-run scheduler and the dashboard publish are skipped.
REM   4 : games are scheduled but NOTHING was predicted: the odds source (SBR) or the NBA scoreboard failed after retries ("ODDS SOURCE
REM       FAILURE" / "NBA SCHEDULE FAILURE" in the log), or every scheduled game was skipped ("ALL GAMES SKIPPED"). The re-run
REM       scheduler is skipped: it only re-runs games already logged without a line, so with nothing logged it would write a false
REM       "every game has its lines" status for the dashboard. The dashboard publish STILL runs: it carries the results the morning
REM       sync just settled and is safe to repeat. The exit code stays 4.
REM   5 : predictions were made and logged, but the injury report was missing or stale, so NO BET was recommended ("BETS SUPPRESSED"
REM       in the log; bet_suppressed_reason in bet_recommendations_log). Both later steps run: the dashboard still publishes. The
REM       exit code stays 5 so Task Scheduler shows a problem.
REM   any other non-zero : treated like 1.
cd /d "%~dp0.."
set EXITCODE=0
echo ===== %DATE% %TIME% ===== >> "data\raw\daily_inference.log"
".venv\Scripts\python.exe" scripts\run_inference_v2.py >> "data\raw\daily_inference.log" 2>&1
set INFER_RC=%ERRORLEVEL%
if not "%INFER_RC%"=="0" set EXITCODE=%INFER_RC%
if not "%INFER_RC%"=="0" echo ===== INFERENCE EXIT CODE %INFER_RC% %DATE% %TIME% ===== >> "data\raw\daily_inference.log"
if "%INFER_RC%"=="3" (
    echo ===== TOO EARLY: nothing predicted, re-run scheduler and dashboard publish skipped ===== >> "data\raw\daily_inference.log"
    exit /b %INFER_RC%
)
if "%INFER_RC%"=="4" echo ===== NOTHING PREDICTED, odds source or schedule failed or every game skipped: re-run scheduler skipped, dashboard publish still runs ===== >> "data\raw\daily_inference.log"
if "%INFER_RC%"=="5" echo ===== NO BETS RECOMMENDED, injury report missing or stale: predictions logged, dashboard publish still runs ===== >> "data\raw\daily_inference.log"
if "%INFER_RC%"=="4" goto PUBLISH
REM games whose lines are not posted yet: schedule a re-run (scripts\run_odds_rerun.bat) and write the dashboard's status file
".venv\Scripts\python.exe" scripts\schedule_odds_rerun.py >> "data\raw\daily_inference.log" 2>&1
set STEP_RC=%ERRORLEVEL%
if "%EXITCODE%"=="0" set EXITCODE=%STEP_RC%
:PUBLISH
".venv\Scripts\python.exe" scripts\export_dashboard_data.py --publish >> "data\raw\daily_inference.log" 2>&1
set STEP_RC=%ERRORLEVEL%
if "%EXITCODE%"=="0" set EXITCODE=%STEP_RC%
echo ===== DONE exit code %EXITCODE% %DATE% %TIME% ===== >> "data\raw\daily_inference.log"
exit /b %EXITCODE%
