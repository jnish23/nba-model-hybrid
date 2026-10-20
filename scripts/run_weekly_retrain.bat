@echo off
REM NOT registered anywhere yet: proposed weekly Task Scheduler job ("NBA Weekly Retrain", e.g. Mondays 04:00 in-season).
REM Rebuilds the training set from the current database, then trains a CANDIDATE model set (new suffix) and gates it against the
REM deployed set. It never changes the deployed set: promotion = DEPLOYED_SUFFIX in team\model_registry.py, in a reviewed commit.
REM Exit code 2 = the candidate is worse than the deployed set on the gate window.
cd /d "%~dp0.."
echo ===== %DATE% %TIME% ===== >> "data\raw\weekly_retrain.log"
".venv\Scripts\python.exe" scripts\generate_training_set.py >> "data\raw\weekly_retrain.log" 2>&1
".venv\Scripts\python.exe" scripts\retrain_candidate.py >> "data\raw\weekly_retrain.log" 2>&1
