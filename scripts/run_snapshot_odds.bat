@echo off
REM Scheduled by Windows Task Scheduler ("NBA Odds Snapshots"): timestamped odds + latest injury report.
cd /d "%~dp0.."
".venv\Scripts\python.exe" scripts\snapshot_odds.py >> "data\raw\snapshot_odds.log" 2>&1
