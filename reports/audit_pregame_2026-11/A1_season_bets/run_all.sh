#!/bin/sh
# run from the worktree root C:/Users/jorda/nba-pregame-audit
PY=C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe
D=reports/audit_pregame_2026-11/A1_season_bets
for s in 01_build_games 02_bets 03_enrich 04_results 05_model_vs_lines 06_dashboard 07_extra 08_exploratory; do
  $PY $D/$s.py > $D/log_$s.txt 2>&1 || echo "FAILED $s"
done
