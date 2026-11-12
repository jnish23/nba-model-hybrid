#!/bin/bash
# Re-runs every A2 check and stores stdout in logs/. Run from the worktree root:  bash reports/audit_pregame_2026-11/A2_backtest_validity/run_all.sh
PY=C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe
D=reports/audit_pregame_2026-11/A2_backtest_validity
mkdir -p $D/logs
for s in check1_recompute check1b_leakprobe check1c_diff_identity check2_segments check3_bpm_recompute check3b_bpm_oddities check4_injury_timing check4b_inj_n_out \
         check5a_snapshots check5a2_close_is_tip_line check5b_clean_odds check5c_big_moves_detail check6_folds check7_metrics check7c_readme_and_ml_sensitivity \
         check8_placebo check9_p2 check9c_open_chosen_close_settled; do
  $PY $D/$s.py 2>&1 | grep -v "Warning\|return np.where\|INFO\|stddev\|spearmanr\|ps2.loc\|^  c /=" > $D/logs/$s.log
done
# same prediction-based checks on E13_combined_winners (the deployed form: E1 spread residual + E3 ML stack + E4 sigma)
for s in check7_metrics check9_p2 check9c_open_chosen_close_settled; do
  A2_RUN=E13_combined_winners $PY $D/$s.py 2>&1 | grep -v "Warning\|return np.where\|INFO" > $D/logs/${s}_E13.log
done
# tests (slow): bash is not re-run here; see check10_pytest_*.txt
