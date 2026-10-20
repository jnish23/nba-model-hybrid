#!/bin/bash
# Descriptive feature-group ablations: one group dropped at a time, retrained on one fold with the frozen v2_base families.
# Dev fold 2023-24 first, then the validation season 2024-25. Nothing is selected from these; they are read in the Phase 4 report.
PY=/c/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe
mkdir -p data/raw/ablation_reports
for season in 2023-24 2024-25; do
  for g in injury teammates opponent team_context rest schedule role conversion; do
    v="abl_${g}_${season}"
    if [ -d "data/props_models/$v" ]; then echo "skip $v"; continue; fi
    echo "$(date +%T) start $v"
    $PY scripts/train_props_models.py --version "$v" --folds "$season" --families-from v2_base --exclude-groups "$g" --report "data/raw/ablation_reports/$v.md" > "data/raw/ablation_reports/$v.log" 2>&1
    echo "$(date +%T) done $v exit $?"
  done
done
echo ALL_DONE
