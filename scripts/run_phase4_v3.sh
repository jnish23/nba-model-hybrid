#!/bin/bash
# Audit-fix runs on DEVELOPMENT folds only (frozen v2_base rate families): v3 dataset (injury publish lag, strict env timing) + forward-chaining minutes bank.
PY=/c/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe
mkdir -p data/raw/ablation_reports
echo "$(date +%T) start v3_norefit_2023-24"
$PY scripts/train_props_models.py --version v3_norefit_2023-24 --features-db data/props_features_v3.db --folds 2023-24 --families-from v2_base --report data/raw/ablation_reports/v3_norefit_2023-24.md > data/raw/ablation_reports/v3_norefit_2023-24.log 2>&1
echo "$(date +%T) done exit $?"
echo "$(date +%T) start v3_dev (all fixes, refit-all)"
$PY scripts/train_props_models.py --version v3_dev --features-db data/props_features_v3.db --folds 2021-22 2022-23 2023-24 --families-from v2_base --refit-all --report reports/props/phase4_v3_dev_run.md > data/raw/ablation_reports/v3_dev.log 2>&1
echo "$(date +%T) done exit $?"
echo ALL_DONE
