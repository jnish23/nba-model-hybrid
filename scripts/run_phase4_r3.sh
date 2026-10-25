#!/bin/bash
# R3 (pre-registered in reports/props/phase4_protocol_update.md): recency weights, half-life 730 days, on the corrected pipeline with refit-all. Dev folds 2022-23 and 2023-24 only.
PY=/c/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe
until grep -q ALL_DONE data/raw/v3_chain.log; do sleep 30; done
echo "$(date +%T) start v3r_dev"
$PY scripts/train_props_models.py --version v3r_dev --features-db data/props_features_v3.db --folds 2022-23 2023-24 --families-from v2_base --refit-all --recency-half-life-days 730 --report data/raw/ablation_reports/v3r_dev.md > data/raw/ablation_reports/v3r_dev.log 2>&1
echo "$(date +%T) done exit $?"
echo ALL_DONE
