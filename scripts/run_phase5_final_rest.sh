#!/bin/bash
# 2025-26 scoring, declared sensitivities and Phase 4 criteria (pre-registration: reports/props/phase5_revised_rule_preregistration.md). Each step runs once.
PY=/c/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe
$PY scripts/props_phase5_dev.py --season 2025-26 --p5-version v3_final_p5 --full-version v3_final_2025-26 --carry-season 2024-25 --carry-version v3_seen_2024-25 --frozen-rule reports/props/phase5_revised_rule.frozen_rule.json --out reports/props/phase5_final_2025-26_S1_R2b.md > data/raw/final_S1.log 2>&1; echo "S1 $?" >> data/raw/final_rest.log
$PY scripts/props_phase5_dev.py --season 2025-26 --p5-version v3_final_p5 --full-version v3_final_2025-26 --no-r2b --frozen-rule reports/props/phase5_dev_2023-24_noR2b.frozen_rule.json --out reports/props/phase5_final_2025-26_S2_original.md > data/raw/final_S2.log 2>&1; echo "S2 $?" >> data/raw/final_rest.log
$PY scripts/props_phase4_eval.py --version v3_final_2025-26 --season 2025-26 --features-db data/props_features_v3.db --out reports/props/phase4_final_2025-26.md > data/raw/final_p4.log 2>&1; echo "P4 $?" >> data/raw/final_rest.log
$PY scripts/props_phase4_eval.py --version v3_final_2025-26 --season 2025-26 --features-db data/props_features_v3.db --r2b --r2b-carry-version v3_seen_2024-25 --r2b-carry-season 2024-25 --out reports/props/phase4_final_2025-26_R2b.md > data/raw/final_p4r.log 2>&1; echo "P4R $?" >> data/raw/final_rest.log
echo ALL >> data/raw/final_rest.log
