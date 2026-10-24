#!/bin/bash
# The single 2025-26 scoring pipeline, steps 1-3 (reports/props/phase5_revised_rule_preregistration.md). No code changes between the freeze commit and the end of the run.
PY=/c/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe
echo "$(date +%T) 1/3 train the 2025-26 fold"
$PY scripts/train_props_models.py --version v3_final_2025-26 --features-db data/props_features_v3.db --folds 2025-26 --families-from v2_base --refit-all --report data/raw/ablation_reports/v3_final_2025-26.md > data/raw/final_train.log 2>&1 || { echo "TRAIN FAILED"; exit 1; }
echo "$(date +%T) 2/3 load 2025-26 prices"
$PY - > data/raw/final_prices.log 2>&1 <<'PYEOF' || { echo "PRICES FAILED"; exit 1; }
import pickle
from props.betting import prices
cons, bets = prices.load_season('data/props_odds.db', '2025-10-01', '2026-06-30')
pickle.dump((cons, bets), open('data/props_betting/prices_2025-26.pkl', 'wb'))
print('consensus rows', len(cons), 'bettable quotes', len(bets)); print(bets.groupby('book').size().to_string())
PYEOF
echo "$(date +%T) 3/3 20,000-draw predictions"
$PY scripts/props_phase5_predict.py --version v3_final_2025-26 --season 2025-26 --prices data/props_betting/prices_2025-26.pkl --out-version v3_final_p5 > data/raw/final_predict.log 2>&1 || { echo "PREDICT FAILED"; exit 1; }
echo "$(date +%T) CHAIN_DONE"
