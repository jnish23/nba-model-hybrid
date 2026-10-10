"""
Ablation: does adding the injury-report features (inj_*) improve the model, against the market?

Runs the same walk-forward backtest twice on the newest training set, with identical settings
(default hyperparameters, calibration, start season 2021-22 so every fold has >= 2 seasons of
injury-report data to learn from):
  A. all features including inj_*        -> models/saved/backtest_predictions_inj_on.csv
  B. the same set with inj_* removed     -> models/saved/backtest_predictions_inj_off.csv
Then compare both with scripts/evaluate_vs_market.py --preds ... (run automatically here).

Usage: python scripts/run_injury_ablation.py [--use-odds]
"""
import argparse
import glob
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_baseline import BaselineModel

START_SEASON = '2021-22'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--use-odds', action='store_true', help='Include opening odds as model inputs')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    data_path = Path(max(glob.glob('data/processed/training_set_v1_reduced_*.csv')))
    logging.info(f"Training set: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    inj_cols = [c for c in df.columns if c.startswith(('inj_', 'diff_inj_'))]
    if not inj_cols:
        raise SystemExit("Training set has no inj_* columns; regenerate it first")
    off_path = data_path.with_name(data_path.stem + '_no_inj.csv')
    df.drop(columns=inj_cols).to_csv(off_path, index=False)
    logging.info(f"Arm B training set without {len(inj_cols)} inj columns: {off_path}")

    tag = '_odds' if args.use_odds else ''
    for arm, path in (('inj_on', data_path), ('inj_off', off_path)):
        logging.info(f"=== Arm {arm} ===")
        model = BaselineModel(path)
        model.backtest_walk_forward(start_season=START_SEASON, use_tuned_params=False, use_odds=args.use_odds,
                                    save_suffix=f'_{arm}{tag}', calibrate=True)

    for arm in ('inj_on', 'inj_off'):
        preds = f'models/saved/backtest_predictions_{arm}{tag}.csv'
        out = Path(f'reports/market_eval_{arm}{tag}.txt')
        with open(out, 'w', encoding='utf-8') as f:
            subprocess.run([sys.executable, 'scripts/evaluate_vs_market.py', '--preds', preds], stdout=f,
                           stderr=subprocess.STDOUT, check=False)
        logging.info(f"Market evaluation for {arm}: {out}")


if __name__ == "__main__":
    main()
