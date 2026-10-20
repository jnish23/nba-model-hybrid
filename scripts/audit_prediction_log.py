"""
Check that a saved prediction log belongs to the model set on disk (read-only).

  1. provenance: do the rows carry model_set_hash, and does each hash match a model_manifest*.json whose files are unchanged?
  2. re-score:   score the logged games again with --suffix's model set and report the gap to what was logged
                 (distinct ml_prob values, max probability / spread / total gaps, log loss of both).

Usage:
    python scripts/audit_prediction_log.py --season 2025-26 --suffix _oos_2025-26 [--model-dir models/saved] [--log predictions/predictions_2025-26.csv]
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from team.data_paths import latest_training_set
from team.log_audit import provenance_report, rescore_gap
from team.market_models import predict_markets
from team.model_registry import MODEL_DIR, load_bundle
from team.train_baseline import BaselineModel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', default='2025-26')
    ap.add_argument('--suffix', help='model set to re-score with (default: _oos_<season>)')
    ap.add_argument('--model-dir', default=str(MODEL_DIR))
    ap.add_argument('--log', help='prediction CSV (default: predictions/predictions_<season>.csv)')
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    suffix = args.suffix or f'_oos_{args.season}'
    log = pd.read_csv(args.log or f'predictions/predictions_{args.season}.csv', dtype={'game_id': str})

    print(json.dumps({'provenance': provenance_report(log, args.model_dir)}, indent=2))

    bundle = load_bundle(suffix, Path(args.model_dir))
    loader = BaselineModel(latest_training_set())
    loader.load_data(use_odds=True)
    frame = loader.df[loader.df['seasonYear'] == args.season]
    print(json.dumps({'rescore': rescore_gap(log, bundle, frame, predict_markets)}, indent=2))


if __name__ == '__main__':
    main()
