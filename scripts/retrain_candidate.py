"""
Retrain a candidate model set on all games to date and gate it against the deployed set. NEVER changes what is deployed.

Writes models/saved/*_with_odds_cand_<YYYYMMDD>.* (a new suffix; an existing one is refused) and reports/retrain/retrain_<tag>.json.
See team/retrain.py. Runs in minutes with the default --tuner grid (deterministic); --tuner flaml spends --time-budget seconds per model.

Usage:
    python scripts/retrain_candidate.py
    python scripts/retrain_candidate.py --tuner flaml --time-budget 600
Scheduled (weekly during the season): scripts/run_weekly_retrain.bat
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from team.data_paths import latest_training_set
from team.retrain import GATE_DAYS, run_retrain
from team.train_baseline import CALIB_HOLDOUT_DAYS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tuner', choices=['grid', 'flaml'], default='grid')
    ap.add_argument('--time-budget', type=int, default=600, help='FLAML seconds per model (only with --tuner flaml)')
    ap.add_argument('--holdout-days', type=int, default=CALIB_HOLDOUT_DAYS, help='calibration block = the last N days')
    ap.add_argument('--gate-days', type=int, default=GATE_DAYS, help='held-out recent window used to compare with the deployed set')
    ap.add_argument('--tag', help='suffix tag (default: today as YYYYMMDD)')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    report = run_retrain(latest_training_set(), tuner=args.tuner, time_budget=args.time_budget, holdout_days=args.holdout_days,
                         gate_days=args.gate_days, tag=args.tag)
    verdict = report['gate'].get('verdict')
    print(f"candidate {report['candidate_suffix']} trained through {report['candidate']['trained_through']}; "
          f"gate: {'PASS' if verdict and verdict['ok'] else 'FAIL ' + str(verdict['worse']) if verdict else report['gate']}")
    sys.exit(0 if (verdict is None or verdict['ok']) else 2)


if __name__ == '__main__':
    main()
