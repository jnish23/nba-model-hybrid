"""What the opt-in does at a retrain: trains a scratch candidate on the 2026-11-01 training set with total_calibration=True (tuner='grid', deterministic),
into reports/audit_pregame_2026-11/A6_totals_calibration/scratch/optin_candidate/ - never models/saved/. Reports the fitted weights, the raw versus calibrated
calibration-block figures, the set hash, and the candidate's forecast on the last 120 days of games it has seen (in-sample for the refit models: illustrative only).
  C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe reports/audit_pregame_2026-11/A6_totals_calibration/scratch_retrain_with_optin.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import matplotlib                                                                  # noqa: E402
matplotlib.use('Agg')
from team import market_models as mm, model_registry as mr                            # noqa: E402
from team.data_paths import latest_training_set                                      # noqa: E402
from team.train_baseline import BaselineModel                                        # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'scratch' / 'optin_candidate'


def main():
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)
    assert 'models' + '/saved' not in str(OUT).replace('\\', '/')
    csv = latest_training_set()
    m = BaselineModel(csv, model_dir=str(OUT))
    suffix = '_a6_optin_scratch'
    m.train_and_save_all(use_odds=True, save_suffix=suffix, time_budget=1, tuner='grid', total_calibration=True, overwrite=True)
    meta = json.loads((OUT / f'calibration_meta{suffix}.json').read_text())
    man = mr.read_manifest(suffix, OUT)
    b = mr.load_bundle(suffix, OUT)
    cal = b.calibrator_total
    lines = np.arange(205, 251, 5.0)
    info = {'training_set': str(csv), 'calib_set_size': meta['calib_set_size'], 'calib_date_range': meta['calib_date_range'],
            'calibrator_params': cal.params, 'weight_by_opening_total': {str(int(x)): float(cal.weight(np.array([x]))[0]) for x in lines},
            'total_mae_raw_calib': meta['total_mae_raw'], 'total_mae_calibrated_calib': meta['total_mae'],
            'total_std_raw_calib': meta['total_std_raw'], 'total_std_calibrated_calib': meta['total_std'],
            'total_bias_raw_calib': meta['total_bias_raw'], 'total_bias_calibrated_calib': meta['total_bias'],
            'sigma_total_mean_calib': meta.get('total_sigma_mean_calib'), 'files': sorted(man['files']), 'model_set_hash': man['model_set_hash'],
            'note': 'calibration-block MAE/bias/std are in-sample for the weights (2 parameters); the walk-forward evidence is E27'}
    (HERE / 'scratch_retrain_with_optin.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
