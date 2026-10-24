"""
Train a LIVE props model: the frozen v3 pipeline (rate families from v2_base, refit on fit + calibration, dispersion floor 1, no rolling recalibration) on every completed season,
for the season about to be played. A new fold of the same configuration, never a new model. The version is immutable; the retrain policy is in reports/props/live_pipeline_plan.md.

    python scripts/train_props_live.py --version v3_live_2026-27 --next-season 2026-27
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.models import data, folds as folds_mod, pipeline, registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--next-season', required=True)
    ap.add_argument('--features-db', default='data/props_features_v3.db')
    ap.add_argument('--families-from', default='v2_base')
    ap.add_argument('--sims', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    df = data.load_frame(a.features_db)
    fold = folds_mod.make_live_fold(df, a.next_season)
    spec = data.FeatureSpec('base')
    families = json.loads((registry.version_dir(a.families_from) / 'manifest.json').read_text(encoding='utf-8'))['families']
    trained_through = str(df.iloc[fold.calib_idx]['game_date'].max())
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    registry.write_manifest(a.version, {'kind': 'live', 'next_season': a.next_season, 'trained_through': trained_through, 'policy': 'pre-season training; monthly refits are new immutable versions',
                                        'spec': 'base', 'families': families, 'families_from': a.families_from, 'refit_all': True, 'recalibration': 'none', 'dispersion_floor': 1.0,
                                        'fit_rows': int(len(fold.fit_idx)), 'calib_rows': int(len(fold.calib_idx)), 'calibration_starts': fold.cutoff_date, 'sims': a.sims, 'seed': a.seed,
                                        'features_db': a.features_db, 'features_db_sha256': registry.sha256_file(a.features_db), 'rows': int(len(df)), 'code_commit': head})
    t = time.time()
    bundle = pipeline.train_fold(df, fold, spec, score_on_test=False, refit_all=True)
    logging.info(f'trained in {time.time() - t:.0f}s ({len(fold.fit_idx)} fit rows, {len(fold.calib_idx)} calibration rows, through {trained_through})')
    scaler, calibrator = pipeline.calibrate_fold(df, fold, bundle, families, a.sims, a.seed)
    bundle.update({'dispersion': scaler, 'calibrator': calibrator, 'families': families})
    registry.save_bundle(a.version, a.next_season, bundle)
    registry.update_manifest(a.version, {'dispersion_c': {f'{k[0]}:{k[1]}': round(v, 4) for k, v in scaler.c.items()}, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
    logging.info(f'saved {a.version} in {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
