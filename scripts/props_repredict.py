"""
Re-predict saved folds with the current calibration step (no retraining of the models): the fold's own models are loaded, the dispersion scaler and NB dispersion are re-fitted
on the calibration window exactly as in training (now with a scaler that can narrow as well as widen), and the test season is predicted again.

    python scripts/props_repredict.py --pairs v3_dev:2021-22 v3_dev:2022-23 v3_dev:2023-24 v3_seen_2024-25:2024-25 --out-version v4

Writes predictions and a small bundle (scaler, isotonic maps, the source version) under the new version; the models stay in the source version.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.features import build  # noqa: E402
from props.models import data, folds as folds_mod, pipeline, registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', nargs='+', required=True, help='source_version:season')
    ap.add_argument('--out-version', required=True)
    ap.add_argument('--features-db', default='data/props_features_v3.db')
    ap.add_argument('--sims', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    df = data.load_frame(a.features_db)
    registry.write_manifest(a.out_version, {'pairs': a.pairs, 'sims': a.sims, 'seed': a.seed, 'features_db': a.features_db, 'features_db_sha256': registry.sha256_file(a.features_db),
                                            'change': 'dispersion scaler may narrow (MIN_C) as well as widen; everything else as the source versions'})
    for pair in a.pairs:
        src, season = pair.split(':')
        t = time.time()
        fold = folds_mod.make_fold(df, season)
        bundle = registry.load_bundle(src, season)
        idx, pmfs, calibrator, played, scaler = pipeline.predict_fold(df, fold, bundle, bundle['families'], a.sims, a.seed)
        registry.save_bundle(a.out_version, season, {'source_version': src, 'families': bundle['families'], 'dispersion': scaler, 'calibrator': calibrator})
        registry.save_predictions(a.out_version, season, idx, pmfs)
        logging.info(f'{src} {season}: re-predicted in {time.time() - t:.0f}s; c table {({k: round(v, 3) for k, v in scaler.c.items() if k[0] in ("points", "assists", "pra")})}')


if __name__ == '__main__':
    main()
