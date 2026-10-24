"""
High-draw predictions (default 20,000 simulations) for the played player-games that have a consensus market quote, from a saved fold of the frozen pipeline.

    python scripts/props_phase5_predict.py --version v3_dev --season 2023-24 --prices data/props_betting/prices_2023-24.pkl --out-version v3_dev_p5

Uses the fold's own models (refit on fit + calibration when the fold was trained with --refit-all) and its saved dispersion scaler. The rolling recalibration R2b is NOT
applied here: it is applied afterwards from the full-season 4,000-draw forecasts (props/betting/evaluate), because it needs every completed game, not only the quoted ones.
"""
import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.models import data, folds as folds_mod, outcome as oc, pipeline, registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--season', required=True)
    ap.add_argument('--prices', required=True)
    ap.add_argument('--out-version', required=True)
    ap.add_argument('--features-db', default='data/props_features_v3.db')
    ap.add_argument('--n-sims', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--chunk-rows', type=int, default=30000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cons, _ = pickle.load(open(a.prices, 'rb'))
    keys = cons[['game_id', 'player_id', 'minutes_before']].drop_duplicates()
    frame = data.load_frame(a.features_db)
    fold = folds_mod.make_fold(frame, a.season)
    bundle = registry.load_bundle(a.version, a.season)
    final = dict(bundle, **bundle['refit']) if 'refit' in bundle else bundle
    if 'refit' in bundle:
        final['rates'].nb_k = bundle['rates'].nb_k
    test = frame.iloc[fold.test_idx]
    played = test[(test['lbl_played'] == 1) & (test['lbl_minutes'] > 0)]
    rows = played.merge(keys, on=['game_id', 'player_id', 'minutes_before'], how='inner')
    logging.info(f'{len(rows)} quoted played rows of {len(played)} played rows')
    registry.write_manifest(a.out_version, {'source_version': a.version, 'season': a.season, 'n_sims': a.n_sims, 'seed': a.seed, 'rows': int(len(rows)),
                                            'features_db': a.features_db, 'features_db_sha256': registry.sha256_file(a.features_db), 'recalibration': 'none here (R2b applied downstream)'})
    parts, mins = {s: [] for s in oc.CAPS}, []
    t0 = time.time()
    for lo in range(0, len(rows), a.chunk_rows):
        chunk = rows.iloc[lo:lo + a.chunk_rows]
        res = pipeline.distributions(final, chunk, bundle['families'], a.n_sims, a.seed + lo)
        res.pop('sim_means', None)
        mm = res.pop('minutes_mean')
        for s in oc.CAPS:
            parts[s].append(bundle['dispersion'].transform(s, res[s], mm))
        mins.append(mm)
        logging.info(f'{min(lo + a.chunk_rows, len(rows))}/{len(rows)} rows, {time.time() - t0:.0f}s')
    idx = rows[['game_id', 'player_id', 'minutes_before', 'season', 'game_date']].reset_index(drop=True)
    idx['minutes_mean'] = np.concatenate(mins)
    registry.save_predictions(a.out_version, a.season, idx, {s: np.concatenate(v) for s, v in parts.items()})
    logging.info('saved')


if __name__ == '__main__':
    main()
