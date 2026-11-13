"""
High-draw predictions (default 20,000 simulations) for the played player-games that have a consensus market quote, from a saved fold of the frozen pipeline.

    python scripts/props_phase5_predict.py --version v3_dev --season 2023-24 --prices data/props_betting/prices_2023-24.pkl --out-version v3_dev_p5

Uses the fold's own models (refit on fit + calibration when the fold was trained with --refit-all) and its saved dispersion scaler. The rolling recalibration R2b is NOT
applied here: it is applied afterwards from the full-season 4,000-draw forecasts (props/betting/evaluate), because it needs every completed game, not only the quoted ones.

Research options (plan reports/props/audit_2026-11/WP6_WP7_experiment_plan.md, steps S4a, S4b, S13). Without them the script behaves exactly as before.

    --all-played     no price file: predict EVERY played test row of the season (lbl_played == 1 and lbl_minutes > 0, all three decision times), e.g.
                     python scripts/props_phase5_predict.py --version a11_r0 --season 2023-24 --all-played --n-sims 4000 --out-version a11_r0_raw --registry-root <dir> --save-raw
    --save-raw       also store each stat's PRE-dispersion pmf as raw_<stat>.npy next to pred_<stat>.npy; row i of every array is row i of index.csv
    --registry-root  RESEARCH mode: read --version and write --out-version under this folder only (no fallback to data/props_models); refuses --season 2025-26 and later.
                     Another season can be added to an existing --out-version in research mode (a new <season> folder; nothing is overwritten); the manifest keeps one entry per season.

Memory held for the pmfs of one season: rows x 463 columns x 4 bytes per set (stat caps 60, 30, 30, 15, 100, 80, 80, 60 give 463 columns with the zero bin),
i.e. 1,852 bytes per row: about 0.16 GB per set for the 84k played rows of a season, 0.31 GB with --save-raw (the arrays are preallocated, not concatenated).
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

PMF_COLUMNS = sum(cap + 1 for cap in oc.CAPS.values())
MANIFEST_COMPAT = ('source_version', 'n_sims', 'seed', 'features_db_sha256', 'mode', 'save_raw', 'recalibration')


def select_rows(played, keys=None):
    """The rows to predict: every played test row (--all-played), or the played rows that carry a consensus quote (default; the inner merge keeps the played rows' order)."""
    if keys is None:
        return played.reset_index(drop=True)
    return played.merge(keys, on=['game_id', 'player_id', 'minutes_before'], how='inner')


def simulate(final, families, dispersion, rows, n_sims, seed, chunk_rows, save_raw=False):
    """Chunked simulation. Returns (minutes_mean, dispersed pmfs by stat, raw pmfs by stat or None); the raw pmf is the one before DispersionScaler.transform."""
    n = len(rows)
    disp = {s: np.empty((n, cap + 1), dtype=np.float32) for s, cap in oc.CAPS.items()}
    raw = {s: np.empty((n, cap + 1), dtype=np.float32) for s, cap in oc.CAPS.items()} if save_raw else None
    mins = []                                                                       # concatenated like before, so index.csv keeps the dtype of minutes_mean
    t0 = time.time()
    for lo in range(0, n, chunk_rows):
        hi = min(lo + chunk_rows, n)
        chunk = rows.iloc[lo:hi]
        res = pipeline.distributions(final, chunk, families, n_sims, seed + lo)
        res.pop('sim_means', None)
        mm = res.pop('minutes_mean')
        for s in oc.CAPS:
            disp[s][lo:hi] = dispersion.transform(s, res[s], mm)
            if save_raw:
                raw[s][lo:hi] = res[s]
        mins.append(mm)
        logging.info(f'{hi}/{n} rows, {time.time() - t0:.0f}s')
    return np.concatenate(mins), disp, raw


def write_manifest(a, root, research, rows, fdb_sha, prices_sha):
    mode = 'all_played' if a.all_played else 'quoted'
    if not (research or a.all_played or a.save_raw):                               # the existing mode: the manifest exactly as before
        registry.write_manifest(a.out_version, {'source_version': a.version, 'season': a.season, 'n_sims': a.n_sims, 'seed': a.seed, 'rows': int(rows),
                                                'features_db': a.features_db, 'features_db_sha256': fdb_sha, 'recalibration': 'none here (R2b applied downstream)'}, root=root)
        return
    common = {'source_version': a.version, 'n_sims': a.n_sims, 'seed': a.seed, 'features_db': a.features_db, 'features_db_sha256': fdb_sha,
              'recalibration': 'none here (R2b applied downstream)', 'mode': mode, 'save_raw': bool(a.save_raw), 'registry_root': str(root)}
    entry = {'rows': int(rows), 'prices': a.prices, 'prices_sha256': prices_sha, 'chunk_rows': a.chunk_rows}
    path = registry.version_dir(a.out_version, root) / 'manifest.json'
    if research and path.exists():                                                  # another season of the same prediction set: new folder, compatible settings, nothing overwritten
        old = registry.find_manifest(a.out_version, (root,))[0]
        diff = [k for k in MANIFEST_COMPAT if old.get(k) != common[k]]
        if diff:
            raise SystemExit(f'--out-version {a.out_version} exists with different settings ({diff}): a version is immutable, use a new name')
        if a.season in old.get('seasons', {}):
            raise FileExistsError(f'{a.out_version}/{a.season} already exists')
        registry.update_manifest(a.out_version, {'seasons': {**old.get('seasons', {}), a.season: entry}}, root=root)
        return
    registry.write_manifest(a.out_version, {**common, 'seasons': {a.season: entry}}, root=root)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', required=True)
    ap.add_argument('--season', required=True)
    ap.add_argument('--prices', default=None, help='price file (cons, bets) of the season; required unless --all-played')
    ap.add_argument('--out-version', required=True)
    ap.add_argument('--features-db', default='data/props_features_v3.db')
    ap.add_argument('--n-sims', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--chunk-rows', type=int, default=30000)
    ap.add_argument('--all-played', action='store_true', help='no price file: predict every played test row of the season')
    ap.add_argument('--save-raw', action='store_true', help='also store the pre-dispersion pmfs as raw_<stat>.npy')
    ap.add_argument('--registry-root', default=None, help='RESEARCH mode: read --version and write --out-version under this folder only; refuses 2025-26 and later')
    a = ap.parse_args(argv)
    if a.all_played and a.prices:
        ap.error('--all-played uses no price file: drop --prices')
    if not a.all_played and not a.prices:
        ap.error('--prices is required unless --all-played')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    root = registry.resolve_root(a.registry_root)
    research = a.registry_root is not None
    if research:
        registry.refuse_holdout([a.season], '--season')
        if registry.version_exists(a.out_version, registry.ROOT):
            raise FileExistsError(f'--out-version {a.out_version!r} already exists under {registry.ROOT}: a research version never shares a name with a production one')
        if not (registry.version_dir(a.version, root) / a.season / 'bundle.pkl').exists():
            raise FileNotFoundError(f'{a.version}/{a.season}/bundle.pkl not found under the research root {root} (research mode never reads data/props_models)')
        if (registry.version_dir(a.out_version, root) / a.season).exists():
            raise FileExistsError(f'{registry.version_dir(a.out_version, root) / a.season} already exists: a registry version is immutable')
    keys, prices_sha = None, None
    if not a.all_played:
        cons, _ = pickle.load(open(a.prices, 'rb'))
        keys = cons[['game_id', 'player_id', 'minutes_before']].drop_duplicates()
        prices_sha = registry.sha256_file(a.prices) if research else None
    frame = data.load_frame(a.features_db)
    if research:                                                                    # structural guard: no holdout row is in memory
        frame = frame[frame['season'] < registry.HOLDOUT_SEASON].reset_index(drop=True)
    fold = folds_mod.make_fold(frame, a.season)
    bundle = registry.load_bundle(a.version, a.season, root=root)
    final = dict(bundle, **bundle['refit']) if 'refit' in bundle else bundle
    if 'refit' in bundle:
        final['rates'].nb_k = bundle['rates'].nb_k
    test = frame.iloc[fold.test_idx]
    played = test[(test['lbl_played'] == 1) & (test['lbl_minutes'] > 0)]
    rows = select_rows(played, keys)
    logging.info(f'{len(rows)} ' + ('played rows (all played)' if a.all_played else f'quoted played rows of {len(played)} played rows'))
    write_manifest(a, root, research, len(rows), registry.sha256_file(a.features_db), prices_sha)
    mins, disp, raw = simulate(final, bundle['families'], bundle['dispersion'], rows, a.n_sims, a.seed, a.chunk_rows, a.save_raw)
    idx = rows[['game_id', 'player_id', 'minutes_before', 'season', 'game_date']].reset_index(drop=True)
    idx['minutes_mean'] = mins
    registry.save_predictions(a.out_version, a.season, idx, disp, root=root)
    if raw is not None:
        registry.save_raw_predictions(a.out_version, a.season, raw, root=root)
    logging.info('saved')


if __name__ == '__main__':
    main()
