"""
Train a LIVE props model: the frozen v3 pipeline (rate families from v2_base, refit on fit + calibration, dispersion floor 1, no rolling recalibration) on every completed season,
for the season about to be played. A new fold of the same configuration, never a new model. The version is immutable (registry.write_manifest refuses an existing manifest, and runs
before any training); the retrain policy is in reports/props/live_pipeline_plan.md.

    python scripts/train_props_live.py --version v3_live_2026-27 --next-season 2026-27

A candidate (plan reports/props/audit_2026-11/A2_live_plan.md) uses its own features database and, when decided, the other feature spec:

    python scripts/train_props_live.py --version v4_live_2026-27c1 --next-season 2026-27 --features-db data/props_features_v4_pregame_all.db --spec base

The frame must hold no row of --next-season or later (make_live_fold raises): build a candidate database with --max-date 2026-07-31.
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

SPECS = ('base', 'with_env')


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--next-season', required=True)
    ap.add_argument('--features-db', default='data/props_features_v3.db')
    ap.add_argument('--families-from', default='v2_base')
    ap.add_argument('--spec', default='base', choices=SPECS, help='feature set (props.models.data.FeatureSpec); with_env adds the market-environment group')
    ap.add_argument('--sims', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args(argv)


def features_db_sidecar(features_db):
    """The <db>.manifest.json that scripts/build_props_features.py --membership writes next to a research build (membership, max_date, hashes), or None (the production database has none)."""
    p = Path(str(features_db) + '.manifest.json')
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding='utf-8'))
    return {k: m.get(k) for k in ('membership', 'max_date', 'sha256', 'code_commit', 'linked_cache_sha256')}


def build_manifest(a, df, fold, families, head, features_db_sha256):
    """The manifest written before training. Default arguments give exactly the keys and values of the original script."""
    trained_through = str(df.iloc[fold.calib_idx]['game_date'].max())
    man = {'kind': 'live', 'next_season': a.next_season, 'trained_through': trained_through, 'policy': 'pre-season training; monthly refits are new immutable versions',
           'spec': a.spec, 'families': families, 'families_from': a.families_from, 'refit_all': True, 'recalibration': 'none', 'dispersion_floor': 1.0,
           'fit_rows': int(len(fold.fit_idx)), 'calib_rows': int(len(fold.calib_idx)), 'calibration_starts': fold.cutoff_date, 'sims': a.sims, 'seed': a.seed,
           'features_db': a.features_db, 'features_db_sha256': features_db_sha256, 'rows': int(len(df)), 'code_commit': head}
    sidecar = features_db_sidecar(a.features_db)
    if sidecar is not None:                                    # only a research-built database has one: the default call writes no extra key
        man['features_db_build'] = sidecar
    return man


def main(argv=None):
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    df = data.load_frame(a.features_db)
    fold = folds_mod.make_live_fold(df, a.next_season)
    spec = data.FeatureSpec(a.spec)
    families = json.loads((registry.version_dir(a.families_from) / 'manifest.json').read_text(encoding='utf-8'))['families']
    trained_through = str(df.iloc[fold.calib_idx]['game_date'].max())
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    registry.write_manifest(a.version, build_manifest(a, df, fold, families, head, registry.sha256_file(a.features_db)))
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
