"""
Train the props models walk-forward and write a versioned registry entry plus a smoke report.

    python scripts/train_props_models.py --version v1_base                 # development folds 2021-22, 2022-23, 2023-24, base features
    python scripts/train_props_models.py --version v1_env --spec with_env   # the market-environment variant
    python scripts/train_props_models.py --version v1_no_tm --exclude-groups teammates   # an ablation
    python scripts/train_props_models.py --version a11_r0 --features-db data/props_features_v3.db --folds 2021-22 2022-23 2023-24 2024-25 --families-from v2_base --refit-all --registry-root <dir>   # research mode (see props/models/registry.py)

The 2024-25 validation season and the 2025-26 holdout are NOT run by default; ask for them explicitly with --folds, and only after the configuration is
frozen. Every model configuration tried is appended to data/props_models/config_log.jsonl (the count that multiple-testing corrections need).
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.features import build  # noqa: E402
from props.models import common, data, folds as folds_mod, pipeline, registry, report  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', required=True)
    ap.add_argument('--spec', default='base', choices=['base', 'with_env'])
    ap.add_argument('--exclude-groups', nargs='*', default=[])
    ap.add_argument('--recency-half-life-days', type=float, default=None, help='exponential recency weights on training rows (age relative to the end of the training window)')
    ap.add_argument('--features-db', default=None, help='dataset database (default data/props_features.db)')
    ap.add_argument('--refit-all', action='store_true', help='refit the models on fit + calibration windows before predicting the test season (dispersion maps still come from the fit-only models)')
    ap.add_argument('--exclude-features', nargs='*', default=[], help='single feature columns to drop (e.g. the R1 season-position features)')
    ap.add_argument('--folds', nargs='*', default=list(folds_mod.DEV_TEST_SEASONS))
    ap.add_argument('--sims', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--report', default=None)
    ap.add_argument('--families-from', default=None, help='registry version whose chosen rate families are reused (frozen configuration): required for 2024-25 and 2025-26')
    ap.add_argument('--registry-root', default=None, help='RESEARCH mode: write the new version (manifest, bundles, pmfs, config log) under this folder instead of data/props_models; refuses the 2025-26 holdout and later folds; '
                                                          '--families-from is looked up here first, then in data/props_models')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    root = registry.resolve_root(args.registry_root)
    research = args.registry_root is not None
    if research:
        registry.refuse_holdout(args.folds, '--folds')
        registry.assert_new_version(args.version, root)                     # before the frame is loaded: fail in seconds, not after the read
        common.set_config_log(root / 'config_log.jsonl')                    # this run's configurations are counted under its own root
    held_out = set(args.folds) & {folds_mod.VALIDATION_SEASON, folds_mod.FINAL_SEASON}
    if held_out and not args.families_from:
        raise SystemExit(f'{sorted(held_out)} are not development folds: pass --families-from <frozen version> so nothing is selected on them')
    fdb = args.features_db or str(build.OUT_DB)
    df = data.load_frame(fdb)
    if research:                                                            # structural guard: no holdout row is in memory, whatever a fold asks for
        df = df[df['season'] < registry.HOLDOUT_SEASON].reset_index(drop=True)
    spec = data.FeatureSpec(args.spec, frozenset(args.exclude_groups), exclude_features=frozenset(args.exclude_features))
    registry.write_manifest(args.version, {'spec': args.spec, 'exclude_groups': args.exclude_groups, 'exclude_features': args.exclude_features, 'refit_all': args.refit_all, 'recency_half_life_days': args.recency_half_life_days, 'features_db': fdb, 'features_db_sha256': registry.sha256_file(fdb), 'folds': args.folds, 'sims': args.sims, 'seed': args.seed,
                                           'rows': int(len(df)), 'calib_frac': folds_mod.CALIB_FRAC, 'first_season': folds_mod.FIRST_SEASON}, root=root)
    if research:
        registry.update_manifest(args.version, {'registry_root': str(root), 'research_mode': True}, root=root)
    fold_objs, bundles = {}, {}
    for s in args.folds:
        t = time.time()
        fold_objs[s] = folds_mod.make_fold(df, s)
        bundles[s] = pipeline.train_fold(df, fold_objs[s], spec, score_on_test=not args.families_from, refit_all=args.refit_all, recency_half_life_days=args.recency_half_life_days)
        logging.info(f'{s}: trained in {time.time() - t:.0f}s {fold_objs[s].describe()}')
    if args.families_from:
        # research root first, then data/props_models (default mode: data/props_models only, as before). Only 'families' is needed: the surviving v2_base manifest is a stub without 'family_table'.
        frozen, frozen_path = registry.find_manifest(args.families_from, (root, registry.ROOT) if research else (registry.ROOT,))
        chosen = frozen['families']
        family_table = frozen.get('family_table') or [{'component': c, 'chosen': f} for c, f in chosen.items()]
        table = pd.DataFrame(family_table)
        registry.update_manifest(args.version, {'families': chosen, 'family_table': family_table, 'families_from': args.families_from, **({'families_from_manifest': str(frozen_path)} if research else {})}, root=root)
    else:
        chosen, table = pipeline.select_families(list(bundles.values()))
        registry.update_manifest(args.version, {'families': chosen, 'family_table': table.to_dict('records')}, root=root)
    logging.info('rate families chosen:\n' + table.to_string(index=False))
    smoke, calib = {}, {}
    for s in args.folds:
        t = time.time()
        idx, pmfs, calibrator, played, scaler = pipeline.predict_fold(df, fold_objs[s], bundles[s], chosen, args.sims, args.seed)
        bundles[s]['calibrator'], bundles[s]['dispersion'], bundles[s]['families'] = calibrator, scaler, chosen
        common.log_config('dispersion_scaler', season=s, c={f'{k[0]}:{k[1]}': round(v, 4) for k, v in scaler.c.items()})
        registry.save_bundle(args.version, s, bundles[s], root=root)
        registry.save_predictions(args.version, s, idx, pmfs, root=root)
        smoke[s] = report.smoke_table(played, pmfs)
        calib[s] = report.calibration_table(played, pmfs, calibrator)
        logging.info(f'{s}: predicted and calibrated in {time.time() - t:.0f}s\n' + smoke[s].round(3).to_string(index=False))
    out = args.report or (str(root / args.version / 'training_report.md') if research else f'reports/props/phase3_training_report_{args.version}.md')
    notes = (f'Version `{args.version}`, feature set `{args.spec}`, excluded groups {args.exclude_groups}, folds {args.folds}, {args.sims} simulations per player-game. '
             'The stored pmfs are dispersion-calibrated (Stage F0); the isotonic layer is reported (calibrated columns) but not applied. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.\n')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(report.markdown(f'Phase 3 training report ({args.version})', smoke, calib, table, {}, notes), encoding='utf-8')
    logging.info(f'wrote {out}')


if __name__ == '__main__':
    main()
