"""
Train the props models walk-forward and write a versioned registry entry plus a smoke report.

    python scripts/train_props_models.py --version v1_base                 # development folds 2021-22, 2022-23, 2023-24, base features
    python scripts/train_props_models.py --version v1_env --spec with_env   # the market-environment variant
    python scripts/train_props_models.py --version v1_no_tm --exclude-groups teammates   # an ablation

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
from props.models import common, data, folds as folds_mod, pipeline, registry, report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', required=True)
    ap.add_argument('--spec', default='base', choices=['base', 'with_env'])
    ap.add_argument('--exclude-groups', nargs='*', default=[])
    ap.add_argument('--folds', nargs='*', default=list(folds_mod.DEV_TEST_SEASONS))
    ap.add_argument('--sims', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--report', default=None)
    ap.add_argument('--families-from', default=None, help='registry version whose chosen rate families are reused (frozen configuration): required for 2024-25 and 2025-26')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    held_out = set(args.folds) & {folds_mod.VALIDATION_SEASON, folds_mod.FINAL_SEASON}
    if held_out and not args.families_from:
        raise SystemExit(f'{sorted(held_out)} are not development folds: pass --families-from <frozen version> so nothing is selected on them')
    df = data.load_frame()
    spec = data.FeatureSpec(args.spec, frozenset(args.exclude_groups))
    registry.write_manifest(args.version, {'spec': args.spec, 'exclude_groups': args.exclude_groups, 'folds': args.folds, 'sims': args.sims, 'seed': args.seed,
                                           'rows': int(len(df)), 'calib_frac': folds_mod.CALIB_FRAC, 'first_season': folds_mod.FIRST_SEASON})
    fold_objs, bundles = {}, {}
    for s in args.folds:
        t = time.time()
        fold_objs[s] = folds_mod.make_fold(df, s)
        bundles[s] = pipeline.train_fold(df, fold_objs[s], spec, score_on_test=not args.families_from)
        logging.info(f'{s}: trained in {time.time() - t:.0f}s {fold_objs[s].describe()}')
    if args.families_from:
        frozen = json.loads((registry.version_dir(args.families_from) / 'manifest.json').read_text(encoding='utf-8'))
        chosen, table = frozen['families'], pd.DataFrame(frozen['family_table'])
        registry.update_manifest(args.version, {'families': chosen, 'family_table': frozen['family_table'], 'families_from': args.families_from})
    else:
        chosen, table = pipeline.select_families(list(bundles.values()))
        registry.update_manifest(args.version, {'families': chosen, 'family_table': table.to_dict('records')})
    logging.info('rate families chosen:\n' + table.to_string(index=False))
    smoke, calib = {}, {}
    for s in args.folds:
        t = time.time()
        idx, pmfs, calibrator, played, scaler = pipeline.predict_fold(df, fold_objs[s], bundles[s], chosen, args.sims, args.seed)
        bundles[s]['calibrator'], bundles[s]['dispersion'], bundles[s]['families'] = calibrator, scaler, chosen
        common.log_config('dispersion_scaler', season=s, c={f'{k[0]}:{k[1]}': round(v, 4) for k, v in scaler.c.items()})
        registry.save_bundle(args.version, s, bundles[s])
        registry.save_predictions(args.version, s, idx, pmfs)
        smoke[s] = report.smoke_table(played, pmfs)
        calib[s] = report.calibration_table(played, pmfs, calibrator)
        logging.info(f'{s}: predicted and calibrated in {time.time() - t:.0f}s\n' + smoke[s].round(3).to_string(index=False))
    out = args.report or f'reports/props/phase3_training_report_{args.version}.md'
    notes = (f'Version `{args.version}`, feature set `{args.spec}`, excluded groups {args.exclude_groups}, folds {args.folds}, {args.sims} simulations per player-game. '
             'The stored pmfs are dispersion-calibrated (Stage F0); the isotonic layer is reported (calibrated columns) but not applied. Smoke check only: no confidence intervals, no ablations, no market comparison. Phase 4 is the formal evaluation.\n')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(report.markdown(f'Phase 3 training report ({args.version})', smoke, calib, table, {}, notes), encoding='utf-8')
    logging.info(f'wrote {out}')


if __name__ == '__main__':
    main()
