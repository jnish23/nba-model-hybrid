"""E31b - does the (post-audit) feature selector help? All features versus selected, under the post-audit harness defaults.
Registered in reports/experiments/PREREGISTRATION.md ("E31b") before any E31b run. Exploratory: decides nothing.

python -m experiments.e31b_nosel --out <dir> [E31b_stage1 E31b_all]

<dir> holds the reference run `E25_with_odds` made under the harness defaults (residual selection, capped calibration block, refit, Platt;
python -m experiments.e25_regen with the dir as lib.OUT) and receives the arms (features.json, predictions.csv, RESULT.md). The script
refuses a reference made under the old scheme, so it cannot silently compare an arm with a different pipeline (E31 did that comparison
on the old one). The arms are cut from the selector's own intermediate lists (`stop_after`), so the reference is a subset of E31b_stage1,
which is a subset of E31b_all, by construction; a violation aborts the arm.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import lib                                                  # noqa: E402
from experiments.lib import TEST_SEASONS, load_games, load_run, run_split_params, run_walk_forward, split_fold   # noqa: E402
from experiments.report import compare, fmt_compare                          # noqa: E402

REFERENCE = 'E25_with_odds'
ARMS = {'E31b_stage1': 'stage1', 'E31b_all': 'pool'}
FLOOR = {'ll': 0.003, 'ae_spread': 0.011, 'ae_total': 0.005}


def require_new_defaults(name):
    """SystemExit unless run `name` was made with the post-audit defaults; returns its split parameters."""
    params = run_split_params(name)
    sel = lib.OUT / name / 'selection.json'
    rule = json.loads(sel.read_text()).get('selection') if sel.exists() else None
    if params['calib_cap'] is None or not params['refit_on_all'] or params['ml_calibrator'] != 'platt' or rule != 'residual':
        raise SystemExit(f'reference {name!r} was not made under the post-audit defaults (split {params}, selection {rule!r}); regenerate it first')
    return params


def arm_lists(m, sel_df, which):
    """{'ml','spread','total'} lists of the arm: the selector's pool ('pool') or clustering output ('stage1') for the residual targets, odds_* appended."""
    return m.select_market_features(sel_df, seed=42, stop_after=which)


def write_features(name, which):
    m, df = load_games()
    ref = json.loads((lib.OUT / REFERENCE / 'features.json').read_text())
    out, counts = {}, {}
    cap = require_new_defaults(REFERENCE)['calib_cap']
    for s in TEST_SEASONS:
        mt, calib, test, n_fl = split_fold(df, s, calib_cap=cap)
        sel_df = mt.iloc[:n_fl]
        stage1, pool = arm_lists(m, sel_df, 'stage1'), arm_lists(m, sel_df, 'pool')
        out[s] = {}
        for kind in ('ml', 'spread', 'total'):
            if not (set(ref[s][kind]) <= set(stage1[kind]) <= set(pool[kind])):
                raise SystemExit(f'check failed: lists are not nested ({s}, {kind}): reference - stage1 = {sorted(set(ref[s][kind]) - set(stage1[kind]))[:5]}, '
                                 f'stage1 - all = {sorted(set(stage1[kind]) - set(pool[kind]))[:5]}')
            out[s][kind] = pool[kind] if which == 'pool' else stage1[kind]
            counts.setdefault(s, {})[kind] = {'reference': len(ref[s][kind]), 'stage1': len(stage1[kind]), 'all': len(pool[kind])}
    d = lib.OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'features.json').write_text(json.dumps(out))
    return counts


def run_arm(name):
    which = ARMS[name]
    params = require_new_defaults(REFERENCE)
    t0 = time.time()
    counts = write_features(name, which)
    logging.info(f'{name}: feature lists written, nesting check passed ({time.time() - t0:.0f}s)')
    test, calib, info = run_walk_forward(name, features_from=name, **params)
    ref, _ = load_run(REFERENCE)
    cmp = compare(name, base=REFERENCE, exp_test=test, base_test=ref, save=False)
    lines = [f'# {name}', '', f'Exploratory (E31b), post-audit defaults {params}. Reference `{REFERENCE}`. Nesting check passed in every fold and target. Total time {time.time() - t0:.0f}s.', '',
             '## Features per fold (reference / stage 1 / all)', '', '```', *[f"{s}: " + ', '.join(f"{k} {v['reference']}/{v['stage1']}/{v['all']}" for k, v in c.items()) for s, c in counts.items()], '```', '',
             '## Paired comparison (arm - reference; negative = arm better)', '', fmt_compare(cmp).to_markdown(index=False), '', '## Per season', '']
    for c, lab in (('ll', 'ML log loss'), ('ae_spread', 'Spread MAE'), ('ae_total', 'Total MAE')):
        lines.append(f"- {lab}: {json.dumps({k: round(v, 4) for k, v in cmp[c]['per_season'].items()})}  (pooled {cmp[c]['diff']:+.5f}; seed-only noise floor about {FLOOR[c]})")
    (lib.OUT / name / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='experiments directory holding the post-audit-defaults E25_with_odds reference')
    ap.add_argument('arms', nargs='*', default=list(ARMS))
    a = ap.parse_args()
    lib.OUT = Path(a.out).resolve()
    for arm in a.arms:
        run_arm(arm)
