"""E29 - forced-inclusion walk-forward test of the E28 candidates. Registered in reports/experiments/PREREGISTRATION.md ("E29")
before any run. python -m experiments.e29_run [arm ...]   (arms: E29_F1_total E29x_pace_total E29x_F1pace_total E29x_F1pool_total)

Each fold's total feature list = the reference run's list (E25_with_odds) + the arm's columns; ML and spread lists are the reference's, so
those two models must reproduce the reference predictions exactly. Written to reports/experiments/<arm>/ (features.json, predictions.csv, RESULT.md)."""
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import decide, lib, metrics as M                       # noqa: E402
from experiments import features_tracking as ft                          # noqa: E402
from experiments.lib import OUT, TEST_SEASONS, load_run, run_walk_forward   # noqa: E402
from experiments.report import compare, fmt_compare                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VARIANT = ROOT / 'data' / 'processed' / 'experiments' / 'e28_training_set.csv'
F1_POOL = ROOT / 'data' / 'processed' / 'experiments' / 'e29_f1pool_training_set.csv'
REFERENCE = 'E25_with_odds'
PACE_COLS = ['trk_prev_opp_pace_home', 'trk_prev_opp_pace_away', 'diff_trk_prev_opp_pace']
NOISE_FLOOR_TOTAL_MAE = 0.0050          # largest seed-only |difference| in reports/experiments/noise_floor.md
TRIVIAL_GAIN_SHARE = 0.05
ARMS = {
    'E29_F1_total': {'cols': ft.family_columns('F1'), 'registered': True},
    'E29x_pace_total': {'cols': PACE_COLS, 'registered': False},
    'E29x_F1pace_total': {'cols': ft.family_columns('F1') + PACE_COLS, 'registered': False},
    'E29x_F1pool_total': {'cols': ft.family_columns('F1'), 'registered': False, 'fresh': True},
}


class GainRecorder:
    """Wraps lib.fit_tuned to record, for every total-model fit, the share of LightGBM gain taken by given columns."""

    def __init__(self, cols):
        self.cols, self.rows = set(cols), []
        self._orig = lib.fit_tuned

    def __enter__(self):
        def wrapped(kind, X, y, n_fit, init=None, grid=lib.GRID):
            est, info = self._orig(kind, X, y, n_fit, init=init, grid=grid)
            if getattr(y, 'name', None) == 'target_total_pts':
                booster = est.booster_
                gain = booster.feature_importance(importance_type='gain')
                names = booster.feature_name()
                tot = float(gain.sum()) or 1.0
                self.rows.append({'n_features': len(names), 'family_share': float(sum(g for n, g in zip(names, gain) if n in self.cols) / tot),
                                  'top5': sorted(zip(names, (gain / tot).round(4)), key=lambda t: -t[1])[:5]})
            return est, info
        lib.fit_tuned = wrapped
        return self

    def __exit__(self, *a):
        lib.fit_tuned = self._orig


def write_features(name, cols):
    ref = json.loads((OUT / REFERENCE / 'features.json').read_text())
    new = copy.deepcopy(ref)
    for s in TEST_SEASONS:
        new[s]['total'] = list(ref[s]['total']) + [c for c in cols if c not in ref[s]['total']]
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'features.json').write_text(json.dumps(new))
    return ref, new


def build_f1_pool():
    df = pd.read_csv(VARIANT, low_memory=False)
    keep = [c for c in df.columns if not (c.startswith('trk_') or c.startswith('diff_trk_'))] + ft.family_columns('F1')
    df[keep].to_csv(F1_POOL, index=False)


def run_arm(name):
    arm = ARMS[name]
    logging.info(f'== {name}: {len(arm["cols"])} columns')
    ref_test, _ = load_run(REFERENCE)
    gains = GainRecorder(arm['cols'])
    if arm.get('fresh'):
        build_f1_pool()
        with gains:
            test, calib, info = run_walk_forward(name, csv=str(F1_POOL), seasons=TEST_SEASONS)
        ref_feats = new_feats = None
    else:
        ref_feats, new_feats = write_features(name, arm['cols'])
        with gains:
            test, calib, info = run_walk_forward(name, csv=str(VARIANT), seasons=TEST_SEASONS)
    # ---- validity checks
    checks = {}
    checks['same_games_same_order'] = bool(list(test['gameId']) == list(ref_test['gameId']))
    if not arm.get('fresh'):
        checks['ml_identical'] = bool(np.allclose(test['ml_prob_raw'], ref_test['ml_prob_raw'], rtol=0, atol=1e-12))
        checks['spread_identical'] = bool(np.allclose(test['spread_pred'], ref_test['spread_pred'], rtol=0, atol=1e-9))
    base = ref_test
    cmp = compare(name, base=REFERENCE, exp_test=test, base_test=base, save=False)
    tot = cmp['ae_total']
    d = {'pooled_total_mae_diff': tot['diff'], 'ci': [tot['ci_lo'], tot['ci_hi']], 'seasons_better': f"{tot['seasons_improved']}/{tot['seasons']}"}
    nh = {k: decide.nh(cmp[c]) for k, c in (('total MSE', 'se_total'), ('ML log loss', 'll'), ('spread MAE', 'ae_spread'))}
    adopt = bool(decide.mc(tot) and all(nh.values()))
    cand = bool(tot['diff'] < -NOISE_FLOOR_TOTAL_MAE and tot['seasons_improved'] >= 4)
    verdict = 'ADOPT' if adopt else ('CANDIDATE' if cand else 'REJECT')
    nontrivial = sum(1 for r in gains.rows if r['family_share'] >= TRIVIAL_GAIN_SHARE)
    lines = [f'# {name}', '', f"Arm columns ({len(arm['cols'])}): {arm['cols']}", f"Registered decision-bearing arm: {arm['registered']}. Reference: `{REFERENCE}`.", '',
             '## Validity checks', '', json.dumps(checks), '', '## Paired comparison vs the reference (exp - ref; negative = better)', '', fmt_compare(cmp).to_markdown(index=False), '',
             '## Per season, total MAE difference', '', json.dumps({k: round(v, 4) for k, v in tot['per_season'].items()}), '',
             f"Noise floor for total MAE (largest seed-only difference): {NOISE_FLOOR_TOTAL_MAE}. Pooled difference {tot['diff']:+.5f}: "
             f"{'exceeds' if abs(tot['diff']) > NOISE_FLOOR_TOTAL_MAE else 'is inside'} the floor.", '',
             '## Share of total-model gain taken by the arm\'s columns, per fold', '',
             *[f"- {s}: {r['family_share']:.3f}  (top: {r['top5'][:3]})" for s, r in zip(TEST_SEASONS, gains.rows)],
             f"Folds with at least {TRIVIAL_GAIN_SHARE:.0%}: {nontrivial}/{len(gains.rows)}", '',
             f"## Verdict (registered rule{'' if arm['registered'] else '; EXPLORATORY arm, decides nothing'}): **{verdict}**",
             f"- decide.mc on total MAE: {decide.mc(tot)}; no-harm: {nh}; candidate rule (beats the floor and >= 4/7 seasons): {cand}"]
    (OUT / name / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / name / 'decision.json').write_text(json.dumps({'verdict': verdict, 'registered': arm['registered'], 'checks': checks, **d,
                                                          'gain_shares': [r['family_share'] for r in gains.rows]}, indent=1, default=float))
    print('\n'.join(lines))
    return verdict


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    for a in (sys.argv[1:] or list(ARMS)):
        run_arm(a)
