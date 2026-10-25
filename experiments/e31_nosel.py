"""E31 - all features versus the selector's list (exploratory; registered in PREREGISTRATION.md before the run).
python -m experiments.e31_nosel [E31_all E31_stage1]   -> reports/experiments/<arm>/ (features.json, predictions.csv, RESULT.md)"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import decide                                              # noqa: E402
from experiments.lib import OUT, TARGETS, TEST_SEASONS, load_games, load_run, run_split_params, run_walk_forward, split_fold   # noqa: E402
from experiments.report import compare, fmt_compare                          # noqa: E402

REFERENCE = 'E25_with_odds'
FLOOR = {'ll': 0.003, 'ae_spread': 0.011, 'ae_total': 0.005}


def pool_lists(m, df_sel, target):
    """(all, stage1) feature lists: the selector's own first steps (pool, zero-variance filter, Spearman clustering at 0.15 distance,
    representative = highest proxy importance, ties by name)."""
    import lightgbm as lgb
    pool = m._get_pool_for_target(target)
    cols = sorted(f for f in pool if f in df_sel.columns)
    X = df_sel[cols].copy().replace([np.inf, -np.inf], np.nan)
    X = X[X.var()[X.var() > 1e-10].index.tolist()]
    y = df_sel[target]
    mdl = (lgb.LGBMClassifier if 'win' in target else lgb.LGBMRegressor)(n_estimators=100, max_depth=4, verbose=-1, random_state=42, n_jobs=-1,
                                                                           deterministic=True, force_row_wise=True, importance_type='split')
    mdl.fit(X, y)
    imp = dict(zip(X.columns, mdl.feature_importances_))
    dist = np.clip(1 - X.corr(method='spearman').abs().fillna(0), 0, 1)
    labels = sch.fcluster(sch.linkage(squareform(dist, checks=False), method='average'), t=0.15, criterion='distance')
    members = {}
    for f, lab in zip(X.columns, labels):
        members.setdefault(lab, []).append(f)
    stage1 = sorted(min(v, key=lambda f: (-imp[f], f)) for v in members.values())
    return list(X.columns), stage1


def write_features(name, which):
    m, df = load_games()
    ref = json.loads((OUT / REFERENCE / 'features.json').read_text())
    out, counts = {}, {}
    for s in TEST_SEASONS:
        mt, calib, test, n_fl = split_fold(df, s, calib_cap=run_split_params(REFERENCE)['calib_cap'])      # the reference run's split
        out[s] = {}
        for kind, target in TARGETS.items():
            allf, st1 = pool_lists(m, mt.iloc[:n_fl], target)
            if not set(ref[s][kind]) <= set(st1):
                raise SystemExit(f'check failed: reference list not inside my stage-1 list ({s}, {kind}): {sorted(set(ref[s][kind]) - set(st1))[:5]}')
            out[s][kind] = allf if which == 'all' else st1
            counts.setdefault(s, {})[kind] = {'reference': len(ref[s][kind]), 'stage1': len(st1), 'all': len(allf)}
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'features.json').write_text(json.dumps(out))
    return counts


def run_arm(name):
    which = 'all' if name == 'E31_all' else 'stage1'
    t0 = time.time()
    counts = write_features(name, which)
    logging.info(f'{name}: feature lists written, subset check passed ({time.time() - t0:.0f}s)')
    test, calib, info = run_walk_forward(name, features_from=name, **run_split_params(REFERENCE))      # features_from: use the lists written above
    ref, _ = load_run(REFERENCE)
    cmp = compare(name, base=REFERENCE, exp_test=test, base_test=ref, save=False)
    lines = [f'# {name}', '', f'Exploratory (E31). Reference `{REFERENCE}`. Subset check passed in every fold and target. Total time {time.time() - t0:.0f}s.', '',
             '## Features per fold (reference / stage 1 / all)', '', '```', *[f"{s}: " + ', '.join(f"{k} {v['reference']}/{v['stage1']}/{v['all']}" for k, v in c.items()) for s, c in counts.items()], '```', '',
             '## Paired comparison (arm - reference; negative = arm better)', '', fmt_compare(cmp).to_markdown(index=False), '', '## Per season', '']
    for c, lab in (('ll', 'ML log loss'), ('ae_spread', 'Spread MAE'), ('ae_total', 'Total MAE')):
        lines.append(f"- {lab}: {json.dumps({k: round(v, 4) for k, v in cmp[c]['per_season'].items()})}  (pooled {cmp[c]['diff']:+.5f}; seed-only noise floor about {FLOOR[c]})")
    (OUT / name / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    for a in (sys.argv[1:] or ['E31_stage1', 'E31_all']):
        run_arm(a)
