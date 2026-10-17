"""E6 - offseason regression to the mean. kappa in {1.0 (baseline), 0.75, 0.5}; the value used for a fold is chosen from
the VALIDATION blocks of that fold's training window (mean rank of the three targets' validation scores), never from the
test season. python -m experiments.e06_run [--skip-runs]"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import decide, metrics as M                          # noqa: E402
from experiments.feature_exp import MAIN, standard_decision           # noqa: E402
from experiments.lib import OUT, TEST_SEASONS, load_run, run_walk_forward   # noqa: E402
from experiments.report import compare, evaluate, fmt_compare, per_season_table   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NAME = 'E06_offseason_regression'
KAPPAS = {0.75: 'E06_k075', 0.5: 'E06_k050'}


def csv_for(k):
    return next((ROOT / 'data' / 'processed').glob(f'training_set_v1_reduced_*_carryover{k:g}.csv'))


def main(skip_runs=False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    if not skip_runs:
        for k, nm in KAPPAS.items():
            run_walk_forward(nm, csv=str(csv_for(k)))
    infos = {1.0: json.loads((OUT / 'baseline' / 'fold_info.json').read_text())}
    preds = {1.0: load_run('baseline')[0]}
    for k, nm in KAPPAS.items():
        infos[k] = json.loads((OUT / nm / 'fold_info.json').read_text())
        preds[k] = load_run(nm)[0]
    chosen, rows = {}, []
    for s in TEST_SEASONS:
        val = pd.DataFrame({k: {t: infos[k][s][t]['val_score'] for t in ('ml', 'spread', 'total')} for k in infos})
        ranks = val.rank(axis=1)                       # per target, 1 = best (lowest validation loss)
        score = ranks.mean(axis=0)
        # ties go to the production value (kappa = 1)
        best = min(score.index, key=lambda k: (score[k], k != 1.0))
        chosen[s] = float(best)
        rows.append({'season': s, 'chosen kappa': best, **{f'val rank k={k}': round(score[k], 2) for k in score.index}})
    base = preds[1.0]
    parts = [preds[chosen[s]][preds[chosen[s]]['seasonYear'] == s] for s in TEST_SEASONS]
    test = pd.concat(parts, ignore_index=True)
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    test.to_csv(OUT / NAME / 'predictions.csv', index=False)
    res = evaluate(NAME, test)
    cmp = compare(NAME, exp_test=test, base_test=base)
    keep, passed, nh_ok = standard_decision(cmp)
    # October-November subset (reported separately; also part of the pre-registered keep rule as a no-harm check)
    on = base.loc[pd.to_datetime(base['gameDate']).dt.month.isin([10, 11]), 'gameId']
    cmp_on = compare(NAME, exp_test=test, base_test=base, subset=set(on), save=False)
    on_ok = all(cmp_on[c]['diff'] <= 0 for c in MAIN.values())
    rest = base.loc[~base['gameId'].isin(on), 'gameId']
    cmp_rest = compare(NAME, exp_test=test, base_test=base, subset=set(rest), save=False)
    keep = bool(keep and on_ok)
    lines = [f'# {NAME}\n', '## Kappa chosen per fold (validation blocks only)\n', pd.DataFrame(rows).to_markdown(index=False), '',
             '## Paired comparison vs baseline, all games\n', fmt_compare(cmp).to_markdown(index=False), '',
             f'## October-November games only ({len(on)} games)\n', fmt_compare(cmp_on).to_markdown(index=False), '',
             '## Rest of the season\n', fmt_compare(cmp_rest).to_markdown(index=False), '',
             '## Per season (value, difference vs baseline)\n', per_season_table(NAME).to_markdown(), '',
             '## Fixed-kappa runs (not the registered rule; for context)\n']
    for k, nm in KAPPAS.items():
        c = compare(nm, exp_test=preds[k], base_test=base, save=False)
        lines += [f'### kappa = {k} in every fold', fmt_compare(c).to_markdown(index=False), '']
    lines += ['## Decision (pre-registered)\n', f'- **{"KEEP" if keep else "REJECT"}** (main criterion: {passed}; no-harm: {nh_ok}; Oct-Nov point estimate not worse on all three: {on_ok})']
    for k, c in MAIN.items():
        lines.append(f'- {k}: {decide.describe(cmp[c])}')
    (OUT / NAME / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'decision.json').write_text(json.dumps({'decision': keep, 'kappa_by_fold': chosen}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main(skip_runs='--skip-runs' in sys.argv)
