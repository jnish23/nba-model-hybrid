"""Generic runner for experiments that change the feature pool (E5-E8): walk-forward with fresh feature selection,
paired comparison with the baseline, RESULT.md and the pre-registered decision."""
import json
import logging

import numpy as np
import pandas as pd

from experiments import decide, metrics as M
from experiments.lib import OUT, load_games, load_run, run_walk_forward
from experiments.report import compare, evaluate, fmt_compare, per_season_table

MAIN = {'ML log loss': 'll', 'Spread MAE': 'ae_spread', 'Total MAE': 'ae_total'}
COMPANION = {'ll': 'brier', 'ae_spread': 'se_spread', 'ae_total': 'se_total'}


def standard_decision(cmp, min_pass=2, **_):
    """MC on at least `min_pass` of the three main metrics and NH on all three (pre-registered for E5/E6)."""
    passed = {k: decide.mc(cmp[c]) for k, c in MAIN.items()}
    nh_ok = {k: decide.nh(cmp[c]) for k, c in MAIN.items()}
    return bool(sum(passed.values()) >= min_pass and all(nh_ok.values())), passed, nh_ok


def run_feature_experiment(name, csv, title, extra_sections=None, seasons=None, subset_fn=None, decision_fn=None):
    from experiments.lib import TEST_SEASONS
    seasons = seasons or TEST_SEASONS
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    test, calib, info = run_walk_forward(name, csv=str(csv), seasons=seasons)
    res = evaluate(name, test)
    base, _ = load_run('baseline')
    base = base[base['seasonYear'].isin(seasons)]
    cmp = compare(name, exp_test=test, base_test=base)
    keep, passed, nh_ok = (decision_fn or standard_decision)(cmp, test=test, base=base)
    lines = [f'# {name}: {title}\n', '## Paired comparison vs baseline (exp - base)\n', fmt_compare(cmp).to_markdown(index=False), '',
             '## Per season (value, difference vs baseline)\n', per_season_table(name).to_markdown(), '',
             '## Pooled primary metrics\n', pd.DataFrame({'baseline': M.core_metrics(base), name: res['pooled']}).round(4).to_markdown(), '',
             '## Selected-feature counts and tuned configs by fold\n', json.dumps({s: {k: [v['n_features'], v['cfg']] for k, v in f.items()} for s, f in info.items()}), '']
    feats = json.loads((OUT / name / 'features.json').read_text())
    new_feats = {s: {k: [f for f in v if f.startswith(('e5_', 'diff_e5_', 'e6_', 'e7_', 'e8_', 'diff_e7', 'diff_e8')) or 'e5_' in f or 'e7_' in f or 'e8_' in f] for k, v in fs.items()} for s, fs in feats.items()}
    lines += ['## New features chosen by selection (per fold: ml / spread / total)\n', '```',
              *[f"{s}: {', '.join(str(len(v)) for v in fs.values())} -> " + ' | '.join(f"{k}: {v[:6]}" for k, v in fs.items()) for s, fs in new_feats.items()], '```', '']
    bets = res['bets_5pct']
    lines += ['## Betting (5% rule), experiment\n', pd.DataFrame({k: {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a in ('bets', 'hit_rate', 'roi', 'max_drawdown')} for k, v in bets.items()}).to_markdown(), '',
              f"CLV points (mean): {res['clv_points']}", '']
    if extra_sections:
        lines += extra_sections(test, base, cmp)
    lines += ['## Decision (pre-registered)\n', f'- **{"KEEP" if keep else "REJECT"}**  (main criterion per metric: {passed}; no-harm per metric: {nh_ok})']
    for k, c in MAIN.items():
        lines.append(f'- {k}: {decide.describe(cmp[c])}')
    (OUT / name / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / name / 'decision.json').write_text(json.dumps({'decision': keep, 'main_criterion': passed, 'no_harm': nh_ok}, indent=1))
    print('\n'.join(lines))
    return keep, test, cmp
