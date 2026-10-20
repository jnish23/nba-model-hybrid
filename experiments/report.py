"""Evaluation + comparison reports for a run (baseline or experiment)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import metrics as M
from experiments.lib import OUT, load_run

PRIMARY = {  # column in game_losses -> (label, lower is better)
    'll': 'ML log loss', 'brier': 'ML Brier',
    'ae_spread': 'Spread MAE', 'se_spread': 'Spread MSE', 'ae_total': 'Total MAE', 'se_total': 'Total MSE'}


def evaluate(name, test=None, save=True, entry=None):
    """Pooled and per-season primary metrics, reliability table, bets (5% edge rule) with CIs, CLV.

    ROI and CLV are priced at the opening line unless `entry` (core.odds_clean.load_entry_odds) is given; see metrics.OPEN_BASIS_CAVEAT."""
    if test is None:
        test, _ = load_run(name)
    res = {'pooled': M.core_metrics(test), 'per_season': M.per_season(test).round(5).to_dict('index')}
    res['reliability'] = M.reliability(test['target_home_win'], test['ml_prob']).round(4).to_dict('records')
    bets = M.make_bets(test, thr=0.05, entry=entry)
    res['entry_basis'] = bets.attrs.get('entry_basis')
    if res['entry_basis'] == 'open':
        res['entry_basis_caveat'] = M.OPEN_BASIS_CAVEAT
    res['bets_5pct'] = {mk: M.bet_metrics(bets[bets['market'] == mk]) for mk in ('ml', 'spread', 'total')}
    clv = M.clv_points(bets)
    res['clv_points'] = {mk: {'n': int(len(clv[bets.loc[clv.index, 'market'] == mk])),
                              'mean': float(clv[bets.loc[clv.index, 'market'] == mk].mean())} for mk in ('spread', 'total')}
    res['bets_per_season'] = {s: {mk: M.bet_metrics(g[g['market'] == mk], n_boot=200).get('roi') for mk in ('ml', 'spread', 'total')}
                              for s, g in bets.groupby('seasonYear')}
    if save:
        (OUT / name / 'metrics.json').write_text(json.dumps(res, indent=1, default=float))
    return res


def compare(name, base='baseline', cols=tuple(PRIMARY), subset=None, save=True, exp_test=None, base_test=None):
    """Paired comparison (experiment - baseline) of per-game losses on common games."""
    bt = base_test if base_test is not None else load_run(base)[0]
    et = exp_test if exp_test is not None else load_run(name)[0]
    bl, el = M.game_losses(bt), M.game_losses(et)
    out = {c: M.paired(bl, el, c, subset=subset) for c in cols if c in bl and c in el}
    if save:
        (OUT / name / f'vs_{base}.json').write_text(json.dumps(out, indent=1, default=float))
    return out


def fmt_compare(res):
    rows = []
    for c, r in res.items():
        rows.append({'metric': PRIMARY.get(c, c), 'n': r['n'], 'diff (exp-base)': round(r['diff'], 5),
                     '95% CI': f"[{r['ci_lo']:.5f}, {r['ci_hi']:.5f}]", 'seasons improved': f"{r['seasons_improved']}/{r['seasons']}",
                     'CI excludes 0 (better)': r['ci_excludes_zero_in_favor']})
    return pd.DataFrame(rows)


def per_season_table(name, base='baseline', metrics=('ml_logloss', 'ml_brier', 'spread_mae', 'total_mae')):
    t, _ = load_run(name)
    b, _ = load_run(base)
    pe, pb = M.per_season(t), M.per_season(b)
    rows = {}
    for s in pe.index:
        rows[s] = {m_: f"{pe.loc[s, m_]:.4f} ({pe.loc[s, m_] - pb.loc[s, m_]:+.4f})" for m_ in metrics}
    return pd.DataFrame(rows).T
