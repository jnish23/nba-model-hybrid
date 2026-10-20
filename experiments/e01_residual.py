"""E1 - market-residual targets (see PREREGISTRATION.md).

spread target  = margin + opening spread        (prediction = -spread + f(X))
total target   = total - opening total          (prediction =  total  + f(X))
ML             = LightGBM init_score = logit(de-vigged opening home probability)
Same selected features as the baseline; only the targets/offsets change.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import decide, metrics as M                      # noqa: E402
from experiments.lib import OUT, load_games, load_run, run_walk_forward   # noqa: E402
from experiments.report import compare, evaluate, fmt_compare, per_season_table   # noqa: E402

NAME = 'E01_residual_targets'


def _imp(o):
    o = np.asarray(o, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def devig_open_prob(df):
    h, a = _imp(df['odds_home_ml']), _imp(df['odds_away_ml'])
    return h / (h + a)


def init_scores(kind, frames, ctx):
    if kind != 'ml':
        return None
    mt, calib, test = frames['mt'], frames['calib'], frames['test']
    # in-fold fallback when the ML price is missing: logistic regression of home win on the opening spread
    lr = LogisticRegression(C=100.0).fit(mt[['odds_home_spread']], mt['target_home_win'])
    out = []
    for fr in (mt, calib, test):
        p = devig_open_prob(fr)
        fb = lr.predict_proba(fr[['odds_home_spread']])[:, 1]
        p = np.where(np.isfinite(p), p, fb)
        out.append(logit(np.clip(p, 1e-4, 1 - 1e-4)))
    return tuple(out)


def target(kind, mt, ctx):
    if kind == 'spread':
        return mt['target_point_diff'] + mt['odds_home_spread']
    if kind == 'total':
        return mt['target_total_pts'] - mt['odds_total']
    return mt['target_home_win']


def pred_offset(kind, frame, ctx):
    return (-frame['odds_home_spread'].to_numpy(float)) if kind == 'spread' else frame['odds_total'].to_numpy(float)


def tails(base, exp):
    rows = []
    masks = {'all games': np.ones(len(base), bool)}
    masks['|spread| > 12'] = (base['odds_home_spread'].abs() > 12).to_numpy()
    masks['total > 242'] = (base['odds_total'] > 242).to_numpy()
    for lab, mk in masks.items():
        b, e = base[mk], exp.set_index('gameId').reindex(base['gameId'][mk]).reset_index()
        row = {'bucket': lab, 'games': int(mk.sum())}
        for tag, df in (('base', b), ('E1', e)):
            for mkt, pred, act in (('spread', 'spread_pred', 'target_point_diff'), ('total', 'total_pred', 'target_total_pts')):
                if lab == '|spread| > 12' and mkt == 'total' or lab == 'total > 242' and mkt == 'spread':
                    continue
                err = df[act] - df[pred]
                row[f'{tag} {mkt} MAE'] = round(float(err.abs().mean()), 3)
                row[f'{tag} {mkt} bias'] = round(float(err.mean()), 3)
            bets = M.make_bets(df.dropna(subset=['spread_pred', 'total_pred']), thr=0.05, markets=('spread', 'total'))
            for mkt in ('spread', 'total'):
                if lab == '|spread| > 12' and mkt == 'total' or lab == 'total > 242' and mkt == 'spread':
                    continue
                bm = bets[bets['market'] == mkt]
                row[f'{tag} {mkt} bets'] = len(bm)
                row[f'{tag} {mkt} ROI%'] = round(float(bm['profit'].sum() / (100 * len(bm)) * 100), 2) if len(bm) else None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    m, df = load_games()
    df = df.dropna(subset=['odds_home_spread', 'odds_total']).reset_index(drop=True)
    test, calib, info = run_walk_forward(NAME, features_from='baseline', df_override=(m, df),
                                         hooks={'init_scores': init_scores, 'target': target, 'pred_offset': pred_offset})
    res = evaluate(NAME, test)
    base, _ = load_run('baseline')
    cmp = compare(NAME, exp_test=test, base_test=base)
    # decisions (pre-registered): per market, MC on the market's main loss + NH on its companion metric
    dec = {'ML': decide.mc(cmp['ll']) and decide.nh(cmp['brier']),
           'SPREAD': decide.mc(cmp['ae_spread']) and decide.nh(cmp['se_spread']),
           'TOTAL': decide.mc(cmp['ae_total']) and decide.nh(cmp['se_total'])}
    tl = tails(base, test)
    lines = [f"# {NAME}\n", '## Paired comparison vs baseline (exp - base)\n', fmt_compare(cmp).to_markdown(index=False), '',
             '## Per season (value, difference vs baseline)\n', per_season_table(NAME).to_markdown(), '',
             '## Pooled primary metrics\n', pd.DataFrame({'baseline': M.core_metrics(base), 'E1': res['pooled']}).round(4).to_markdown(), '',
             '## Tails (descriptive; bets under the 5% rule)\n', tl.to_markdown(index=False), '',
             '## Betting (5% rule) E1\n', json.dumps({k: {a: b for a, b in v.items()} for k, v in res['bets_5pct'].items()}, default=float, indent=1), '',
             '## Decision (pre-registered)\n'] + [f"- KEEP-{k}: **{v}**" for k, v in dec.items()]
    for k, c in (('ML log loss', 'll'), ('Spread MAE', 'ae_spread'), ('Total MAE', 'ae_total')):
        lines.append(f"- {k}: {decide.describe(cmp[c])}")
    (OUT / NAME / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'decision.json').write_text(json.dumps({'decision': dec}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
