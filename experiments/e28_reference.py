"""Deployed-form out-of-sample predictions on the 10-19 training set, used as the reference for E28 (residual screen) and E29
(forced-inclusion walk-forward).

  total   E25_with_odds  (level model, opening odds in the pool)
  spread  E25_residual   (E1: market as an offset; spread_pred is the predicted HOME margin)
  ML      E3 stack = logistic on [logit(raw ML probability of E25_with_odds), E1 margin / sigma of E25_residual], fitted per
          fold on that fold's calibration set, as experiments/e13_combined.py does (games common to both runs)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.lib import TEST_SEASONS, label_calib_folds, load_games, load_run   # noqa: E402

WITH_ODDS, RESIDUAL = 'E25_with_odds', 'E25_residual'


def lg(p):
    return logit(np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4))


def stack_ml(base, base_cal, res, res_cal, df_full, df_lines, seasons=TEST_SEASONS):
    """Per-fold logistic stack. Returns a Series of stacked home-win probabilities indexed like `res` (games common to both)."""
    base_cal = base_cal.copy()
    res_cal = res_cal.copy()
    base_cal['fold'] = label_calib_folds(df_full, base_cal)
    res_cal['fold'] = label_calib_folds(df_lines, res_cal)
    b = base.set_index('gameId')
    out = pd.Series(np.nan, index=res['gameId'])
    for s in seasons:
        cb = base_cal[base_cal['fold'] == s].set_index('gameId')
        cr = res_cal[res_cal['fold'] == s].set_index('gameId')
        ids = cb.index.intersection(cr.index)
        Xc = np.column_stack([lg(cb.loc[ids, 'ml_prob_raw']), cr.loc[ids, 'spread_pred'] / cr.loc[ids, 'sigma_spread']])
        yc = cb.loc[ids, 'target_home_win']
        t = res[res['seasonYear'] == s]
        t = t[t['gameId'].isin(b.index)]
        Xt = np.column_stack([lg(b.loc[t['gameId'], 'ml_prob_raw']), t['spread_pred'] / t['sigma_spread']])
        out.loc[t['gameId'].to_numpy()] = LogisticRegression(C=1.0).fit(Xc, yc).predict_proba(Xt)[:, 1]
    return out


def deployed_form():
    """One row per out-of-sample game (2019-20 .. 2025-26, regular season and playoffs) common to both runs:
    gameId, gameDate, seasonYear, targets, opening lines, total_pred, spread_pred, sigma_spread, ml_raw, ml_stack."""
    base, base_cal = load_run(WITH_ODDS)
    res, res_cal = load_run(RESIDUAL)
    _, df = load_games()
    df_lines = df[df['odds_home_spread'].notna() & df['odds_total'].notna()].reset_index(drop=True)
    st = stack_ml(base, base_cal, res, res_cal, df, df_lines)
    g = res.merge(base[['gameId', 'total_pred', 'ml_prob_raw']].rename(columns={'total_pred': 'total_pred_level', 'ml_prob_raw': 'ml_raw'}), on='gameId', how='inner')
    g['ml_stack'] = st.reindex(g['gameId']).to_numpy()
    g['total_pred'] = g['total_pred_level']
    return g.drop(columns=['total_pred_level', 'ml_prob', 'ml_prob_raw'], errors='ignore')


if __name__ == '__main__':
    g = deployed_form()
    y = g['target_home_win'].to_numpy(float)
    for c in ('ml_raw', 'ml_stack'):
        p = np.clip(g[c].to_numpy(float), 1e-4, 1 - 1e-4)
        ok = np.isfinite(p)
        print(c, ok.sum(), 'log loss', float(-(y[ok] * np.log(p[ok]) + (1 - y[ok]) * np.log(1 - p[ok])).mean()))
    print('spread MAE', float((g['target_point_diff'] - g['spread_pred']).abs().mean()), 'total MAE', float((g['target_total_pts'] - g['total_pred']).abs().mean()))
    print(g.shape, g['seasonYear'].value_counts().sort_index().to_dict())
