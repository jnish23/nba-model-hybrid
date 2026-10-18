"""
Is player-availability information priced by the market, and when?

Using realized availability (core/availability_features.py) and clean odds (core/odds_clean.py),
regress three quantities on home-minus-away availability features:
  move        = close_mkt_margin - open_mkt_margin   (does availability news move the line after the open?)
  open_error  = margin - open_mkt_margin             (does the OPENING line underprice it?)
  close_error = margin - close_mkt_margin            (does even the CLOSING line underprice it?)
Market margin = -home_spread. Coefficients are points of home margin per unit of feature
(units ~ points per 100 possessions of player value lost). Negative = losing value hurts the home side.

Reading the results:
  - big |t| on `move` + small on `close_error`  -> the news is real but gets priced before close;
    edge exists only if you learn it before the market does (timing / scraper).
  - |t| > 2 on `close_error`                     -> the market underreacts even at close.

Usage: python scripts/evaluate_availability.py
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.odds_clean import load_clean_odds
from core.availability_features import build_availability_features, to_game_level

SEASONS = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
FEATURES = ['diff_avail_lost_new', 'diff_avail_lost_old', 'diff_avail_returning']


def ols(X, y, names):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - resid.var() / y.var()
    return pd.DataFrame({'coef': beta, 'se': se, 't': beta / se}, index=['const'] + names), r2


def main():
    logging.basicConfig(level=logging.WARNING)
    db = Path('data/nba_stats.db')
    odds = load_clean_odds(db)
    feats = build_availability_features(db, SEASONS)
    df = to_game_level(feats, odds)
    df['margin'] = df['home_score'] - df['away_score']
    df['open_mkt'] = -df['open_home_spread']
    df['close_mkt'] = -df['close_home_spread']
    df['move'] = df['close_mkt'] - df['open_mkt']
    df['open_error'] = df['margin'] - df['open_mkt']
    df['close_error'] = df['margin'] - df['close_mkt']
    df = df.dropna(subset=['open_mkt', 'close_mkt'] + FEATURES)
    print(f"Games: {len(df)} (2019-20 onward, open + close spread available)\n")

    print("Feature scale (home-minus-away):")
    print(df[FEATURES].describe().T[['mean', 'std', 'min', 'max']].round(2).to_string(), "\n")

    for target in ('move', 'open_error', 'close_error'):
        tab, r2 = ols(df[FEATURES].values, df[target].values, FEATURES)
        print(f"=== {target}  (R^2 = {r2:.4f}, sd(target) = {df[target].std():.2f})")
        print(tab.round(3).to_string(), "\n")

    # How much of a fresh absence is priced by the open vs by the close?
    print("=== Fresh absences, by size: mean line move and errors (home-minus-away lost_new, binned)")
    bins = [-np.inf, -6, -3, -1, 1, 3, 6, np.inf]
    df['bin'] = pd.cut(df['diff_avail_lost_new'], bins)
    g = df.groupby('bin', observed=True).agg(games=('move', 'size'), move=('move', 'mean'),
                                              open_error=('open_error', 'mean'),
                                              close_error=('close_error', 'mean'))
    print(g.round(2).to_string())
    print("\n(Positive diff = home lost more value. If the open has not priced it, open_error is negative\n"
          " for positive bins; if the close has, close_error is ~0.)")

    print("\n=== By season: coefficient on diff_avail_lost_new")
    rows = []
    for s, g in df.groupby('seasonYear'):
        for target in ('move', 'open_error', 'close_error'):
            tab, _ = ols(g[FEATURES].values, g[target].values, FEATURES)
            rows.append({'season': s, 'target': target, 'coef': tab.loc['diff_avail_lost_new', 'coef'],
                         't': tab.loc['diff_avail_lost_new', 't']})
    print(pd.DataFrame(rows).pivot(index='season', columns='target', values=['coef', 't']).round(2).to_string())


if __name__ == "__main__":
    main()
