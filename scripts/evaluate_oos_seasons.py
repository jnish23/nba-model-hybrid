"""
Score out-of-sample season predictions against the betting lines, per season.

Inputs: reports/oos_predictions_<season>.csv (scripts/generate_season_predictions.py --eval-only)
and predictions/predictions_2025-26.csv. Each season's models were trained only on earlier
seasons. Bets use the shared rule in team/betting.py (the same one live picks and the dashboard
use): bet when P(win) = Phi(point edge / residual std) beats the -110 break-even by more than 5%,
which is a point edge of T = std * Phi^-1(0.5238 + 0.05) on each market (about 2.5 pts on spreads and
3.4 pts on totals). Lines in the files are the DraftKings openers (the model's own inputs); results
are also scored at the consensus closing line, with closing line value (CLV).

The primary rule is the 5% probability edge above; flat point thresholds are shown for context only
(a flat 1.0 pt is only a +0.6% edge on spreads and -0.2% on totals).

CAVEAT: the opening lines are posted before the game-day-noon injury report the model reads, so the 'open'
ROI and the CLV below include line movement the model could not have bet at. Only the closing-line
comparisons are valid until bets are priced at the first line stamped at/after the feature cutoff
(core.odds_clean.load_entry_odds, collected by scripts/snapshot_odds.py).

Usage: python scripts/evaluate_oos_seasons.py
"""
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.odds_clean import load_clean_odds
from team.betting import points_threshold, DEFAULT_SPREAD_STD, DEFAULT_TOTAL_STD, DEFAULT_PROB_EDGE

PAY = 100 / 110
BREAK_EVEN = 110 / 210
PRIMARY_T = {'spread': round(points_threshold(DEFAULT_SPREAD_STD, DEFAULT_PROB_EDGE), 2),
             'total': round(points_threshold(DEFAULT_TOTAL_STD, DEFAULT_PROB_EDGE), 2)}
THRESHOLDS = [0.0, 1.0, 2.0, 3.0]


def season_of(date):
    d = pd.Timestamp(date)
    y = d.year if d.month >= 10 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def load():
    files = sorted(glob.glob('reports/oos_predictions_*.csv')) + ['predictions/predictions_2025-26.csv']
    df = pd.concat([pd.read_csv(f, dtype={'game_id': str}) for f in files if Path(f).exists()], ignore_index=True)
    df = df.drop_duplicates('game_id', keep='last')
    df['season'] = df['game_date'].map(season_of)
    odds = load_clean_odds(Path('data/nba_stats.db'))
    odds['game_id'] = odds['gameId'].astype(str).str.zfill(10)
    return df.merge(odds[['game_id', 'close_home_spread', 'close_total']], on='game_id', how='left')


def bets(df, market, t):
    if market == 'spread':
        edge = df['spread_pred'] + df['odds_home_spread']
        side = np.sign(edge)
        res_open = side * (df['home_score'] - df['away_score'] + df['odds_home_spread'])
        res_close = side * (df['home_score'] - df['away_score'] + df['close_home_spread'])
        clv = side * (df['odds_home_spread'] - df['close_home_spread'])
    else:
        edge = df['total_pred'] - df['odds_total']
        side = np.sign(edge)
        res_open = side * (df['home_score'] + df['away_score'] - df['odds_total'])
        res_close = side * (df['home_score'] + df['away_score'] - df['close_total'])
        clv = side * (df['close_total'] - df['odds_total'])
    sel = edge.abs() > t
    return res_open[sel], res_close[sel], clv[sel]


def summarize(res):
    res = res.dropna()
    n = len(res)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    wins, losses = (res > 0).sum(), (res < 0).sum()
    wr = wins / (wins + losses)
    roi = (wins * PAY - losses) / n
    z = (wr - BREAK_EVEN) / np.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / (wins + losses))
    return wr, roi, z, n


def main():
    logging.basicConfig(level=logging.WARNING)
    pd.set_option('display.width', 220)
    df = load()
    rows = []
    for market in ('spread', 'total'):
        for t in sorted(set(THRESHOLDS + [PRIMARY_T[market]])):
            for season, g in list(df.groupby('season')) + [('ALL', df)]:
                o, c, clv = bets(g, market, t)
                wr_o, roi_o, z_o, n = summarize(o)
                wr_c, roi_c, z_c, _ = summarize(c)
                clv = clv.dropna()
                rows.append({'market': market, 'T': t, 'season': season, 'bets': n,
                             'win% open': wr_o * 100, 'ROI% open': roi_o * 100, 'z open': z_o,
                             'win% close': wr_c * 100, 'ROI% close': roi_c * 100, 'z close': z_c,
                             'CLV pts': clv.mean(), 'CLV t': clv.mean() / clv.std() * np.sqrt(len(clv)) if len(clv) > 1 else np.nan})
    out = pd.DataFrame(rows).round(2)
    print(f"PRIMARY RULE (5% probability edge: spread T = {PRIMARY_T['spread']} pts, total T = {PRIMARY_T['total']} pts), "
          f"by season. Break-even win% at -110 = {BREAK_EVEN:.2%}")
    print("NOTE: 'open' results and CLV use the opening line, which predates the game-day injury report the model reads; "
          "only closing-line comparisons are valid (see experiments.metrics.OPEN_BASIS_CAVEAT).")
    primary = out[[abs(t - PRIMARY_T[m]) < 1e-9 for m, t in zip(out['market'], out['T'])]]
    print(primary.to_string(index=False))
    print("\nAll thresholds, pooled over seasons (context only)")
    print(out[out['season'] == 'ALL'].drop(columns='season').to_string(index=False))
    out.to_csv('reports/oos_multi_season_eval.csv', index=False)


if __name__ == "__main__":
    main()
