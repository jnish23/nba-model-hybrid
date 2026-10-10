"""
Evaluate backtest predictions against the betting market (opening and closing lines).

Answers, per market (moneyline / spread / total):
  1. How good is the market itself (opening vs closing)?
  2. Is the model more or less accurate than the opening / closing line?
  3. Does the model carry information the line does not? (regression of the line's error on the
     model's disagreement with it; walk-forward blend fitted on prior seasons only)
  4. Closing line value (CLV): when the model disagrees with the opening line, does the line move
     toward the model by close? CLV is far less noisy than win/loss.

Usage:
    python scripts/evaluate_vs_market.py [--preds models/saved/backtest_predictions_calibrated.csv]
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.odds_clean import load_clean_odds, american_to_payout

SPREAD_THRESHOLDS = [0.0, 2.5, 5.0]
TOTAL_THRESHOLDS = [0.0, 3.0, 6.0]
ML_THRESHOLDS = [0.0, 0.03, 0.05]
STANDARD_PAYOUT = 100.0 / 110.0  # -110


# ---------------------------------------------------------------- helpers

def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ols(x, y):
    """y = a + b*x. Returns (a, b, se_b)."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(y) - 2)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return beta[0], beta[1], np.sqrt(cov[1, 1])


def _logit_fit(X, y, offset, iters=50):
    """Logistic regression with an offset (Newton-Raphson). Returns (beta, se)."""
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(offset + X @ beta)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta, np.sqrt(np.diag(np.linalg.inv(H)))


def _section(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


# ---------------------------------------------------------------- data

def load(preds_path, db_path):
    preds = pd.read_csv(preds_path)
    odds = load_clean_odds(db_path)
    odds['gameId'] = odds['gameId'].astype(int)
    df = preds.drop(columns=[c for c in preds.columns if c.startswith('odds_')]).merge(
        odds.drop(columns=['gameDate', 'teamId_home', 'teamId_away', 'seasonYear']), on='gameId', how='inner')

    df['margin'] = df['home_score'] - df['away_score']
    df['total_pts'] = df['home_score'] + df['away_score']
    df['home_win'] = (df['margin'] > 0).astype(int)
    mismatch = (df['margin'] != df['spread_actual']).sum()
    if mismatch:
        logging.warning(f"{mismatch} games where prediction-file margin != team_game_full margin")

    # Market's implied home margin (spread is negative when home is favored)
    df['open_mkt_margin'] = -df['open_home_spread']
    df['close_mkt_margin'] = -df['close_home_spread']
    logging.info(f"Joined {len(df)} of {len(preds)} predictions to clean odds")
    return df


# ---------------------------------------------------------------- 1 + 2: accuracy

def accuracy_tables(df):
    _section("1-2. ACCURACY BY SEASON: model vs opening vs closing line "
             "(games with model + open + close available)")
    d = df.dropna(subset=['open_home_spread', 'close_home_spread', 'open_total', 'close_total',
                          'open_home_prob', 'close_home_prob'])
    rows = []
    for season, g in list(d.groupby('seasonYear')) + [('ALL', d)]:
        rows.append({
            'season': season, 'games': len(g),
            'ML ll model': _logloss(g.home_win, g.ml_prob),
            'ML ll open': _logloss(g.home_win, g.open_home_prob),
            'ML ll close': _logloss(g.home_win, g.close_home_prob),
            'Spr MAE model': (g.margin - g.spread_pred).abs().mean(),
            'Spr MAE open': (g.margin - g.open_mkt_margin).abs().mean(),
            'Spr MAE close': (g.margin - g.close_mkt_margin).abs().mean(),
            'Tot MAE model': (g.total_pts - g.total_pred).abs().mean(),
            'Tot MAE open': (g.total_pts - g.open_total).abs().mean(),
            'Tot MAE close': (g.total_pts - g.close_total).abs().mean(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nLower is better. ML ll = log loss of home-win probability (market = no-vig consensus).")


# ---------------------------------------------------------------- 3: incremental information

def information_tests(df):
    _section("3. DOES THE MODEL ADD INFORMATION BEYOND THE LINE?")
    print("Spread/Total: regress (actual - line) on (model - line). Slope b > 0 (t > 2) means the model's\n"
          "disagreement with the line predicts the line's error. b = 1 would mean trust the model fully.\n"
          "ML: logistic regression of home_win on logit(model) - logit(market), with the market as offset.\n")
    rows = []
    for snap in ('open', 'close'):
        for label, actual, model, line in (
                ('Spread', 'margin', 'spread_pred', f'{snap}_mkt_margin'),
                ('Total', 'total_pts', 'total_pred', f'{snap}_total')):
            g = df.dropna(subset=[line])
            a, b, se = _ols((g[model] - g[line]).values, (g[actual] - g[line]).values)
            rows.append({'market': label, 'line': snap, 'n': len(g), 'slope b': b, 'se': se, 't': b / se})
        g = df.dropna(subset=[f'{snap}_home_prob'])
        off = _logit(g[f'{snap}_home_prob'].values)
        x = (_logit(g['ml_prob'].values) - off)[:, None]
        beta, se = _logit_fit(x, g['home_win'].values.astype(float), off)
        rows.append({'market': 'ML', 'line': snap, 'n': len(g), 'slope b': beta[1], 'se': se[1], 't': beta[1] / se[1]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    _section("3b. WALK-FORWARD BLEND vs OPENING LINE (blend weight fit on prior seasons only)")
    print("blend = line + b*(model - line). 'gain' = market error - blend error on the held-out season\n"
          "(positive = blending in the model would have improved on the opening line).\n")
    seasons = sorted(df['seasonYear'].unique())
    rows = []
    for i, s in enumerate(seasons[1:], start=1):
        train, test = df[df.seasonYear.isin(seasons[:i])], df[df.seasonYear == s]
        row = {'season': s}
        for label, actual, model, line in (('Spr', 'margin', 'spread_pred', 'open_mkt_margin'),
                                           ('Tot', 'total_pts', 'total_pred', 'open_total')):
            tr, te = train.dropna(subset=[line]), test.dropna(subset=[line])
            a, b, _ = _ols((tr[model] - tr[line]).values, (tr[actual] - tr[line]).values)
            blend = te[line] + a + b * (te[model] - te[line])
            row[f'{label} b'] = b
            row[f'{label} MAE gain'] = (te[actual] - te[line]).abs().mean() - (te[actual] - blend).abs().mean()
        tr, te = train.dropna(subset=['open_home_prob']), test.dropna(subset=['open_home_prob'])
        off_tr, off_te = _logit(tr.open_home_prob.values), _logit(te.open_home_prob.values)
        beta, _ = _logit_fit((_logit(tr.ml_prob.values) - off_tr)[:, None], tr.home_win.values.astype(float), off_tr)
        p_blend = 1 / (1 + np.exp(-(off_te + beta[0] + beta[1] * (_logit(te.ml_prob.values) - off_te))))
        row['ML b'] = beta[1]
        row['ML ll gain'] = _logloss(te.home_win, te.open_home_prob) - _logloss(te.home_win, p_blend)
        rows.append(row)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


# ---------------------------------------------------------------- 4: CLV

def _summarize(bets, clv_col, won_col, profit_col, threshold, extra=None):
    n = len(bets)
    if n == 0:
        return {'threshold': threshold, 'bets': 0}
    clv = bets[clv_col]
    decided = bets[won_col].dropna()
    row = {
        'threshold': threshold, 'bets': n,
        'mean CLV': clv.mean(), 'CLV t': clv.mean() / (clv.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        '% beat close': (clv > 0).mean() * 100, '% worse': (clv < 0).mean() * 100,
        'win %': decided.mean() * 100 if len(decided) else np.nan,
        'ROI %': bets[profit_col].sum() / n * 100,
    }
    if extra:
        row.update(extra(bets))
    return row


def clv_tables(df):
    _section("4. CLOSING LINE VALUE: bets placed at the OPENING line when the model disagrees with it")
    print("Spread/Total CLV is in points (positive = the line moved your way by close). Win % excludes\n"
          "pushes; ROI assumes -110. Break-even at -110 is 52.38%. |CLV t| > 2 is the signal to look for.\n")

    d = df.dropna(subset=['open_home_spread', 'close_home_spread'])
    edge = d.spread_pred - d.open_mkt_margin
    rows = []
    for thr in SPREAD_THRESHOLDS:
        b = d[edge.abs() > thr].copy()
        side = np.sign(edge[edge.abs() > thr])  # +1 = home, -1 = away
        b['clv'] = side * (b.open_home_spread - b.close_home_spread)
        cover = side * (b.margin + b.open_home_spread)
        b['won'] = np.where(cover > 0, 1.0, np.where(cover < 0, 0.0, np.nan))
        b['profit'] = np.where(cover > 0, STANDARD_PAYOUT, np.where(cover < 0, -1.0, 0.0))
        rows.append(_summarize(b, 'clv', 'won', 'profit', thr))
    print("SPREAD (model margin vs opening line, points)")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    d = df.dropna(subset=['open_total', 'close_total'])
    edge = d.total_pred - d.open_total
    rows = []
    for thr in TOTAL_THRESHOLDS:
        b = d[edge.abs() > thr].copy()
        side = np.sign(edge[edge.abs() > thr])  # +1 = over
        b['clv'] = side * (b.close_total - b.open_total)
        res = side * (b.total_pts - b.open_total)
        b['won'] = np.where(res > 0, 1.0, np.where(res < 0, 0.0, np.nan))
        b['profit'] = np.where(res > 0, STANDARD_PAYOUT, np.where(res < 0, -1.0, 0.0))
        rows.append(_summarize(b, 'clv', 'won', 'profit', thr))
    print("\nTOTAL (model total vs opening total, points)")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # Moneyline: CLV measured as the expected ROI of the opening price under the CLOSING no-vig probability
    d = df.dropna(subset=['open_home_prob', 'close_home_prob', 'open_home_ml', 'open_away_ml'])
    edge = d.ml_prob - d.open_home_prob
    rows = []
    for thr in ML_THRESHOLDS:
        b = d[edge.abs() > thr].copy()
        home = (edge[edge.abs() > thr] > 0).values
        p_close = np.where(home, b.close_home_prob, 1 - b.close_home_prob)
        pay_med = np.where(home, american_to_payout(b.open_home_ml), american_to_payout(b.open_away_ml))
        pay_best = np.where(home, american_to_payout(b.open_home_ml_best), american_to_payout(b.open_away_ml_best))
        won = np.where(home, b.home_win, 1 - b.home_win).astype(float)
        b['clv'] = p_close * pay_med - (1 - p_close)        # expected ROI per unit vs close
        b['won'] = won
        b['profit'] = np.where(won == 1, pay_med, -1.0)
        b['profit_best'] = np.where(won == 1, pay_best, -1.0)
        b['clv_best'] = p_close * pay_best - (1 - p_close)
        b['dog'] = np.where(home, b.open_home_prob < 0.5, b.open_home_prob > 0.5)
        rows.append(_summarize(b, 'clv', 'won', 'profit', thr, extra=lambda x: {
            'CLV best px': x.clv_best.mean(), 'ROI best px %': x.profit_best.sum() / len(x) * 100,
            '% dogs': x.dog.mean() * 100}))
    # Baseline: a random side (average of both sides) at the same prices, valued by the closing probability
    def _ev(p, pay):
        return p * pay - (1 - p)
    base_med = (_ev(d.close_home_prob, american_to_payout(d.open_home_ml))
                + _ev(1 - d.close_home_prob, american_to_payout(d.open_away_ml))) / 2
    base_best = (_ev(d.close_home_prob, american_to_payout(d.open_home_ml_best))
                 + _ev(1 - d.close_home_prob, american_to_payout(d.open_away_ml_best))) / 2
    rows.append({'threshold': 'random side', 'bets': len(d), 'mean CLV': base_med.mean(),
                 'CLV best px': base_best.mean()})
    print("\nMONEYLINE (model prob vs opening no-vig prob). 'mean CLV' = expected ROI per unit of the opening\n"
          "median price if the closing no-vig probability is the truth; 'best px' = best price across books.")
    out = pd.DataFrame(rows)
    for c in ('mean CLV', 'CLV best px'):
        out[c] = out[c] * 100
    print(out.round(2).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preds', default='models/saved/backtest_predictions_calibrated.csv')
    parser.add_argument('--db', default='data/nba_stats.db')
    parser.add_argument('--out', help='Optional CSV path for the joined prediction + odds frame')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    df = load(Path(args.preds), Path(args.db))
    accuracy_tables(df)
    information_tests(df)
    clv_tables(df)
    if args.out:
        df.to_csv(args.out, index=False)
        logging.info(f"Saved joined frame to {args.out}")


if __name__ == "__main__":
    main()
