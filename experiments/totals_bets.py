"""Flat -110 over/under profits per game at the opening total, the closing total and the best-book line and quoted price (E12 method).
Shared by the totals-angle experiments from E19 on (E16 and E17 carry their own copy)."""
import numpy as np

from experiments.e12_shopping import load_multibook, payout

WIN = 100 / 110


def best_line_and_price(lines, odds, pick_min):
    bl = (lines.min(axis=1) if pick_min else lines.max(axis=1)).to_numpy()
    pr = np.full(len(bl), np.nan)
    for i, (lrow, orow, b) in enumerate(zip(lines.to_numpy(), odds.to_numpy(), bl)):
        if np.isnan(b):
            continue
        q = orow[lrow == b]
        q = q[~np.isnan(q)]
        pr[i] = q.max() if len(q) else -110.0
    return bl, pr


def settlement_profits(df):
    """df needs gameId, odds_total, close_total, target_total_pts. Returns (profits dict, availability masks dict)."""
    mats, _ = load_multibook()
    tots, over_odds, under_odds = (mats[k].reindex(df['gameId']) for k in ('tot', 'over', 'under'))
    a = df['target_total_pts'].to_numpy(float)
    ob_line, ob_pr = best_line_and_price(tots, over_odds, True)
    ub_line, ub_pr = best_line_and_price(tots, under_odds, False)

    def prof(line, side, price=None):
        r = (a - np.asarray(line, float)) * (1 if side == 'over' else -1)
        pay = np.full(len(df), WIN) if price is None else payout(np.nan_to_num(price, nan=-110.0))
        return np.where(r > 0, pay, np.where(r < 0, -1.0, 0.0))
    P = {'over_open': prof(df['odds_total'], 'over'), 'over_close': prof(df['close_total'], 'over'), 'over_best': prof(ob_line, 'over', ob_pr),
         'under_open': prof(df['odds_total'], 'under'), 'under_close': prof(df['close_total'], 'under'), 'under_best': prof(ub_line, 'under', ub_pr)}
    ok = {'close': df['close_total'].notna().to_numpy(), 'best': ~np.isnan(ob_line)}
    return P, ok
