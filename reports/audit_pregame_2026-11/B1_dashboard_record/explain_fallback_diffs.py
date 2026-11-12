"""B1: why the dashboard's fallback path (no bet_recommendations_log: recompute with the deployed sigmas and -110) differs from A1's reconstruction, bet by bet.
For each game where the two disagree, the edge the page computes (sigma from the model set, -110) next to A1's (snapshot / recovered price; for totals the 18.94 stand-in sigma),
and which of the two ingredients flips the decision. Reads A1's recon_games.csv (rc_*_A columns) and bets.csv; read-only. Run from the worktree root."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from team.betting import get_spread_bet, get_total_bet

A1 = ROOT / 'reports/audit_pregame_2026-11/A1_season_bets'
meta = json.loads((ROOT / 'models/saved/calibration_meta_with_odds_cand_20261019.json').read_text())
SP, TT = meta['spread_std'], meta['total_std']
TT_A1 = json.loads((A1 / 'recon.json').read_text())['implied_total_sigma_median']
R = pd.read_csv(A1 / 'recon_games.csv', dtype={'game_id': str})
B = pd.read_csv(A1 / 'bets.csv', dtype={'game_id': str})
R = R[R.home_score.notna()]          # completed games only: the dashboard's record has no pending bets

lines = ["| market | game | line | model number | A1 decision (price, sigma) | A1 edge | page decision (-110, model-set sigma) | page edge | which ingredient flips it |", "|---|---|---|---|---|---|---|---|---|"]
rows = []
for mk, name in (('Spread', 'spread'), ('Total', 'total')):
    a1 = B[B.market == mk].set_index('game_id')                  # all A1 bets, settled or pending
    for r in R.itertuples():
        gid = r.game_id
        if name == 'spread':
            side_p, edge_p = get_spread_bet(r.spread_pred, r.odds_home_spread, SP, 0.05)
            px = r.rc_spread_price_A if isinstance(r.rc_spread_bet_A, str) else np.nan
            a1_side = a1.side.get(gid)
            # A1 price for the side the model leans to (also when A1 did not bet): the price A1 used in its edge
            a1_edge = r.rc_spread_edge_A
            lean_home = r.spread_pred + r.odds_home_spread > 0
            a1_price = r.rc_spread_price_A if pd.notna(r.rc_spread_price_A) else -110
            pr_only, _ = get_spread_bet(r.spread_pred, r.odds_home_spread, SP, 0.05, home_price=a1_price if lean_home else None, away_price=None if lean_home else a1_price)
            num, line = r.spread_pred, r.odds_home_spread
            sig_a1 = SP
        else:
            side_p, edge_p = get_total_bet(r.total_pred, r.odds_total, TT, 0.05)
            a1_side = a1.side.get(gid)
            a1_edge = r.rc_total_edge_A
            lean_over = r.total_pred - r.odds_total > 0
            a1_price = r.rc_total_price_A if pd.notna(r.rc_total_price_A) else -110
            pr_only, _ = get_total_bet(r.total_pred, r.odds_total, TT, 0.05, over_price=a1_price if lean_over else None, under_price=None if lean_over else a1_price)
            num, line = r.total_pred, r.odds_total
            sig_a1 = TT_A1
        if (a1_side if isinstance(a1_side, str) else None) == side_p:
            continue
        a1_price_txt = f"{a1_price:+.0f}"
        a1_dec = a1_side if isinstance(a1_side, str) else None
        if name == 'spread':
            sig_only, _ = get_spread_bet(r.spread_pred, r.odds_home_spread, SP, 0.05)          # A1's sigma (the model set's) at -110: identical to the page, so never flips
        else:
            sig_only, _ = get_total_bet(r.total_pred, r.odds_total, TT_A1, 0.05)               # A1's stand-in sigma at -110
        flips = []
        if pr_only == a1_dec:
            flips.append("price")
        if sig_only == a1_dec:
            flips.append(f"sigma ({TT:.2f} vs A1's stand-in {TT_A1:.2f})")
        if not flips:
            flips.append("price and sigma together")
        lines.append(f"| {mk} | {gid} {r.matchup} {r.game_date} | {line:g} | {num:.1f} | {a1_side if isinstance(a1_side, str) else 'no bet'} ({a1_price_txt}, {sig_a1:.2f}) | {a1_edge:+.4f} | "
                     f"{side_p or 'no bet'} | {edge_p:+.4f} | {' or '.join(flips)} |")
        rows.append(dict(market=mk, game_id=gid))
Path(Path(__file__).parent / 'fallback_differences.md').write_text("\n".join(lines) + "\n", encoding='utf-8')
print("\n".join(lines))
print(len(rows), 'games differ')
