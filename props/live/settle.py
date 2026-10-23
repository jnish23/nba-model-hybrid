"""
Settlement of recorded bets from the box scores: outcome, the three early-exit treatments, profit under the three pricing views (best, median, flat -110) and CLV to T-5.
Pure functions of the bet rows, the labels and the signals; the job writes the result rows. Same functions as the backtest (props.betting.evaluate / export).
"""
import numpy as np
import pandas as pd

from dashboard_props import logic
from props.betting import evaluate as be, export, settle as settle_mod

LABEL_COLS = ['game_id', 'player_id', 'lbl_played', 'lbl_minutes'] + be.LABEL_COLS


def labels_from_history(hist):
    """Label frame in the dataset's format from the box-score history (rows.load_history)."""
    lab = hist.rename(columns={'minutes': 'lbl_minutes'})
    lab = lab.assign(game_id=lab['game_id'].astype(str))
    return lab[LABEL_COLS].drop_duplicates(['game_id', 'player_id'])


def signal_context(signals):
    """Consensus line and fair P(over) per (game, player, stat, slot) from the recorded signals, in the form the backtest's CLV function expects."""
    cons = signals.rename(columns={'slot_minutes': 'minutes_before', 'line': 'consensus_line', 'p_market': 'consensus_fair_over'})
    cons = cons[['game_id', 'player_id', 'stat', 'minutes_before', 'consensus_line', 'consensus_fair_over']].drop_duplicates(['game_id', 'player_id', 'stat', 'minutes_before'])
    hist = cons.pivot_table(index=['game_id', 'player_id', 'stat'], columns='minutes_before', values=['consensus_line', 'consensus_fair_over'], aggfunc='first')
    hist.columns = [f"{'line' if a == 'consensus_line' else 'mkt'}_{b}" for a, b in hist.columns]
    return cons, hist.reset_index()


def settle_bets(bets, labels, signals, settled_utc, run_id):
    """bets: prop_bets rows of games whose box score is in `labels`. Returns prop_results rows."""
    if bets.empty:
        return pd.DataFrame()
    b = bets.rename(columns={'side': 'best_side', 'dec_best': 'dec_price'}).copy()
    b = be.settle_all(be.attach_outcomes(b.reset_index(drop=True), labels))
    out = pd.DataFrame({'bet_id': b['bet_id'], 'settled_utc': settled_utc, 'y': b['y'], 'played': b['played'].astype(int), 'minutes_actual': b['lbl_minutes'],
                        'status_stands': b['status_stands'], 'status_flagged': b['status_early_exit_flagged_books'], 'status_all': b['status_early_exit_all_books'],
                        'profit_stands_best': b['profit_stands'], 'run_id': run_id})
    for name, col in (('median', 'dec_median'), ('flat', 'dec_flat')):
        prof, _ = settle_mod.settle(b['best_side'].to_numpy(), b['line'], b[col], b['y'], b['played'], b['lbl_minutes'], b['expected_minutes'], b['book'].to_numpy(), 'stands')
        out[f'profit_stands_{name}'] = prof
    if signals is not None and len(signals):
        cons, hist = signal_context(signals)
        sel = b.assign(consensus_fair_over=b['p_over_market'])
        out['clv_prob'], out['line_move_favour'], out['close_line'] = export.clv_per_bet(sel, cons)
        h = b[['bet_id', 'game_id', 'player_id', 'stat']].merge(hist, on=['game_id', 'player_id', 'stat'], how='left')
        for c in ('line_240', 'line_60', 'line_5', 'mkt_240', 'mkt_60', 'mkt_5'):
            out[c] = h[c].to_numpy() if c in h.columns else np.nan
    return out
