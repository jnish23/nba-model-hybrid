"""
Settlement of recorded bets from the box scores: outcome, the three early-exit treatments, profit under the three pricing views (best, median, flat -110) and CLV to T-5.
Pure functions of the bet rows, the labels and the signals; the job writes the result rows. Same functions as the backtest (props.betting.evaluate / export).
"""
import numpy as np
import pandas as pd

from core import props_board_logic as logic
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
    for c in ('clv_prob', 'line_move_favour', 'close_line', 'line_240', 'line_60', 'line_5', 'mkt_240', 'mkt_60', 'mkt_5'):
        out[c] = np.nan                                 # bets with no recorded signals (restored after the 2026-10-31 loss) get the columns, blank
    if signals is not None and len(signals):
        cons, hist = signal_context(signals)
        sel = b.assign(consensus_fair_over=b['p_over_market'])
        out['clv_prob'], out['line_move_favour'], out['close_line'] = export.clv_per_bet(sel, cons)
        h = b[['bet_id', 'game_id', 'player_id', 'stat']].merge(hist, on=['game_id', 'player_id', 'stat'], how='left')
        for c in ('line_240', 'line_60', 'line_5', 'mkt_240', 'mkt_60', 'mkt_5'):
            out[c] = h[c].to_numpy() if c in h.columns else np.nan
    return out


def clv_audit_rows(res, signals, written_utc, run_id):
    """
    prop_clv_audit rows (source 'signals') for the settled bets in `res` that have a closing quote. The snapshot time and minutes are those of the recorded slot-5 signal the CLV
    was computed from (the same row signal_context pivots), so the provenance is the signal and not a re-derivation.
    """
    cols = ['bet_id', 'written_utc', 'source', 'close_snapshot_ts_utc', 'close_snapshot_minutes', 'close_line', 'mkt_close', 'clv_prob', 'line_move_favour', 'run_id']
    if res is None or res.empty or signals is None or signals.empty:
        return pd.DataFrame(columns=cols)
    r = res[res['close_line'].notna()]
    if r.empty:
        return pd.DataFrame(columns=cols)
    ids = r['bet_id'].str.split('|', expand=True)                          # bet_id = season|game_id|player_id|stat
    keys = pd.DataFrame({'bet_id': r['bet_id'].to_numpy(), 'game_id': ids[1].astype(str).to_numpy(), 'player_id': ids[2].astype(int).to_numpy(), 'stat': ids[3].to_numpy()})
    s5 = signals[signals['slot_minutes'] == 5].assign(game_id=lambda d: d['game_id'].astype(str))
    s5 = s5.sort_values('snapshot_ts_utc').drop_duplicates(['game_id', 'player_id', 'stat'], keep='last')
    s5 = s5[['game_id', 'player_id', 'stat', 'snapshot_ts_utc', 'snapshot_minutes']]
    k = keys.merge(s5, on=['game_id', 'player_id', 'stat'], how='left')
    return pd.DataFrame({'bet_id': r['bet_id'].to_numpy(), 'written_utc': written_utc, 'source': 'signals', 'close_snapshot_ts_utc': k['snapshot_ts_utc'].to_numpy(),
                         'close_snapshot_minutes': k['snapshot_minutes'].to_numpy(), 'close_line': r['close_line'].to_numpy(), 'mkt_close': r['mkt_5'].to_numpy(),
                         'clv_prob': r['clv_prob'].to_numpy(), 'line_move_favour': r['line_move_favour'].to_numpy(), 'run_id': run_id})[cols]


def detect_corrections(bets, results, labels, detected_utc=None, run_id=None):
    """
    Re-settle the already-settled bets with the current box-score labels and return the rows where y or the 'stands' status differs from the stored prop_results row (which is never rewritten).
    bets: prop_bets rows; results: prop_results rows (bet_id, y, status_stands, profit_stands_best); labels as for settle_bets. Bets whose game is not in `labels` are skipped.
    Columns: bet_id, y_old, y_new, status_old, status_new, profit_best_old, profit_best_new, profit_median_new, profit_flat_new (+ detected_utc, run_id when given).
    """
    cols = ['bet_id', 'y_old', 'y_new', 'status_old', 'status_new', 'profit_best_old', 'profit_best_new', 'profit_median_new', 'profit_flat_new']
    extra = (['detected_utc', 'run_id'] if detected_utc is not None or run_id is not None else [])
    empty = pd.DataFrame(columns=cols + extra)
    if bets is None or bets.empty or results is None or results.empty or labels is None or labels.empty:
        return empty
    b = bets[bets['bet_id'].isin(set(results['bet_id'])) & bets['game_id'].astype(str).isin(set(labels['game_id'].astype(str)))]
    if b.empty:
        return empty
    new = settle_bets(b.assign(game_id=b['game_id'].astype(str)), labels.assign(game_id=labels['game_id'].astype(str)), None, detected_utc, run_id)
    old = results[['bet_id', 'y', 'status_stands', 'profit_stands_best']].rename(columns={'y': 'y_old', 'status_stands': 'status_old', 'profit_stands_best': 'profit_best_old'})
    m = new[['bet_id', 'y', 'status_stands', 'profit_stands_best', 'profit_stands_median', 'profit_stands_flat']].rename(
        columns={'y': 'y_new', 'status_stands': 'status_new', 'profit_stands_best': 'profit_best_new', 'profit_stands_median': 'profit_median_new', 'profit_stands_flat': 'profit_flat_new'}
    ).merge(old, on='bet_id', how='inner', validate='one_to_one')
    y_old, y_new = m['y_old'].to_numpy(float), m['y_new'].to_numpy(float)
    y_changed = ~((y_old == y_new) | (np.isnan(y_old) & np.isnan(y_new)))
    s_changed = (m['status_old'].fillna('') != m['status_new'].fillna('')).to_numpy()
    out = m[y_changed | s_changed][cols].reset_index(drop=True)
    if extra:
        out = out.assign(detected_utc=detected_utc, run_id=run_id)
    return out
