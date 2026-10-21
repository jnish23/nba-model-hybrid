"""
Selection rules for simulated betting (reports/props/phase5_plan.md, section 4). Nothing here is fitted to a result; k and the thresholds come from the development season
and are frozen before any later season is looked at.

For a quote at the consensus line:  p_mkt = consensus de-vigged P(over);  p_mod = the model's P(over | no push);  push = the model's P(push) at that line.
Shrunk probability  p* = p_mkt + k * (p_mod - p_mkt)  (k = information slope, floor 0, cap K_CAP).
EV per unit staked (a push refunds):  over = (1 - push) * (p* * (dec_over - 1) - (1 - p*)),  under = (1 - push) * ((1 - p*) * (dec_under - 1) - p*).
"""
import numpy as np
import pandas as pd

K_CAP = 0.5
TIMES = (240, 60, 5)


def shrunk(p_mkt, p_mod, k):
    return np.asarray(p_mkt, float) + np.asarray(k, float) * (np.asarray(p_mod, float) - np.asarray(p_mkt, float))


def ev_over_under(p_star, push, dec_over, dec_under):
    p_star, push = np.asarray(p_star, float), np.asarray(push, float)
    ev_o = (1.0 - push) * (p_star * (np.asarray(dec_over, float) - 1.0) - (1.0 - p_star))
    ev_u = (1.0 - push) * ((1.0 - p_star) * (np.asarray(dec_under, float) - 1.0) - p_star)
    return ev_o, ev_u


def clip_k(slope):
    return float(np.clip(slope, 0.0, K_CAP))


def add_ev(q, k_by_stat):
    """q: one row per bettable quote with p_mkt (consensus_fair_over), p_mod, push, over_dec, under_dec, stat. Adds p_star, ev_over, ev_under, best_side, best_ev."""
    k = q['stat'].map(k_by_stat).fillna(0.0).to_numpy(float)
    q = q.copy()
    q['p_star'] = shrunk(q['consensus_fair_over'], q['p_mod'], k)
    q['ev_over'], q['ev_under'] = ev_over_under(q['p_star'], q['push'], q['over_dec'], q['under_dec'])
    q['best_side'] = np.where(q['ev_over'] >= q['ev_under'], 'over', 'under')
    q['best_ev'] = np.maximum(q['ev_over'], q['ev_under'])
    q['dec_price'] = np.where(q['best_side'] == 'over', q['over_dec'], q['under_dec'])
    return q


def select(q, e, time_mode='earliest', min_books=3):
    """
    One bet per (game, player, stat). Among bettable quotes with at least min_books agreeing books, at each decision time take the quote with the highest EV; the bet is
    placed at the first allowed time whose best EV is at least e ('earliest') or only at the given time (240/60/5). Both sides of one market are never both bet.
    """
    q = q[(q['n_at_line'] >= min_books) & (q['best_ev'] >= e)]
    if time_mode != 'earliest':
        q = q[q['minutes_before'] == int(time_mode)]
    q = q.sort_values(['game_id', 'player_id', 'stat', 'minutes_before', 'best_ev'], ascending=[True, True, True, False, False])
    first_t = q.groupby(['game_id', 'player_id', 'stat'])['minutes_before'].transform('max')            # larger minutes_before = earlier in time
    q = q[q['minutes_before'] == first_t]
    return q.drop_duplicates(['game_id', 'player_id', 'stat']).reset_index(drop=True)
