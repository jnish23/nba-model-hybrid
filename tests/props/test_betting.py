import numpy as np
import pandas as pd

from props.betting import prices, rules, settle


def test_american_to_decimal():
    assert np.allclose(prices.american_to_decimal([100, -110, 150, -200]), [2.0, 1 + 100 / 110, 2.5, 1.5])


def test_settlement_stands_push_void_and_early_exit_treatments():
    kw = dict(line=np.array([20.5, 20.5, 20.0, 20.5, 20.5, 20.5]), dec_price=np.array([1.9] * 6),
              y=np.array([25, 18, 20, 25, 25, np.nan]), played=np.array([1, 1, 1, 1, 1, 0]),
              minutes=np.array([30, 30, 30, 5, 5, 0.0]), expected_minutes=np.array([30, 30, 30, 30, 30, 30.0]),
              book=np.array(['fanduel', 'fanduel', 'fanduel', 'fanatics', 'fanduel', 'fanduel']))
    side = np.array(['over', 'over', 'over', 'over', 'over', 'over'])
    p, s = settle.settle(side, treatment='stands', **kw)
    assert list(s) == ['win', 'loss', 'push', 'win', 'win', 'void'] and np.allclose(p, [0.9, -1, 0, 0.9, 0.9, 0])
    p, s = settle.settle(side, treatment='early_exit_flagged_books', **kw)                 # the 5-minute night at Fanatics is voided, at FanDuel it stands
    assert list(s) == ['win', 'loss', 'push', 'void', 'win', 'void']
    p, s = settle.settle(side, treatment='early_exit_all_books', **kw)
    assert list(s) == ['win', 'loss', 'push', 'void', 'void', 'void']
    under = np.array(['under'] * 6)                                                          # unders are never voided by an early exit
    _, s = settle.settle(under, treatment='early_exit_all_books', **kw)
    assert list(s) == ['loss', 'win', 'push', 'loss', 'loss', 'void']


def test_ev_matches_a_fair_coin_and_refunds_pushes():
    ev_o, ev_u = rules.ev_over_under(0.5, 0.0, 2.0, 2.0)
    assert abs(ev_o) < 1e-12 and abs(ev_u) < 1e-12
    ev_o, ev_u = rules.ev_over_under(0.55, 0.0, 1.909090909, 1.909090909)                 # -110 both ways
    assert abs(ev_o - (0.55 * 0.909090909 - 0.45)) < 1e-9 and ev_u < 0
    ev_o2, _ = rules.ev_over_under(0.55, 0.2, 1.909090909, 1.909090909)                    # a 20% push chance scales the whole EV by 0.8
    assert abs(ev_o2 - 0.8 * ev_o) < 1e-9


def test_shrinkage_and_selection_one_bet_per_market_earliest_time():
    q = pd.DataFrame({'game_id': ['g'] * 5, 'player_id': [1] * 5, 'stat': ['points'] * 5, 'minutes_before': [240, 240, 60, 60, 5], 'book': list('abcde'),
                      'consensus_fair_over': [0.5] * 5, 'p_mod': [0.5, 0.7, 0.7, 0.7, 0.7], 'push': [0.0] * 5, 'over_dec': [1.9, 1.9, 2.0, 1.95, 2.1],
                      'under_dec': [1.9] * 5, 'n_at_line': [4, 4, 4, 2, 6]})
    k = {'points': 0.5}
    q = rules.add_ev(q, k)
    assert np.allclose(q['p_star'].iloc[1], 0.6) and q['best_side'].iloc[1] == 'over'
    sel = rules.select(q, e=0.03, time_mode='earliest', min_books=3)
    assert len(sel) == 1 and sel['minutes_before'].iloc[0] == 240 and sel['book'].iloc[0] == 'b'           # earliest time with a qualifying quote; best EV within it
    sel5 = rules.select(q, e=0.03, time_mode='earliest', min_books=5)
    assert len(sel5) == 1 and sel5['minutes_before'].iloc[0] == 5                                             # only the 6-book quote at T-5 qualifies
    assert len(rules.select(q, e=0.03, time_mode='60', min_books=3)) == 1
    assert rules.clip_k(-0.1) == 0.0 and rules.clip_k(0.9) == rules.K_CAP and rules.clip_k(0.2) == 0.2


def test_line_move_sign_in_clv_table():
    import pandas as pd
    from props.betting import evaluate as be
    n = 200
    sel = pd.DataFrame({'game_id': [f'g{i}' for i in range(n)], 'player_id': 1, 'stat': 'points', 'minutes_before': 60, 'line': 20.5, 'best_side': ['over'] * (n // 2) + ['under'] * (n // 2),
                        'consensus_fair_over': 0.5})
    cons = pd.DataFrame({'game_id': sel['game_id'], 'player_id': 1, 'stat': 'points', 'minutes_before': 5, 'consensus_line': 21.5, 'consensus_fair_over': 0.5})      # the line rose by one point
    t, counts = be.clv_table(sel, cons, n_boot=100)
    row = t[t['measure'].str.startswith('line moved')].iloc[0]
    assert counts['line_moved'] == n and abs(row['mean']) < 1e-9          # +1 for the over bettors, -1 for the under bettors: they cancel in a 50/50 mix
    over_only = be.clv_table(sel[sel['best_side'] == 'over'].reset_index(drop=True).assign(game_id=lambda d: d['game_id']), cons, n_boot=100)[0]
    assert over_only[over_only['measure'].str.startswith('line moved')]['mean'].iloc[0] > 0.99       # an over bettor gains when the line rises
