import numpy as np
import pandas as pd

from props.live import settle


def _bets():
    base = {'season': 'S', 'game_id': 'g1', 'stat': 'assists', 'dec_median': 1.9, 'dec_flat': 1.909091, 'expected_minutes': 30.0, 'book': 'fanduel', 'minutes_before': 240,
            'p_over_market': 0.5}
    return pd.DataFrame([{**base, 'bet_id': 'b1', 'player_id': 1, 'side': 'over', 'line': 4.5, 'dec_best': 2.0},
                         {**base, 'bet_id': 'b2', 'player_id': 2, 'side': 'under', 'line': 6.5, 'dec_best': 1.8},
                         {**base, 'bet_id': 'b3', 'player_id': 3, 'side': 'over', 'line': 3.5, 'dec_best': 1.9},
                         {**base, 'bet_id': 'b4', 'player_id': 4, 'side': 'over', 'line': 5.0, 'dec_best': 1.9}])


def _labels():
    return pd.DataFrame({'game_id': 'g1', 'player_id': [1, 2, 3, 4], 'lbl_played': [1, 1, 0, 1], 'lbl_minutes': [30.0, 30.0, 0.0, 30.0],
                         'y_points': 10, 'y_rebounds': 5, 'y_assists': [6, 8, 0, 5], 'y_threes': 1, 'y_pra': 20})


def _signals():
    rows = []
    for pid in (1, 2, 3, 4):
        for slot, line, mkt in ((240, 4.5, 0.50), (5, 4.5, 0.56)):
            rows.append({'game_id': 'g1', 'player_id': pid, 'stat': 'assists', 'slot_minutes': slot, 'line': line if pid != 2 else 6.5, 'p_market': mkt})
    return pd.DataFrame(rows)


def test_settlement_wins_losses_voids_and_pushes_at_the_three_prices():
    r = settle.settle_bets(_bets(), _labels(), _signals(), '2026-10-24T08:00:00Z', 'r1').set_index('bet_id')
    assert r.loc['b1', 'status_stands'] == 'win' and np.isclose(r.loc['b1', 'profit_stands_best'], 1.0) and np.isclose(r.loc['b1', 'profit_stands_median'], 0.9)
    assert r.loc['b2', 'status_stands'] == 'loss' and r.loc['b2', 'profit_stands_best'] == -1.0 and r.loc['b2', 'profit_stands_flat'] == -1.0
    assert r.loc['b3', 'status_stands'] == 'void' and r.loc['b3', 'profit_stands_best'] == 0.0 and r.loc['b3', 'played'] == 0
    assert r.loc['b4', 'status_stands'] == 'push' and r.loc['b4', 'profit_stands_median'] == 0.0


def test_clv_is_the_fair_probability_move_of_the_bet_side_and_blank_when_the_line_moved():
    sig = _signals()
    r = settle.settle_bets(_bets(), _labels(), sig, 'now', 'r1').set_index('bet_id')
    assert np.isclose(r.loc['b1', 'clv_prob'], 0.06)                                   # over at 0.50, closing fair 0.56
    assert np.isclose(r.loc['b2', 'clv_prob'], -0.06) and r.loc['b2', 'close_line'] == 6.5      # under at 0.50, closing fair 0.44
    moved = sig.copy()
    moved.loc[(moved.player_id == 1) & (moved.slot_minutes == 5), 'line'] = 5.5
    r2 = settle.settle_bets(_bets(), _labels(), moved, 'now', 'r1').set_index('bet_id')
    assert np.isnan(r2.loc['b1', 'clv_prob']) and r2.loc['b1', 'line_move_favour'] == 1.0 and r2.loc['b1', 'line_5'] == 5.5
