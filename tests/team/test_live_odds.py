import numpy as np

from team.live_odds import first_book_value, split_open_and_current


def _g(**markets):
    return {k: {'draftkings': v} for k, v in markets.items()}


def test_model_inputs_are_opening_and_current_kept_separately():
    curr = _g(home_ml=-150, away_ml=130, home_spread=-4.0, away_spread=4.0, total=221.5)
    opened = _g(home_ml=-135, away_ml=115, home_spread=-3.0, away_spread=3.0, total=219.0)
    out = split_open_and_current(curr, opened)
    assert out['odds_home_spread'] == -3.0 and out['curr_odds_home_spread'] == -4.0
    assert out['odds_total'] == 219.0 and out['curr_odds_total'] == 221.5
    assert out['odds_home_ml'] == -135 and out['curr_odds_home_ml'] == -150
    assert out['odds_open_missing'] == 0


def test_pickem_spread_zero_is_a_real_value():
    assert first_book_value({'draftkings': 0}) == 0.0
    assert np.isnan(first_book_value({'draftkings': None, 'fanduel': ''}))
    assert first_book_value({'draftkings': None, 'fanduel': '-3.5'}) == -3.5


def test_missing_opening_line_is_never_replaced_by_the_current_line():
    """The spread model learns what the OPENING line misses; the current line already contains the news since the open."""
    curr = _g(home_ml=-150, away_ml=130, home_spread=-4.0, away_spread=4.0, total=221.5)
    opened = _g(home_spread=-3.0, away_spread=3.0, total=219.0)      # no moneylines opened
    out = split_open_and_current(curr, opened)
    assert np.isnan(out['odds_home_ml']) and np.isnan(out['odds_away_ml'])
    assert out['curr_odds_home_ml'] == -150                                 # still available for bets
    assert out['odds_home_spread'] == -3.0 and out['odds_open_missing'] == 2
    out = split_open_and_current(curr, None)
    assert out['odds_open_missing'] == 5 and np.isnan(out['odds_total']) and out['curr_odds_total'] == 221.5


def test_current_prices_are_carried_for_edge_calculations():
    curr = {**_g(home_spread=-4.0, away_spread=4.0, total=221.5), 'home_spread_odds': {'draftkings': -115},
            'away_spread_odds': {'draftkings': -105}, 'over_odds': {'draftkings': -120}}
    out = split_open_and_current(curr, curr)
    assert out['curr_price_home_spread_odds'] == -115 and out['curr_price_away_spread_odds'] == -105
    assert out['curr_price_over_odds'] == -120 and np.isnan(out['curr_price_under_odds'])


def test_every_field_of_a_game_comes_from_one_book():
    """Old rule: field-by-field fill, so the spread could be one book's and the moneyline another's."""
    curr = {'home_spread': {'draftkings': -3.5, 'fanduel': -4.0}, 'away_spread': {'draftkings': 3.5, 'fanduel': 4.0},
            'total': {'draftkings': 221.5, 'fanduel': 222.0}, 'home_ml': {'fanduel': -160}, 'away_ml': {'fanduel': 140}}   # DK has no ML
    out = split_open_and_current(curr, curr)
    assert (out['odds_home_spread'], out['odds_total'], out['odds_home_ml'], out['odds_away_ml']) == (-4.0, 222.0, -160, 140)   # all FanDuel


def test_a_book_far_from_the_consensus_is_not_used():
    quote = lambda dk, fd, mgm: {'draftkings': dk, 'fanduel': fd, 'betmgm': mgm}
    curr = {'home_spread': quote(-9.5, -3.5, -3.5), 'away_spread': quote(9.5, 3.5, 3.5), 'total': quote(231.0, 221.5, 222.0),
            'home_ml': quote(-900, -160, -165), 'away_ml': quote(600, 140, 140)}       # DK is an in-game quote
    out = split_open_and_current(curr, curr)
    assert out['odds_home_spread'] == -3.5 and out['odds_total'] == 221.5 and out['odds_home_ml'] == -160


# ------------------------------------------------------------------ bet log row

import pytest

from team.live_odds import bet_log_row


def test_split_open_and_current_reports_which_book_supplied_each_side():
    curr = {'home_spread': {'fanduel': -4.0}, 'away_spread': {'fanduel': 4.0}, 'total': {'fanduel': 221.5},
            'home_ml': {'fanduel': -150}, 'away_ml': {'fanduel': 130}}
    opened = _g(home_ml=-135, away_ml=115, home_spread=-3.0, away_spread=3.0, total=219.0)
    out = split_open_and_current(curr, opened)
    assert (out['open_book'], out['curr_book']) == ('draftkings', 'fanduel')
    assert split_open_and_current(curr, None)['open_book'] is None


def _logged_game():
    return {'gameId': '0022600200', 'gameDate': '2026-11-11', 'odds_source': 'sbr', 'open_book': 'draftkings', 'curr_book': 'draftkings',
            'odds_home_ml': -135.0, 'odds_away_ml': 115.0, 'odds_home_spread': -3.0, 'odds_total': 219.0,
            'curr_odds_home_ml': -150.0, 'curr_odds_away_ml': 130.0, 'curr_odds_home_spread': -4.0, 'curr_odds_total': 221.5,
            'curr_price_home_spread_odds': -115.0, 'curr_price_away_spread_odds': -105.0,
            'curr_price_over_odds': -108.0, 'curr_price_under_odds': -112.0}


def test_bet_log_row_own_view_and_line_move_and_their_identities():
    pred = {'ml_prob': 0.62, 'ml_prob_raw': 0.60, 'spread_pred': 5.0, 'total_pred': 225.0, 'spread_std': 13.0, 'total_std': 18.0}
    row = bet_log_row(_logged_game(), pred, {'ml': ('home', 0.03), 'spread': ('home', 0.07), 'total': (None, 0.02)},
                      '2026-11-11T16:19:00.000001Z', 'hash1', '2026-11-11 11:19:00-05:00', 'r.pdf')
    # spread: model margin 5.0 against the opening line -3.0 (implied margin 3.0) = +2.0; line moved from -3.0 to -4.0 = +1.0
    assert row['spread_own_view'] == 2.0 and row['spread_line_move'] == 1.0
    assert row['spread_pred'] + row['curr_odds_home_spread'] == row['spread_own_view'] - row['spread_line_move']        # edge in points at the current line
    assert row['total_own_view'] == 6.0 and row['total_line_move'] == -2.5
    assert 225.0 - 221.5 == row['total_own_view'] + row['total_line_move']
    open_fair = (135 / 235) / (135 / 235 + 100 / 215)
    curr_fair = (150 / 250) / (150 / 250 + 100 / 230)
    assert row['ml_own_view'] == pytest.approx(0.62 - open_fair) and row['ml_line_move'] == pytest.approx(open_fair - curr_fair)
    assert (row['ml_bet'], row['spread_bet'], row['total_bet']) == ('home', 'home', None)
    assert row['spread_edge'] == 0.07 and row['total_edge'] == 0.02 and row['ml_prob_raw'] == 0.60
    assert (row['game_id'], row['model_set_hash'], row['injury_report_file'], row['curr_book']) == ('0022600200', 'hash1', 'r.pdf', 'draftkings')


def test_bet_log_row_missing_lines_are_null_not_zero():
    g = {**_logged_game(), 'curr_odds_home_spread': np.nan, 'curr_odds_home_ml': np.nan, 'curr_price_over_odds': None, 'curr_book': None}
    pred = {'ml_prob': 0.62, 'spread_pred': 5.0, 'total_pred': 225.0, 'spread_std': 13.0, 'total_std': 18.0}      # no ml_prob_raw
    row = bet_log_row(g, pred, {'ml': (None, 0.0), 'spread': (None, 0.0), 'total': ('over', 0.06)}, 't', None, None, None)
    assert row['spread_edge'] is None and row['spread_line_move'] is None and row['spread_own_view'] == 2.0
    assert row['ml_edge'] is None and row['ml_line_move'] is None and row['ml_own_view'] is not None
    assert row['ml_prob_raw'] is None and row['curr_price_over_odds'] is None and row['curr_book'] is None
    assert row['total_edge'] == 0.06 and row['injury_cutoff_et'] is None and row['injury_report_file'] is None
    from core.prediction_logger import BET_LOG_NAMES
    assert set(row) == set(BET_LOG_NAMES)


def test_bet_log_row_carries_the_suppression_reason_and_keeps_the_edge():
    pred = {'ml_prob': 0.62, 'ml_prob_raw': 0.60, 'spread_pred': 5.0, 'total_pred': 225.0, 'spread_std': 13.0, 'total_std': 18.0}
    bets = {'ml': (None, 0.03), 'spread': (None, 0.07), 'total': (None, 0.09)}                # sides withheld, edges as computed
    row = bet_log_row(_logged_game(), pred, bets, 't', 'h', None, None, bet_suppressed_reason='injury_report_stale')
    assert row['bet_suppressed_reason'] == 'injury_report_stale'
    assert (row['ml_bet'], row['spread_bet'], row['total_bet']) == (None, None, None)
    assert (row['ml_edge'], row['spread_edge'], row['total_edge']) == (0.03, 0.07, 0.09)
    assert bet_log_row(_logged_game(), pred, bets, 't')['bet_suppressed_reason'] is None          # default: a normal day
    from core.prediction_logger import BET_LOG_NAMES
    assert set(row) == set(BET_LOG_NAMES)
