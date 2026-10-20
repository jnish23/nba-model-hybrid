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
