import numpy as np
import pandas as pd

from core.odds_pick import pick_book_odds
from team.game_features_v2 import GameFeatureBuilder


def book(hs, aml, hml, total):
    return {'home_spread': hs, 'away_spread': -hs if hs is not None else None, 'home_ml': hml, 'away_ml': aml, 'total': total}


def test_priority_book_wins_when_it_has_every_field():
    out = pick_book_odds({'fanduel': book(-4, 140, -160, 222), 'draftkings': book(-3.5, 135, -155, 221.5)})
    assert out['book'] == 'draftkings' and out['home_spread'] == -3.5 and out['home_ml'] == -155 and out['total'] == 221.5


def test_fields_are_never_mixed_across_books_when_one_book_has_all_of_them():
    out = pick_book_odds({'draftkings': book(-3.5, None, None, 221.5), 'fanduel': book(-4, 140, -160, 222)})
    assert out['book'] == 'fanduel' and (out['home_spread'], out['home_ml'], out['total']) == (-4, -160, 222)


def test_without_a_complete_book_use_the_first_with_spread_and_total_and_leave_the_moneyline_empty():
    out = pick_book_odds({'draftkings': book(-3.5, None, None, 221.5), 'fanduel': book(-4, None, -160, 222)})
    assert out['book'] == 'draftkings' and np.isnan(out['home_ml']) and out['home_spread'] == -3.5


def test_in_game_or_stale_quote_is_dropped_by_the_consensus_filter():
    out = pick_book_odds({'draftkings': book(-14.5, 900, -2500, 236.0), 'fanduel': book(-3.5, 140, -160, 221.5),
                          'betmgm': book(-3.5, 142, -165, 222.0)})
    assert out['book'] == 'fanduel'


def test_junk_values_are_cleaned_and_pickem_is_a_real_spread():
    junk = pick_book_odds({'draftkings': {'home_spread': 40, 'home_ml': 2147483647, 'away_ml': -100000, 'total': 500}})
    assert junk['book'] is None or np.isnan(junk['home_spread'])
    out = pick_book_odds({'draftkings': book(0.0, 110, -130, 220.0)})
    assert out['home_spread'] == 0.0 and out['away_spread'] == 0.0 or out['away_spread'] == -0.0
    assert pick_book_odds({})['book'] is None


def test_training_build_applies_the_same_rule_to_the_wide_opening_table():
    opening = pd.DataFrame({
        'home_spread_draftkings': [-3.5, -3.5], 'away_spread_draftkings': [3.5, 3.5], 'total_draftkings': [221.5, 221.5],
        'home_ml_draftkings': [np.nan, -155], 'away_ml_draftkings': [np.nan, 135],
        'home_spread_fanduel': [-4.0, -4.0], 'away_spread_fanduel': [4.0, 4.0], 'total_fanduel': [222.0, 222.0],
        'home_ml_fanduel': [-160, -160], 'away_ml_fanduel': [140, 140]}, index=[10, 11])
    out = GameFeatureBuilder.pick_opening_odds(opening)
    assert list(out.index) == [10, 11]
    assert out.loc[10, ['odds_home_spread', 'odds_total', 'odds_home_ml', 'odds_away_ml']].tolist() == [-4.0, 222.0, -160, 140]   # all FanDuel
    assert out.loc[11, ['odds_home_spread', 'odds_total', 'odds_home_ml', 'odds_away_ml']].tolist() == [-3.5, 221.5, -155, 135]  # all DraftKings
