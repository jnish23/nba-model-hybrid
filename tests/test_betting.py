import numpy as np
import pandas as pd
import pytest

from models import betting as b


def test_implied_prob_and_payout_known_values():
    assert b.calculate_implied_prob(-110) == pytest.approx(110 / 210)
    assert b.calculate_implied_prob(150) == pytest.approx(0.4)
    assert b.calculate_payout(-110) == pytest.approx(100 / 1.1)
    assert b.calculate_payout(150) == pytest.approx(150.0)
    assert b.calculate_payout(-200) == pytest.approx(50.0)
    assert np.isnan(b.calculate_payout(np.nan))


def test_ev_matches_payout():
    # fair coin at +100 -> EV 0; at -110 -> -2.27%
    assert b.calculate_bet_ev(0.5, 100) == pytest.approx(0.0)
    assert b.calculate_bet_ev(0.5, -110) == pytest.approx(0.5 * (100 / 110) - 0.5)


# A typical priced game: home -170 / away +150 -> implied 0.6296 + 0.4 = 1.0296 (2.96% overround)
GAME = {'odds_home_ml': -170, 'odds_away_ml': 150}


def test_away_bet_is_priced_against_its_own_moneyline_not_one_minus_home():
    """Old rule: away edge = (1-p) - (1 - home_implied) = home_implied - p; with p=0.40 that is 0.2296.
    Correct: (1-p) - away_implied = 0.60 - 0.40 = 0.20. The 0.0296 difference is the vig."""
    side, edge = b.get_ml_bet({'ml_prob': 0.40, **GAME}, threshold=0.05)
    assert side == 'away'
    assert edge == pytest.approx(0.20)
    old_style = (1 - 0.40) - (1 - b.calculate_implied_prob(-170))
    assert old_style - edge == pytest.approx(b.calculate_implied_prob(-170) + b.calculate_implied_prob(150) - 1)


def test_away_bet_that_only_clears_the_threshold_thanks_to_the_vig_is_rejected():
    # home implied 0.6296, away implied 0.40. p_home=0.575 -> away prob 0.425 -> true edge 0.025
    side, edge = b.get_ml_bet({'ml_prob': 0.575, **GAME}, threshold=0.05)
    assert side is None and edge == pytest.approx(0.025)
    # the old home-only rule would have shown an away edge of 0.6296 - 0.575 = 0.0546 > 5% and bet
    assert b.calculate_implied_prob(-170) - 0.575 > 0.05


def test_home_bet_uses_home_price():
    side, edge = b.get_ml_bet({'ml_prob': 0.72, **GAME}, threshold=0.05)
    assert side == 'home' and edge == pytest.approx(0.72 - 170 / 270)


def test_no_bet_without_both_prices_or_probability():
    assert b.get_ml_bet({'ml_prob': 0.7, 'odds_home_ml': -170}) == (None, 0.0)
    assert b.get_ml_bet({'ml_prob': np.nan, **GAME}) == (None, 0.0)
    assert b.get_ml_bet(pd.Series({'ml_prob': 0.7, 'odds_home_ml': np.nan, 'odds_away_ml': 150})) == (None, 0.0)


def test_both_sides_never_clear_the_threshold_together_when_prices_have_vig():
    for p in np.linspace(0.01, 0.99, 99):
        h, a = b.ml_side_edges(p, -170, 150)
        assert not (h > 0.0 and a > 0.0)


def test_ml_bets_settle_at_their_own_odds_not_minus_110():
    assert b.settle_bet(150, True) == pytest.approx(150.0)
    assert b.settle_bet(-250, True) == pytest.approx(40.0)
    assert b.settle_bet(-250, False) == -100
    # ten +200 winners and ten losers at the same price: +$1000 vs a flat -110 settlement of +$909.09
    profit = sum(b.settle_bet(200, w) for w in [True] * 10 + [False] * 10)
    flat = sum(b.settle_bet(-110, w) for w in [True] * 10 + [False] * 10)
    assert profit == pytest.approx(1000.0) and flat == pytest.approx(10 * 100 / 1.1 - 1000)


def test_evaluate_vs_market_reports_roi_at_actual_odds(capsys):
    from models.train_baseline import BaselineModel
    df = pd.DataFrame({'ml_prob': [0.8, 0.8, 0.2, 0.2], 'ml_actual': [1, 0, 0, 1],
                       'odds_home_ml': [-110, -110, 130, 130], 'odds_away_ml': [-110, -110, -150, -150]})
    BaselineModel('unused.csv').evaluate_vs_market(df)
    out = capsys.readouterr().out
    assert 'ROI at the actual moneylines' in out
