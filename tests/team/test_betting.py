import numpy as np
import pandas as pd
import pytest

from team import betting as b


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
    from team.train_baseline import BaselineModel
    df = pd.DataFrame({'ml_prob': [0.8, 0.8, 0.2, 0.2], 'ml_actual': [1, 0, 0, 1],
                       'odds_home_ml': [-110, -110, 130, 130], 'odds_away_ml': [-110, -110, -150, -150]})
    BaselineModel('unused.csv').evaluate_vs_market(df)
    out = capsys.readouterr().out
    assert 'ROI at the actual moneylines' in out


# ---- item 11: one threshold definition for spreads / totals --------------------------------

def test_flat_one_point_rule_is_not_a_five_percent_edge():
    # residual stds of the deployed models (spread ~13.4, total ~18.2 points)
    assert b.implied_prob_edge(1.0, 13.4) == pytest.approx(0.0059, abs=2e-4)     # barely above break-even
    assert b.implied_prob_edge(1.0, 18.2) < 0                                    # a losing bet by the model's own odds
    assert b.points_threshold(13.4, 0.05) == pytest.approx(2.49, abs=0.02)
    assert b.points_threshold(18.2, 0.05) == pytest.approx(3.39, abs=0.02)


def test_points_threshold_is_the_inverse_of_implied_prob_edge():
    for std in (12.0, 13.4, 18.2):
        for edge in (0.0, 0.02, 0.05, 0.1):
            assert b.implied_prob_edge(b.points_threshold(std, edge), std) == pytest.approx(edge)


def test_spread_and_total_sides():
    # model: home by 6.5; line home -3.5 -> 3.0 points on home; at std 13.4 that is a 5.9% edge
    side, edge = b.get_spread_bet(6.5, -3.5, 13.4, 0.05)
    assert side == 'home' and edge == pytest.approx(b.implied_prob_edge(3.0, 13.4))
    assert b.get_spread_bet(6.5, -3.5, 13.4, 0.10)[0] is None
    assert b.get_spread_bet(-1.0, -3.5, 13.4, 0.05)[0] == 'away'       # model: home by -1 -> 4.5 on away
    assert b.get_total_bet(226.0, 221.5, 18.2, 0.05)[0] == 'over'
    assert b.get_total_bet(216.0, 221.5, 18.2, 0.05)[0] == 'under'
    assert b.get_total_bet(222.5, 221.5, 18.2, 0.05)[0] is None
    assert b.get_spread_bet(np.nan, -3.5) == (None, 0.0)


def test_backtest_and_live_pick_the_same_bets():
    """scripts/backtest_analysis.py and the live engine both route through team.betting."""
    import backtest_analysis as bt
    df = pd.DataFrame({'spread_pred': [6.5, -1.0, 1.0], 'odds_home_spread': [-3.5, -3.5, -3.5],
                       'home_score': [110, 100, 100], 'away_score': [100, 95, 95],
                       'total_pred': [226.0, 216.0, 222.5], 'odds_total': [221.5, 221.5, 221.5],
                       'seasonYear': ['2024-25'] * 3, 'spread_std': [13.4] * 3, 'total_std': [18.2] * 3})
    sp = bt.get_spread_bets(df, 0.05)
    assert [s.lower() for s in sp['side']] == [b.get_spread_bet(r.spread_pred, r.odds_home_spread, 13.4)[0]
                                              for r in df.itertuples()
                                              if b.get_spread_bet(r.spread_pred, r.odds_home_spread, 13.4)[0]]
    tl = bt.get_totals_bets(df, 0.05)
    assert list(tl['side'].str.lower()) == ['over', 'under']


def test_live_engine_uses_the_shared_rule():
    import inspect
    import run_inference_v2
    src = inspect.getsource(run_inference_v2.InferenceEngine.run)
    assert 'get_spread_bet' in src and 'get_total_bet' in src and 'get_ml_bet' in src
    assert 'raw_edge_over' not in src          # the old flat 5/7-point total rule is gone


def test_pushes_are_not_losses_in_the_backtest():
    import backtest_analysis as bt
    df = pd.DataFrame({'spread_pred': [10.0], 'odds_home_spread': [-3.0], 'home_score': [103], 'away_score': [100],
                       'total_pred': [200.0], 'odds_total': [200.0], 'seasonYear': ['2024-25']})
    bets = bt.get_spread_bets(df, 0.0)
    assert bool(bets.loc[0, 'push']) and bets.loc[0, 'profit'] == 0.0


def test_betting_module_does_not_need_scipy():
    # The hosted dashboard installs only dashboard/requirements.txt (no scipy); team.betting must import without it.
    import subprocess, sys
    code = ("import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'scipy' or name.startswith('scipy.'):\n"
            "            raise ImportError('scipy blocked')\n"
            "sys.meta_path.insert(0, Block())\n"
            "from team import betting as b\n"
            "assert abs(b.cover_probability(0.0, 13.4) - 0.5) < 1e-12\n"
            "assert 'scipy' not in sys.modules\n")
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=str(__import__('pathlib').Path(__file__).resolve().parents[2]))
    assert r.returncode == 0, r.stderr
