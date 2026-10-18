"""
Betting math shared by the live picks, the backtests and the dashboard.

Conventions
  * American odds in, profit per $100 stake out (calculate_payout); a loss is -100.
  * Every side is judged against ITS OWN price. The two moneyline prices of a game carry a vig
    (their implied probabilities sum to > 1), so the away side must never be priced as
    "1 - home implied": that hands the away bet the overround for free.
  * One bet-selection rule for all three markets: bet when the model's win probability exceeds the
    price's break-even probability by more than `threshold` (a probability edge, default 5%).
    For spreads and totals the win probability is Phi(point edge / residual std), so the rule is
    expressed in points through points_threshold() when a points view is wanted.
"""
import numpy as np
import pandas as pd
from statistics import NormalDist

# Standard normal from the standard library: the hosted dashboard imports this module with only its own light
# requirements (dashboard/requirements.txt), so this module must not depend on scipy.
_NORM = NormalDist()

STANDARD_PRICE = -110            # spread / total price when the real one is unknown
DEFAULT_PROB_EDGE = 0.05
# Out-of-sample residual std of the spread / total models (points); live and backtests read the
# value saved with each model set (calibration_meta*.json) and fall back to these.
DEFAULT_SPREAD_STD = 13.4
DEFAULT_TOTAL_STD = 18.2


def calculate_implied_prob(american_odds):
    """Break-even probability of an American price (vig included)."""
    if pd.isna(american_odds):
        return np.nan

    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)


def calculate_payout(american_odds, stake=100):
    """Profit on a winning bet at American odds."""
    if pd.isna(american_odds):
        return np.nan

    if american_odds > 0:
        return stake * (american_odds / 100)
    else:
        return stake * (100 / -american_odds)


def calculate_bet_ev(model_prob, american_odds):
    """
    Expected value of a $1 bet: P(win) * payout - P(lose) * stake.
    """
    if pd.isna(model_prob) or pd.isna(american_odds):
        return np.nan

    return model_prob * calculate_payout(american_odds, stake=1) - (1 - model_prob)


def settle_bet(american_odds, won, stake=100):
    """Profit of a settled bet: the payout at the bet's OWN odds if it won, -stake if it lost."""
    return calculate_payout(american_odds, stake) if won else -stake


# ----------------------------------------------------------------------------------------------
# Moneyline

def ml_side_edges(home_win_prob, home_ml_odds, away_ml_odds):
    """(home_edge, away_edge): model probability minus each side's own break-even probability."""
    home_edge = home_win_prob - calculate_implied_prob(home_ml_odds)
    away_edge = (1.0 - home_win_prob) - calculate_implied_prob(away_ml_odds)
    return home_edge, away_edge


def get_ml_bet(row, threshold=DEFAULT_PROB_EDGE):
    """
    Moneyline bet for one game from 'ml_prob' (model P(home wins)), 'odds_home_ml' and
    'odds_away_ml'. Each side is compared with its own price; the side with the larger edge is
    bet if that edge exceeds `threshold`.

    Returns ('home' | 'away' | None, edge of the best side). Without both prices there is no bet
    (the away price is never inferred from the home price).
    """
    model_prob = row.get('ml_prob')
    home_odds, away_odds = row.get('odds_home_ml'), row.get('odds_away_ml')
    if pd.isna(model_prob) or pd.isna(home_odds) or pd.isna(away_odds):
        return None, 0.0

    home_edge, away_edge = ml_side_edges(model_prob, home_odds, away_odds)
    if home_edge >= away_edge:
        side, edge = 'home', home_edge
    else:
        side, edge = 'away', away_edge
    return (side, edge) if edge > threshold else (None, edge)


# ----------------------------------------------------------------------------------------------
# Spread / total: probability edge from a point edge

def cover_probability(point_edge, std):
    """P(a bet with `point_edge` points of model edge wins), Normal residual with std `std`."""
    return _NORM.cdf(float(point_edge) / float(std))


def points_threshold(std, prob_edge=DEFAULT_PROB_EDGE, price=STANDARD_PRICE):
    """Point edge at which P(win) - break_even(price) == prob_edge."""
    return std * _NORM.inv_cdf(calculate_implied_prob(price) + prob_edge)


def implied_prob_edge(point_edge, std, price=STANDARD_PRICE):
    """The probability edge a given point edge represents (negative = losing bet even if the model is right)."""
    return cover_probability(point_edge, std) - calculate_implied_prob(price)


def get_spread_bet(spread_pred, home_line, std=DEFAULT_SPREAD_STD, threshold=DEFAULT_PROB_EDGE,
                   price=STANDARD_PRICE):
    """
    spread_pred: model's predicted home margin; home_line: the home spread (-3.5 = home favored).
    Home covers when margin + line > 0, so the model's point edge on home is spread_pred + home_line.
    Returns ('home' | 'away' | None, probability edge of the better side).
    """
    if pd.isna(spread_pred) or pd.isna(home_line):
        return None, 0.0
    point_edge = spread_pred + home_line
    edge = implied_prob_edge(abs(point_edge), std, price)
    if edge <= threshold:
        return None, edge
    return ('home' if point_edge > 0 else 'away'), edge


def get_total_bet(total_pred, total_line, std=DEFAULT_TOTAL_STD, threshold=DEFAULT_PROB_EDGE,
                  price=STANDARD_PRICE):
    """Returns ('over' | 'under' | None, probability edge of the better side)."""
    if pd.isna(total_pred) or pd.isna(total_line):
        return None, 0.0
    point_edge = total_pred - total_line
    edge = implied_prob_edge(abs(point_edge), std, price)
    if edge <= threshold:
        return None, edge
    return ('over' if point_edge > 0 else 'under'), edge
