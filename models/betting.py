"""
Betting math shared by the live picks, the backtests and the dashboard.

Conventions
  * American odds in, profit per $100 stake out (calculate_payout); a loss is -100.
  * Every side is judged against ITS OWN price. The two moneyline prices of a game carry a vig
    (their implied probabilities sum to > 1), so the away side must never be priced as
    "1 - home implied": that hands the away bet the overround for free.
  * A moneyline bet is made when the model's win probability exceeds the side's break-even
    probability by more than `threshold` (a probability edge, default 5%).
"""
import numpy as np
import pandas as pd

DEFAULT_PROB_EDGE = 0.05


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
