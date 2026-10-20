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

def is_whole_line(line):
    """True for a whole-number line (-3, 221): the bet can push. Half-point lines (-3.5) cannot."""
    return line is not None and not pd.isna(line) and float(line) == round(float(line))


def cover_probability(point_edge, std, line=None):
    """
    P(a bet with `point_edge` points of model edge wins), Normal residual with std `std`.

    Margins and totals are whole numbers. On a half-point line the bet wins when the result lands past the half point,
    which is Phi(edge / std). On a WHOLE-number line (pass `line=`) a result exactly on the line is a push (stake returned):
    with continuity correction win = Phi((edge - 0.5)/std), loss = Phi((-edge - 0.5)/std), and the number to compare with the
    price's break-even is the win probability among bets that settle, win / (win + loss). Without `line` the half-point
    formula is used (the pre-audit behaviour).
    """
    e, sd = float(point_edge), float(std)
    if is_whole_line(line):
        win, loss = _NORM.cdf((e - 0.5) / sd), _NORM.cdf((-e - 0.5) / sd)
        return win / (win + loss)
    return _NORM.cdf(e / sd)


def cover_probabilities(point_edges, stds, lines=None):
    """cover_probability for arrays (backtests): same formula, vectorized with numpy only."""
    e = np.asarray(point_edges, float)
    sd = np.broadcast_to(np.asarray(stds, float), e.shape)
    cdf = np.vectorize(_NORM.cdf, otypes=[float])
    half = cdf(e / sd)
    if lines is None:
        return half
    ln = np.broadcast_to(np.asarray(lines, float), e.shape)
    whole = np.isfinite(ln) & (ln == np.round(ln))
    win, loss = cdf((e - 0.5) / sd), cdf((-e - 0.5) / sd)
    return np.where(whole, win / (win + loss), half)


def points_threshold(std, prob_edge=DEFAULT_PROB_EDGE, price=STANDARD_PRICE):
    """Point edge at which P(win) - break_even(price) == prob_edge."""
    return std * _NORM.inv_cdf(calculate_implied_prob(price) + prob_edge)


def implied_prob_edge(point_edge, std, price=STANDARD_PRICE, line=None):
    """The probability edge a given point edge represents (negative = losing bet even if the model is right)."""
    return cover_probability(point_edge, std, line=line) - calculate_implied_prob(price)


def _side_price(price, side_price):
    """The bet's own price when it is known, else the standard one."""
    return price if side_price is None or pd.isna(side_price) else side_price


def get_spread_bet(spread_pred, home_line, std=DEFAULT_SPREAD_STD, threshold=DEFAULT_PROB_EDGE,
                   price=STANDARD_PRICE, home_price=None, away_price=None):
    """
    spread_pred: model's predicted home margin; home_line: the home spread (-3.5 = home favored).
    Home covers when margin + line > 0, so the model's point edge on home is spread_pred + home_line.
    The side's own price (home_price / away_price, American odds) is used when given, else `price` (-110): at -115 or -120
    the break-even is 53.5% / 54.5% and a 5% edge needs more points. Whole-number lines can push (see cover_probability).
    Returns ('home' | 'away' | None, probability edge of the better side).
    """
    if pd.isna(spread_pred) or pd.isna(home_line):
        return None, 0.0
    point_edge = spread_pred + home_line
    side_price = _side_price(price, home_price if point_edge > 0 else away_price)
    edge = implied_prob_edge(abs(point_edge), std, side_price, line=home_line)
    if edge <= threshold:
        return None, edge
    return ('home' if point_edge > 0 else 'away'), edge


def get_total_bet(total_pred, total_line, std=DEFAULT_TOTAL_STD, threshold=DEFAULT_PROB_EDGE,
                  price=STANDARD_PRICE, over_price=None, under_price=None):
    """Returns ('over' | 'under' | None, probability edge of the better side). Prices and pushes as in get_spread_bet."""
    if pd.isna(total_pred) or pd.isna(total_line):
        return None, 0.0
    point_edge = total_pred - total_line
    side_price = _side_price(price, over_price if point_edge > 0 else under_price)
    edge = implied_prob_edge(abs(point_edge), std, side_price, line=total_line)
    if edge <= threshold:
        return None, edge
    return ('over' if point_edge > 0 else 'under'), edge
