
import numpy as np
import pandas as pd

def calculate_implied_prob(american_odds):
    """
    Calculates implied probability from American odds.
    Negative odds: -odds / (-odds + 100)
    Positive odds: 100 / (odds + 100)
    """
    if pd.isna(american_odds):
        return np.nan
    
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)

def calculate_payout(american_odds, stake=100):
    """Calculates profit from American odds."""
    if pd.isna(american_odds):
        return np.nan
        
    if american_odds > 0:
        return stake * (american_odds / 100)
    else:
        return stake * (100 / -american_odds)

def calculate_bet_ev(model_prob, american_odds):
    """
    Calculate expected value of a $1 bet.
    EV = P(win) * payout - P(lose) * stake
    Stake is assumed to be 1.
    """
    if pd.isna(model_prob) or pd.isna(american_odds):
        return np.nan
        
    if american_odds > 0:
        payout = american_odds / 100
    else:
        payout = 100 / abs(american_odds)
    
    # EV = P(win) * payout - P(lose) * stake
    # stake = 1
    return model_prob * payout - (1 - model_prob)  # * 1

def get_ml_bet(row, threshold=0.05):
    """
    Determine ML bet based on edge over market.
    Requires: 'ml_prob', 'odds_home_ml' (or market_prob pre-calculated)
    
    The user snippet uses row['market_prob']. 
    We will try to use that if present, otherwise calculate it from odds.
    
    Returns: ('home', edge) or ('away', edge) or (None, edge)
    """
    
    # Get model probability for home win
    model_prob = row.get('ml_prob')
    if pd.isna(model_prob):
        return None, 0.0
        
    # Get market probability (implied from odds)
    market_prob = row.get('market_prob')
    if pd.isna(market_prob):
        # Try to calculate from odds_home_ml
        # We need the "Fair" market prob, but usually we just use the vig-inclusive implied odds
        # or we remove vig. 
        # The user's query implies comparing against "market_prob". 
        # Let's assume we use the implied prob from the Home Odds.
        odds = row.get('odds_home_ml')
        if pd.notna(odds):
            market_prob = calculate_implied_prob(odds)
        else:
            return None, 0.0

    # Calculate Edge
    # Edge = Model_Home_Prob - Market_Home_Prob
    edge = model_prob - market_prob
    
    if edge > threshold:
        return 'home', edge
    elif edge < -threshold:
        return 'away', abs(edge) # Edge for away is (1-model) - (1-market) = -edge?
        # Proof:
        # Home Edge = P_h - M_h
        # Away Edge = P_a - M_a = (1 - P_h) - (1 - M_h) = -P_h + M_h = -(P_h - M_h) = -Home Edge
        # So if Home Edge is -0.10 (favored away), Away Edge is +0.10.
    else:
        return None, edge
