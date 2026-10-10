
import pandas as pd
import numpy as np
import logging
from models.bpm_builder import BPMBuilder

class PlayerFeatureBuilder:
    def __init__(self, db_path):
        self.db_path = db_path
        self.bpm_builder = BPMBuilder(db_path)
        
    def build_roster_features(self, seasons=None):
        """
        Builds 'roster_bpm' for every game using PRIOR information only.
        """
        logging.info("Building Player Features (BPM Aggregation - No Leakage)...")
        
        # 1. Calculate BPM for all players/games
        self.bpm_builder.load_data(seasons)
        df_bpm = self.bpm_builder.calculate_rolling_bpm()
        
        # 2. Get prior game's roster and minutes for each team
        # This serves as our "expected roster" for the current game
        
        df_bpm = df_bpm.sort_values(['teamId', 'personId', 'gameDate'])
        
        # For each player-team, get their BPM and minutes from PRIOR games
        # (The BPM should already be shifted if BPMBuilder is fixed)
        
        # We need to aggregate at team level using PRIOR game's roster
        # Approach: For each game, use the roster from that team's previous game
        
        logging.info("Building prior-game roster proxy...")
        
        # Step 1: Get each team's games in order
        team_games = df_bpm.groupby('teamId')[['gameId', 'gameDate']].first().reset_index()
        # Actually, we need all games per team
        
        # Better approach: shift the entire aggregation
        # First, aggregate current game roster (this is "leaky" version)
        # Then shift by team to get "prior roster BPM"
        
        df_bpm['weighted_val'] = df_bpm['bpm'] * df_bpm['e_minutes']
        
        # Aggregate per game-team (this still uses actual roster, but we'll shift it)
        grouped = df_bpm.groupby(['gameId', 'teamId', 'gameDate']).agg({
            'weighted_val': 'sum',
            'e_minutes': 'sum'
        }).reset_index()
        
        grouped['roster_bpm_current'] = grouped['weighted_val'] / grouped['e_minutes'].replace(0, 1)
        
        # Sort by team and date
        grouped = grouped.sort_values(['teamId', 'gameDate'])
        
        # Shift to get PRIOR game's roster BPM as proxy for current game
        grouped['roster_bpm'] = grouped.groupby('teamId')['roster_bpm_current'].shift(1)
        
        # For first game of season, use league average or team's prior season ending value
        # Simple fallback: fill with 0 (league average BPM)
        grouped['roster_bpm'] = grouped['roster_bpm'].fillna(0)
        
        # Alternative: Use EWMA of roster BPM for more stability
        grouped['roster_bpm_ewma'] = grouped.groupby('teamId')['roster_bpm_current'].transform(
            lambda x: x.shift(1).ewm(span=10, adjust=False).mean()
        ).fillna(0)
        
        features = grouped[['gameId', 'teamId', 'roster_bpm', 'roster_bpm_ewma']]
        
        logging.info(f"Aggregated {len(features)} team-games (leak-free).")
        self._grouped = grouped
        
        return features

    def next_game_roster_features(self, seasons=None):
        """
        roster_bpm / roster_bpm_ewma for each team's NEXT (unplayed) game, computed exactly as
        build_roster_features computes them for a historical game: the previous game's roster BPM
        and the EWMA (span 10) of roster BPM through the previous game. Use this for live
        inference so features match training. Returns a DataFrame indexed by teamId.
        """
        self.build_roster_features(seasons)
        g = self._grouped.sort_values(['teamId', 'gameDate'])
        out = g.groupby('teamId').agg(roster_bpm=('roster_bpm_current', 'last'))
        out['roster_bpm_ewma'] = g.groupby('teamId')['roster_bpm_current'].apply(
            lambda x: x.ewm(span=10, adjust=False).mean().iloc[-1])
        return out

if __name__ == "__main__":
    pfb = PlayerFeatureBuilder(r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db')
    df = pfb.build_roster_features(seasons=['2025-26'])
    print(df.head())
    print(df['roster_bpm'].describe())
