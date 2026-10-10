
import pandas as pd
import numpy as np
import sqlite3
import logging

class BPMBuilder:
    """
    Calculates Box Plus/Minus (BPM) 2.0 using official coefficients.
    Source: https://www.basketball-reference.com/about/bpm2.html
    """
    def __init__(self, db_path):
        self.db_path = db_path
        
        # Coefficients from User Prompt
        self.coeffs_pos = {
            'pts': 0.860, # Constant
            '3pm': 0.389, # Constant
            'ast': (0.580, 1.034), # (PG, C)
            'tov': -0.964, # Constant
            'orb': (0.613, 0.181),
            'drb': (0.116, 0.181),
            'stl': (1.369, 1.008),
            'blk': (1.327, 0.703),
            'pf': -0.367 # Constant
        }
        
        self.coeffs_role = {
            'fga': (-0.560, -0.780), # (Creator, Receiver)
            'fta': (-0.246, -0.343)
        }
        
    def load_data(self, seasons=None):
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM player_basic_stats"
            if seasons:
                query += f" WHERE seasonYear IN ({','.join(repr(s) for s in seasons)})"
            self.df_stats = pd.read_sql(query, conn)
            
            # Load Positions
            self.df_info = pd.read_sql("SELECT PERSON_ID as personId, POSITION FROM player_info", conn)
            
            # Load Team Advanced Stats (For BPM Adjustment)
            # JOIN with basic stats to get seasonYear
            query_team = """
                SELECT adv.teamId, adv.gameId, bst.seasonYear, adv.netRating 
                FROM team_advanced_stats_v3 adv
                JOIN team_basic_stats bst ON adv.gameId = bst.gameId AND adv.teamId = bst.teamId
            """
            if seasons:
                # filter by seasonYear from basic stats
                query_team += f" WHERE bst.seasonYear IN ({','.join(repr(s) for s in seasons)})"
            self.df_team_adv = pd.read_sql(query_team, conn)
            
        self.df = self.df_stats.merge(self.df_info, on='personId', how='left')

    def _map_position(self, pos_str):
        """Maps Position String to 1.0 (PG) - 5.0 (C)"""
        if not isinstance(pos_str, str): return 3.0 # Default SF
        
        # Handle multi-position (take avg? or primary?)
        # "Guard-Forward" -> ?
        p = pos_str.upper()
        # (These checks used to compare mixed-case words against the upper-cased string, so every
        # player got the 3.0 default.)
        if 'GUARD' in p and 'FORWARD' in p: return 2.5 # GF / FG
        if 'FORWARD' in p and 'CENTER' in p: return 4.5 # FC / CF
        if 'GUARD' in p: return 1.5 # G (Generic)
        if 'FORWARD' in p: return 3.5 # F (Generic)
        if 'CENTER' in p: return 5.0
        
        # Specifics
        mapping = {
            'PG': 1.0, 'SG': 2.0, 'SF': 3.0, 'PF': 4.0, 'C': 5.0,
            'G': 1.5, 'F': 3.5
        }
        return mapping.get(p, 3.0)

    def calculate_rolling_bpm(self):
        logging.info("Calculating Rolling BPM (Strict 2.0)...")
        
        # Ensure clean state (idempotency)
        if hasattr(self, 'df_stats') and hasattr(self, 'df_info'):
             self.df = self.df_stats.merge(self.df_info, on='personId', how='left')
        
        self.df['gameDate'] = pd.to_datetime(self.df['gameDate'])
        self.df = self.df.sort_values(['personId', 'gameDate'])
        
        stat_cols = ['minutes', 'fieldGoalsMade', 'fieldGoalsAttempted', 
                    'threePointersMade', 'threePointersAttempted', 'freeThrowsMade', 'freeThrowsAttempted', 
                    'offensiveRebounds', 'defensiveRebounds', 'assists', 'steals', 'blocks', 'turnovers', 'personalFouls', 'points']
        
        self.df[stat_cols] = self.df[stat_cols].fillna(0)
        
        # 1. Rolling EWMA Stats (this part is fine - ewm looks backward only)
        ewma_stats = self.df.groupby('personId')[stat_cols].transform(
            lambda x: x.shift(1).ewm(span=40, adjust=False).mean()  # ADD shift(1) to exclude current game
        )
        ewma_stats.columns = [f"e_{c}" for c in stat_cols]
        self.df = pd.concat([self.df, ewma_stats], axis=1)

        # 2. Per 100 Possessions (same as before)
        self.df['est_poss'] = self.df['e_minutes'] * (100/48)
        poss_divisor = self.df['est_poss'].replace(0, 1)
        
        p100 = {}
        for c in stat_cols:
            p100[c] = (self.df[f"e_{c}"] / poss_divisor) * 100
            

        # =====================================================
        # NEW: Reliability weight based on cumulative minutes
        # =====================================================
        # Players with few minutes should regress toward 0 (league average)
        MIN_MINUTES_FULL_WEIGHT = 200  # ~8-10 games of starter minutes
        
        # Calculate cumulative minutes (shifted - only prior games)
        self.df['cumulative_minutes'] = self.df.groupby('personId')['minutes'].transform(
            lambda x: x.shift(1).cumsum()
        ).fillna(0)
        
        self.df['reliability'] = (self.df['cumulative_minutes'] / MIN_MINUTES_FULL_WEIGHT).clip(0, 1)
        
        logging.info(f"Reliability: min={self.df['reliability'].min():.3f}, "
                    f"median={self.df['reliability'].median():.3f}, "
                    f"max={self.df['reliability'].max():.3f}")
            
        # 3. Position & Role Scalars
        self.df['pos_scalar'] = self.df['POSITION'].apply(self._map_position)
        
        ast_p100 = p100['assists']
        self.df['role_scalar'] = 5.0 - (0.4 * ast_p100)
        self.df['role_scalar'] = self.df['role_scalar'].clip(1.0, 5.0)



        # 3. Position & Role Scalars (same as before)
        self.df['pos_scalar'] = self.df['POSITION'].apply(self._map_position)
        
        ast_p100 = p100['assists']
        self.df['role_scalar'] = 5.0 - (0.4 * ast_p100)
        self.df['role_scalar'] = self.df['role_scalar'].clip(1.0, 5.0)
        
        # 4. Calculate Raw BPM Terms (same as before)
        raw_bpm = 0.0
        
        def get_coeff(c_tuple, scalar):
            slope = (c_tuple[1] - c_tuple[0]) / 4.0
            return c_tuple[0] + slope * (scalar - 1.0)
            
        raw_bpm += p100['points'] * self.coeffs_pos['pts']
        raw_bpm += p100['threePointersMade'] * self.coeffs_pos['3pm']
        c_ast = self.df['pos_scalar'].apply(lambda x: get_coeff(self.coeffs_pos['ast'], x))
        raw_bpm += p100['assists'] * c_ast
        raw_bpm += p100['turnovers'] * self.coeffs_pos['tov']
        c_orb = self.df['pos_scalar'].apply(lambda x: get_coeff(self.coeffs_pos['orb'], x))
        raw_bpm += p100['offensiveRebounds'] * c_orb
        c_drb = self.df['pos_scalar'].apply(lambda x: get_coeff(self.coeffs_pos['drb'], x))
        raw_bpm += p100['defensiveRebounds'] * c_drb
        c_stl = self.df['pos_scalar'].apply(lambda x: get_coeff(self.coeffs_pos['stl'], x))
        raw_bpm += p100['steals'] * c_stl
        c_blk = self.df['pos_scalar'].apply(lambda x: get_coeff(self.coeffs_pos['blk'], x))
        raw_bpm += p100['blocks'] * c_blk
        raw_bpm += p100['personalFouls'] * self.coeffs_pos['pf']
        
        c_fga = self.df['role_scalar'].apply(lambda x: get_coeff(self.coeffs_role['fga'], x))
        raw_bpm += p100['fieldGoalsAttempted'] * c_fga
        
        c_fta = self.df['role_scalar'].apply(lambda x: get_coeff(self.coeffs_role['fta'], x))
        raw_bpm += (p100['freeThrowsAttempted'] * 0.44) * c_fta 
        
        # =====================================================
        # Apply reliability weighting (regress low-sample to 0)
        # =====================================================
        self.df['raw_bpm_unweighted'] = raw_bpm
        self.df['raw_bpm'] = raw_bpm * self.df['reliability']
    
        logging.info(f"Raw BPM (unweighted): {self.df['raw_bpm_unweighted'].min():.1f} to {self.df['raw_bpm_unweighted'].max():.1f}")
        logging.info(f"Raw BPM (weighted):   {self.df['raw_bpm'].min():.1f} to {self.df['raw_bpm'].max():.1f}")
    

        # Center on the league mean of raw BPM over player-games in the 365 days BEFORE this date.
        # Point-in-time (a full-sample mean used future games), and the trailing window makes the
        # value independent of where the loaded data starts once a year of history is loaded,
        # so training and live inference agree even if they load different season ranges.
        daily = self.df.groupby('gameDate')['raw_bpm'].agg(['sum', 'count']).sort_index()
        trailing = daily.rolling('365D', closed='left').sum()
        prior_mean = (trailing['sum'] / trailing['count']).fillna(0.0)
        self.df['bpm'] = self.df['raw_bpm'] - self.df['gameDate'].map(prior_mean)

        # # =====================================================
        # # 5. FIXED Team Adjustment - ROLLING, NOT FULL SEASON
        # # =====================================================
        # logging.info("Calculating Rolling Team Adjustment (no leakage)...")
        
        # # First, we need rolling team net rating (not season average)
        # # Sort team stats by date
        # self.df_team_adv = self.df_team_adv.sort_values(['teamId', 'seasonYear'])
        
        # # Merge gameDate into team_adv for proper sorting
        # # We need game-level team stats with dates
        # # Assuming df_team_adv has one row per game per team
        
        # # Calculate rolling team net rating (EWMA, shifted to exclude current game)
        # self.df_team_adv['rolling_net_rating'] = (
        #     self.df_team_adv.groupby('teamId')['netRating']
        #     .transform(lambda x: x.shift(1).ewm(span=40, adjust=False).mean())
        # )
        
        # # Merge rolling net rating back (need gameId for proper join)
        # # If df_team_adv doesn't have gameId, we need to add it
        # # For now, assume we can merge on teamId + gameId or similar
        
        # # Alternative approach: Calculate team adjustment at the player-game level
        # # using only prior games
        
        # self.df = self.df.sort_values(['teamId', 'gameDate'])
        
        # # Calculate rolling team raw BPM (minutes-weighted, from prior games only)
        # def rolling_team_raw_bpm(group):
        #     """Calculate rolling minutes-weighted team raw BPM."""
        #     group = group.sort_values('gameDate')
            
        #     # For each game, calculate team raw BPM from PRIOR games only
        #     results = []
        #     team_history = []
            
        #     for idx, row in group.iterrows():
        #         if len(team_history) == 0:
        #             # No history yet - use 0 or league average
        #             results.append(0.0)
        #         else:
        #             history_df = pd.DataFrame(team_history)
        #             total_min = history_df['e_minutes'].sum()
        #             if total_min > 0:
        #                 weighted_bpm = (history_df['raw_bpm'] * history_df['e_minutes']).sum() / total_min * 5.0
        #             else:
        #                 weighted_bpm = 0.0
        #             results.append(weighted_bpm)
                
        #         # Add current game to history (for next iteration)
        #         team_history.append({'raw_bpm': row['raw_bpm'], 'e_minutes': row['e_minutes']})
                
        #         # Optional: limit history window (e.g., last 40 games worth of player-games)
        #         # This prevents very old data from dominating
        #         if len(team_history) > 500:  # Roughly ~40 games * ~12 players
        #             team_history = team_history[-500:]
            
        #     return pd.Series(results, index=group.index)
        
        # logging.info("Computing rolling team raw BPM (this may take a minute)...")
        # self.df['team_raw_rolling'] = self.df.groupby('teamId').apply(
        #     rolling_team_raw_bpm, include_groups=False
        # ).reset_index(level=0, drop=True)
        
        # # Similarly, get rolling team net rating
        # # Merge team-level rolling net rating
        # if 'rolling_net_rating' in self.df_team_adv.columns:
        #     # Need to merge carefully - assuming df_team_adv has gameId
        #     team_rating_lookup = self.df_team_adv[['teamId', 'gameId', 'rolling_net_rating']].drop_duplicates()
        #     self.df = self.df.merge(team_rating_lookup, on=['teamId', 'gameId'], how='left')
        # else:
        #     # Fallback: calculate inline
        #     self.df['rolling_net_rating'] = (
        #         self.df.groupby('teamId')['netRating']
        #         .transform(lambda x: x.shift(1).ewm(span=40, adjust=False).mean())
        #         if 'netRating' in self.df.columns else 0
        #     )
        
        # # Calculate adjustment constant (using rolling values)
        # self.df['rolling_net_rating'] = self.df['rolling_net_rating'].fillna(0)
        # self.df['team_raw_rolling'] = self.df['team_raw_rolling'].fillna(0)
        
        # self.df['team_adjustment_constant'] = (self.df['rolling_net_rating'] - self.df['team_raw_rolling']) / 5.0
        
        # # Final BPM
        # self.df['bpm'] = self.df['raw_bpm'] + self.df['team_adjustment_constant']
        
        # Handle any remaining NaNs (early season games with no history)
        self.df['bpm'] = self.df['bpm'].fillna(self.df['raw_bpm'])
        
        return self.df[['gameId', 'gameDate', 'personId', 'playerName', 'teamId', 'bpm', 'e_minutes']]


    def save_player_bpm_to_db(self):
        """
        Calculate BPM for all players/games and save to database.
        Run this after updating player_basic_stats.
        """
        logging.info("Calculating and saving player BPM to database...")
        
        # Calculate BPM (this uses EWMA with shift(1) - no leakage)
        bpm_df = self.calculate_rolling_bpm()
        
        if bpm_df.empty:
            logging.warning("No BPM data calculated.")
            return
        
        # Select columns to save
        save_cols = ['gameId', 'gameDate', 'personId', 'playerName', 'teamId', 'bpm', 'e_minutes']
        if 'reliability' in bpm_df.columns:
            save_cols.append('reliability')
        
        bpm_df = bpm_df[save_cols]
        
        logging.info(f"Saving {len(bpm_df)} player-game BPM records...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Create table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS player_bpm (
                    gameId TEXT,
                    gameDate TEXT,
                    personId INTEGER,
                    playerName TEXT,
                    teamId INTEGER,
                    bpm REAL,
                    e_minutes REAL,
                    reliability REAL,
                    PRIMARY KEY (gameId, personId)
                )
            """)
            
            # Clear existing data (or you could do incremental updates)
            conn.execute("DELETE FROM player_bpm")
            
            # Insert new data
            bpm_df.to_sql('player_bpm', conn, if_exists='append', index=False)
            
            # Create index for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_player_bpm_team_date ON player_bpm(teamId, gameDate)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_player_bpm_person ON player_bpm(personId, gameDate)")
        
        logging.info("player_bpm table saved successfully.")



if __name__ == "__main__":
    db_path = r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db'
    b = BPMBuilder(db_path)
    b.load_data(['2023-24', '2024-25', '2025-26'])
    df = b.calculate_rolling_bpm()
    print(df.sort_values('bpm', ascending=False).head(20))

    # Calculate and save
    b.save_player_bpm_to_db()


        # Show top players (most recent game's BPM)
    with sqlite3.connect(db_path) as conn:
        top_players = pd.read_sql("""
            SELECT playerName, teamId, bpm, e_minutes, reliability, gameDate
            FROM player_bpm
            WHERE gameDate = (SELECT MAX(gameDate) FROM player_bpm)
            ORDER BY bpm DESC
            LIMIT 20
        """, conn)
    
    print("\n" + "="*60)
    print("TOP PLAYERS BY BPM (Most Recent Game)")
    print("="*60)
    print(top_players.to_string(index=False))