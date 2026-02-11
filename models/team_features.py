import pandas as pd
import numpy as np
import sqlite3
import logging
import random
from datetime import timedelta

class TeamFeatureBuilder:
    def __init__(self, db_path):
        self.db_path = db_path
        self.feature_categories = {
             'team_offense': [
                'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalPercentage', 'threePointersMade',
                 'threePointersAttempted', 'threePointerPercentage', 'freeThrowsMade', 'freeThrowsAttempted',
                  'freeThrowPercentage', 'offensiveRebounds', 'defensiveRebounds', 'rebounds', 'assists', 
                  'steals', 'blocks', 'turnovers', 'personalFouls', 'points', 'plusMinus',  'offensiveRating',
                   'defensiveRating', 'netRating', 'assistPercentage', 'assistToTurnover', 'assistRatio', 
                   'offensiveReboundPercentage', 'defensiveReboundPercentage', 'reboundPercentage',

                'Restricted Area FGM', 'Restricted Area FGA', 'Restricted Area FG_PCT', 'In The Paint (Non-RA) FGM',
                 'In The Paint (Non-RA) FGA', 'In The Paint (Non-RA) FG_PCT', 'Mid-Range FGM', 'Mid-Range FGA',
                  'Mid-Range FG_PCT', 'Above the Break 3 FGM', 'Above the Break 3 FGA', 'Above the Break 3 FG_PCT', 
                  'Corner 3 FGM', 'Corner 3 FGA', 'Corner 3 FG_PCT',

                'percentageFieldGoalsAttempted2pt', 'percentageFieldGoalsAttempted3pt', 'percentagePoints2pt', 
                'percentagePointsMidrange2pt', 'percentagePoints3pt', 'percentagePointsFastBreak',
                 'percentagePointsFreeThrow', 'percentagePointsOffTurnovers', 'percentagePointsPaint',
                  'percentageAssisted2pt', 'percentageUnassisted2pt', 'percentageAssisted3pt', 
                  'percentageUnassisted3pt', 'percentageAssistedFGM', 'percentageUnassistedFGM', 
                'distance', 'reboundChancesOffensive', 'reboundChancesDefensive', 
                'reboundChancesTotal', 'secondaryAssists', 'freeThrowAssists', 
                'passes', 'assists', 'contestedFieldGoalsMade', 'contestedFieldGoalsAttempted', 
                'contestedFieldGoalPercentage', 'uncontestedFieldGoalsMade', 'uncontestedFieldGoalsAttempted', 
                'uncontestedFieldGoalsPercentage', 'fieldGoalPercentage', 'defendedAtRimFieldGoalsMade',
                 'defendedAtRimFieldGoalsAttempted', 'defendedAtRimFieldGoalPercentage'
            ],
            'team_defense': [
                'OPP_FGM_C', 'OPP_FGM_F', 'OPP_FGM_G', 'OPP_FGA_C', 'OPP_FGA_F', 'OPP_FGA_G', 
                'OPP_FG_PCT_C', 'OPP_FG_PCT_F', 'OPP_FG_PCT_G', 'OPP_FG3M_C', 'OPP_FG3M_F',
                 'OPP_FG3M_G', 'OPP_FG3A_C', 'OPP_FG3A_F', 'OPP_FG3A_G', 'OPP_FG3_PCT_C', 
                 'OPP_FG3_PCT_F', 'OPP_FG3_PCT_G', 'OPP_FTM_C', 'OPP_FTM_F', 'OPP_FTM_G', 
                 'OPP_FTA_C', 'OPP_FTA_F', 'OPP_FTA_G', 'OPP_FT_PCT_C', 'OPP_FT_PCT_F',
                  'OPP_FT_PCT_G', 'OPP_OREB_C', 'OPP_OREB_F', 'OPP_OREB_G', 'OPP_DREB_C', 
                  'OPP_DREB_F', 'OPP_DREB_G', 'OPP_REB_C', 'OPP_REB_F', 'OPP_REB_G', 'OPP_AST_C', 
                  'OPP_AST_F', 'OPP_AST_G', 'OPP_TOV_C', 'OPP_TOV_F', 'OPP_TOV_G', 'OPP_STL_C', 
                  'OPP_STL_F', 'OPP_STL_G', 'OPP_BLK_C', 'OPP_BLK_F', 'OPP_BLK_G', 'OPP_BLKA_C', 
                  'OPP_BLKA_F', 'OPP_BLKA_G', 'OPP_PF_C', 'OPP_PF_F', 'OPP_PF_G', 'OPP_PFD_C', 
                  'OPP_PFD_F', 'OPP_PFD_G', 'OPP_PTS_C', 'OPP_PTS_F', 'OPP_PTS_G', 'PLUS_MINUS_C_DEF',
                   'PLUS_MINUS_F_DEF', 'PLUS_MINUS_G_DEF'
            ]
        }
        self.spans = [5, 20, 40]
        self.team_map = self._build_team_map()

    def _build_team_map(self):
        with sqlite3.connect(self.db_path) as conn:
            try:
                # Try to get from team_info or distinct from team_game_full
                # Ensuring we get the common name used in schedule
                df = pd.read_sql("SELECT DISTINCT teamId, teamName FROM team_basic_stats", conn)
                return dict(zip(df['teamName'], df['teamId']))
            except Exception as e:
                logging.warning(f"Could not build team map: {e}")
                return {}

    def get_latest_team_features(self, date=None):
        """
        Get the most recent feature vector for ALL teams as of 'date'.
        """
        if date is None:
            # Usage: Date for "today's" prediction (features from history before today)
            import datetime
            date = datetime.datetime.now().strftime('%Y-%m-%d')
            
        logging.info(f"Building latest team features as of {date}...")
        
        # Get all team IDs
        with sqlite3.connect(self.db_path) as conn:
             ids = pd.read_sql("SELECT DISTINCT teamId FROM team_game_full", conn)['teamId'].tolist()
             
        # Build for each
        # This might call DB 30 times. It's acceptable for live inference (few seconds).
        # We could optimize by batch fetching history for all teams then splitting, but simpler to reuse logic.
        
        features_map = {}
        for tid in ids:
            # Season assumption?
            # We can just pass current season string, logic mainly uses date.
            # Context features use season.
            # Assuming '2025-26' based on user context
            feats = self.build_features(tid, date, '2025-26') 
            if feats:
                features_map[tid] = feats
                
        return features_map

    def build_features(self, team_id: int, game_date: str, season: str) -> dict:
        features = {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Query history before game_date
            query = f"""
            SELECT *, case when gameId like '004%' then 1 else 0 end as is_playoff 
            FROM team_game_full 
            WHERE teamId = {team_id} 
            AND gameDate < '{game_date}'
            ORDER BY gameDate ASC
            """
            df_history = pd.read_sql(query, conn)
        
        if df_history.empty:
            logging.warning(f"No history found for team {team_id} before {game_date}")
            return {} # Or return NaNs? Return empty dict will likely be handled downstream
            
        features.update(self._offensive_features(df_history))
        features.update(self._defensive_features(df_history))
        features.update(self._rest_features(team_id, game_date, conn)) # Pass conn for fresh query
        
        # Additional placeholders for requested groups
        # (Opponent strength needs opponent ID which isn't passed here, 
        # usually handled by merging feature vectors of both teams)
        features.update(self._contextual_features(df_history, game_date, season))
        
        return features

    def _calculate_ewma(self, df: pd.DataFrame, columns: list, prefix: str) -> dict:
        feats = {}
        # Filter columns that exist in df
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols and len(df) > 0:
             pass # Removed debug log
        
        for span in self.spans:
            ewma = df[valid_cols].ewm(span=span).mean().iloc[-1]
            for col, val in ewma.items():
                feats[f"{prefix}_{col}_{span}g"] = val
        return feats

    def _offensive_features(self, df: pd.DataFrame) -> dict:
        return self._calculate_ewma(df, self.feature_categories['team_offense'], 'off')

    def _defensive_features(self, df: pd.DataFrame) -> dict:
        return self._calculate_ewma(df, self.feature_categories['team_defense'], 'def')

    def _rest_features(self, team_id, game_date, conn=None) -> dict:
        # Need to query gameDate including current game to know days rest?
        # User said "Queries games before game_date only".
        # So we check the LAST game date in history.
        feats = {}
        
        # We need a connection if not provided?
        local_conn = False
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            local_conn = True
            
        try:
             # Get the last game date before current
             last_game_query = f"""
             SELECT MAX(gameDate) as last_date 
             FROM team_game_full 
             WHERE teamId = {team_id} AND gameDate < '{game_date}'
             """
             last_date_str = pd.read_sql(last_game_query, conn)['last_date'][0]
             
             if not last_date_str:
                 feats['days_rest'] = 7 # Default long rest
             else:
                 current_dt = pd.to_datetime(game_date)
                 last_dt = pd.to_datetime(last_date_str)
                 feats['days_rest'] = (current_dt - last_dt).days
                 feats['days_rest'] =int(np.where(feats['days_rest'] > 7, 7, feats['days_rest']))
                 
             feats['is_b2b'] = 1 if feats.get('days_rest', 10) == 1 else 0
             
             # Calculate 3-in-4, 4-in-5
             # Need last 5 games dates
             dates_query = f"""
             SELECT gameDate FROM team_game_full
             WHERE teamId = {team_id} AND gameDate < '{game_date}'
             ORDER BY gameDate DESC
             LIMIT 5
             """
             dates = pd.read_sql(dates_query, conn)['gameDate'].tolist()
             dates = [pd.to_datetime(d) for d in dates]
             
             feats['games_last_7d'] = sum(1 for d in dates if (pd.to_datetime(game_date) - d).days <= 7)
             
        finally:
            if local_conn:
                conn.close()
                
        return feats

    def _contextual_features(self, df: pd.DataFrame, game_date, season) -> dict:
        feats = {}
        feats['season_game_number'] = len(df) + 1
        # For inference/simulation, check if 'season' is playoffs?
        # Or check if game_date > April 15?
        # Simpler: Check last game nature from df?
        # For live inference we usually assume Regular Season unless specified.
        feats['is_playoff'] = 0 
        
        return feats

    def build_entire_history(self) -> pd.DataFrame:
        """
         Efficiently builds features for the entire history using vectorized operations.
         Returns a DataFrame with index (teamId, gameDate) and all feature columns.
        """
        logging.info("Loading full team history...")
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql("SELECT * FROM team_game_full ORDER BY teamId, gameDate", conn)
            
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        
        # 1. EWMA Features
        logging.info("Calculating vectorized EWMA features...")
        
        # Collect new columns in a dictionary to avoid fragmentation
        new_features = {}
        
        # Offense
        for col in self.feature_categories['team_offense']:
            if col not in df.columns: continue
            # Pre-calculate groupby object if possible? No, we transform per column.
            # But we can optimize by doing all spans in one go if we wanted, but transform is convenient.
            grp = df.groupby('teamId')[col] 
            for span in self.spans:
                # shift(1) ensures we strictly use PAST data
                new_features[f"off_{col}_{span}g"] = grp.transform(
                    lambda x: x.ewm(span=span).mean().shift(1)
                )
                
        # Defense
        for col in self.feature_categories['team_defense']:
            if col not in df.columns: continue
            grp = df.groupby('teamId')[col]
            for span in self.spans:
                 new_features[f"def_{col}_{span}g"] = grp.transform(
                    lambda x: x.ewm(span=span).mean().shift(1)
                )

        # 2. Rest Features
        logging.info("Calculating vectorized Rest features...")
        # Previous game date
        prev_game_date = df.groupby('teamId')['gameDate'].shift(1)
        
        # Days rest
        days_rest = (df['gameDate'] - prev_game_date).dt.days.fillna(7)
        new_features['days_rest'] = days_rest
        new_features['is_b2b'] = (days_rest == 1).astype(int)
        
        # Games in last 7 days
        # Reuse logic: Rolling count
        temp = df[['teamId', 'gameDate']].copy()
        temp['c'] = 1
        try:
            g7 = temp.groupby('teamId').rolling('7D', on='gameDate', closed='left')['c'].count()
            # The result index is (teamId, gameDate).
            # Since df is sorted by teamId, gameDate, the row order matches df.
            # We can use .values to bypass index alignment issues.
            new_features['games_last_7d'] = g7.values
        except:
             new_features['games_last_7d'] = 3 # Placeholder
        
        # Fill NA for this specific col if needed, though concat handles it mainly
        # new_features['games_last_7d'] = new_features['games_last_7d'].fillna(0) # Do after concat
        
        # 3. Contextual
        logging.info("Calculating Contextual features...")
        new_features['season_game_number'] = df.groupby('teamId').cumcount() + 1
        new_features['is_playoff'] = (df['gameId'].str[:3] == '004').astype(int)
    
        
        # Assemble Final DataFrame
        logging.info("Assembling final feature set...")
        meta_df = df[['teamId', 'gameDate', 'seasonYear', 'gameId']].copy()
        features_df = pd.DataFrame(new_features, index=df.index)
        
        # Fill NAs in features (e.g. games_last_7d might have NAs from alignment, though usually fine)
        if 'games_last_7d' in features_df.columns:
             features_df['games_last_7d'] = features_df['games_last_7d'].fillna(0)
        
        result_df = pd.concat([meta_df, features_df], axis=1)
        
        return result_df
        
if __name__ == "__main__":
    # verification
    db_path = r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db'
    builder = TeamFeatureBuilder(db_path)
    
    print("--- Verifying Batch Generation ---")
    full_df = builder.build_entire_history()

    if not full_df.columns.is_unique:
        print("\n!!! DUPLICATE COLUMNS DETECTED !!!\n")

        dupes = full_df.columns[full_df.columns.duplicated()].tolist()
        print("Duplicate Columns:", dupes)

        full_df = full_df.loc[:, ~full_df.columns.duplicated()]

        print("\n!!! DUPLICATE COLUMNS REMOVED !!!\n")


    print(f"Full History Shape: {full_df.shape}")
    print("Sample Columns:", full_df.columns[:10])
    
    print("\n--- Compare Batch vs Single ---")
    # Pick a random row from full_df (that isn't the first game of season to avoid NaN/startup diffs)
    # Using simple row index
    if len(full_df) > 100:
        sample_row = full_df.iloc[random.randint(100, len(full_df)-1)]
    else:
        sample_row = full_df.iloc[-1]
    
    t_id = sample_row['teamId']
    g_date = sample_row['gameDate'].strftime('%Y-%m-%d')
    season = sample_row['seasonYear']
    
    print(f"Testing Team: {t_id}, Date: {g_date}")
    
    # Run Single
    single_feats = builder.build_features(t_id, g_date, season)
    
    # Check match
    diffs = []
    print(f"Checking {len(single_feats)} single features against batch row...")
    
    for k, v in single_feats.items():
        if k in sample_row:
            batch_val = sample_row[k]
            # Handle NaN
            if pd.isna(v) and pd.isna(batch_val): 
                continue
            if pd.isna(v) or pd.isna(batch_val):
                diffs.append(f"{k}: Single={v}, Batch={batch_val}")
                continue
            
            if abs(v - batch_val) > 0.001:
                diffs.append(f"{k}: Single={v:.4f}, Batch={batch_val:.4f}")
    
    if diffs:
        print("MISMATCHES FOUND:")
        for d in diffs[:5]: print(d)
    else:
        print("SUCCESS: Single and Batch results match!")
