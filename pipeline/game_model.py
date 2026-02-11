import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

class GameModel:
    def __init__(self):
        pass

    def prepare_final_dataset(self, ridge_df, delta_df, games_df):
        """
        Merges Ridge Ratings + Roster Deltas + Context.
        """
        # 1. Prepare Games
        columns_to_keep = ['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ID_opp', 'PTS', 'IS_HOME', 'PACE']
        if 'SEASON' in games_df.columns:
            columns_to_keep.append('SEASON')
            
        # Handle case where PACE might not be in games_df (backwards compatibility)
        avail_cols = [c for c in columns_to_keep if c in games_df.columns]
        base_df = games_df[avail_cols].copy()
        
        base_df = base_df.rename(columns={
            'GAME_ID': 'gameId',
            'GAME_DATE': 'gameDate', 
            'TEAM_ID': 'teamId',
            'TEAM_ID_opp': 'opponentId',
            'PTS': 'points',
            'IS_HOME': 'isHome',
            'PACE': 'pace'
        })
        
        # Calculate Season End
        if 'season_end' not in base_df.columns:
            if 'SEASON' in games_df.columns:
                base_df['season_end'] = base_df['SEASON'].str[:4].astype(int) + 1
            else:
                 # Fallback?
                 base_df['season_end'] = base_df['gameDate'].dt.year # Approximate
        
        base_df = base_df.drop_duplicates(subset=['gameId', 'teamId'])
        base_df['gameId'] = base_df['gameId'].astype(int)
        base_df['teamId'] = base_df['teamId'].astype(int)
        
        # 2. Merge Ridge Ratings
        ridge_df = ridge_df.copy()
        ridge_df = ridge_df.drop_duplicates(subset=['gameDate', 'teamId'])
        ridge_df['teamId'] = ridge_df['teamId'].astype(int)
        
        df = pd.merge(base_df, ridge_df, on=['gameDate', 'teamId'], how='inner')
        
        ridge_opp = ridge_df.copy()
        # Rename ALL columns to opp_
        ridge_opp.columns = ['gameDate', 'opponentId'] + [f'opp_{c}' for c in ridge_opp.columns if c not in ['gameDate', 'teamId']]
        ridge_opp['opponentId'] = ridge_opp['opponentId'].astype(int)
        
        df = pd.merge(df, ridge_opp, on=['gameDate', 'opponentId'], how='inner')
        
        # 3. Merge Roster Deltas
        delta_df = delta_df.copy()
        delta_df = delta_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])
        delta_df['GAME_ID'] = delta_df['GAME_ID'].astype(int)
        delta_df['TEAM_ID'] = delta_df['TEAM_ID'].astype(int)
        
        # Team Delta
        df = pd.merge(df, delta_df[['GAME_ID', 'TEAM_ID', 'roster_delta']], 
                      left_on=['gameId', 'teamId'], right_on=['GAME_ID', 'TEAM_ID'], how='left')
        
        # Opponent Delta
        delta_opp = delta_df[['GAME_ID', 'TEAM_ID', 'roster_delta']].copy()
        delta_opp.columns = ['GAME_ID', 'opponentId', 'opp_roster_delta']
        df = pd.merge(df, delta_opp, left_on=['gameId', 'opponentId'], right_on=['GAME_ID', 'opponentId'], how='left')
        
        df['roster_delta'] = df['roster_delta'].fillna(0)
        df['opp_roster_delta'] = df['opp_roster_delta'].fillna(0)
        
        # 4. Rest Days
        df = df.sort_values(['teamId', 'gameDate'])
        df['rest_days'] = df.groupby('teamId')['gameDate'].diff().dt.days.fillna(3).clip(1, 4)
        
        # 5. Opponent Rest
        rest_lookup = df[['gameDate', 'teamId', 'rest_days']].copy()
        rest_lookup.columns = ['gameDate', 'opponentId', 'opp_rest_days']
        df = pd.merge(df, rest_lookup, on=['gameDate', 'opponentId'], how='left')
        df['opp_rest_days'] = df['opp_rest_days'].fillna(3)
        
        # 6. Interaction Features
        df['rest_advantage'] = df['rest_days'] - df['opp_rest_days']
        
        if 'off_orb' in df.columns and 'opp_def_orb' in df.columns: 
             df['rebound_mismatch'] = df['off_orb'] - df['opp_def_orb']
             
        if 'off_efg' in df.columns and 'opp_def_efg' in df.columns:
             df['shooting_mismatch'] = df['off_efg'] - df['opp_def_efg']
             
        return df.dropna()

    def train_final_score_model(self, df):
        """Trains XGBoost to predict team points."""
        
        # Explicit feature selection (Causal: My Offense vs Opponent Defense)
        
        # 1. Team Offense
        # We grab all 'off_' cols (ratings) and 'avg_' (league baselines)
        off_cols = [c for c in df.columns if c.startswith('off_')]
        
        # 2. Opponent Defense (Matching the offense)
        opp_def_cols = [c for c in df.columns if c.startswith('opp_def_')]
        
        # 3. Context & Era
        context_cols = ['isHome', 'roster_delta', 'opp_roster_delta', 'rest_days', 'opp_rest_days']
        
        if 'pace' in df.columns:
            context_cols.append('pace')
            
        # Era Baseline (one is enough, but grab available)
        baseline_cols = [c for c in df.columns if c.startswith('avg_')]
        if len(baseline_cols) > 1:
             baseline_cols = [baseline_cols[0]] # Just one to capture inflation
        
        # 4. Interactions
        interaction_cols = ['rest_advantage']
        if 'rebound_mismatch' in df.columns: interaction_cols.append('rebound_mismatch')
        if 'shooting_mismatch' in df.columns: interaction_cols.append('shooting_mismatch')
        
        # Combine
        features = off_cols + opp_def_cols + context_cols + baseline_cols + interaction_cols
        features = [f for f in features if f in df.columns]
        
        target = 'points'
        
        # Monthly Walk-Forward Validation
        dates = pd.to_datetime(df['gameDate'])
        min_date = dates.min()
        max_date = dates.max()
        
        start_date = pd.Timestamp("2018-10-01")
        sim_dates = pd.date_range(start_date, max_date, freq='MS')
        
        preds = []
        print(f"Game Model Features ({len(features)}): {features}")
        
        for sim_month in sim_dates:
            train = df[df['gameDate'] < sim_month]
            test = df[(df['gameDate'].dt.year == sim_month.year) & 
                      (df['gameDate'].dt.month == sim_month.month)]
            
            if len(test) == 0: continue
            if len(train) < 2000:
                 print(f"   Skipping {sim_month.strftime('%Y-%m')}: Insufficient history ({len(train)})")
                 continue
                 
            model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=3)
            model.fit(train[features], train[target])
            
            test_preds = model.predict(test[features])
            
            test = test.copy()
            test['pred_points'] = test_preds
            preds.append(test)
            
            mae = mean_absolute_error(test[target], test_preds)
            print(f"   Month {sim_month.strftime('%Y-%m')} MAE: {mae:.2f} (Train: {len(train)})")
            
        return pd.concat(preds)
