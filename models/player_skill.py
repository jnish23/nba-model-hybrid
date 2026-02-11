import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

class PlayerSkillModel:
    def __init__(self):
        self.best_model = None
        # Define features upfront so _engineer_features can use them
        self.features = [
             'shotDistanceFixed', 'shotAngle', 
             'is_Arc3', 'is_Corner3', 'is_Rim', 'is_ShortMidRange', 'is_LongMidRange', 'is_BackCourt',
             'putback', 'isRegularSeason', 'home', '3PT',
             'is_Off3BlockedOreb',
             'is_Off3Oreb', 'is_OffArc3Make', 'is_OffArc3Miss',
             'is_OffBackCourtBlockedOreb', 'is_OffBackCourtMake',
             'is_OffBackCourtMiss', 'is_OffBackCourtOreb', 'is_OffCorner3Make',
             'is_OffCorner3Miss', 'is_OffDeadball', 'is_OffLongMidRangeBlockedOreb',
             'is_OffLongMidRangeMake', 'is_OffLongMidRangeMiss',
             'is_OffLongMidRangeOreb', 'is_OffRimBlockedOreb', 'is_OffRimMake',
             'is_OffRimMiss', 'is_OffRimOreb', 'is_OffShortMidRangeBlockedOreb',
             'is_OffShortMidRangeMake', 'is_OffShortMidRangeMiss',
             'is_OffShortMidRangeOreb', 'is_OffTurnover',
             'total_seconds_remaining', 'is_heave'
        ]

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (Same cleaning logic) ...
        # 1. Basic cleaning
        df = df.sort_values(by=['gameId', 'period', 'actionNumber']).reset_index(drop=True)
        df['actionType'] = df['actionType'].str.strip()
        
        action_map = {'Made Shot': 1, 'Missed Shot': 2, 'Rebound': 4, 'Turnover': 5}
        df['actionCode'] = df['actionType'].map(action_map)
        
        # 2. Zone Mapping
        zone_map = {
            'Restricted Area': 'Rim',
            'Left Corner 3': 'Corner3',
            'Right Corner 3': 'Corner3',
            'Above the Break 3': 'Arc3',
            'Mid-Range': 'LongMidRange',
            'In The Paint (Non-RA)': 'ShortMidRange',
            'Backcourt': 'BackCourt'
        }
        df['ShotZone'] = df['SHOT_ZONE_BASIC'].map(zone_map)
        df['description'] = df['description'].fillna('Jump Ball')
        
        # 3. Complex Shot Context (StartType, Orebs)
        mask_shot = df['actionCode'].isin([1, 2])
        df['LatestShotZone'] = df['ShotZone'].where(mask_shot).ffill()
        df['LatestShooterTeam'] = df['teamId'].where(mask_shot).ffill()
        
        df['prev_actionCode'] = df.groupby('gameId')['actionCode'].shift(1)
        df['prev_teamId'] = df.groupby('gameId')['teamId'].shift(1)
        df['prev_ShotZone'] = df.groupby('gameId')['ShotZone'].shift(1)
        df['is_block_event'] = df['description'].str.contains('BLOCK', case=False, na=False)
        df['prev_was_block'] = (df['is_block_event'].shift(1) == True)
        
        # EventStartType Logic
        df['EventStartType'] = pd.Series([None] * len(df), dtype='object')
        
        cond_off_make = (df['prev_actionCode'] == 1) & (df['teamId'] != df['prev_teamId'])
        df.loc[cond_off_make, 'EventStartType'] = 'Off' + df.loc[cond_off_make, 'prev_ShotZone'] + 'Make'
        
        cond_turnover = (df['prev_actionCode'] == 5) & (df['teamId'] != df['prev_teamId'])
        df.loc[cond_turnover, 'EventStartType'] = 'OffTurnover'
        
        cond_def_reb = (df['actionCode'] == 4) & (df['teamId'] != df['prev_teamId']) & (df['prev_actionCode'] == 2)
        is_blocked_miss = (cond_def_reb) & (df['prev_was_block'])
        is_clean_miss = (cond_def_reb) & (~df['prev_was_block'])
        
        df.loc[is_blocked_miss, 'EventStartType'] = 'Off' + df.loc[is_blocked_miss, 'prev_ShotZone'] + 'Block'
        df.loc[is_clean_miss, 'EventStartType'] = 'Off' + df.loc[is_clean_miss, 'prev_ShotZone'] + 'Miss'
        
        df['StartType'] = df.groupby('gameId')['EventStartType'].ffill().fillna('OffDeadball')
        
        # OREB Logic
        is_oreb = (df['actionCode'] == 4) & (df['teamId'] == df['LatestShooterTeam'])
        cond_blocked_oreb = (is_oreb) & (df['prev_was_block'])
        cond_standard_oreb = (is_oreb) & (~df['prev_was_block'])
        
        df['OrebShotType'] = pd.Series([None] * len(df), dtype='object')
        df.loc[cond_blocked_oreb, 'OrebShotType'] = df.loc[cond_blocked_oreb, 'LatestShotZone'] + 'Blocked'
        df.loc[cond_standard_oreb, 'OrebShotType'] = df.loc[cond_standard_oreb, 'LatestShotZone']
        
        df['PossessionID'] = df['EventStartType'].notna().cumsum()
        df['OrebShotType'] = df.groupby('PossessionID')['OrebShotType'].ffill()
        
        def get_play_start_type(row):
            if pd.isna(row['OrebShotType']):
                return row['StartType']
            ost = str(row['OrebShotType'])
            if ost in ['Arc3Blocked', 'Corner3Blocked']: return 'Off3BlockedOreb'
            elif ost in ['Corner3', 'Arc3']: return 'Off3Oreb'
            else: return 'Off' + ost + 'Oreb'

        df['PlayStartType'] = df.apply(get_play_start_type, axis=1)
        
        # Shot Physics
        raw_angle = np.abs(np.degrees(np.arctan2(df['xLegacy'], df['yLegacy'])))  
        df['shotAngle'] = np.where(raw_angle < 90, raw_angle, 180 - raw_angle)
        df['shotAngle'] = df['shotAngle'].round(1)
        
        df['shotDistanceFixed'] = df['shotDistance'].copy()
        df.loc[(df['SHOT_TYPE'] == '3PT Field Goal') & (df['shotDistance'] < 22), 'shotDistanceFixed'] = df['SHOT_DISTANCE'] 
        
        # Boolean FLags
        df['putback'] = df['description'].str.contains('putback', case=False) 
        df['home'] = (df['location']=='h').astype(int)
        df['3PT'] = (df['SHOT_TYPE'] == '3PT Field Goal').astype(int)
        df['shotResult'] = (df['shotResult'] == 'Made').astype(int)
        
        # Time / Heave Logic
        # Parse 'clock' if minutes/seconds columns are missing
        if 'MINUTES_REMAINING' in df.columns and 'SECONDS_REMAINING' in df.columns:
             df['total_seconds_remaining'] = df['MINUTES_REMAINING'] * 60 + df['SECONDS_REMAINING']
        else:
             # Fallback: Parse 'clock' e.g. PT11M43.00S
             # Simplified regex or string split
             try:
                 df['clock_str'] = df['clock'].astype(str).str.replace('PT', '').str.replace('S', '')
                 # Split on 'M'
                 split_time = df['clock_str'].str.split('M', expand=True)
                 df['min_temp'] = pd.to_numeric(split_time[0], errors='coerce').fillna(0)
                 df['sec_temp'] = pd.to_numeric(split_time[1], errors='coerce').fillna(0)
                 df['total_seconds_remaining'] = df['min_temp'] * 60 + df['sec_temp']
             except:
                 df['total_seconds_remaining'] = 720 # Default to full quarter? Or 0?
                 
        # Define Heave: < 4 seconds, > 40 feet (Half court is ~47, but heave usually > 40)
        df['is_heave'] = ((df['total_seconds_remaining'] < 4) & (df['shotDistance'] > 40)).astype(int)
        
        cols_to_dummy = ['season', 'ShotZone', 'OrebShotType', 'PlayStartType']
        df = pd.get_dummies(df, columns=cols_to_dummy, prefix='is')
        
        # Fill Missing Features
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = 0
                
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def prepare_shot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._engineer_features(df)
        
        season_cols = [c for c in df.columns if 'is_20' in c and '-' in c]
        df['season_end'] = 0
        for col in season_cols:
            # Fix: Handle 'is_2023-24' format correctly
            try:
                # col is like 'is_2023-24'
                clean_col = col.replace('is_', '')
                year_end = int(clean_col.split('-')[0]) + 1 # 2023-24 -> 2024
            except Exception as e:
                # print(f"Error parsing {col}: {e}")
                year_end = 0 
            df.loc[df[col] == 1, 'season_end'] = year_end
            
        target = 'shotResult'
        
        # self.features is already defined in __init__
        
        df = df[df['season_end'] > 0].copy()
        df = df.loc[df['isFieldGoal'] == 1]
        
        # print(df.columns)
        # print(df[target].value_counts(dropna=False))

        df = df.dropna(subset=[target] + self.features)
        return df, target

    def tune_xgboost(self, X_train, y_train, X_val, y_val):
        split_index = [-1]*len(X_train) + [0]*len(X_val)
        pds = PredefinedSplit(test_fold=split_index)
        
        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)
        
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 300, 500],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1, 0.5]
        }
        
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        
        search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=10, cv=pds, scoring='neg_log_loss', n_jobs=2, verbose=0
        )
        
        search.fit(X_combined, y_combined)
        
        # Logging (Requested by User)
        print(f"   Best Params: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_

    def run_expanding_window(self, df, target):
        seasons = sorted(df['season_end'].unique())
        start_test_idx = 2
        all_predictions = []
        
        print(f"Player Skill: Data spans {seasons[0]} to {seasons[-1]}")

        for i in range(start_test_idx, len(seasons)):
            test_season = seasons[i]
            validation_season = seasons[i-1]
            
            # print(f"Processing {test_season}...")
            
            train_mask = df['season_end'] < validation_season
            val_mask = df['season_end'] == validation_season
            test_mask = df['season_end'] == test_season
            
            X_train = df.loc[train_mask, self.features]
            y_train = df.loc[train_mask, target]
            
            if len(X_train) < 1000:
                print(f"Skipping {test_season} (Insufficient Training Data: {len(X_train)})")
                continue
            X_val = df.loc[val_mask, self.features]
            y_val = df.loc[val_mask, target]
            X_test = df.loc[test_mask, self.features]
            
            if len(X_val) == 0: continue

            best_model, params = self.tune_xgboost(X_train, y_train, X_val, y_val)
            self.best_model = best_model # Keep last
            
            probs = best_model.predict_proba(X_test)[:, 1]
            
            # Ensure we keep the date column if it exists (try 'gameDate')
            cols_to_keep = ['season_end', target, 'personId', 'gameId', 'teamId']
            if 'gameDate' in df.columns: cols_to_keep.append('gameDate')
            elif 'GAME_DATE' in df.columns: cols_to_keep.append('GAME_DATE')
            
            chunk = df.loc[test_mask, cols_to_keep].copy()
            chunk['xFG'] = probs
            chunk['residual'] = chunk[target] - chunk['xFG']
            
            all_predictions.append(chunk)

        return pd.concat(all_predictions)

    def calculate_rolling_player_stats(self, shot_df, k_short=100, k_long=300):
        """
        Calculates player skill using Empirical Bayes Shrinkage (Robust Method).
        FIX: Aggregate to game-level first to avoid within-game leakage.
        """
        # Step 1: Aggregate residuals to game-level (mean residual per game)
        game_agg = shot_df.groupby(['personId', 'gameId']).agg(
            residual_mean=('residual', 'mean'),
            shot_count=('residual', 'count')
        ).reset_index()
        
        game_agg = game_agg.sort_values(['personId', 'gameId'])
        
        # Step 2: Calculate game-level EWM (now span is in games, not shots)
        # Adjust spans: 200 shots ~= 15 games, 1500 shots ~= 100 games
        span_short_games = 15
        span_long_games = 100
        
        grouped = game_agg.groupby('personId')['residual_mean']
        
        # EWM with shift(1) to exclude current game
        game_agg['ewm_mean_short'] = grouped.transform(
            lambda x: x.ewm(span=span_short_games, min_periods=1).mean().shift(1)
        )
        game_agg['ewm_mean_long'] = grouped.transform(
            lambda x: x.ewm(span=span_long_games, min_periods=1).mean().shift(1)
        )
        
        # Cumulative game count for effective N
        game_agg['game_num'] = grouped.cumcount()  # 0-indexed, so game 1 has game_num=0
        
        # Effective N approximation (in games now)
        alpha_short = 2 / (span_short_games + 1)
        alpha_long = 2 / (span_long_games + 1)
        
        game_agg['eff_n_short'] = (1 - (1 - alpha_short) ** game_agg['game_num']) / alpha_short
        game_agg['eff_n_long'] = (1 - (1 - alpha_long) ** game_agg['game_num']) / alpha_long
        
        # Shrinkage toward 0 (league average)
        # Posterior = (EWM * eff_n) / (eff_n + k)
        game_agg['entering_skill_short'] = (
            (game_agg['ewm_mean_short'] * game_agg['eff_n_short']) / 
            (game_agg['eff_n_short'] + k_short)
        )
        game_agg['entering_skill_long'] = (
            (game_agg['ewm_mean_long'] * game_agg['eff_n_long']) / 
            (game_agg['eff_n_long'] + k_long)
        )
        
        # Fill NaN (first game has no history)
        game_agg['entering_skill_short'] = game_agg['entering_skill_short'].fillna(0)
        game_agg['entering_skill_long'] = game_agg['entering_skill_long'].fillna(0)
        
        return game_agg[['personId', 'gameId', 'entering_skill_long', 'entering_skill_short']]

    def normalize_player_skill(self, player_features, player_basic_stats):
        """
        Converts 'Total Residual' into 'Skill Points Per Minute'.
        """
        pbs = player_basic_stats.copy()
        try:
             pbs['gameId'] = pbs['gameId'].astype(int)
        except:
             pass 
        
        df = pd.merge(player_features, pbs, left_on=['personId', 'gameId'], right_on=['personId', 'gameId'])
        df['minutes'] = df['minutes'].replace(0, 1)
        
        # FIX: shift(1) to avoid using current game's MIN/FGA (data leakage)
        df['min_ewm'] = df.groupby(['personId'])['minutes'].transform(lambda x: x.ewm(span=10).mean().shift(1))
        df['fga_ewm'] = df.groupby(['personId'])['fieldGoalsAttempted'].transform(lambda x: x.ewm(span=10).mean().shift(1))
        
        # Avoid division by zero
        df['min_ewm'] = df['min_ewm'].replace(0, 1)
        
        # Skill (pts/shot) * Shots/Game = Skill Points/Game
        # Skill Points/Game / Minutes/Game = Skill Points/Minute
        df['skill_per_minute'] = (df['entering_skill_short'] * df['fga_ewm']) / df['min_ewm']
        
        return df[['personId', 'gameId', 'skill_per_minute']]

