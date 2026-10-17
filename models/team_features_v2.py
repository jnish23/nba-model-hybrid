import pandas as pd
import numpy as np
import sqlite3
import logging

from models.season_utils import season_for_date
import random
from datetime import timedelta


OPP_STATS = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
             'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
             'AST', 'TOV', 'STL', 'BLK', 'PTS']


# ---- Win/loss momentum: ONE implementation shared by the batch build (build_entire_history)
# and live inference (TeamFeatureBuilder.build_features). Each function takes one team's
# results in game order and returns, for every row, the value known BEFORE that game
# (shift(1)); live inference appends a NaN row for the unplayed game and reads the last value.

def win_pct_last10(win: pd.Series) -> pd.Series:
    """Share of wins in the previous 10 games (needs at least 3)."""
    return win.shift(1).rolling(10, min_periods=3).mean()


def win_streak(win: pd.Series) -> pd.Series:
    """Win (+n) / loss (-n) streak entering each game; 0 before the first game."""
    s = win.shift(1)
    streaks = []
    current = 0
    for val in s:
        if pd.isna(val):
            streaks.append(0)
            continue
        if val == 1:
            current = max(1, current + 1) if current > 0 else 1
        else:
            current = min(-1, current - 1) if current < 0 else -1
        streaks.append(current)
    return pd.Series(streaks, index=win.index)


def home_win_pct_20g(home_win: pd.Series, is_home: pd.Series) -> pd.Series:
    """Win rate in home games among the previous 20 games (needs at least 5 games)."""
    return (home_win.shift(1).rolling(20, min_periods=5).sum() /
            is_home.shift(1).rolling(20, min_periods=5).sum())


def is_home_game(matchup: pd.Series) -> pd.Series:
    return matchup.str.contains('vs.', na=False).astype(int)


def normalize_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """team_game_full stores WL / MATCHUP upper-case. The momentum code looks for 'wl' / 'matchup',
    so without this rename those features were silently never built (in training or live)."""
    return df.rename(columns={c: c.lower() for c in ('WL', 'MATCHUP') if c in df.columns and c.lower() not in df.columns})



def ewma_pre_game(values, seasons, span, carryover=1.0, season_means=None):
    """
    Exponentially weighted mean of a team's PAST games for every game (what `ewm(span).mean().shift(1)`
    gives), optionally regressing it toward the league mean when a new season starts.

    carryover = 1.0 reproduces the plain EWMA that runs straight across seasons. With carryover k < 1,
    the first game of a new season sees  k * EWMA_end_of_last_season + (1 - k) * league_mean_of_last_season
    (the weight of the history is unchanged; only its mean is shrunk, once per season boundary).
    `season_means` maps season -> league mean of this stat in that season (known once it is over).
    """
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    out = np.full(len(values), np.nan)
    num = den = 0.0
    prev_season = None
    for i, (x, season) in enumerate(zip(values, seasons)):
        if prev_season is not None and season != prev_season and carryover < 1.0 and den > 0:
            mu = season_means.get(prev_season, np.nan) if season_means else np.nan
            if not np.isnan(mu):
                num = carryover * num + (1.0 - carryover) * mu * den
        out[i] = num / den if den > 0 else np.nan
        num *= decay
        den *= decay
        if not np.isnan(x):
            num += x
            den += 1.0
        prev_season = season
    return out



def games_in_previous_7_days(team_ids, game_dates):
    """Number of the team's games in [date - 7 days, date) for every row (same definition as
    groupby.rolling('7D', closed='left').count()); rows may be in any order."""
    team_ids = pd.Series(np.asarray(team_ids))
    dates = pd.Series(pd.to_datetime(np.asarray(game_dates)))
    out = np.zeros(len(dates))
    for _, idx in team_ids.groupby(team_ids).groups.items():
        d = dates.loc[idx].to_numpy()
        order = np.argsort(d, kind='stable')
        ds = d[order]
        before_today = np.searchsorted(ds, ds, side='left')
        in_window = np.searchsorted(ds, ds - np.timedelta64(7, 'D'), side='left')
        counts = np.empty(len(ds))
        counts[order] = before_today - in_window
        out[np.asarray(list(idx))] = counts
    return out


class TeamFeatureBuilder:
    def __init__(self, db_path, season_carryover=1.0):
        """season_carryover < 1 is an EXPERIMENT (batch build only): shrink each team's EWMA features
        toward the league mean when a new season starts. 1.0 (default) = the production behaviour,
        EWMAs that run unchanged across the offseason."""
        self.db_path = db_path
        self.season_carryover = season_carryover
        self.feature_categories = {
             'team_offense': [
                # Efficiency (don't include made + attempted + pct — just pct and volume)
                'offensiveRating', 'defensiveRating', 'netRating',
                'fieldGoalPercentage', 'threePointerPercentage', 'freeThrowPercentage',
                'points', 'plusMinus',
                # Rebounding
                'offensiveRebounds', 'defensiveRebounds',
                # Playmaking & Turnovers
                'assists', 'turnovers', 'assistToTurnover',
                # Shot distribution (pick representative subset, not every zone)
                'percentagePointsPaint', 'percentagePointsFastBreak',
                'percentagePointsOffTurnovers', 'percentageFieldGoalsAttempted3pt',
                # Volume context
                'fieldGoalsAttempted', 'threePointersAttempted', 'freeThrowsAttempted',
                # Hustle/Defense
                'steals', 'blocks',
            ],
            'team_defense': [
                # Aggregate opponent stats only — drop the per-position (C/F/G) breakdown
                # These need to exist in team_game_full; if they don't, we'll derive them below
                'OPP_FGM', 'OPP_FGA', 'OPP_FG_PCT',
                'OPP_FG3M', 'OPP_FG3A', 'OPP_FG3_PCT',
                'OPP_FTM', 'OPP_FTA', 'OPP_FT_PCT',
                'OPP_OREB', 'OPP_DREB', 'OPP_REB',
                'OPP_AST', 'OPP_TOV', 'OPP_STL', 'OPP_BLK', 'OPP_PTS',
            ]
        }
        self.spans = [5, 20]
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

    def get_latest_team_features(self, date=None, game_ids=None):
        """
        Feature vector for every team as of 'date' (history strictly before it).

        game_ids: optional {teamId: gameId} of the games being predicted (sets is_playoff from the
        NBA game id, as the batch build does).
        """
        if date is None:
            # Usage: Date for "today's" prediction (features from history before today)
            import datetime
            date = datetime.datetime.now().strftime('%Y-%m-%d')
            
        logging.info(f"Building latest team features as of {date}...")
        
        # One query for all teams (was one per team), then split in memory
        with sqlite3.connect(self.db_path) as conn:
            history = pd.read_sql(
                "SELECT * FROM team_game_full WHERE gameDate < ? ORDER BY teamId, gameDate ASC",
                conn, params=[str(date)])
        season = season_for_date(date)
        game_ids = game_ids or {}
        features_map = {}
        for tid, df_team in history.groupby('teamId'):
            feats = self._features_from_history(df_team.reset_index(drop=True), tid, date, season,
                                                game_ids.get(tid))
            if feats:
                features_map[tid] = feats
        return features_map

    def build_features(self, team_id: int, game_date: str, season: str, game_id: str = None) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            # Query history before game_date
            df_history = pd.read_sql(
                "SELECT * FROM team_game_full WHERE teamId = ? AND gameDate < ? ORDER BY gameDate ASC",
                conn, params=[int(team_id), str(game_date)])
        return self._features_from_history(df_history, team_id, game_date, season, game_id)

    def _features_from_history(self, df_history: pd.DataFrame, team_id, game_date, season, game_id=None) -> dict:
        """Features for one team's next game from its game log (rows before game_date, oldest first)."""
        features = {}
        df_history = self._with_opponent_totals(normalize_result_columns(df_history))

        if df_history.empty:
            logging.warning(f"No history found for team {team_id} before {game_date}")
            return {} # Or return NaNs? Return empty dict will likely be handled downstream

        features.update(self._offensive_features(df_history))
        features.update(self._defensive_features(df_history))
        features.update(self._momentum_features(df_history))
        features.update(self._rest_features(df_history, game_date))
        features.update(self._contextual_features(df_history, game_date, season, game_id))
        return features

    def _with_opponent_totals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check that team_game_full carries the exact OPP_<stat> opponent box-score totals.

        There is deliberately no fallback to the positional OPP_<stat>_<C|F|G> columns: those
        buckets overlap (hybrid positions fall in more than one), so their sum overstates every
        opponent total by ~22%. A table without the exact columns must be rebuilt with
        NBADatabaseSync.rebuild_team_game_full() (scripts/create_team_full_table.py does).
        """
        if df.empty:
            return df
        missing = [f'OPP_{stat}' for stat in OPP_STATS if f'OPP_{stat}' not in df.columns]
        if missing:
            raise ValueError(
                f"team_game_full has no exact opponent columns ({missing[:3]}...). Rebuild it with "
                f"scripts/create_team_full_table.py (NBADatabaseSync.rebuild_team_game_full); the "
                f"positional C/F/G columns cannot be summed into opponent totals.")
        return df

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
                
        # Trend features
        for col in valid_cols:
            col_5 = f"{prefix}_{col}_5g"
            col_20 = f"{prefix}_{col}_20g"
            if col_5 in feats and col_20 in feats:
                feats[f"{prefix}_{col}_trend"] = feats[col_5] - feats[col_20]
                
        return feats

    def _offensive_features(self, df: pd.DataFrame) -> dict:
        return self._calculate_ewma(df, self.feature_categories['team_offense'], 'off')

    def _defensive_features(self, df: pd.DataFrame) -> dict:
        return self._calculate_ewma(df, self.feature_categories['team_defense'], 'def')

    def _momentum_features(self, df: pd.DataFrame) -> dict:
        """win_pct_last10, win_streak, home_win_pct_20g for the NEXT game: the batch functions
        applied to the game log plus one unplayed (NaN) row, reading the last value."""
        feats = {}
        if 'wl' not in df.columns:
            return feats
        nan_row = pd.Series([np.nan])
        win = pd.concat([(df['wl'] == 'W').astype(int), nan_row], ignore_index=True)
        feats['win_pct_last10'] = win_pct_last10(win).iloc[-1]
        feats['win_streak'] = win_streak(win).iloc[-1]
        if 'matchup' in df.columns:
            is_home = pd.concat([is_home_game(df['matchup']), nan_row], ignore_index=True)
            home_win = pd.concat([is_home_game(df['matchup']) * (df['wl'] == 'W').astype(int), nan_row],
                                 ignore_index=True)
            feats['home_win_pct_20g'] = home_win_pct_20g(home_win, is_home).iloc[-1]
        return feats

    def _rest_features(self, df: pd.DataFrame, game_date) -> dict:
        """days_rest / is_b2b / games_last_7d exactly as build_entire_history defines them:
        days since the previous game (7 if none; NOT capped), and games in [date - 7d, date)."""
        feats = {}
        current = pd.to_datetime(game_date)
        dates = pd.to_datetime(df['gameDate'])
        if dates.empty:
            days_rest = 7
        else:
            days_rest = (current - dates.max()).days
        feats['days_rest'] = days_rest
        feats['is_b2b'] = 1 if days_rest == 1 else 0
        feats['games_last_7d'] = int((dates >= current - pd.Timedelta(days=7)).sum())
        return feats

    def _contextual_features(self, df: pd.DataFrame, game_date, season, game_id=None) -> dict:
        feats = {}
        # Game number within the season, matching build_entire_history
        if 'seasonYear' in df.columns:
            feats['season_game_number'] = int((df['seasonYear'] == season).sum()) + 1
        else:
            feats['season_game_number'] = len(df) + 1
        # Playoff games carry NBA game ids starting '004' (as in the batch build)
        feats['is_playoff'] = int(str(game_id)[:3] == '004') if game_id is not None else 0
        return feats

    def _ewma_feature(self, df, grp, col, span):
        """Past-games EWMA of `col` per team (the production definition unless season_carryover < 1)."""
        if self.season_carryover >= 1.0:
            return grp.transform(lambda x: x.ewm(span=span).mean().shift(1))
        # Experiment: regress to the league mean at season boundaries. season_means[s] = league mean of
        # `col` over season s (a finished season only enters the next one, so nothing from the future).
        season_means = df.groupby('seasonYear')[col].mean().to_dict()
        out = pd.Series(np.nan, index=df.index)
        for _, g in df.groupby('teamId'):
            out.loc[g.index] = ewma_pre_game(g[col].to_numpy(dtype=float), g['seasonYear'].to_numpy(), span,
                                             self.season_carryover, season_means)
        return out

    def build_entire_history(self) -> pd.DataFrame:
        """
         Efficiently builds features for the entire history using vectorized operations.
         Returns a DataFrame with index (teamId, gameDate) and all feature columns.
        """
        logging.info("Loading full team history...")
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql("SELECT * FROM team_game_full ORDER BY teamId, gameDate", conn)
            
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        
        df = self._with_opponent_totals(normalize_result_columns(df))
        
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
                new_features[f"off_{col}_{span}g"] = self._ewma_feature(df, grp, col, span)
                
        # Defense
        for col in self.feature_categories['team_defense']:
            if col not in df.columns: continue
            grp = df.groupby('teamId')[col]
            for span in self.spans:
                 new_features[f"def_{col}_{span}g"] = self._ewma_feature(df, grp, col, span)

        # Trend features
        logging.info("Calculating trend features...")
        for category in ['team_offense', 'team_defense']:
            prefix = 'off' if category == 'team_offense' else 'def'
            for col in self.feature_categories[category]:
                col_5 = f"{prefix}_{col}_5g"
                col_20 = f"{prefix}_{col}_20g"
                if col_5 in new_features and col_20 in new_features:
                    new_features[f"{prefix}_{col}_trend"] = new_features[col_5] - new_features[col_20]
                    
        # Win/loss momentum
        logging.info("Calculating momentum features...")
        if 'wl' in df.columns:
            df['_win'] = (df['wl'] == 'W').astype(int)
            
            # Last-10 record (shifted to avoid leakage)
            new_features['win_pct_last10'] = df.groupby('teamId')['_win'].transform(win_pct_last10)

            # Win streak (positive = win streak, negative = loss streak)
            new_features['win_streak'] = df.groupby('teamId')['_win'].transform(win_streak)
            df.drop(columns=['_win'], inplace=True)

        # Home/away historical performance
        if 'matchup' in df.columns:
            df['_is_home'] = is_home_game(df['matchup'])
            if 'wl' in df.columns:
                df['_home_win'] = df['_is_home'] * (df['wl'] == 'W').astype(int)
                new_features['home_win_pct_20g'] = df.groupby('teamId').apply(
                    lambda g: home_win_pct_20g(g['_home_win'], g['_is_home'])
                ).reset_index(level=0, drop=True)
            df.drop(columns=[c for c in ['_is_home', '_home_win'] if c in df.columns], inplace=True)

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
        except (ValueError, KeyError, TypeError) as e:
            # e.g. a non-monotonic date column. Compute the same count another way instead of filling a
            # constant (a made-up feature for every row is worse than a failure).
            logging.error(f"rolling('7D') failed for games_last_7d ({type(e).__name__}: {e}); "
                          f"using the exact searchsorted count")
            new_features['games_last_7d'] = games_in_previous_7_days(df['teamId'], df['gameDate'])
        
        # Fill NA for this specific col if needed, though concat handles it mainly
        # new_features['games_last_7d'] = new_features['games_last_7d'].fillna(0) # Do after concat
        
        # 3. Contextual
        logging.info("Calculating Contextual features...")
        # Game number within the season (was a cumulative count over all seasons: 1..1200)
        new_features['season_game_number'] = df.groupby(['teamId', 'seasonYear']).cumcount() + 1
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
