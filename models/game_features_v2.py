import pandas as pd
import numpy as np
import sqlite3
import logging

from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.team_features_v2 import TeamFeatureBuilder
from models.player_features import PlayerFeatureBuilder
from models.travel_builder import TravelFeatureBuilder


class GameFeatureBuilder:
    # Injury-report snapshot used for inj_* features. Must match when live inference runs
    # (see models/injury_report_features.SNAPSHOT_CUTOFFS); recorded in the training-set log.
    INJURY_SNAPSHOT = 'game_day_noon'
    INJURY_FIRST_SEASON = '2019-20'  # official report archive coverage

    def __init__(self, db_path, team_features_df=None, injury_snapshot=None, season_carryover=1.0):
        self.db_path = db_path
        self.season_carryover = season_carryover   # <1: offseason regression experiment (team_features_v2)
        self.team_features_df = team_features_df
        self.injury_snapshot = injury_snapshot or self.INJURY_SNAPSHOT
        
    def _get_team_features(self):
        """Get or build team features"""
        if self.team_features_df is not None:
            return self.team_features_df
            
        logging.info("Building team features from scratch...")
        builder = TeamFeatureBuilder(self.db_path, season_carryover=self.season_carryover)
        return builder.build_entire_history()

    def _get_player_features(self, seasons=None):
        logging.info("Building player features...")
        builder = PlayerFeatureBuilder(self.db_path)
        return builder.build_roster_features(seasons)

    def _get_travel_features(self):
        logging.info("Building Travel Features...")
        builder = TravelFeatureBuilder(self.db_path)
        df = builder.build_features()
        # Drop non-feature cols
        return df.drop(columns=['gameDate', 'venueTricode', 'lat', 'long'])


    def _load_db_odds(self):
        """Load odds from DB (Using Opening Odds ONLY)"""
        with sqlite3.connect(self.db_path) as conn:
            # Load OPENING odds as the primary source
            try:
                opening = pd.read_sql("SELECT * FROM nba_odds_opening", conn)
            except:
                logging.warning("Could not load nba_odds_opening")
                opening = pd.DataFrame()

            # Build team name → teamId lookup to fill missing IDs
            # (the scraper often omits teamId for newer seasons)
            try:
                team_map_df = pd.read_sql(
                    "SELECT DISTINCT teamId, teamName FROM team_basic_stats WHERE teamId IS NOT NULL",
                    conn
                )
                name_to_id = dict(zip(team_map_df['teamName'].str.strip(), team_map_df['teamId']))
            except Exception as e:
                logging.warning(f"Could not build team name→ID map: {e}")
                name_to_id = {}

        # Fill missing teamId_home / teamId_away from team name columns
        if not opening.empty and name_to_id:
            if 'teamId_home' in opening.columns and 'home_team' in opening.columns:
                missing_home = opening['teamId_home'].isna()
                if missing_home.any():
                    filled = opening.loc[missing_home, 'home_team'].str.strip().map(name_to_id)
                    opening.loc[missing_home, 'teamId_home'] = filled
                    logging.info(f"Filled {filled.notna().sum()} missing teamId_home values from team name lookup")

            if 'teamId_away' in opening.columns and 'away_team' in opening.columns:
                missing_away = opening['teamId_away'].isna()
                if missing_away.any():
                    filled = opening.loc[missing_away, 'away_team'].str.strip().map(name_to_id)
                    opening.loc[missing_away, 'teamId_away'] = filled

        # Helper to extract best odds across books
        def get_best_odds(df, suffix=""):
            # Priority: DraftKings > FanDuel > BetMGM > Bet365 > Caesars > Avg
            books = ['draftkings', 'fanduel', 'betmgm', 'bet365', 'caesars', 'bet_rivers_ny']
            
            out = pd.DataFrame(index=df.index)
            # Base keys
            for k in ['home_spread', 'away_spread', 'home_ml', 'away_ml', 'total']:
                target = f"odds_{k}{suffix}"
                # Try each book
                found = False
                for b in books:
                    col = f"{k}_{b}{suffix}"
                    if col in df.columns:
                        if not found:
                            out[target] = df[col]
                            found = True
                        else:
                            out[target] = out[target].combine_first(df[col])
            return out

        # Process Opening
        if not opening.empty and 'teamId_home' in opening.columns:
            # Drop rows still missing teamId_home (can't merge without a key)
            opening = opening.dropna(subset=['teamId_home'])
            opening['teamId_home'] = opening['teamId_home'].astype(int)
            opening['teamId_away'] = pd.to_numeric(opening['teamId_away'], errors='coerce').astype('Int64')

            # We pass suffix="" so they map to 'odds_home_ml', etc.
            o_vals = get_best_odds(opening, "")
            opening_processed = pd.concat([opening[['gameDate', 'teamId_home', 'teamId_away']], o_vals], axis=1)
        else:
            opening_processed = pd.DataFrame()

        return opening_processed

    def _load_archive_odds(self, csv_path):
        """Load archive odds (now with pre-calculated IDs)"""
        if not csv_path or not os.path.exists(csv_path):
            return pd.DataFrame()
            
        logging.info(f"Loading archive odds from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Date Parsing (YYYYMMDD)
        # 1. Convert to numeric first, coercing errors to NaN
        df['date'] = pd.to_numeric(df['date'], errors='coerce')

        # 2. Drop rows where date is missing
        df = df.dropna(subset=['date']).copy()

        # 3. Convert to datetime
        # Using .astype(int) on the numeric column prevents scientific notation issues
        df['gameDate'] = pd.to_datetime(df['date'].astype(int).astype(str), format='%Y%m%d')
        
        # Schema Mapping - SWITCH TO OPENING ONLY
        # Note: Archive doesn't have Opening ML, so odds_home_ml will be missing for older seasons.
        schema = {
            'home_open_spread': 'odds_home_spread', 'away_open_spread': 'odds_away_spread',
            'open_over_under': 'odds_total'
        }
        df = df.rename(columns=schema)

        # VALIDATE spreads before returning
        for col in ['odds_home_spread', 'odds_away_spread', 'odds_home_spread_open', 'odds_away_spread_open']:
            if col in df.columns:
                bad_mask = df[col].abs() > 25
                if bad_mask.sum() > 0:
                    logging.warning(f"Archive: Setting {bad_mask.sum()} invalid {col} values to NaN")
                    df.loc[bad_mask, col] = np.nan

        # Keep only relevant
        cols = ['gameDate', 'teamId_home'] + list(schema.values())
        cols = [c for c in cols if c in df.columns]
        
        return df[cols]

    ROSTER_COLS = ('roster_bpm', 'roster_bpm_ewma')
    MATCHUP_PAIRS = (
        # (home_offense_stat, away_defense_stat, name)
        ('off_offensiveRating_5g_home', 'def_defensiveRating_5g_away', 'matchup_off_vs_def_rating_5g'),
        ('off_offensiveRating_20g_home', 'def_defensiveRating_20g_away', 'matchup_off_vs_def_rating_20g'),
        ('off_threePointersAttempted_5g_home', 'def_OPP_FG3A_5g_away', 'matchup_3pa_vs_opp_3pa_5g'),
    )
    NON_FEATURE_PREFIXES = ('teamId', 'matchup', 'wl', 'points', 'gameId', 'is_home', 'seasonYear', 'gameDate')

    @classmethod
    def add_roster_features(cls, game_df, p_feats):
        """Attach roster_bpm / roster_bpm_ewma for each side as *_home / *_away.

        p_feats has one row per (gameId, teamId). Shared by the training-set build and live
        inference so the column names cannot drift apart.
        """
        p = p_feats[['gameId', 'teamId', *cls.ROSTER_COLS]]
        for side in ('home', 'away'):
            side_p = p.rename(columns={'teamId': f'teamId_{side}', **{c: f'{c}_{side}' for c in cls.ROSTER_COLS}})
            game_df = game_df.merge(side_p, on=['gameId', f'teamId_{side}'], how='left')
        return game_df

    @classmethod
    def add_derived_features(cls, game_df):
        """diff_<x> = x_home - x_away for every numeric home/away pair, plus the matchup features.

        One implementation for the batch training-set build and live inference.
        """
        logging.info("Generating differential and matchup features...")
        home_feat_cols = [c for c in game_df.columns
                          if c.endswith('_home') and not c.startswith(cls.NON_FEATURE_PREFIXES)]
        new_cols = {}
        for hcol in home_feat_cols:
            base = hcol.rsplit('_home', 1)[0]
            acol = f"{base}_away"
            if acol in game_df.columns and pd.api.types.is_numeric_dtype(game_df[hcol]) \
                    and not pd.api.types.is_bool_dtype(game_df[hcol]):
                new_cols[f"diff_{base}"] = game_df[hcol] - game_df[acol]
        for home_col, away_col, name in cls.MATCHUP_PAIRS:
            if home_col in game_df.columns and away_col in game_df.columns:
                new_cols[name] = game_df[home_col] - game_df[away_col]
            else:
                missing = [c for c in (home_col, away_col) if c not in game_df.columns]
                logging.debug(f"Matchup feature skipped (missing cols {missing}): {name}")
        return pd.concat([game_df, pd.DataFrame(new_cols, index=game_df.index)], axis=1) if new_cols else game_df

    def build_training_set(self, seasons=None, archive_path=None):
        """
        Build a single-row-per-game training set. combines data from various sources.
        """
        # 1. Get Base Features
        df_features = self._get_team_features().copy()
        
        # 1b. Travel
        travel = self._get_travel_features()
        df_features = df_features.merge(travel, on=['teamId', 'gameId'], how='left')

        if seasons:
            df_features = df_features[df_features['seasonYear'].isin(seasons)]
            
        # 2. Get Metadata (Home/Away info)
        with sqlite3.connect(self.db_path) as conn:
            meta = pd.read_sql("SELECT gameId, teamId, matchup, wl, points, gameDate, seasonYear FROM team_game_full", conn)
        
        meta = meta.rename(columns={'MATCHUP': 'matchup', 'WL': 'wl'})
        meta['gameDate'] = pd.to_datetime(meta['gameDate'])
        
        df_features = df_features.merge(meta, on=['teamId', 'gameDate', 'seasonYear'], suffixes=('', '_meta'))
        if 'gameId_meta' in df_features.columns:
            df_features['gameId'] = df_features['gameId'].fillna(df_features['gameId_meta'])
            df_features = df_features.drop(columns=['gameId_meta'])

        # 3. Identify Home/Away
        df_features['is_home'] = df_features['matchup'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
        
        # 4. Filter into Home and Away sets
        home_df = df_features[df_features['is_home'] == 1].copy()
        away_df = df_features[df_features['is_home'] == 0].copy()
        
        # 5. Merge on gameId
        game_df = pd.merge(home_df, away_df, on=['gameId', 'seasonYear', 'gameDate'], suffixes=('_home', '_away'))
        
        # 5b. Player (roster BPM) and injury-report features. They are merged BEFORE the diff step so
        # that diff_roster_bpm / diff_roster_bpm_ewma / diff_inj_* are created from the _home/_away
        # pairs (merged after it, the two roster merges collided into roster_bpm_ewma_x/_y, no diff
        # was ever built, and roster_bpm_home/away were dropped from the ML/Spread pool).
        p_feats = self._get_player_features(seasons)
        game_df = self.add_roster_features(game_df, p_feats)

        inj_seasons = sorted(s for s in game_df['seasonYear'].unique() if s >= self.INJURY_FIRST_SEASON)
        if inj_seasons:
            from models.injury_report_features import build_injury_features, add_injury_features
            logging.info(f"Adding injury report features (snapshot={self.injury_snapshot}) for {inj_seasons}")
            inj = build_injury_features(self.db_path, inj_seasons, snapshot_name=self.injury_snapshot)
            game_df = add_injury_features(game_df, inj)

        # 5c. Differential + matchup features (same code path as live inference)
        game_df = self.add_derived_features(game_df)

        # 6. Calculate Targets
        targets = pd.DataFrame({
            'target_home_win': game_df['wl_home'].apply(lambda x: 1 if x == 'W' else 0),
            'target_point_diff': game_df['points_home'] - game_df['points_away'],
            'target_total_pts': game_df['points_home'] + game_df['points_away'],
        }, index=game_df.index)
        game_df = pd.concat([game_df, targets], axis=1)

        # 7. Add Odds (Priority: Archive > DB)
        logging.info("Merging Odds data (OPENING ONLY)...")
        
        # A. Load Archive Odds FIRST (as base/fallback)
        archive_path = Path.cwd() / 'data' / 'nba_odds_archive.csv'
        archive_odds = self._load_archive_odds(archive_path)
        
        # B. Load DB Odds (Opening)
        db_odds = self._load_db_odds()
        
        if not db_odds.empty:
            db_odds['gameDate'] = pd.to_datetime(db_odds['gameDate'])
        
        # Select final odds cols
        keep_cols = ['gameDate', 'teamId_home', 'odds_home_ml', 'odds_away_ml', 
                    'odds_home_spread', 'odds_away_spread', 'odds_total']
        keep_cols = [c for c in keep_cols if c in db_odds.columns]
        if not db_odds.empty:
            db_odds = db_odds[keep_cols]

        # C. Combine: DB takes full priority; archive fills in older seasons
        # Use concat+drop_duplicates so DB-only rows (e.g. 2026) are NOT dropped.
        # combine_first only fills NaN values in the *left* index — rows that don't
        # exist in the archive (any date after 2023) were silently dropped.
        odds_frames = []
        if not db_odds.empty:
            odds_frames.append(db_odds)       # DB first = higher priority
        if not archive_odds.empty:
            # Align archive columns to DB schema (may be missing odds_home_ml etc.)
            archive_aligned = archive_odds.reindex(columns=db_odds.columns if not db_odds.empty else archive_odds.columns)
            odds_frames.append(archive_aligned)

        if odds_frames:
            combined = pd.concat(odds_frames, ignore_index=True)
            combined['gameDate'] = pd.to_datetime(combined['gameDate'])
            # Keep first occurrence per game (DB wins because it's first in the list)
            final_odds = combined.drop_duplicates(subset=['gameDate', 'teamId_home'], keep='first').copy()
        else:
            final_odds = pd.DataFrame()
        

        # D. VALIDATE: Clean obviously bad spread values
        for col in ['odds_home_spread', 'odds_away_spread']:
            if col in final_odds.columns:
                final_odds[col] = pd.to_numeric(final_odds[col], errors='coerce')
                bad_mask = final_odds[col].abs() > 25
                bad_count = bad_mask.sum()
                if bad_count > 0:
                    logging.warning(f"Cleaning {bad_count} invalid {col} values (abs > 25)")
                    final_odds.loc[bad_mask, col] = np.nan
        
        # E. Log data source breakdown
        if not final_odds.empty:
            final_odds['gameDate'] = pd.to_datetime(final_odds['gameDate'])
            for year in sorted(final_odds['gameDate'].dt.year.unique()):
                year_data = final_odds[final_odds['gameDate'].dt.year == year]
                spread_coverage = year_data['odds_home_spread'].notna().mean() if 'odds_home_spread' in year_data else 0
                logging.info(f"  {year}: {len(year_data)} games, spread coverage: {spread_coverage:.1%}")
        

        # Merge into Game DF

        game_df = pd.merge(game_df, final_odds, left_on=['gameDate', 'teamId_home'], right_on=['gameDate', 'teamId_home'], how='left')
        spread_cov = game_df['odds_home_spread'].notna().mean() if 'odds_home_spread' in game_df.columns else 0
        ml_cov = game_df['odds_home_ml'].notna().mean() if 'odds_home_ml' in game_df.columns else 0
        logging.info(f"Built training set with {len(game_df)} games. Spread coverage: {spread_cov:.2%}, ML coverage: {ml_cov:.2%} (Archive has spreads/totals; ML from DB)")

        return game_df


    def verify_no_leakage(self, df):
        """
        Sanity check: roster_bpm should not be perfectly correlated with outcome.
        """
        print("\n" + "="*60)
        print("LEAKAGE VERIFICATION")
        print("="*60)
        
        # Correlation with target
        if 'roster_bpm_home' in df.columns and 'target_home_win' in df.columns:
            corr = df['roster_bpm_home'].corr(df['target_home_win'])
            print(f"roster_bpm_home correlation with target: {corr:.4f}")
            
            if abs(corr) > 0.5:
                print("⚠️  WARNING: Correlation > 0.5 suggests possible leakage!")
            else:
                print("✓ Correlation looks reasonable (no obvious leakage)")
        
        # Check if roster_bpm has same value for consecutive home games
        # (it should, if we're using prior game correctly)
        df_sorted = df.sort_values(['teamId_home', 'gameDate'])
        df_sorted['roster_bpm_home_lag'] = df_sorted.groupby('teamId_home')['roster_bpm_home'].shift(1)
        
        # The current game's roster_bpm should equal the previous game's roster_bpm
        # (since both are based on the game BEFORE them)
        same_count = (df_sorted['roster_bpm_home'] == df_sorted['roster_bpm_home_lag']).sum()
        total = df_sorted['roster_bpm_home_lag'].notna().sum()
        
        print(f"\nConsecutive games with identical roster_bpm: {same_count}/{total}")
        print("(Should be LOW - if HIGH, the shift isn't working)")


if __name__ == "__main__":
    # Verification
    db_path = r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db'
    builder = GameFeatureBuilder(db_path)
    df = builder.build_training_set()

    print(df.head())
    print(df.columns)
    builder.verify_no_leakage(df)
