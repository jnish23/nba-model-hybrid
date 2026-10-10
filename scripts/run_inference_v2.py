import pandas as pd
import numpy as np
import sqlite3
import logging
import datetime
import joblib
import sys
import os
import json
from scipy import stats
from bs4 import BeautifulSoup
from pathlib import Path 
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.team_features_v2 import TeamFeatureBuilder
from models.travel_builder import TravelFeatureBuilder
from models.live_rotation_v2 import LiveRotationBuilder
from models.bpm_builder import BPMBuilder
from models.prediction_logger import PredictionLogger
from models.player_features import PlayerFeatureBuilder
from models import injury_report_features as irf
from models.game_features_v2 import GameFeatureBuilder
from update_database import NBADatabaseSync
from build_nba_odds_table import sync_odds_data
from nba_api.stats.endpoints import scoreboardv3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from models.season_utils import season_for_date

current_season = season_for_date(datetime.date.today())



def get_nba_schedule(game_date=None):
    """
    Get today's games with official NBA gameIds.
    Returns DataFrame with gameId, home_team, away_team, gameDate.
    """
    if game_date is None:
        game_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    try:
        games = scoreboardv3.ScoreboardV3(game_date=game_date).get_data_frames()[1]
        # Fix: UTC time might be next day for late games. Use the requested/local game_date.
        games['gameDate'] = game_date
        games['away_team_abbr'] = games['gameCode'].str[-6:-3]
        games['home_team_abbr'] = games['gameCode'].str[-3:]

        return games[['gameId', 'gameDate', 'home_team_abbr', 'away_team_abbr']]
    except Exception as e:
        logging.error(f"Failed to fetch NBA schedule: {e}")
        return pd.DataFrame()

class InferenceEngine:
    def __init__(self, db_path, model_path_ml, target_date=None):
        self.db_path = db_path
        self.target_date = target_date  # Optional: override datetime.now() for backfilling
        
        # 1. Parse Paths using the input ML model as the anchor
        # Example input: "models/saved/model_ml_with_odds_v3.pkl"
        model_path = Path(model_path_ml)
        model_dir = model_path.parent
        model_name = model_path.name  # "model_ml_with_odds_v3.pkl"
        
        # 2. Load the Base ML Model
        logging.info(f"Loading ML Model from {model_path}")
        self.model_ml = joblib.load(model_path)
        
        # 3. Load the Matching Calibrator
        # Logic: Replace "model_ml" with "calibrator_ml" in the filename
        if 'model_ml' in model_name:
            calib_name = model_name.replace('model_ml', 'calibrator_ml')
            calib_path = model_dir / calib_name
            
            if calib_path.exists():
                self.calibrator_ml = joblib.load(calib_path)
                logging.info(f"Loaded Calibrator: {calib_name}")
            else:
                logging.warning(f"Calibrator not found at {calib_path}")
                self.calibrator_ml = None
        else:
            logging.warning("Filename format mismatch. Expected 'model_ml...'")
            self.calibrator_ml = None

        # 4. Load Spread/Total Models (Using the same suffix pattern)
        # We assume if ML model is "model_ml_v3.pkl", Spread is "model_spread_v3.pkl"
        suffix = model_name.replace('model_ml', '')  # e.g., "_with_odds_v3.pkl"
        
        try:
            spread_path = model_dir / f'model_spread{suffix}'
            self.model_spread = joblib.load(spread_path)
            logging.info(f"Loaded Spread Model: {spread_path.name}")
        except Exception as e:
            logging.warning(f"Could not load spread model: {e}")
            self.model_spread = None

        try:
            total_path = model_dir / f'model_total{suffix}'
            self.model_total = joblib.load(total_path)
            logging.info(f"Loaded Total Model: {total_path.name}")
        except Exception as e:
            logging.warning(f"Could not load total model: {e}")
            self.model_total = None

        # 5. Load Feature Configuration
        # Your training script named this "feature_config{suffix}.json"
        # NOT "selected_features.json"
        config_name = f'feature_config{suffix}'.replace('.pkl', '.json')
        config_path = model_dir / config_name
        
        self.selected_features = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.selected_features = json.load(f)
            logging.info(f"Loaded feature config from {config_name}")
        else:
            logging.warning(f"Feature config not found at {config_path}")

        # 6. Load Calibration Metadata
        # Your training script saved this to model_dir, NOT root
        meta_name = f'calibration_meta{suffix}'.replace('.pkl', '.json')
        meta_path = model_dir / meta_name
        
        self.calib_meta = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.calib_meta = json.load(f)
            logging.info(f"Loaded calibration meta from {meta_name}")
        else:
            logging.warning(f"Calibration meta not found at {meta_path}")

        # Initialize Logger
        self.logger = PredictionLogger(db_path) 
        logging.info("Inference Engine initialization complete.")
            
        logging.info("Models loaded successfully.")
        
    def get_todays_games(self):
        logging.info("Fetching Today's Odds & Schedule...")
        # Use target_date if provided (for backfilling), otherwise use today
        from scripts.scrape_injury_reports import now_et
        today = self.target_date if self.target_date else now_et().strftime('%Y-%m-%d')
        
        # 1. Try DB (for archive/completeness, but for LIVE we prefer SBR usually if DB is stale)
        # Check if DB has today's games? 
        # For this refactor, let's rely on SBR for live lines to be freshest.
        nba_schedule = get_nba_schedule(game_date=today)
        if nba_schedule.empty:
            logging.warning("No games from NBA API")
            return pd.DataFrame()
        # 2. Direct SBR Fetch
        logging.info(f"Fetching from SBR for {today}...")
        try:
            from sbrscrape import Scoreboard
            
            # Fetch Current
            sb_curr = Scoreboard(sport='NBA', date=today, current_line=True)
            if not sb_curr.games:
                logging.warning("No games found on SBR (Current).")
                return pd.DataFrame()
                
            # Fetch Open
            sb_open = Scoreboard(sport='NBA', date=today, current_line=False)
            
            # Helper to extract value
            def get_val(item):
                if isinstance(item, dict):
                    # Priority: generic books
                    for k in ['draftkings', 'fanduel', 'betmgm', 'bet365', 'caesars']:
                        if k in item and item[k]: return item[k]
                return None

            # Build dict of Opening Odds by Matchup key
            open_map = {}
            if sb_open.games:
                for g in sb_open.games:
                    # Key: Away @ Home
                    key = f"{g.get('away_team')} @ {g.get('home_team')}"
                    open_map[key] = {
                    'odds_away_ml_open': get_val(g.get('away_ml')),
                        'odds_home_ml_open': get_val(g.get('home_ml')),
                        'odds_home_spread_open': get_val(g.get('home_spread')),
                        'odds_away_spread_open': get_val(g.get('away_spread')),
                        'odds_total_open': get_val(g.get('total'))
                    }
            
            games = []
            for g in sb_curr.games:
                key = f"{g.get('away_team')} @ {g.get('home_team')}"
                open_data = open_map.get(key, {})
                
                row = {
                    'gameDate': today,
                    'away_team': g.get('away_team'),
                    'home_team': g.get('home_team'),
                    'away_team_abbr' : g.get('away_team_abbr'),
                    'home_team_abbr' : g.get('home_team_abbr'),
                    # Current
                    'odds_away_ml': get_val(g.get('away_ml')),
                    'odds_home_ml': get_val(g.get('home_ml')),
                    'odds_home_spread': get_val(g.get('home_spread')),
                    'odds_away_spread': get_val(g.get('away_spread')),
                    'odds_total': get_val(g.get('total')),
                    # Open (Fallback to Current if missing, or np.nan? Model likely needs numeric. 
                    # If open is missing, "No movement" assumption -> use current is safest proxy)
                    'odds_away_ml_open': open_data.get('odds_away_ml_open') or get_val(g.get('away_ml')),
                    'odds_home_ml_open': open_data.get('odds_home_ml_open') or get_val(g.get('home_ml')),
                    'odds_home_spread_open': open_data.get('odds_home_spread_open') or get_val(g.get('home_spread')),
                    'odds_away_spread_open': open_data.get('odds_away_spread_open') or get_val(g.get('away_spread')),
                    'odds_total_open': open_data.get('odds_total_open') or get_val(g.get('total'))
                }
                games.append(row)
                
            df = pd.DataFrame(games)

            abbr_map = { 'BK':'BKN',
                        'CHAR':'CHA',
                        'GS':'GSW',
                        'NO':'NOP',
                        'NY':'NYK',
                        'PHO':'PHX',
                        'SA':'SAS'}
            df['away_team_abbr'] = df['away_team_abbr'].replace(abbr_map)
            df['home_team_abbr'] = df['home_team_abbr'].replace(abbr_map)
                            # Map Team IDs
            with sqlite3.connect(self.db_path) as conn:
                try:
                    # Columns are likely camelCase now: teamId, teamName, teamTricode
                    team_info = pd.read_sql("SELECT DISTINCT teamId, teamName, teamTricode FROM team_basic_stats", conn)
                except:
                    # Fallback try original uppercase just in case
                    team_info = pd.read_sql("SELECT DISTINCT TEAM_ID as teamId, TEAM_NAME as teamName, TEAM_ABBREVIATION as teamTricode FROM team_basic_stats", conn)
                
            name_map = {}
            for _, t in team_info.iterrows():
                name_map[t['teamName']] = t['teamId']
                name_map[t['teamTricode']] = t['teamId']
                # Create 'SimpleName' manually (Last word)
                simple = t['teamName'].split(' ')[-1]
                name_map[simple] = t['teamId']
            
            # Manual Fixes
            name_map['LA Clippers'] = 1610612746
            name_map['L.A. Clippers'] = 1610612746
            name_map['LA Lakers'] = 1610612747
            name_map['L.A. Lakers'] = 1610612747
            name_map['Clippers'] = 1610612746
            name_map['Lakers'] = 1610612747
            
            # Debug Prints
            logging.info(f"SBR Away Teams: {df['away_team'].unique()}")
            logging.info(f"SBR Home Teams: {df['home_team'].unique()}")
            # logging.info(f"Map Keys Sample: {list(name_map.keys())[:10]}")
            
            def map_id(name):
                if not isinstance(name, str): return None
                if name in name_map: return name_map[name]
                # Fuzzy
                for k, v in name_map.items():
                    if k in name or name in k: return v
                return None
                
            df['teamId_home'] = df['home_team'].apply(map_id)
            df['teamId_away'] = df['away_team'].apply(map_id)
            
            df = df.dropna(subset=['teamId_home', 'teamId_away'])
            
            merged = nba_schedule.merge(
                df,
                on=['home_team_abbr', 'away_team_abbr'],
                how='left',
                suffixes=('', '_x')
            )
            
            merged = merged.drop(columns=['gameDate_x'])
            
            return merged
            
        except Exception as e:
            logging.error(f"SBR Fetch Error: {e}")
            return pd.DataFrame()

    def get_latest_team_features(self):
        logging.info("Building Full Team Features...")
        
        # 1. Load Data
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql("SELECT * FROM team_game_full", conn)
            
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        df = df.sort_values(['teamId', 'gameDate'])
        
        # Ensure we have aggregated OPP columns
        for stat in ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                     'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
                     'AST', 'TOV', 'STL', 'BLK', 'PTS']:
            pos_cols = [f'OPP_{stat}_{p}' for p in ['C', 'F', 'G']]
            existing = [c for c in pos_cols if c in df.columns]
            if existing:
                if 'PCT' in stat:
                    df[f'OPP_{stat}'] = df[existing].mean(axis=1)
                else:
                    df[f'OPP_{stat}'] = df[existing].sum(axis=1)
        
        builder = TeamFeatureBuilder(self.db_path)
        off_cols = builder.feature_categories['team_offense']
        def_cols = builder.feature_categories['team_defense']
        spans = [5, 20] # Note: spans updated to [5, 20]
        
        final_feats = pd.DataFrame(index=df['teamId'].unique())
        final_feats.index.name = 'teamId'
        
        feat_dict = {}

        # Offense
        for col in off_cols:
            if col in df.columns:
                grp = df.groupby('teamId')[col]
                for span in spans:
                    series = grp.transform(lambda x: x.ewm(span=span).mean())
                    feat_dict[f"off_{col}_{span}g"] = series.groupby(df['teamId']).last()
                    
        # Defense
        for col in def_cols:
            if col in df.columns:
                grp = df.groupby('teamId')[col]
                for span in spans:
                    series = grp.transform(lambda x: x.ewm(span=span).mean())
                    feat_dict[f"def_{col}_{span}g"] = series.groupby(df['teamId']).last()

        # Trend & Momentum Features
        # Trend
        for category in ['team_offense', 'team_defense']:
            prefix = 'off' if category == 'team_offense' else 'def'
            cat_cols = off_cols if category == 'team_offense' else def_cols
            for col in cat_cols:
                col_5 = f"{prefix}_{col}_5g"
                col_20 = f"{prefix}_{col}_20g"
                if col_5 in feat_dict and col_20 in feat_dict:
                    feat_dict[f"{prefix}_{col}_trend"] = feat_dict[col_5] - feat_dict[col_20]
                    
        # Momentum
        if 'wl' in df.columns:
            df['_win'] = (df['wl'] == 'W').astype(int)
            # win_pct_last10 (using the *completed* history)
            feat_dict['win_pct_last10'] = df.groupby('teamId')['_win'].rolling(10, min_periods=3).mean().groupby('teamId').last()
            
            def streak(series):
                streaks = []
                current = 0
                for val in series:
                    if pd.isna(val):
                        streaks.append(0)
                        continue
                    if val == 1:
                        current = max(1, current + 1) if current > 0 else 1
                    else:
                        current = min(-1, current - 1) if current < 0 else -1
                    streaks.append(current)
                return pd.Series(streaks, index=series.index)
            
            feat_dict['win_streak'] = df.groupby('teamId')['_win'].apply(streak).groupby('teamId').last()
            
            # home_win_pct_20g
            if 'matchup' in df.columns:
                df['_is_home'] = df['matchup'].str.contains('vs.', na=False).astype(int)
                df['_home_win'] = df['_is_home'] * df['_win']
                
                feat_dict['home_win_pct_20g'] = df.groupby('teamId').apply(
                    lambda g: g['_home_win'].rolling(20, min_periods=5).sum() / 
                              g['_is_home'].rolling(20, min_periods=5).sum()
                ).groupby('teamId').last()

        final_feats = pd.DataFrame.from_dict(feat_dict)
        final_feats.index.name = 'teamId'

        # 4. Rest & Context
        last_dates = df.groupby('teamId')['gameDate'].max()
        today = pd.to_datetime(datetime.datetime.now().strftime('%Y-%m-%d'))
        
        days_rest = (today - last_dates).dt.days - 1 
        
        final_feats['days_rest'] = days_rest.clip(lower=0)
        final_feats['is_b2b'] = (final_feats['days_rest'] == 0).astype(int)
        
        cutoff = today - pd.Timedelta(days=7)
        games_7d = df[df['gameDate'] >= cutoff].groupby('teamId').size()
        final_feats['games_last_7d'] = games_7d.reindex(final_feats.index, fill_value=0)
        
        # Season Game Num
        final_feats['season_game_number'] = df.groupby('teamId').count()['gameDate'] + 1
        
        final_feats['is_playoff'] = 0 
        
        return final_feats

    def get_travel_features(self, schedule):
        logging.info("Building Travel Features...")
        tb = TravelFeatureBuilder(self.db_path)
        tb.load_data()
        
        feats = []
        teams = set(schedule['teamId_home']).union(set(schedule['teamId_away']))
        
        for tid in teams:
            # Last Game
            t_games = tb.df_games[tb.df_games['teamId'] == tid].sort_values('gameDate')
            if t_games.empty: continue
            
            last = t_games.iloc[-1]
            last_dt = pd.to_datetime(last['gameDate'])
            last_lat = last.get('lat', 0)
            last_lon = last.get('long', 0)
            last_tz = last.get('timeZone', 0)
            
            # Current Game
            is_home = tid in schedule['teamId_home'].values
            if is_home:
                # My Arena
                ainfo = tb.df_arenas[tb.df_arenas['teamId'] == tid]
                row_sched = schedule[schedule['teamId_home'] == tid].iloc[0]
                opp_tid = row_sched['teamId_away']
            else:
                # Host Arena (for travel metrics)
                row_sched = schedule[schedule['teamId_away'] == tid].iloc[0]
                host = row_sched['teamId_home']
                ainfo = tb.df_arenas[tb.df_arenas['teamId'] == host]
                opp_tid = host
                
            if ainfo.empty: continue
            
            curr_lat = ainfo['lat'].values[0]
            curr_lon = ainfo['long'].values[0]
            curr_alt = ainfo['altitude'].values[0]
            curr_tz = ainfo['tz_offset'].values[0]
            
            dist = tb._haversine_np(last_lat, last_lon, curr_lat, curr_lon)
            today = pd.to_datetime(datetime.datetime.now().strftime('%Y-%m-%d'))
            rest = max(0, (today - last_dt).days - 1)
            tz_delta = curr_tz - last_tz
            
            # Altitude Diff Calculation
            # Need My Home Altitude and Opp Home Altitude
            try:
                my_home_alt = tb.df_arenas.loc[tb.df_arenas['teamId'] == tid, 'altitude'].values[0]
                opp_home_alt = tb.df_arenas.loc[tb.df_arenas['teamId'] == opp_tid, 'altitude'].values[0]
                
                if is_home:
                     alt_diff = my_home_alt - opp_home_alt
                else:
                     alt_diff = opp_home_alt - my_home_alt
            except:
                alt_diff = 0
            
            feats.append({
                'teamId': tid,
                'dist_miles': dist,
                'tz_delta': tz_delta,
                'body_clock_lag': 0,
                'eff_rest_days': min(rest, 3),
                'altitude': curr_alt,
                'game_altitude_diff': alt_diff,
                'road_trip_count': 0
            })
            
        return pd.DataFrame(feats).set_index('teamId')

    def get_training_consistent_roster_features(self):
        """roster_bpm / roster_bpm_ewma computed exactly as in the training set (previous-game roster)."""
        seasons = sorted(pd.read_sql("SELECT DISTINCT seasonYear FROM team_game_full",
                                     sqlite3.connect(self.db_path))['seasonYear'])
        return PlayerFeatureBuilder(self.db_path).next_game_roster_features(seasons)

    def get_injury_features(self, games_df):
        """
        inj_* features for today's games from the official NBA injury report, built with the same
        function and snapshot as the training set (models/injury_report_features.py). The cutoff
        is min(training snapshot, now), so a run before the snapshot time uses the latest report.
        """
        from scripts.scrape_injury_reports import live as fetch_report, now_et
        games_df = games_df.dropna(subset=['teamId_home', 'teamId_away'])  # games SBR did not list
        game_date = pd.Timestamp(games_df['gameDate'].iloc[0]).normalize()
        season = season_for_date(game_date)
        # Fetch the report the training snapshot would use: latest at/before the snapshot time
        # (or now, if earlier). Training caps the cutoff at tip - 30 min; for normal evening tips
        # the clock time is earlier, so the cap only matters for early tips (handled in training).
        clock = irf.SNAPSHOT_CLOCK[GameFeatureBuilder.INJURY_SNAPSHOT]
        now = now_et()
        cutoff = min(now, game_date + clock) if clock is not None else now
        report_path = fetch_report(as_of=cutoff)  # saves into data/raw/injury_reports
        logging.info(f"Injury report for snapshot {GameFeatureBuilder.INJURY_SNAPSHOT} (cutoff {cutoff:%Y-%m-%d %H:%M} ET): {report_path}")
        reports = irf.parse_all_reports()
        upcoming = pd.concat([
            pd.DataFrame({'gameId': games_df['gameId'].astype(str), 'teamId': games_df['teamId_home'].astype(int)}),
            pd.DataFrame({'gameId': games_df['gameId'].astype(str), 'teamId': games_df['teamId_away'].astype(int)}),
        ], ignore_index=True)
        upcoming['gameDate'] = game_date.strftime('%Y-%m-%d')
        upcoming['seasonYear'] = season
        inj = irf.build_injury_features(self.db_path, [season], snapshot_name=GameFeatureBuilder.INJURY_SNAPSHOT,
                                        as_of=now, upcoming=upcoming, reports=reports)
        inj = inj[inj['gameId'].isin(upcoming['gameId'])]
        logging.info(f"Injury features for {len(inj)} team-games; missing report for {int(inj['inj_missing'].sum())}")
        return inj.set_index(['gameId', 'teamId'])

    def get_roster_features(self):
        """
        Get injury-adjusted roster BPM for all teams.
        Combines:
        1. Historical EWMA of team roster_bpm (from DB)
        2. Today's injury adjustments (from LiveRotationBuilder)
        """
        logging.info("Building Roster Features (injury-adjusted)...")
        
        # 1. Get historical roster_bpm_ewma from player_bpm table
        historical_ewma = self._get_historical_roster_bpm_ewma()
        
        # 2. Get today's injury-adjusted roster_bpm
        rb = LiveRotationBuilder(self.db_path)
        today_roster = rb.build_roster_features(season=current_season)
        
        if today_roster.empty:
            logging.warning("No roster features built")
            return pd.DataFrame()
        
        # 3. Combine: use today's injury-adjusted BPM, but historical EWMA
        if not historical_ewma.empty:
            today_roster = today_roster.join(historical_ewma, how='left')
            today_roster['roster_bpm_ewma'] = today_roster['roster_bpm_ewma'].fillna(today_roster['roster_bpm'])
        else:
            today_roster['roster_bpm_ewma'] = today_roster['roster_bpm']
        
        logging.info(f"Roster features built for {len(today_roster)} teams")
        
        return today_roster[['roster_bpm', 'roster_bpm_ewma']]


    def _get_historical_roster_bpm_ewma(self):
        """
        Calculate team-level roster_bpm EWMA from player_bpm table.
        This matches the training methodology.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get team roster_bpm for each game
                query = """
                    SELECT 
                        teamId,
                        gameDate,
                        SUM(bpm * e_minutes) / NULLIF(SUM(e_minutes), 0) as game_roster_bpm
                    FROM player_bpm
                    WHERE gameDate >= date('now', '-90 days')
                    GROUP BY teamId, gameDate
                    ORDER BY teamId, gameDate
                """
                df = pd.read_sql(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            df['gameDate'] = pd.to_datetime(df['gameDate'])
            df = df.sort_values(['teamId', 'gameDate'])
            
            # Calculate EWMA per team
            df['roster_bpm_ewma'] = df.groupby('teamId')['game_roster_bpm'].transform(
                lambda x: x.ewm(span=10, adjust=False).mean()
            )
            
            # Get latest EWMA per team
            latest = df.groupby('teamId')['roster_bpm_ewma'].last()
            
            return latest.to_frame()
            
        except Exception as e:
            logging.warning(f"Could not get historical roster EWMA: {e}")
            return pd.DataFrame()


    def _get_model_cols(self, model):
        if not model: return []
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_.tolist()
        elif hasattr(model, 'feature_name_'):
            return model.feature_name_
        elif hasattr(model, 'booster_'):
            return model.booster_.feature_name()
        return []

    def verify_feature_alignment(self, row_df, model_name, expected_cols):
        """Check and report feature misalignment."""
        available = set(row_df.columns)
        expected = set(expected_cols)
        
        missing = expected - available
        
        if missing:
            logging.warning(f"{model_name}: Missing {len(missing)} features: {list(missing)[:5]}...")
            # Fill missing with 0 or NaN
            for col in missing:
                row_df[col] = 0.0
        
        # Ensure correct order
        return row_df[expected_cols]

    def run(self):
        logging.info("Starting Inference Run...")
        
        # 1. Get Schedule
        games_df = self.get_todays_games()
        if games_df.empty:
            logging.info("No games found today.")
            return

        # 2. Build Features
        logging.info("Building Live Features...")
        
        # Team Features
        tb = TeamFeatureBuilder(self.db_path)
        team_feats = tb.get_latest_team_features()
        
        # Travel Features
        trb = TravelFeatureBuilder(self.db_path)
        travel_feats = trb.get_travel_features(games_df)
        
        # Roster features: computed exactly as in training (previous-game roster). Injuries enter
        # through the inj_* features below, not through an injury-adjusted roster_bpm (training
        # never saw an injury-adjusted roster_bpm, so mixing them was a train/inference mismatch).
        try:
            roster_feats = self.get_training_consistent_roster_features()
            logging.info(f"Roster Feats Shape: {roster_feats.shape}")
        except Exception as e:
            logging.warning(f"Roster features skipped: {e}")
            roster_feats = pd.DataFrame()

        # Injury report features (same code path and snapshot as the training set)
        try:
            inj_feats = self.get_injury_features(games_df)
        except Exception as e:
            # Same values training uses for a team with no report: inj_missing=1, zeros elsewhere
            logging.warning(f"Injury features unavailable, treating as missing report: {e}")
            inj_feats = pd.DataFrame()
            
        # 3. Predict Loop
        preds = []
        
        # Determine Model Columns (Robust Fallback)
        cols_ml_model = self._get_model_cols(self.model_ml)
        cols_spread_model = self._get_model_cols(self.model_spread)
        cols_total_model = self._get_model_cols(self.model_total)

        logging.info(f"ML Model expects {len(cols_ml_model)} features.")
        logging.info(f"Spread Model expects {len(cols_spread_model)} features.")
        logging.info(f"Total Model expects {len(cols_total_model)} features.")

        # Set index for travel feats to allow lookup by Team Name
        if not travel_feats.empty and 'team' in travel_feats.columns:
            travel_feats = travel_feats.set_index('team')

        for _, g in games_df.iterrows():
            # Build Row using a dictionary to prevent fragmentation
            hid = tb.team_map.get(g['home_team'])
            aid = tb.team_map.get(g['away_team'])
            
            if not hid or not aid:
                logging.warning(f"Team ID not found for {g['home_team']} or {g['away_team']}")
                continue
                
            row = {}
            
            def safe_float(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return np.nan

            # Add Odds (Base & Open)
            row['odds_home_ml'] = g.get('odds_home_ml')
            row['odds_away_ml'] = g.get('odds_away_ml')
            row['odds_home_spread'] = safe_float(g.get('odds_home_spread'))
            row['odds_away_spread'] = safe_float(g.get('odds_away_spread'))
            row['odds_total'] = safe_float(g.get('odds_total'))
            
            # Add Open Odds
            for open_key in ['odds_home_ml_open', 'odds_away_ml_open', 'odds_home_spread_open', 'odds_away_spread_open', 'odds_total_open']:
                row[open_key] = safe_float(g.get(open_key))


            # Home Feats
            if hid in team_feats:
                for k,v in team_feats[hid].items(): row[f"{k}_home"] = v
            else:
                logging.warning(f"No team stats for Home ID {hid}")

            # travel_feats
            if g['home_team'] in travel_feats.index:
                for k,v in travel_feats.loc[g['home_team']].items(): 
                     row[f"{k}_home"] = v
            
        # Roster/BPM Logic
            if hid in roster_feats.index:
                row['roster_bpm_home'] = roster_feats.loc[hid, 'roster_bpm']
                row['roster_bpm_ewma_x'] = roster_feats.loc[hid, 'roster_bpm_ewma']
            else:
                row['roster_bpm_home'] = 0.0 # Default/Neutral
                row['roster_bpm_ewma_x'] = 0.0
            
            if aid in roster_feats.index:
                row['roster_bpm_away'] = roster_feats.loc[aid, 'roster_bpm']
                row['roster_bpm_ewma_y'] = roster_feats.loc[aid, 'roster_bpm_ewma']
            else:
                row['roster_bpm_away'] = 0.0
                row['roster_bpm_ewma_y'] = 0.0

            # Injury report features; when unavailable, the values training uses for a team with
            # no report (inj_missing=1, zeros)
            for side, tid in (('home', hid), ('away', aid)):
                key = (str(g['gameId']), int(tid))
                have = not inj_feats.empty and key in inj_feats.index
                for c in irf.INJ_COLS:
                    row[f'{c}_{side}'] = inj_feats.loc[key, c] if have else (1 if c == 'inj_missing' else 0.0)

            # Away Feats
            if aid in team_feats:
                for k,v in team_feats[aid].items(): row[f"{k}_away"] = v
            else:
                logging.warning(f"No team stats for Away ID {aid}")

            if g['away_team'] in travel_feats.index:
                for k,v in travel_feats.loc[g['away_team']].items(): 
                     row[f"{k}_away"] = v
                
            # Create DF from the fully populated dictionary to avoid fragmentation
            
            # --- DIFFERENTIAL & MATCHUP FEATURES (Calculated directly in Dict) ---
            home_feat_keys = [k for k in row.keys() if k.endswith('_home') and not k.startswith(('teamId', 'matchup', 'wl', 'points', 'gameId', 'is_home', 'seasonYear', 'gameDate'))]
            for hcol in home_feat_keys:
                base = hcol.rsplit('_home', 1)[0]
                acol = f"{base}_away"
                if acol in row:
                    val = row[hcol]
                    if isinstance(val, (int, float, np.number)):
                        row[f"diff_{base}"] = val - row[acol]

            # Generate specific matchup features (home offense vs away defense, etc.)
            matchup_pairs = [
                ('off_offensiveRating_5g_home', 'def_defensiveRating_5g_away', 'matchup_off_vs_def_rating_5g'),
                ('off_offensiveRating_20g_home', 'def_defensiveRating_20g_away', 'matchup_off_vs_def_rating_20g'),
                ('off_threePointersAttempted_5g_home', 'def_OPP_FG3A_5g_away', 'matchup_3pa_vs_opp_3pa_5g'),
            ]
            
            for home_col, away_col, name in matchup_pairs:
                if home_col in row and away_col in row:
                    row[name] = row[home_col] - row[away_col]
                    
            row_df = pd.DataFrame([row])
            
            # Sanitization: All features must use underscores to match FLAML/LGBM model map
            row_df.columns = [str(c).replace(' ', '_') for c in row_df.columns]
            
            # --- ML PREDICTION ---
            cols_ml = self.selected_features.get('ml', [])
            if not cols_ml or len(cols_ml) > 800:
                cols_ml = cols_ml_model # Fallback to model features
            
            # VERIFY ALIGNMENT
            row_df_ml = self.verify_feature_alignment(row_df.copy(), 'ML', cols_ml)
            
            try:
                prob = self.model_ml.predict_proba(row_df_ml)[0, 1]
            except Exception as e:
                logging.error(f"ML Prediction Failed: {e}")
                prob = 0.5
            
            # Apply Calibration if available
            if self.calibrator_ml:
                prob = self.calibrator_ml.predict([prob])[0]
            
            # --- SPREAD PREDICTION ---
            cols_spread = self.selected_features.get('spread', [])
            if not cols_spread or len(cols_spread) > 800:
                cols_spread = cols_spread_model
                
            # VERIFY ALIGNMENT
            row_df_spread = self.verify_feature_alignment(row_df.copy(), 'Spread', cols_spread)

            pred_spread = 0.0
            if self.model_spread:
                try:
                    pred_spread = self.model_spread.predict(row_df_spread)[0]
                except Exception as e:
                     logging.error(f"Spread Prediction Failed: {e}")

            # --- TOTAL PREDICTION ---
            cols_total = self.selected_features.get('total', [])
            if not cols_total or len(cols_total) > 800:
                cols_total = cols_total_model
                
            # VERIFY ALIGNMENT
            row_df_total = self.verify_feature_alignment(row_df.copy(), 'Total', cols_total)
            
            pred_total = 0.0
            if self.model_total:
                try:
                    pred_total = self.model_total.predict(row_df_total)[0]
                except Exception as e:
                     logging.error(f"Total Prediction Failed: {e}")

            # Get Odds
            line = g.get('odds_home_spread', 0)
            total_line = g.get('odds_total', 0)
            ml_home = g.get('odds_home_ml', 0)
            ml_away = g.get('odds_away_ml', 0)
            
            # Implied Prob Logic
            def to_decimal(us_odds):
                try:
                    us = float(us_odds)
                    if us > 0: return round(1 + (us / 100), 2)
                    else: return round(1 + (100 / abs(us)), 2)
                except:
                    return 0.0

            dec_home = to_decimal(ml_home)
            dec_away = to_decimal(ml_away)
            
            imp_home_str = ""
            if dec_home > 0:
                imp_home = (1/dec_home)
                imp_home_str = f"{imp_home:.1%}"
            else:
                est_margin = -1 * float(line) if line else 0
                est_prob = stats.norm.cdf(est_margin / 12.0)
                imp_home_str = f"{est_prob:.1%} (Est)"
                item_ml_dec = 0.0
            
            # --- Calibration Probabilities ---
            std_spread = self.calib_meta.get('spread_std', 12.0)
            std_total = self.calib_meta.get('total_std', 18.0)
            
            spread_edge = pred_spread + line
            spread_cover_prob = stats.norm.cdf(spread_edge / std_spread)
            
            if spread_cover_prob > 0.5:
                pick_spread_team = 'Home'
                pick_spread_prob = spread_cover_prob
            else:
                pick_spread_team = 'Away'
                pick_spread_prob = 1.0 - spread_cover_prob
            
            # Total Prob (Over)
            total_edge = pred_total - total_line
            prob_over = stats.norm.cdf(total_edge / std_total)
            
            if prob_over > 0.5:
                pick_total_type = 'Over'
                pick_total_prob = prob_over
            else:
                pick_total_type = 'Under'
                pick_total_prob = 1.0 - prob_over

            # Log Prediction
            try:
                log_entry = {
                    'gameId': g['gameId'],
                    'gameDate': g['gameDate'],
                    'home_team': g['home_team'],
                    'away_team': g['away_team'],
                    'pred_home_win_prob': prob, # Calibrated Prob
                    'pred_spread': pred_spread,
                    'pred_total': pred_total,
                    'line_spread': line,
                    'line_total': total_line,
                    'line_ml_home': ml_home, # Raw American Odds
                    'line_ml_away': ml_away,
                    'model_version': 'v1.0.0_hybrid',
                }
                self.logger.log_prediction(log_entry)
            except Exception as e:
                logging.error(f"Failed to log prediction: {e}")

            # Recommendation Logic
            # User Criteria: 
            # - Over: Edge >= 5 (Higher confidence if >= 7)
            # - Under: Edge >= 7 (High confidence only)
            
            raw_edge_over = pred_total - total_line
            recommendation = ""
            
            if raw_edge_over >= 7.0:
                recommendation = "STRONG OVER"
            elif raw_edge_over >= 5.0:
                recommendation = "BET OVER"
            elif raw_edge_over <= -7.0:
                recommendation = "STRONG UNDER"
                
            preds.append({
                'game_id': g['gameId'],
                'game_date': g['gameDate'],
                'prediction_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'home_team': g['home_team'],
                'away_team': g['away_team'],
                'matchup': f"{g['away_team']} @ {g['home_team']}",
                'ml_prob': prob,
                'spread_pred': round(pred_spread, 1),
                'total_pred': round(pred_total, 1),
                'odds_home_ml': ml_home,
                'odds_away_ml': ml_away,
                'odds_home_spread': line,
                'odds_total': total_line,
                'home_score': np.nan, # Pending
                'away_score': np.nan, # Pending
                'status': 'PENDING',
                'recommendation': recommendation # Keeping as extra column for CSV utility
            })
            
        res = pd.DataFrame(preds)
        print("\n--- PREDICTIONS ---")
        print(res.to_string(index=False))
        res.to_csv('todays_picks.csv', index=False)


if __name__ == "__main__":

    p = Path(__file__).parent.parent / 'models' / 'saved' / 'model_ml_with_odds_v2_injury.pkl'
    
    seasons_to_check = [current_season]
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    
    syncer = NBADatabaseSync(db_path)

    syncer.sync_database(seasons_to_check)

    engine = InferenceEngine(db_path, p)

    engine.run()
