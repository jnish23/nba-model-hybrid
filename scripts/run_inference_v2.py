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
from models.live_odds import split_open_and_current
from models.live_features import align_features, build_live_game_frame
from models.model_registry import DEPLOYED_SUFFIX, MODEL_DIR, load_bundle
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
    def __init__(self, db_path, suffix=DEPLOYED_SUFFIX, model_dir=MODEL_DIR, target_date=None):
        self.db_path = db_path
        self.target_date = target_date  # Optional: override datetime.now() for backfilling

        # One explicit, validated model set (models/model_registry.py): all pickles, calibrator and
        # the feature config share `suffix`, and the config is checked against each pickle.
        bundle = load_bundle(suffix, model_dir)
        self.bundle = bundle
        self.model_ml = bundle.ml
        self.calibrator_ml = bundle.calibrator
        self.model_spread = bundle.spread
        self.model_total = bundle.total
        self.selected_features = bundle.features
        self.calib_meta = bundle.calib_meta

        # Initialize Logger
        self.logger = PredictionLogger(db_path) 
        logging.info(f"Inference Engine initialized with model set '{suffix}'.")
        
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
            from models.sbr import Scoreboard
            
            # Fetch Current
            sb_curr = Scoreboard(sport='NBA', date=today, current_line=True)
            if not sb_curr.games:
                logging.warning("No games found on SBR (Current).")
                return pd.DataFrame()
                
            # Fetch Open
            sb_open = Scoreboard(sport='NBA', date=today, current_line=False)
            
            # Opening lines are the model inputs (odds_*), current lines are kept for bets
            # (curr_odds_*): see models/live_odds.py
            open_map = {}
            if sb_open.games:
                for g in sb_open.games:
                    open_map[f"{g.get('away_team')} @ {g.get('home_team')}"] = g

            games = []
            for g in sb_curr.games:
                key = f"{g.get('away_team')} @ {g.get('home_team')}"
                if key not in open_map:
                    logging.warning(f"No opening line on SBR for {key}; the current line stands in as model input")
                row = {
                    'gameDate': today,
                    'away_team': g.get('away_team'),
                    'home_team': g.get('home_team'),
                    'away_team_abbr': g.get('away_team_abbr'),
                    'home_team_abbr': g.get('home_team_abbr'),
                    **split_open_and_current(g, open_map.get(key)),
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


    def verify_feature_alignment(self, row_df, model_name, expected_cols):
        """Select the model's columns in order; missing features become NaN and are logged loudly."""
        return align_features(row_df, expected_cols, model_name)

    def run(self):
        logging.info("Starting Inference Run...")
        
        # 1. Get Schedule
        games_df = self.get_todays_games()
        if games_df.empty:
            logging.info("No games found today.")
            return
        unmapped = games_df[games_df[['teamId_home', 'teamId_away']].isna().any(axis=1)]
        for _, g in unmapped.iterrows():
            logging.warning(f"Team ID not found for {g.get('home_team')} or {g.get('away_team')}; skipping game {g.get('gameId')}")
        games_df = games_df.dropna(subset=['teamId_home', 'teamId_away']).reset_index(drop=True)
        if games_df.empty:
            logging.info("No games with resolvable teams today.")
            return

        # 2. Build the model inputs with the same code as the training set (models/live_features.py):
        #    team form, travel, roster BPM, injury report, differences and matchups
        logging.info("Building Live Features...")
        try:
            inj_feats = self.get_injury_features(games_df).reset_index()
        except Exception as e:
            # Same values training uses for a team with no report: inj_missing=1, zeros elsewhere
            logging.warning(f"Injury features unavailable, treating as missing report: {e}")
            inj_feats = None
        frame = build_live_game_frame(self.db_path, games_df, injury_features=inj_feats)

        # 3. Predict (model columns come from the validated feature config)
        cols_ml = self.selected_features['ml']
        cols_spread = self.selected_features['spread']
        cols_total = self.selected_features['total']
        logging.info(f"Models expect {len(cols_ml)} / {len(cols_spread)} / {len(cols_total)} features (ML / Spread / Total).")
        prob_raw = self.model_ml.predict_proba(self.verify_feature_alignment(frame, 'ML', cols_ml))[:, 1]
        probs = self.calibrator_ml.predict(prob_raw)
        pred_spreads = self.model_spread.predict(self.verify_feature_alignment(frame, 'Spread', cols_spread))
        pred_totals = self.model_total.predict(self.verify_feature_alignment(frame, 'Total', cols_total))

        preds = []
        for i, g in games_df.iterrows():
            prob, pred_spread, pred_total = float(probs[i]), float(pred_spreads[i]), float(pred_totals[i])

            # Tradable (current) lines for edges, EV and the prediction log
            line = float(g['curr_odds_home_spread']) if pd.notna(g.get('curr_odds_home_spread')) else np.nan
            total_line = float(g['curr_odds_total']) if pd.notna(g.get('curr_odds_total')) else np.nan
            ml_home = float(g['curr_odds_home_ml']) if pd.notna(g.get('curr_odds_home_ml')) else np.nan
            ml_away = float(g['curr_odds_away_ml']) if pd.notna(g.get('curr_odds_away_ml')) else np.nan

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict today's NBA games with the deployed model set")
    parser.add_argument('--suffix', default=DEPLOYED_SUFFIX,
                        help=f"model set suffix in models/saved (default: deployed set {DEPLOYED_SUFFIX})")
    parser.add_argument('--date', default=None, help='YYYY-MM-DD to backfill instead of today (ET)')
    args = parser.parse_args()

    seasons_to_check = [current_season]
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    
    syncer = NBADatabaseSync(db_path)

    try:
        syncer.sync_database(seasons_to_check)
    except Exception as e:
        # Predict anyway, but make the staleness visible in the log
        with sqlite3.connect(db_path) as conn:
            last = conn.execute("SELECT MAX(gameDate) FROM team_game_full").fetchone()[0]
        logging.error(f"DATABASE SYNC FAILED ({e}). Predicting with data through {last}; "
                      f"features may be stale.")

    engine = InferenceEngine(db_path, suffix=args.suffix, target_date=args.date)

    engine.run()


if __name__ == "__main__":
    main()
