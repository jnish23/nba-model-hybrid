import pandas as pd
import numpy as np
import sqlite3
import logging
import datetime
import sys
import os
from pathlib import Path 
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.prediction_logger import PredictionLogger
from team.live_odds import split_open_and_current
from team.live_features import align_features, build_live_game_frame
from team.betting import (get_ml_bet, get_spread_bet, get_total_bet, DEFAULT_SPREAD_STD,
                            DEFAULT_TOTAL_STD)
from team.model_registry import DEPLOYED_SUFFIX, MODEL_DIR, load_bundle
from core import injury_report_features as irf
from team.game_features_v2 import GameFeatureBuilder
from core.update_database import NBADatabaseSync
from nba_api.stats.endpoints import scoreboardv3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from core.season_utils import season_for_date

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

        # One explicit, validated model set (team/model_registry.py): all pickles, calibrator and
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
        from core.scrape_injury_reports import now_et
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
            from core.sbr import Scoreboard
            
            # Fetch Current
            sb_curr = Scoreboard(sport='NBA', date=today, current_line=True)
            if not sb_curr.games:
                logging.warning("No games found on SBR (Current).")
                return pd.DataFrame()
                
            # Fetch Open
            sb_open = Scoreboard(sport='NBA', date=today, current_line=False)
            
            # Opening lines are the model inputs (odds_*), current lines are kept for bets
            # (curr_odds_*): see team/live_odds.py
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
                except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
                    logging.warning(f"team_basic_stats has no camelCase columns ({e}); trying the upper-case names")
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

    def get_injury_features(self, games_df):
        """
        inj_* features for today's games from the official NBA injury report, built with the same
        function and snapshot as the training set (core/injury_report_features.py). The cutoff
        is min(training snapshot, now), so a run before the snapshot time uses the latest report.
        """
        from core.scrape_injury_reports import live as fetch_report, now_et
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
        if report_path is None:
            logging.error("Injury report download FAILED; using only the reports already on disk")
        logging.info(f"Injury report for snapshot {GameFeatureBuilder.INJURY_SNAPSHOT} (cutoff {cutoff:%Y-%m-%d %H:%M} ET): {report_path}")
        reports = irf.parse_all_reports()      # re-scans the report folder: picks up the PDF just fetched
        latest, age, stale = irf.check_report_freshness(reports, game_date, cutoff)
        if stale:
            logging.error(f"STALE INJURY DATA: newest report for {game_date:%Y-%m-%d} is "
                          f"{'missing' if latest is None else f'{age:.1f} h older than the {cutoff:%H:%M} ET cutoff ({latest})'}; "
                          f"teams without a covering report are flagged inj_missing=1")
        else:
            logging.info(f"Newest injury report for {game_date:%Y-%m-%d}: {latest} ({age:.1f} h before the cutoff)")
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

        # 2. Build the model inputs with the same code as the training set (team/live_features.py):
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

        spread_std = self.calib_meta.get('spread_std', DEFAULT_SPREAD_STD)
        total_std = self.calib_meta.get('total_std', DEFAULT_TOTAL_STD)
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

            # Recommendations: ONE rule for all markets (team/betting.py, shared with the backtests and
            # the dashboard): bet when the model's win probability beats the price's break-even
            # probability by more than DEFAULT_PROB_EDGE. Spread/total win probability is
            # Phi(point edge / residual std) with the std saved with the model set.
            ml_bet, ml_edge = get_ml_bet({'ml_prob': prob, 'odds_home_ml': ml_home, 'odds_away_ml': ml_away})
            spread_bet, spread_edge = get_spread_bet(pred_spread, line, spread_std)
            total_bet, total_edge = get_total_bet(pred_total, total_line, total_std)
            recommendation = "; ".join(
                f"{market} {side.upper()} ({edge:+.1%})"
                for market, side, edge in (('ML', ml_bet, ml_edge), ('SPREAD', spread_bet, spread_edge),
                                           ('TOTAL', total_bet, total_edge))
                if side)

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
                'ml_bet': ml_bet,
                'ml_edge': round(ml_edge, 4),
                'spread_bet': spread_bet,
                'spread_edge': round(spread_edge, 4),
                'total_bet': total_bet,
                'total_edge': round(total_edge, 4),
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
