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
from core.odds_rerun import unpriced_games
from core.odds_fallback import SOURCE, fetch_team_odds, find_game
from team.live_odds import MARKETS, split_open_and_current
from team.live_features import align_features, build_live_game_frame
from team.betting import (get_ml_bet, get_spread_bet, get_total_bet, DEFAULT_SPREAD_STD,
                            DEFAULT_TOTAL_STD)
from team.model_registry import DEPLOYED_SUFFIX, MODEL_DIR, load_bundle
from team.market_models import predict_markets
from team.explain import build_reference, explain_markets, load_oos_history, save_explanations
from team.data_paths import latest_training_set
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

        return games[['gameId', 'gameDate', 'gameStatus', 'gameTimeUTC', 'home_team_abbr', 'away_team_abbr']]
    except Exception as e:
        logging.error(f"Failed to fetch NBA schedule: {e}")
        return pd.DataFrame()

class InferenceEngine:
    def __init__(self, db_path, suffix=DEPLOYED_SUFFIX, model_dir=MODEL_DIR, target_date=None, only_unpriced=False):
        self.db_path = db_path
        self.only_unpriced = only_unpriced  # re-run: predict only the games that have not started and still lack a line
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

            # Games SBR lists without a usable line are filled from The Odds API (core/odds_fallback.py); one bulk call covers them all
            api_games = None
            games = []
            for g in sb_curr.games:
                key = f"{g.get('away_team')} @ {g.get('home_team')}"
                lines = split_open_and_current(g, open_map.get(key))
                source = 'sbr'
                if not any(pd.notna(lines[f'{prefix}{m}']) for m in MARKETS for prefix in ('odds_', 'curr_odds_')):
                    # SBR has nothing usable for the game (no books, or none passing the cross-book checks): ask the API
                    if api_games is None:
                        api_games = fetch_team_odds()
                    backup = find_game(api_games, g.get('away_team'), g.get('home_team'))
                    if backup is not None:
                        lines = split_open_and_current(backup, backup)    # the API only has the current line: it stands in for the open too
                        source = SOURCE
                        logging.info(f"No usable SBR lines for {key}; using The Odds API")
                    else:
                        source = 'none'
                elif key not in open_map:
                    logging.warning(f"No opening line on SBR for {key}; the game is skipped (the current line is not a model input)")
                row = {
                    'gameDate': today,
                    'away_team': g.get('away_team'),
                    'home_team': g.get('home_team'),
                    'away_team_abbr': g.get('away_team_abbr'),
                    'home_team_abbr': g.get('home_team_abbr'),
                    'odds_source': source,
                    **lines,
                }
                games.append(row)

            df = pd.DataFrame(games)
            incomplete = df['odds_open_missing'] > 0
            if incomplete.any():
                logging.warning("Skipping games without an opening line: "
                                + ", ".join(f"{a} @ {h}" for a, h in zip(df.loc[incomplete, 'away_team'], df.loc[incomplete, 'home_team'])))
                df = df[~incomplete].reset_index(drop=True)
            if df.empty:
                return pd.DataFrame()

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
            self.log_odds_sources(merged)
            
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
        # Informational only (never a model input): noon-to-latest status downgrades, and both injury sources side by side
        self.downgrades = {}
        try:
            from core.injury_downgrades import build_downgrade_features
            d, _, _ = build_downgrade_features(self.db_path, [season], as_of=now, upcoming=upcoming, reports=reports)
            d = d[d['gameId'].isin(upcoming['gameId'])]
            self.downgrades = {(r.gameId, int(r.teamId)): r.inj_downgrade_lost for r in d.itertuples()}
        except Exception as e:
            logging.warning(f"Downgrade flags unavailable: {e}")
        from core.injury_source_compare import log_side_by_side
        log_side_by_side(self.db_path, game_date, reports=reports, as_of=now)
        return inj.set_index(['gameId', 'teamId'])

    def verify_feature_alignment(self, row_df, model_name, expected_cols):
        """Select the model's columns in order; missing features become NaN and are logged loudly."""
        return align_features(row_df, expected_cols, model_name)

    def explain_games(self, frame, games_df):
        """
        Why each spread/total differs from the current line (team/explain.py), stored in prediction_explanations next to the
        predictions. An explanation problem is logged and never stops the daily run.
        """
        try:
            feats = sorted({f for cols in self.selected_features.values() for f in cols})
            ref = None
            try:
                train = pd.read_csv(latest_training_set(), usecols=lambda c: c in feats or c == 'seasonYear')
                ref = build_reference(train[train['seasonYear'].isin(sorted(train['seasonYear'].unique())[-3:])], feats)
            except Exception as e:
                logging.warning(f"No league-wide reference for the explanations (percentiles left empty): {e}")
            lines = games_df.assign(gameId=games_df['gameId'].astype(str)).set_index('gameId').reindex(frame['gameId'].astype(str))
            current = pd.DataFrame({'spread_line': pd.to_numeric(lines['curr_odds_home_spread'], errors='coerce').to_numpy(),
                                    'total_line': pd.to_numeric(lines['curr_odds_total'], errors='coerce').to_numpy(),
                                    'home_ml': pd.to_numeric(lines['curr_odds_home_ml'], errors='coerce').to_numpy()})
            expl = explain_markets(self.bundle, frame, current, align=self.verify_feature_alignment, reference=ref,
                                   history=load_oos_history(), prediction_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            n = save_explanations(self.db_path, expl)
            logging.info(f"Stored {n} explanation rows for {expl['game_id'].nunique()} games")
        except Exception as e:
            logging.error(f"Explanations failed (predictions are unaffected): {e}", exc_info=True)

    def log_odds_sources(self, games):
        """Which source priced each game today (sbr / theoddsapi / none), kept in odds_source_log so the backup's use can be reviewed."""
        try:
            rows = [(str(r.gameId), str(r.gameDate), r.odds_source, datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
                    for r in games.itertuples() if isinstance(getattr(r, 'odds_source', None), str)]
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS odds_source_log (game_id TEXT, game_date TEXT, source TEXT, ts_utc TEXT, "
                             "PRIMARY KEY (game_id, ts_utc))")
                conn.executemany("INSERT OR IGNORE INTO odds_source_log VALUES (?,?,?,?)", rows)
            used = [r[0] for r in rows if r[2] == SOURCE]
            if used:
                logging.info(f"{len(used)} game(s) priced by The Odds API instead of SBR")
        except Exception as e:                       # bookkeeping only: never stops the run
            logging.warning(f"Could not log the odds sources: {e}")

    def keep_unpriced(self, games_df):
        """
        Games not yet started (gameStatus 1) whose LOGGED prediction lacks the moneyline, spread or total: the ones a re-run can improve.
        It goes by the log, not the lines fetched now: a game the backup priced a minute ago still has its old no-odds row to replace.
        """
        logged = set(unpriced_games(self.db_path, games_df['gameDate'].iloc[0])['game_id'])
        not_started = pd.to_numeric(games_df['gameStatus'], errors='coerce') == 1
        keep = games_df[games_df['gameId'].astype(str).isin(logged) & not_started].reset_index(drop=True)
        logging.info(f"Re-run: {len(keep)} of {len(games_df)} games were logged without a line and have not started: "
                     + ", ".join(f"{a} @ {h}" for a, h in zip(keep['away_team'], keep['home_team'])))
        return keep

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
        if self.only_unpriced:
            games_df = self.keep_unpriced(games_df)
            if games_df.empty:
                logging.info("Re-run: every game that has not started already has its lines; nothing to predict.")
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
        # models/market_models.predict_markets: handles both model-set generations (isotonic + plain regressors, or the
        # market-aware set: residual spread model, margin-stack ML calibrator, per-game total sigma)
        out = predict_markets(self.bundle, frame, align=self.verify_feature_alignment)
        probs, pred_spreads, pred_totals = out['ml_prob'].to_numpy(), out['spread_pred'].to_numpy(), out['total_pred'].to_numpy()
        spread_stds = out['spread_std'].fillna(DEFAULT_SPREAD_STD).to_numpy()
        total_stds = out['total_std'].fillna(DEFAULT_TOTAL_STD).to_numpy()
        self.explain_games(frame, games_df)
        preds = []
        for i, g in games_df.iterrows():
            prob, pred_spread, pred_total = float(probs[i]), float(pred_spreads[i]), float(pred_totals[i])
            spread_std, total_std = float(spread_stds[i]), float(total_stds[i])

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
                    'model_set_hash': self.bundle.model_set_hash,       # ties the row to the exact files / training CSV
                    'training_set_hash': self.bundle.training_set_hash,
                }
                self.logger.log_prediction(log_entry)
            except Exception as e:
                logging.error(f"Failed to log prediction: {e}")

            # Recommendations: ONE rule for all markets (team/betting.py, shared with the backtests and
            # the dashboard): bet when the model's win probability beats the price's break-even
            # probability by more than DEFAULT_PROB_EDGE. Spread/total win probability is
            # Phi(point edge / residual std) with the std saved with the model set.
            ml_bet, ml_edge = get_ml_bet({'ml_prob': prob, 'odds_home_ml': ml_home, 'odds_away_ml': ml_away})
            def price(name):
                v = g.get(f'curr_price_{name}')
                return float(v) if v is not None and pd.notna(v) else None
            spread_bet, spread_edge = get_spread_bet(pred_spread, line, spread_std,
                                                     home_price=price('home_spread_odds'), away_price=price('away_spread_odds'))
            total_bet, total_edge = get_total_bet(pred_total, total_line, total_std,
                                                  over_price=price('over_odds'), under_price=price('under_odds'))
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
                'inj_downgrade_lost_home': getattr(self, 'downgrades', {}).get((str(g['gameId']), int(g['teamId_home'])), 0.0),
                'inj_downgrade_lost_away': getattr(self, 'downgrades', {}).get((str(g['gameId']), int(g['teamId_away'])), 0.0),
                'recommendation': recommendation # Keeping as extra column for CSV utility
            })
            
        res = pd.DataFrame(preds)
        print("\n--- PREDICTIONS ---")
        print(res.to_string(index=False))
        if self.only_unpriced and Path('todays_picks.csv').exists():
            # a re-run covers a few games: refresh their rows and keep the rest of the day's file
            earlier = pd.read_csv('todays_picks.csv', dtype={'game_id': str})
            res = pd.concat([earlier[~earlier['game_id'].isin(res['game_id'].astype(str))], res], ignore_index=True)
        res.to_csv('todays_picks.csv', index=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict today's NBA games with the deployed model set")
    parser.add_argument('--suffix', default=DEPLOYED_SUFFIX,
                        help=f"model set suffix in models/saved (default: deployed set {DEPLOYED_SUFFIX})")
    parser.add_argument('--date', default=None, help='YYYY-MM-DD to backfill instead of today (ET)')
    parser.add_argument('--only-unpriced', action='store_true',
                        help="re-run: skip the database sync and predict only today's not-yet-started games that still lack a line "
                             "(scripts/run_odds_rerun.bat, scheduled by scripts/schedule_odds_rerun.py)")
    args = parser.parse_args()

    seasons_to_check = [current_season]
    db_path = Path.cwd() / 'data' / 'nba_stats.db'

    if not args.only_unpriced:                 # a re-run uses the morning's sync
        syncer = NBADatabaseSync(db_path)
        try:
            syncer.sync_database(seasons_to_check)
        except Exception as e:
            # Predict anyway, but make the staleness visible in the log
            with sqlite3.connect(db_path) as conn:
                last = conn.execute("SELECT MAX(gameDate) FROM team_game_full").fetchone()[0]
            logging.error(f"DATABASE SYNC FAILED ({e}). Predicting with data through {last}; "
                          f"features may be stale.")

    engine = InferenceEngine(db_path, suffix=args.suffix, target_date=args.date, only_unpriced=args.only_unpriced)

    engine.run()


if __name__ == "__main__":
    main()
