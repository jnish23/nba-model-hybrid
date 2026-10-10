import pandas as pd
from pathlib import Path
import datetime
import logging
import sqlite3

class PredictionLogger:
    def __init__(self, db_path, log_path='predictions/predictions_2025-26.csv'):
        self.db_path = db_path
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize predictions_log table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions_log_v2 (
                    game_id TEXT PRIMARY KEY,
                    game_date TEXT,
                    prediction_date TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    matchup TEXT,
                    ml_prob REAL,
                    spread_pred REAL,
                    total_pred REAL,
                    odds_home_ml REAL,
                    odds_away_ml REAL,
                    odds_home_spread REAL,
                    odds_total REAL,
                    home_score INTEGER,
                    away_score INTEGER,
                    status TEXT DEFAULT 'PENDING'
                )
            """)
            
    def log_prediction(self, prediction: dict, overwrite: bool = True):
        """
        Log a prediction (UPSERT).
        """
        # Map input dict keys to db schema if necessary
        # Assuming input has keys matching our schema or we map them here
        
        # Normalize game id to canonical 10-char NBA id (pad with leading zeros)
        def _normalize_game_id(gid):
            if gid is None:
                return None
            s = str(gid)
            return s.zfill(10) if s.isdigit() else s

        row = {
            'game_id': _normalize_game_id(prediction.get('gameId')),
            'game_date': prediction.get('gameDate'),
            'prediction_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'home_team': prediction.get('home_team'),
            'away_team': prediction.get('away_team'),
            'matchup': f"{prediction.get('away_team')} @ {prediction.get('home_team')}",
            'ml_prob': prediction.get('pred_home_win_prob'),
            'spread_pred': prediction.get('pred_spread'),
            'total_pred': prediction.get('pred_total'),
            'odds_home_ml': prediction.get('line_ml_home'),
            'odds_away_ml': prediction.get('line_ml_away'), # Need to ensure this is passed
            'odds_home_spread': prediction.get('line_spread'),
            'odds_total': prediction.get('line_total'),
            'status': 'PENDING'
        }
        
        # 1. Save to DB
        try:
            with sqlite3.connect(self.db_path) as conn:
                cols = ', '.join(row.keys())
                placeholders = ', '.join(['?'] * len(row))
                verb = "INSERT OR REPLACE" if overwrite else "INSERT OR IGNORE"
                sql = f"{verb} INTO predictions_log_v2 ({cols}) VALUES ({placeholders})"
                conn.execute(sql, list(row.values()))
        except Exception as e:
            logging.error(f"Failed to log prediction to DB: {e}")
            
        # 2. Save to CSV (mirroring DB behavior to avoid duplicates)
        df_row = pd.DataFrame([row])
        if self.log_path.exists():
            # Read the CSV's game_ids once per logger, not on every call
            if getattr(self, '_csv_game_ids', None) is None:
                self._csv_game_ids = set(pd.read_csv(self.log_path, dtype=str, usecols=['game_id'])['game_id'])
            already_exists = row['game_id'] in self._csv_game_ids
            if not already_exists:
                df_row.to_csv(self.log_path, mode='a', header=False, index=False)
            elif overwrite:
                # Replace existing row (same as INSERT OR REPLACE) — only rewrite the file in this case
                existing = pd.read_csv(self.log_path, dtype=str)
                existing = existing[existing['game_id'] != row['game_id']]
                updated = pd.concat([existing, df_row.astype(str)], ignore_index=True)
                updated.to_csv(self.log_path, index=False)
            # else: skip (same as INSERT OR IGNORE)
        else:
            df_row.to_csv(self.log_path, index=False)
            self._csv_game_ids = set()
        self._csv_game_ids.add(row['game_id'])
            
        logging.info(f"Logged prediction for {row['matchup']}")

    def update_actual_scores(self):
        """
        Check for completed games in DB and update predictions_log.
        """
        logging.info("Checking for completed games to update validation logs...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Get pending games
            pending = pd.read_sql("SELECT game_id, game_date, home_team, away_team FROM predictions_log_v2 WHERE home_score IS NULL", conn)
            
            if pending.empty:
                return
                
            # Self-join team_basic_stats to get home/away scores per game
            query = """
                SELECT 
                    h.gameId,
                    h.gameDate,
                    h.teamId as home_teamId,
                    h.teamName AS home_team,
                    h.points AS home_score,
                    a.teamId as away_teamId,
                    a.teamName AS away_team,
                    a.points AS away_score
                FROM team_basic_stats h
                JOIN team_basic_stats a 
                    ON h.gameId = a.gameId 
                    AND h.teamId != a.teamId
                WHERE h.matchup LIKE '%vs.%'
                AND h.WL IS NOT NULL
            """

            try:
                results_df = pd.read_sql(query, conn)
            except:
                logging.warning("Could not query 'team_basic_stats' table. Skipping validation.")
                return
                
            if results_df.empty:
                return
                
            # We need team name mapping to match composite IDs if used
            team_map = pd.read_sql("SELECT teamId, teamName FROM team_basic_stats", conn)
            id_to_name = dict(zip(team_map['teamId'], team_map['teamName']))
            
            # Normalize for matching
            results_df['gameId'] = results_df['gameId'].astype(str)
            results_df['gameDate'] = results_df['gameDate'].astype(str).str[:10]  # Just YYYY-MM-DD
            
            logging.info(f"Found {len(results_df)} completed games, {len(pending)} pending predictions")

            updates = []
            
            for _, p_row in pending.iterrows():
                # Normalize pending game id for matching with team_basic_stats (which uses 10-char ids)
                raw_gid = p_row['game_id']
                gid = str(raw_gid) if raw_gid is not None else ''
                gid_norm = gid.zfill(10) if gid.isdigit() else gid
                p_date = str(p_row['game_date'])[:10]
                
                # Try direct gameId match first (for backfill entries)
                match = results_df[results_df['gameId'] == gid_norm]
                # also accept direct (un-padded) if present
                if match.empty:
                    match = results_df[results_df['gameId'] == gid]
                
                # Fallback: date + BOTH teams must match (for composite ID entries)
                if match.empty:
                    match = results_df[
                        (results_df['gameDate'] == p_date) & 
                        (results_df['home_team'] == p_row['home_team']) &
                        (results_df['away_team'] == p_row['away_team'])
                    ]
                
                if not match.empty:
                    res = match.iloc[0]
                    conn.execute("""
                        UPDATE predictions_log_v2 
                        SET home_score = ?, away_score = ?, status = 'COMPLETED'
                        WHERE game_id = ?
                    """, (int(res['home_score']), int(res['away_score']), p_row['game_id']))
                    updates.append(gid)
            
            if updates:
                conn.commit()
                logging.info(f"Updated scores for {len(updates)} games")
            
            # Debug: show what didn't match
            if len(updates) < len(pending):
                unmatched_ids = set(pending['game_id'].astype(str)) - set(updates)
                sample = pending[pending['game_id'].astype(str).isin(list(unmatched_ids)[:3])]
                logging.warning(f"Failed to match {len(unmatched_ids)} games. Samples:")
                for _, row in sample.iterrows():
                    logging.warning(f"  game_id={row['game_id']}, date={row['game_date']}, "
                                f"home={row['home_team']}, away={row['away_team']}")
                # Also show what's in DB for that date
                sample_date = sample.iloc[0]['game_date'] if len(sample) > 0 else None
                if sample_date:
                    db_sample = results_df[results_df['gameDate'] == str(sample_date)[:10]]
                    logging.warning(f"  DB games on {sample_date}: {db_sample[['home_team', 'away_team']].to_dict('records')}")