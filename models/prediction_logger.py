import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import logging
import sqlite3

TABLE = 'predictions_log_v2'
CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {table} (
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
        status TEXT DEFAULT 'PENDING',
        model_version TEXT
    )
"""
COLUMNS = ['game_id', 'game_date', 'prediction_date', 'home_team', 'away_team', 'matchup', 'ml_prob',
           'spread_pred', 'total_pred', 'odds_home_ml', 'odds_away_ml', 'odds_home_spread', 'odds_total',
           'home_score', 'away_score', 'status', 'model_version']


def ensure_primary_key(conn):
    """
    Make sure predictions_log_v2 has game_id as PRIMARY KEY and the model_version column. A
    `DataFrame.to_sql(if_exists='replace')` rewrite drops the key (and with it the uniqueness that
    INSERT OR REPLACE / ON CONFLICT rely on, so repeated logging silently piled up duplicate rows).
    Repair = copy the table aside, keep one row per game_id (a COMPLETED row beats a PENDING one, then
    the latest prediction_date), and rebuild it with the key. Returns True if a repair was needed.
    """
    info = conn.execute(f"PRAGMA table_info({TABLE})").fetchall()
    cols = [r[1] for r in info]
    if any(r[5] for r in info):                     # r[5] = pk position
        if 'model_version' not in cols:
            conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN model_version TEXT")
        return False

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = f"{TABLE}_prepk_{stamp}"
    conn.execute(f'CREATE TABLE "{backup}" AS SELECT * FROM {TABLE}')
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df['_done'] = (df['status'] == 'COMPLETED').astype(int)
    df = (df.sort_values(['game_id', '_done', 'prediction_date']).drop_duplicates('game_id', keep='last')
            [COLUMNS])
    logging.warning(f"{TABLE} had no PRIMARY KEY; rebuilding it with one ({len(df)} unique games; "
                    f"previous contents kept in {backup})")
    conn.execute(f"DROP TABLE {TABLE}")
    conn.execute(CREATE_TABLE_SQL.format(table=TABLE))
    df.to_sql(TABLE, conn, if_exists='append', index=False)
    return True


def replace_rows_in_date_range(conn, log):
    """
    Replace the rows whose game_date lies in [min, max] of `log` (a DataFrame in the table's columns,
    e.g. a regenerated season) with the new rows, inside the existing table. The table is never
    dropped or recreated (pandas to_sql(if_exists='replace') rebuilds it WITHOUT the primary key).
    Returns (lo, hi).
    """
    conn.execute(CREATE_TABLE_SQL.format(table=TABLE))
    ensure_primary_key(conn)
    lo, hi = log['game_date'].min(), log['game_date'].max()
    conn.execute(f"DELETE FROM {TABLE} WHERE substr(game_date, 1, 10) BETWEEN ? AND ?", (lo, hi))
    log[[c for c in COLUMNS if c in log.columns]].to_sql(TABLE, conn, if_exists='append', index=False)
    return lo, hi


class PredictionLogger:
    def __init__(self, db_path, log_path=None):
        if log_path is None:
            from models.season_utils import season_for_date
            log_path = f'predictions/predictions_{season_for_date()}.csv'
        self.db_path = db_path
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Create predictions_log_v2 (game_id is the PRIMARY KEY) and repair an older copy that lost it."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(CREATE_TABLE_SQL.format(table=TABLE))
            ensure_primary_key(conn)
            
    def log_prediction(self, prediction: dict, overwrite: bool = True):
        """
        Log a prediction (UPSERT keyed on game_id).

        A re-run refreshes the prediction columns but never touches home_score / away_score / status:
        a game that is already COMPLETED keeps its result, and its pre-game prediction is kept as
        well (a post-game re-prediction must not rewrite history). overwrite=False leaves any
        existing row for the game untouched.
        """
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
            'odds_away_ml': prediction.get('line_ml_away'),
            'odds_home_spread': prediction.get('line_spread'),
            'odds_total': prediction.get('line_total'),
            'status': 'PENDING',            # only used when the game is new
            'model_version': prediction.get('model_version'),
        }
        refreshed = [c for c in row if c not in ('game_id', 'status')]

        # 1. Save to DB
        try:
            with sqlite3.connect(self.db_path) as conn:
                cols = ', '.join(row.keys())
                placeholders = ', '.join(['?'] * len(row))
                if overwrite:
                    updates = ', '.join(f"{c} = excluded.{c}" for c in refreshed)
                    sql = (f"INSERT INTO {TABLE} ({cols}) VALUES ({placeholders}) "
                           f"ON CONFLICT(game_id) DO UPDATE SET {updates} "
                           f"WHERE {TABLE}.status IS NOT 'COMPLETED'")
                else:
                    sql = f"INSERT OR IGNORE INTO {TABLE} ({cols}) VALUES ({placeholders})"
                conn.execute(sql, list(row.values()))
        except Exception as e:
            logging.error(f"Failed to log prediction to DB: {e}")

        # 2. Save to CSV, mirroring the DB (same columns as the table; scores/status carried over)
        df_row = pd.DataFrame([row]).reindex(columns=COLUMNS)
        if self.log_path.exists():
            existing = pd.read_csv(self.log_path, dtype=str).reindex(columns=COLUMNS)
            match = existing['game_id'] == row['game_id']
            new_row = df_row.astype(object).where(df_row.notna(), None).astype(str).replace('None', np.nan)
            if not match.any():
                updated = pd.concat([existing, new_row], ignore_index=True)
            elif overwrite and not (existing.loc[match, 'status'] == 'COMPLETED').any():
                keep = existing.loc[match, ['home_score', 'away_score', 'status']].iloc[0]
                new_row[['home_score', 'away_score', 'status']] = keep.values
                updated = pd.concat([existing[~match], new_row], ignore_index=True)
            else:
                updated = existing                  # completed (or overwrite=False): keep the row as it is
            updated.to_csv(self.log_path, index=False)
        else:
            df_row.to_csv(self.log_path, index=False)

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
            except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
                logging.warning(f"Could not query 'team_basic_stats' ({e}). Skipping validation.")
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