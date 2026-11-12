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
        model_version TEXT,
        model_set_hash TEXT,
        training_set_hash TEXT
    )
"""
COLUMNS = ['game_id', 'game_date', 'prediction_date', 'home_team', 'away_team', 'matchup', 'ml_prob',
           'spread_pred', 'total_pred', 'odds_home_ml', 'odds_away_ml', 'odds_home_spread', 'odds_total',
           'home_score', 'away_score', 'status', 'model_version', 'model_set_hash', 'training_set_hash']
PROVENANCE_COLUMNS = ['model_version', 'model_set_hash', 'training_set_hash']   # added to older tables with ALTER TABLE


# ---------------------------------------------------------------------------------------------------------------
# bet_recommendations_log: append-only record of what each inference run read and judged, one row per game per run.
# predictions_log_v2 keeps one (refreshed) row per game with only the current line; this table is what lets a bet
# be reconstructed afterwards (opening line the model read, current line + price it was judged at, edge, sigma).
BET_LOG_TABLE = 'bet_recommendations_log'
BET_LOG_COLUMNS = [
    ('game_id', 'TEXT'), ('game_date', 'TEXT'), ('run_ts_utc', 'TEXT'), ('model_set_hash', 'TEXT'), ('odds_source', 'TEXT'),
    ('open_book', 'TEXT'), ('curr_book', 'TEXT'),
    ('odds_home_ml', 'REAL'), ('odds_away_ml', 'REAL'), ('odds_home_spread', 'REAL'), ('odds_total', 'REAL'),
    ('curr_odds_home_ml', 'REAL'), ('curr_odds_away_ml', 'REAL'), ('curr_odds_home_spread', 'REAL'), ('curr_odds_total', 'REAL'),
    ('curr_price_home_spread_odds', 'REAL'), ('curr_price_away_spread_odds', 'REAL'),
    ('curr_price_over_odds', 'REAL'), ('curr_price_under_odds', 'REAL'),
    ('ml_prob', 'REAL'), ('ml_prob_raw', 'REAL'), ('spread_pred', 'REAL'), ('total_pred', 'REAL'),
    ('spread_std', 'REAL'), ('total_std', 'REAL'),
    ('ml_bet', 'TEXT'), ('ml_edge', 'REAL'), ('ml_own_view', 'REAL'), ('ml_line_move', 'REAL'),
    ('spread_bet', 'TEXT'), ('spread_edge', 'REAL'), ('spread_own_view', 'REAL'), ('spread_line_move', 'REAL'),
    ('total_bet', 'TEXT'), ('total_edge', 'REAL'), ('total_own_view', 'REAL'), ('total_line_move', 'REAL'),
    ('injury_cutoff_et', 'TEXT'), ('injury_report_file', 'TEXT'),
    # Why the bets of this row were withheld although the edge was computed (NULL on a normal day): 'injury_report_missing',
    # 'injury_report_stale', 'injury_snapshot_mismatch' or 'injury_features_failed' (scripts/run_inference_v2.py). Always the LAST
    # column: tables created before it existed get it with ALTER TABLE (ensure_bet_log_schema).
    ('bet_suppressed_reason', 'TEXT'),
]
BET_LOG_NAMES = [c for c, _ in BET_LOG_COLUMNS]
BET_LOG_CREATE_SQL = (f"CREATE TABLE IF NOT EXISTS {BET_LOG_TABLE} ("
                      + ", ".join(f"{c} {t}" for c, t in BET_LOG_COLUMNS) + ", PRIMARY KEY (game_id, run_ts_utc))")


def ensure_bet_log_schema(conn):
    """
    Create bet_recommendations_log if it is missing and add every BET_LOG_COLUMNS column an older copy of the table lacks
    (ALTER TABLE ... ADD COLUMN, so existing rows keep their data and get NULL). Columns are only ever added, never changed.
    Anything that INSERTs into the table with BET_LOG_NAMES (log_bet_recommendations, the backfill script) must call this first.
    """
    conn.execute(BET_LOG_CREATE_SQL)
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({BET_LOG_TABLE})")}
    for name, sql_type in BET_LOG_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE {BET_LOG_TABLE} ADD COLUMN {name} {sql_type}")
            logging.info(f"{BET_LOG_TABLE}: added column {name}")


def log_bet_recommendations(db_path, rows):
    """
    Append `rows` (dicts keyed by BET_LOG_NAMES, see team.live_odds.bet_log_row) to bet_recommendations_log.
    Never updates or deletes: a re-run adds new rows (the primary key is game_id + run_ts_utc). Unknown keys are ignored and
    missing keys are NULL. Bookkeeping only: any failure is logged at ERROR and swallowed, so the daily run goes on.
    Returns the number of rows written (0 on failure).
    """
    try:
        rows = list(rows)
    except Exception as e:
        logging.error(f"Could not write {BET_LOG_TABLE}: rows not iterable ({e})")
        return 0
    try:
        if not rows:
            return 0
        cols = ', '.join(BET_LOG_NAMES)
        marks = ', '.join('?' * len(BET_LOG_NAMES))
        data = [[r.get(c) for c in BET_LOG_NAMES] for r in rows]
        with sqlite3.connect(db_path) as conn:
            ensure_bet_log_schema(conn)
            conn.executemany(f"INSERT INTO {BET_LOG_TABLE} ({cols}) VALUES ({marks})", data)
        return len(data)
    except Exception as e:
        logging.error(f"Could not write {BET_LOG_TABLE} ({len(rows)} rows lost): {e}", exc_info=True)
        return 0


# ---------------------------------------------------------------------------------------------------------------
# inference_skipped_games: append-only record of every game a run did NOT predict, with the reason. Without it a game dropped
# before prediction (no opening line, unknown team name, absent from the odds source, not on the NBA schedule) left only a log line.
SKIPPED_TABLE = 'inference_skipped_games'
SKIPPED_REASONS = ('missing_opening_line', 'unmapped_team', 'no_odds', 'not_on_schedule')
SKIPPED_COLUMNS = ['game_date', 'run_ts_utc', 'game_id', 'home_team', 'away_team', 'reason', 'detail']
# game_id is NULL for a game the NBA schedule does not list (not_on_schedule, and SBR rows dropped before they could be matched),
# so it cannot be part of the key: one row per run, date, matchup and reason.
SKIPPED_CREATE_SQL = (f"CREATE TABLE IF NOT EXISTS {SKIPPED_TABLE} (game_date TEXT NOT NULL, run_ts_utc TEXT NOT NULL, game_id TEXT, "
                      "home_team TEXT NOT NULL, away_team TEXT NOT NULL, reason TEXT NOT NULL, detail TEXT, "
                      "PRIMARY KEY (game_date, run_ts_utc, home_team, away_team, reason))")


def log_skipped_games(db_path, rows, run_ts_utc=None):
    """
    Append `rows` (dicts with game_date, game_id, home_team, away_team, reason, detail; run_ts_utc from each row or the argument) to
    inference_skipped_games. Never updates or deletes; a row that repeats an existing key (same run, matchup and reason) is ignored.
    Bookkeeping only: any failure is logged at ERROR and swallowed. Returns the number of rows offered to the table (0 on failure).
    """
    try:
        rows = list(rows)
        if not rows:
            return 0
        text = lambda v: '' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        data = [[text(r.get('game_date'))[:10], text(r.get('run_ts_utc') or run_ts_utc),
                 None if r.get('game_id') is None else str(r.get('game_id')),
                 text(r.get('home_team')), text(r.get('away_team')), text(r.get('reason')), text(r.get('detail'))] for r in rows]
        with sqlite3.connect(db_path) as conn:
            conn.execute(SKIPPED_CREATE_SQL)
            conn.executemany(f"INSERT OR IGNORE INTO {SKIPPED_TABLE} ({', '.join(SKIPPED_COLUMNS)}) VALUES (?,?,?,?,?,?,?)", data)
        return len(data)
    except Exception as e:
        logging.error(f"Could not write {SKIPPED_TABLE}: {e}", exc_info=True)
        return 0


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
        for c in PROVENANCE_COLUMNS:
            if c not in cols:
                conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN {c} TEXT")
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
            from core.season_utils import season_for_date
            from core.scrape_injury_reports import now_et
            log_path = f'predictions/predictions_{season_for_date(now_et().date())}.csv'      # the season is decided in Eastern time
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
            # prediction_date stays in MACHINE-LOCAL time (Pacific on the production machine), format unchanged: the published dashboard
            # and the CSV mirror read it. Do not use it as a timestamp across zones; bet_recommendations_log.run_ts_utc is the
            # unambiguous one (UTC, 'Z').
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
            'model_set_hash': prediction.get('model_set_hash'),         # which files (and training CSV) produced this row
            'training_set_hash': prediction.get('training_set_hash'),
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

    def _mirror_scores_to_csv(self, conn, game_ids=None):
        """Copy settled games' scores and status from the DB into the CSV log (log_prediction mirrors DB -> CSV
        for new rows, but settling happens in the DB only, so the CSV would otherwise stay PENDING).

        game_ids=None reconciles EVERY COMPLETED DB row whose CSV row is not COMPLETED. Mirroring only the games settled in
        the current call lets one missed run (older code running, a locked file, a CSV restored from git) leave a row PENDING
        in the CSV for good, because a settled game is never 'pending' in the DB again (25 rows, 2026-10-26..28)."""
        if not self.log_path.exists():
            return
        done = pd.read_sql(f"SELECT game_id, home_score, away_score, status FROM {TABLE} WHERE status = 'COMPLETED'", conn)
        if game_ids is not None:
            done = done[done['game_id'].astype(str).isin([str(g) for g in game_ids])]
        if done.empty:
            return
        csv = pd.read_csv(self.log_path, dtype=str).reindex(columns=COLUMNS)
        by_id = done.assign(game_id=done['game_id'].astype(str)).set_index('game_id')
        hit = csv['game_id'].isin(by_id.index)
        if game_ids is None:
            hit &= csv['status'] != 'COMPLETED'
            if not hit.any():
                return
        for col in ('home_score', 'away_score', 'status'):
            vals = csv.loc[hit, 'game_id'].map(by_id[col])
            csv.loc[hit, col] = vals.astype(int).astype(str) if col != 'status' else vals
        csv.to_csv(self.log_path, index=False)
        logging.info(f"Mirrored {int(hit.sum())} settled games into {self.log_path}")

    def reconcile_csv_with_db(self):
        """Make the CSV mirror agree with the DB for every settled game (see _mirror_scores_to_csv). Never raises."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._mirror_scores_to_csv(conn, None)
        except Exception as e:
            logging.error(f"Could not reconcile {self.log_path} with the database: {e}")

    def update_actual_scores(self):
        """
        Check for completed games in DB and update predictions_log, then bring the CSV mirror in line with the DB.
        """
        try:
            self._settle_from_box_scores()
        finally:
            self.reconcile_csv_with_db()

    def _settle_from_box_scores(self):
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
                self._mirror_scores_to_csv(conn, updates)
            
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