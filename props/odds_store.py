"""
Props odds store: its own SQLite file (data/props_odds.db), never nba_stats.db.

prop_snapshots  one row per snapshot x book x player x market x line (both prices on the row).
                player_id / game_id are filled by the ID resolver after insert (NULL = unresolved, never dropped).
team_lines      game spread / total / moneyline per snapshot x book, on the same snapshot_ts_utc as the props.
events          one row per vendor event (tip time, teams, resolved NBA game id and ET game date).
snapshot_log    one row per snapshot x event: what came back (rows, books, markets) and its credit cost. Absence
                is information (a line pulled between two snapshots), so it is recorded, not inferred later.
unmatched_names every vendor player name the matcher could not resolve, with counts.
api_ledger      created by core.odds_vendor.CreditBudget: every request and its credit cost.
All timestamps are UTC ISO-8601 strings ending in Z.
"""
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path('data/props_odds.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS prop_snapshots (
    snapshot_ts_utc TEXT NOT NULL, game_id TEXT, event_id_vendor TEXT NOT NULL, book TEXT NOT NULL,
    player_id INTEGER, player_name_raw TEXT NOT NULL, market TEXT NOT NULL, line REAL NOT NULL,
    over_price REAL, under_price REAL, is_alt_line INTEGER NOT NULL DEFAULT 0, source TEXT,
    commence_time_utc TEXT, home_team TEXT, away_team TEXT, book_last_update_utc TEXT,
    UNIQUE (snapshot_ts_utc, event_id_vendor, book, market, player_name_raw, line, is_alt_line)
);
CREATE INDEX IF NOT EXISTS ix_props_event_ts ON prop_snapshots (event_id_vendor, snapshot_ts_utc);
CREATE INDEX IF NOT EXISTS ix_props_player ON prop_snapshots (player_id, market);
CREATE TABLE IF NOT EXISTS team_lines (
    snapshot_ts_utc TEXT NOT NULL, event_id_vendor TEXT NOT NULL, book TEXT NOT NULL, game_id TEXT,
    home_ml REAL, away_ml REAL, home_spread REAL, home_spread_price REAL, away_spread_price REAL,
    total REAL, over_price REAL, under_price REAL, book_last_update_utc TEXT, source TEXT,
    UNIQUE (snapshot_ts_utc, event_id_vendor, book)
);
CREATE TABLE IF NOT EXISTS events (
    event_id_vendor TEXT PRIMARY KEY, commence_time_utc TEXT NOT NULL, home_team TEXT, away_team TEXT,
    et_game_date TEXT, game_id TEXT, home_team_id INTEGER, away_team_id INTEGER
);
CREATE TABLE IF NOT EXISTS snapshot_log (
    snapshot_ts_utc TEXT NOT NULL, event_id_vendor TEXT NOT NULL, mode TEXT, minutes_to_tip REAL,
    n_prop_rows INTEGER, n_players INTEGER, books TEXT, markets TEXT, credit_cost REAL, raw_path TEXT,
    PRIMARY KEY (snapshot_ts_utc, event_id_vendor)
);
CREATE TABLE IF NOT EXISTS unmatched_names (
    player_name_raw TEXT NOT NULL, team_context TEXT NOT NULL, reason TEXT, n_rows INTEGER,
    first_seen_utc TEXT, last_seen_utc TEXT, PRIMARY KEY (player_name_raw, team_context)
);
"""


def connect(path=DEFAULT_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=120)         # the live collector and a backfill may write at the same time
    conn.executescript(SCHEMA)
    return conn


def _insert_ignore(conn, table, df, cols):
    if df is None or df.empty:
        return 0
    df = df.reindex(columns=cols).astype(object).where(df.reindex(columns=cols).notna(), None)
    before = conn.total_changes
    conn.executemany(f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                     df.itertuples(index=False, name=None))
    return conn.total_changes - before


PROP_INSERT_COLS = ['snapshot_ts_utc', 'event_id_vendor', 'book', 'player_name_raw', 'market', 'line', 'over_price',
                    'under_price', 'is_alt_line', 'source', 'commence_time_utc', 'home_team', 'away_team',
                    'book_last_update_utc']
TEAM_INSERT_COLS = ['snapshot_ts_utc', 'event_id_vendor', 'book', 'home_ml', 'away_ml', 'home_spread',
                    'home_spread_price', 'away_spread_price', 'total', 'over_price', 'under_price',
                    'book_last_update_utc', 'source']


def insert_props(conn, df):
    return _insert_ignore(conn, 'prop_snapshots', df, PROP_INSERT_COLS)


def insert_team_lines(conn, df):
    return _insert_ignore(conn, 'team_lines', df, TEAM_INSERT_COLS)


def upsert_event(conn, event_id, commence_time_utc, home_team, away_team, et_game_date=None, game_id=None,
                 home_team_id=None, away_team_id=None):
    conn.execute("""INSERT INTO events (event_id_vendor, commence_time_utc, home_team, away_team, et_game_date, game_id,
                                        home_team_id, away_team_id) VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(event_id_vendor) DO UPDATE SET commence_time_utc=excluded.commence_time_utc,
                      et_game_date=COALESCE(excluded.et_game_date, et_game_date),
                      game_id=COALESCE(excluded.game_id, game_id),
                      home_team_id=COALESCE(excluded.home_team_id, home_team_id),
                      away_team_id=COALESCE(excluded.away_team_id, away_team_id)""",
                 (event_id, commence_time_utc, home_team, away_team, et_game_date, game_id, home_team_id,
                  away_team_id))


def log_snapshot(conn, snapshot_ts_utc, event_id, mode, minutes_to_tip, props_df, credit_cost, raw_path):
    conn.execute("INSERT OR REPLACE INTO snapshot_log VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (snapshot_ts_utc, event_id, mode, minutes_to_tip, len(props_df),
                  int(props_df['player_name_raw'].nunique()) if len(props_df) else 0,
                  ','.join(sorted(props_df['book'].unique())) if len(props_df) else '',
                  ','.join(sorted(props_df['market'].unique())) if len(props_df) else '', credit_cost, raw_path))


def last_snapshot_ts(conn, event_id):
    row = conn.execute("SELECT MAX(snapshot_ts_utc) FROM snapshot_log WHERE event_id_vendor = ?", (event_id,)).fetchone()
    return row[0]


def read_props(conn, where='1=1', params=()):
    return pd.read_sql(f"SELECT * FROM prop_snapshots WHERE {where}", conn, params=params)
