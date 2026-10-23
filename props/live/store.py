"""
The live store (data/props_live.db): signals, the immutable bet record, results, slot log and pipeline health.

prop_bets is the paper record. Database triggers abort any UPDATE or DELETE, so a bet row is exactly what was written at first qualification; corrections are new rows in other tables.
Everything for one slot (its signals, its new bets and its slot-log row) is written in one transaction, so a crash leaves either all of it or none and a re-run is safe.
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_PATH = 'data/props_live.db'

SCHEMA = """
CREATE TABLE IF NOT EXISTS slot_log (
    game_id TEXT NOT NULL, slot_minutes INTEGER NOT NULL, status TEXT NOT NULL, detail TEXT, run_id TEXT, written_utc TEXT NOT NULL,
    PRIMARY KEY (game_id, slot_minutes)
);
CREATE TABLE IF NOT EXISTS prop_signals (
    signal_id TEXT PRIMARY KEY, run_id TEXT, season TEXT, game_id TEXT NOT NULL, game_date TEXT, tip_ref_utc TEXT, slot_minutes INTEGER NOT NULL, snapshot_ts_utc TEXT,
    snapshot_minutes REAL, player_id INTEGER NOT NULL, player_name TEXT, stat TEXT NOT NULL, line REAL, n_at_line INTEGER, p_market REAL, p_model REAL, p_star REAL, push REAL,
    side TEXT, book TEXT, dec_best REAL, dec_median REAL, ev_best REAL, ev_median REAL, ev_flat REAL, expected_minutes REAL, qualified INTEGER NOT NULL, already_bet INTEGER NOT NULL,
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS ix_signals_game ON prop_signals (game_id, slot_minutes);
CREATE TABLE IF NOT EXISTS prop_bets (
    bet_id TEXT PRIMARY KEY, season TEXT, game_id TEXT NOT NULL, game_date TEXT, tip_ref_utc TEXT, placed_at_utc TEXT, created_utc TEXT NOT NULL, minutes_before INTEGER,
    matchup TEXT, home_team TEXT, away_team TEXT, player_id INTEGER NOT NULL, player_name TEXT, stat TEXT NOT NULL, side TEXT NOT NULL, line REAL NOT NULL, book TEXT,
    dec_best REAL, dec_median REAL, dec_flat REAL, ev_best REAL, ev_median REAL, ev_flat REAL, p_over_model REAL, p_over_star REAL, p_over_market REAL, push REAL,
    n_at_line INTEGER, n_quotes INTEGER, expected_minutes REAL, snapshot_ts_utc TEXT, snapshot_minutes REAL, quote_age_min REAL, pmf_json TEXT, books_json TEXT,
    model_version TEXT, model_sha256 TEXT, features_db_sha256 TEXT, rule_sha256 TEXT, roster_rule TEXT, code_commit TEXT, run_id TEXT,
    UNIQUE (game_id, player_id, stat)
);
CREATE TRIGGER IF NOT EXISTS prop_bets_no_update BEFORE UPDATE ON prop_bets BEGIN SELECT RAISE(ABORT, 'prop_bets is immutable'); END;
CREATE TRIGGER IF NOT EXISTS prop_bets_no_delete BEFORE DELETE ON prop_bets BEGIN SELECT RAISE(ABORT, 'prop_bets is immutable'); END;
CREATE TABLE IF NOT EXISTS prop_results (
    bet_id TEXT PRIMARY KEY, settled_utc TEXT NOT NULL, y REAL, played INTEGER, minutes_actual REAL, status_stands TEXT, status_flagged TEXT, status_all TEXT,
    profit_stands_best REAL, profit_stands_median REAL, profit_stands_flat REAL, clv_prob REAL, line_move_favour REAL, close_line REAL,
    line_240 REAL, line_60 REAL, line_5 REAL, mkt_240 REAL, mkt_60 REAL, mkt_5 REAL, run_id TEXT
);
CREATE TABLE IF NOT EXISTS pipeline_health (
    run_id TEXT NOT NULL, job TEXT NOT NULL, started_utc TEXT NOT NULL, finished_utc TEXT, status TEXT NOT NULL, rows_written INTEGER, detail TEXT,
    PRIMARY KEY (run_id, job)
);
"""


def connect(path=DEFAULT_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=120)
    conn.executescript(SCHEMA)
    return conn


def _insert(conn, table, df, ignore=False):
    if df is None or len(df) == 0:
        return 0
    cols = [r[1] for r in conn.execute(f'PRAGMA table_info({table})')]
    df = df.reindex(columns=cols)
    df = df.astype(object).where(df.notna(), None)
    before = conn.total_changes
    verb = 'INSERT OR IGNORE' if ignore else 'INSERT'
    conn.executemany(f"{verb} INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", df.itertuples(index=False, name=None))
    return conn.total_changes - before


def slot_done(conn, game_id, slot):
    return conn.execute('SELECT status FROM slot_log WHERE game_id = ? AND slot_minutes = ?', (game_id, int(slot))).fetchone()


def existing_bet_keys(conn, game_ids):
    if not game_ids:
        return set()
    ph = ','.join('?' * len(game_ids))
    return set(conn.execute(f'SELECT game_id, player_id, stat FROM prop_bets WHERE game_id IN ({ph})', list(game_ids)).fetchall())


def write_slot(conn, game_id, slot, status, now_utc, run_id, signals=None, bets=None, detail=None):
    """One transaction: signals, new bets and the slot-log row. A bet that already exists for the (game, player, stat) is never rewritten (INSERT OR IGNORE on the unique key)."""
    with conn:
        n_sig = _insert(conn, 'prop_signals', signals, ignore=True)
        n_bet = _insert(conn, 'prop_bets', bets, ignore=True)
        conn.execute('INSERT OR REPLACE INTO slot_log (game_id, slot_minutes, status, detail, run_id, written_utc) VALUES (?,?,?,?,?,?)',
                     (game_id, int(slot), status, json.dumps(detail, default=str) if detail is not None else None, run_id, now_utc))
    return n_sig, n_bet


def health(conn, run_id, job, started_utc, finished_utc, status, rows=None, detail=None):
    with conn:
        conn.execute('INSERT OR REPLACE INTO pipeline_health VALUES (?,?,?,?,?,?,?)',
                     (run_id, job, started_utc, finished_utc, status, rows, json.dumps(detail, default=str) if detail is not None else None))


def read(conn, table, where='', params=()):
    return pd.read_sql(f'SELECT * FROM {table} {where}', conn, params=params)
