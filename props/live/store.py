"""
The live store (data/props_live.db): signals, the immutable bet record, results, slot log and pipeline health.

prop_bets is the paper record. Database triggers abort any UPDATE or DELETE, so a bet row is exactly what was written at first qualification; corrections are new rows in other tables.
Append-only side tables (also guarded by triggers): prop_clv_audit (where each CLV value came from), prop_result_corrections (box-score changes found after settlement) and prop_fill_checks (did the quoted price survive to the next snapshots).
prop_results is write-once per cell (owner decision 2026-11-12): triggers abort any DELETE and any UPDATE that changes a filled cell; filling a blank cell is allowed (the 2026-11-12 CLV backfill did that).
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
CREATE TABLE IF NOT EXISTS prop_clv_audit (
    bet_id TEXT NOT NULL, written_utc TEXT NOT NULL, source TEXT NOT NULL, close_snapshot_ts_utc TEXT, close_snapshot_minutes REAL, close_line REAL, mkt_close REAL,
    clv_prob REAL, line_move_favour REAL, run_id TEXT,
    PRIMARY KEY (bet_id, written_utc)
);
CREATE TRIGGER IF NOT EXISTS prop_clv_audit_no_update BEFORE UPDATE ON prop_clv_audit BEGIN SELECT RAISE(ABORT, 'prop_clv_audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS prop_clv_audit_no_delete BEFORE DELETE ON prop_clv_audit BEGIN SELECT RAISE(ABORT, 'prop_clv_audit is append-only'); END;
CREATE TABLE IF NOT EXISTS prop_result_corrections (
    bet_id TEXT NOT NULL, detected_utc TEXT NOT NULL, y_old REAL, y_new REAL, status_old TEXT, status_new TEXT, profit_best_old REAL, profit_best_new REAL,
    profit_median_new REAL, profit_flat_new REAL, run_id TEXT,
    PRIMARY KEY (bet_id, detected_utc)
);
CREATE TRIGGER IF NOT EXISTS prop_result_corrections_no_update BEFORE UPDATE ON prop_result_corrections BEGIN SELECT RAISE(ABORT, 'prop_result_corrections is append-only'); END;
CREATE TRIGGER IF NOT EXISTS prop_result_corrections_no_delete BEFORE DELETE ON prop_result_corrections BEGIN SELECT RAISE(ABORT, 'prop_result_corrections is append-only'); END;
CREATE TABLE IF NOT EXISTS prop_fill_checks (
    bet_id TEXT PRIMARY KEY, checked_utc TEXT NOT NULL, n_later_snapshots INTEGER, next_snapshot_ts_utc TEXT, next_minutes_after REAL, book_price_next REAL, best_price_next REAL,
    median_price_next REAL, line_next REAL, book_still_at_or_better_next INTEGER, best_still_at_or_better_next INTEGER, minutes_price_available REAL, max_best_price_before_tip REAL,
    run_id TEXT, minutes_price_available_lo REAL, price_ended INTEGER
);
CREATE TRIGGER IF NOT EXISTS prop_fill_checks_no_update BEFORE UPDATE ON prop_fill_checks BEGIN SELECT RAISE(ABORT, 'prop_fill_checks is append-only'); END;
CREATE TRIGGER IF NOT EXISTS prop_fill_checks_no_delete BEFORE DELETE ON prop_fill_checks BEGIN SELECT RAISE(ABORT, 'prop_fill_checks is append-only'); END;
CREATE TABLE IF NOT EXISTS pipeline_health (
    run_id TEXT NOT NULL, job TEXT NOT NULL, started_utc TEXT NOT NULL, finished_utc TEXT, status TEXT NOT NULL, rows_written INTEGER, detail TEXT,
    PRIMARY KEY (run_id, job)
);
"""


def lock_results(conn):
    """
    prop_results is write-once per cell: abort any DELETE and any UPDATE that changes a filled cell (filling a blank cell stays allowed). The UPDATE trigger is built from the table's
    actual columns, so a store created by an older schema is locked on the columns it has.
    """
    cols = [r[1] for r in conn.execute('PRAGMA table_info(prop_results)')]
    when = ' OR '.join(f'(OLD."{c}" IS NOT NULL AND NEW."{c}" IS NOT OLD."{c}")' for c in cols)
    conn.executescript(
        "CREATE TRIGGER IF NOT EXISTS prop_results_no_delete BEFORE DELETE ON prop_results BEGIN SELECT RAISE(ABORT, 'prop_results rows are never deleted'); END; "
        f"CREATE TRIGGER IF NOT EXISTS prop_results_no_overwrite BEFORE UPDATE ON prop_results WHEN {when} "
        "BEGIN SELECT RAISE(ABORT, 'prop_results: a filled cell is never rewritten (blank cells may be filled; corrections go to prop_result_corrections)'); END;")


def connect(path=DEFAULT_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=120)
    conn.executescript(SCHEMA)
    lock_results(conn)
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
    """
    One transaction: signals, new bets and the slot-log row. Bets are inserted WITHOUT ignore-on-error (a constraint violation aborts the whole slot, which is then retried); a bet whose
    (game, player, stat) already exists is dropped before the insert, never rewritten. The writer stamps created_utc and run_id, and the written count must equal the new bets.
    """
    if bets is not None and len(bets):
        have = existing_bet_keys(conn, list(bets['game_id'].unique()))
        bets = bets[[(g, int(p), s) not in have for g, p, s in zip(bets['game_id'], bets['player_id'], bets['stat'])]].assign(created_utc=now_utc, run_id=run_id)
    with conn:
        n_sig = _insert(conn, 'prop_signals', signals, ignore=True)
        n_bet = _insert(conn, 'prop_bets', bets)
        if bets is not None and n_bet != len(bets):
            raise RuntimeError(f'{len(bets)} bets to record but {n_bet} written')
        conn.execute('INSERT OR REPLACE INTO slot_log (game_id, slot_minutes, status, detail, run_id, written_utc) VALUES (?,?,?,?,?,?)',
                     (game_id, int(slot), status, json.dumps(detail, default=str) if detail is not None else None, run_id, now_utc))
    return n_sig, n_bet


def health(conn, run_id, job, started_utc, finished_utc, status, rows=None, detail=None):
    with conn:
        conn.execute('INSERT OR REPLACE INTO pipeline_health VALUES (?,?,?,?,?,?,?)',
                     (run_id, job, started_utc, finished_utc, status, rows, json.dumps(detail, default=str) if detail is not None else None))


def read(conn, table, where='', params=()):
    return pd.read_sql(f'SELECT * FROM {table} {where}', conn, params=params)
