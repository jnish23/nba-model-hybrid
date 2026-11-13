"""prop_results is write-once per cell: blanks may be filled, a filled cell is never rewritten, rows are never deleted (owner decision 2026-11-12)."""
import sqlite3

import pytest

from props.live import store


def _db(tmp_path):
    conn = store.connect(tmp_path / 'live.db')
    with conn:
        conn.execute("INSERT INTO prop_results (bet_id, settled_utc, y, status_stands, profit_stands_best, run_id) VALUES ('b1', '2026-11-12T00:00:00Z', 7.0, 'win', 0.9, 'r1')")
    return conn


def test_a_blank_cell_can_be_filled_once(tmp_path):
    conn = _db(tmp_path)
    with conn:
        n = conn.execute("UPDATE prop_results SET clv_prob = 0.01, close_line = 6.5, line_5 = 6.5, mkt_5 = 0.52 WHERE bet_id = 'b1'").rowcount
    assert n == 1
    assert conn.execute("SELECT clv_prob, close_line FROM prop_results").fetchone() == (0.01, 6.5)
    with conn:                                               # writing the same value again is not a change
        conn.execute("UPDATE prop_results SET clv_prob = 0.01 WHERE bet_id = 'b1'")


@pytest.mark.parametrize('sql', [
    "UPDATE prop_results SET y = 8.0 WHERE bet_id = 'b1'",
    "UPDATE prop_results SET status_stands = 'loss' WHERE bet_id = 'b1'",
    "UPDATE prop_results SET profit_stands_best = -1.0 WHERE bet_id = 'b1'",
    "UPDATE prop_results SET y = NULL WHERE bet_id = 'b1'",
    "UPDATE prop_results SET run_id = 'r2' WHERE bet_id = 'b1'",
    "UPDATE prop_results SET bet_id = 'b2' WHERE bet_id = 'b1'",
    "DELETE FROM prop_results WHERE bet_id = 'b1'",
])
def test_a_filled_cell_is_never_rewritten_and_rows_are_never_deleted(tmp_path, sql):
    conn = _db(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute(sql)
    assert conn.execute("SELECT y, status_stands, profit_stands_best, run_id FROM prop_results WHERE bet_id = 'b1'").fetchone() == (7.0, 'win', 0.9, 'r1')


def test_a_filled_clv_cell_cannot_be_overwritten_after_the_fill(tmp_path):
    conn = _db(tmp_path)
    with conn:
        conn.execute("UPDATE prop_results SET clv_prob = 0.01 WHERE bet_id = 'b1'")
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute("UPDATE prop_results SET clv_prob = 0.02 WHERE bet_id = 'b1'")


def test_settling_twice_still_ignores_the_second_insert(tmp_path):
    conn = _db(tmp_path)
    with conn:
        conn.execute("INSERT OR IGNORE INTO prop_results (bet_id, settled_utc, y) VALUES ('b1', '2026-11-13T00:00:00Z', 9.0)")
    assert conn.execute("SELECT y FROM prop_results").fetchone() == (7.0,)
