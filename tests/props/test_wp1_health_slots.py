"""Health rows for the decision slots: per-slot counts and warnings, and the age of the snapshot behind the T-5 slots (in-memory SQLite built with the live store's SCHEMA)."""
import json
import sqlite3

import pandas as pd

from props.live import health, store

SINCE = '2026-11-05T00:00:00Z'
WRITTEN = '2026-11-10T00:00:00Z'


def _conn():
    conn = sqlite3.connect(':memory:')
    conn.executescript(store.SCHEMA)
    return conn


def _add(conn, n, slot, status, snapshot_minutes=None, written=WRITTEN, start=0):
    for i in range(n):
        detail = {'snapshot_minutes': snapshot_minutes} if snapshot_minutes is not None else {'minutes_to_tip': -3.1}
        conn.execute('INSERT INTO slot_log VALUES (?,?,?,?,?,?)', (f'G{slot}{status}{start + i}', slot, status, json.dumps(detail), 'r', written))


def test_one_structurally_missed_slot_warns_even_when_the_pool_is_healthy():
    conn = _conn()
    _add(conn, 20, 240, 'done')
    _add(conn, 20, 60, 'done')
    _add(conn, 10, 5, 'missed')                                   # pooled: 10 / 50 = 20% would pass the old check; T-5 alone is 100%
    out = health.slot_checks(conn, SINCE)
    row = dict((c, (s, d)) for c, s, d in out)['decision slots (7 days)']
    assert row[0] == 'warn'
    assert 'T-5 100% missed' in row[1] and 'T-240: 20 done, 0 missed, 0 no_quotes of 20' in row[1] and 'T-5: 0 done, 10 missed, 0 no_quotes of 10' in row[1]
    assert 'T-240 ' not in row[1].split(':')[0]                   # the warning names only the failing slot


def test_slots_at_or_below_the_threshold_are_ok_and_no_quotes_counts_as_lost():
    conn = _conn()
    _add(conn, 8, 60, 'done')
    _add(conn, 1, 60, 'missed')
    _add(conn, 1, 60, 'no_quotes')                                # 2 of 10 = 20%: not above the limit
    st, detail = [(s, d) for c, s, d in health.slot_checks(conn, SINCE) if c == 'decision slots (7 days)'][0]
    assert st == 'ok' and 'T-60: 8 done, 1 missed, 1 no_quotes of 10' in detail and 'T-240: no slots' in detail
    _add(conn, 1, 60, 'no_quotes', start=5)                       # 3 of 11 > 20%
    assert [s for c, s, d in health.slot_checks(conn, SINCE) if c == 'decision slots (7 days)'][0] == 'warn'


def test_old_slots_outside_the_window_are_ignored_and_an_empty_log_is_ok():
    conn = _conn()
    assert [(c, s) for c, s, d in health.slot_checks(conn, SINCE)] == [('decision slots (7 days)', 'ok'), ('closing snapshot age', 'ok')]
    _add(conn, 5, 5, 'missed', written='2026-10-01T00:00:00Z')
    assert [s for c, s, d in health.slot_checks(conn, SINCE)] == ['ok', 'ok']


def test_closing_snapshot_age_reports_median_and_max_and_warns_above_15_minutes():
    conn = _conn()
    for k, m in enumerate([9.9, 10.0, 10.1, 11.5]):
        conn.execute('INSERT INTO slot_log VALUES (?,?,?,?,?,?)', (f'A{k}', 5, 'done', json.dumps({'snapshot_minutes': m}), 'r', WRITTEN))
    conn.execute('INSERT INTO slot_log VALUES (?,?,?,?,?,?)', ('Z', 5, 'done', 'not json', 'r', WRITTEN))      # unreadable detail is skipped
    conn.execute('INSERT INTO slot_log VALUES (?,?,?,?,?,?)', ('Y', 60, 'done', json.dumps({'snapshot_minutes': 59.9}), 'r', WRITTEN))   # other slots do not count
    st, detail = [(s, d) for c, s, d in health.slot_checks(conn, SINCE) if c == 'closing snapshot age'][0]
    assert st == 'ok' and 'T-5 slots done: 4' in detail and 'median of 10.1' in detail and 'at most 11.5' in detail
    for k in range(6):
        conn.execute('INSERT INTO slot_log VALUES (?,?,?,?,?,?)', (f'L{k}', 5, 'done', json.dumps({'snapshot_minutes': 30.0}), 'r', WRITTEN))
    assert [s for c, s, d in health.slot_checks(conn, SINCE) if c == 'closing snapshot age'][0] == 'warn'


def test_checks_keeps_its_columns_and_adds_the_closing_snapshot_row(tmp_path):
    odds = tmp_path / 'odds.db'
    with sqlite3.connect(odds) as c:
        c.execute('CREATE TABLE snapshot_log (snapshot_ts_utc TEXT, event_id_vendor TEXT, mode TEXT)')
        c.execute("INSERT INTO snapshot_log VALUES ('2026-11-10T00:10:00Z', 'e', 'slow')")
    (tmp_path / 'r').mkdir()
    conn = store.connect(tmp_path / 'live.db')
    _add(conn, 3, 5, 'done', snapshot_minutes=10.0)
    df = health.checks(conn, odds, tmp_path / 'r', pd.Timestamp('2026-11-10T00:30:00Z'), games_soon=False)
    assert list(df.columns) == ['check', 'status', 'detail']
    assert list(df['check']) == ['odds collector', 'injury reports', 'signal job', 'decision slots (7 days)', 'closing snapshot age', 'settlement']
    assert df.set_index('check').loc['closing snapshot age', 'status'] == 'ok'
