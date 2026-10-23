import sqlite3

import pandas as pd

from props.live import health, store


def test_checks_flag_a_dead_signal_job_only_when_games_are_near(tmp_path):
    odds = tmp_path / 'odds.db'
    with sqlite3.connect(odds) as c:
        c.execute('CREATE TABLE snapshot_log (snapshot_ts_utc TEXT, event_id_vendor TEXT, mode TEXT)')
        c.execute("INSERT INTO snapshot_log VALUES ('2026-10-23T00:10:00Z', 'e', 'slow')")
    (tmp_path / 'r').mkdir()
    (tmp_path / 'r' / 'a.pdf').write_bytes(b'x')
    conn = store.connect(tmp_path / 'live.db')
    now = pd.Timestamp('2026-10-23T00:30:00Z')
    busy = health.checks(conn, odds, tmp_path / 'r', now, games_soon=True).set_index('check')
    assert busy.loc['odds collector', 'status'] == 'ok' and busy.loc['signal job', 'status'] == 'fail'
    quiet = health.checks(conn, odds, tmp_path / 'r', now, games_soon=False).set_index('check')
    assert quiet.loc['signal job', 'status'] == 'ok'
    store.health(conn, 'r1', 'signal', '2026-10-23T00:20:00Z', '2026-10-23T00:21:00Z', 'ok', 0, {})
    assert health.checks(conn, odds, tmp_path / 'r', now, games_soon=True).set_index('check').loc['signal job', 'status'] == 'ok'
