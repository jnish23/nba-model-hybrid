import datetime as dt
import sqlite3

import pandas as pd

from core import odds_rerun as orr

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 23, 16, 15, tzinfo=UTC)          # 9:15 AM Pacific


def _db(tmp_path, rows):
    db = tmp_path / 'x.db'
    cols = ['game_id', 'game_date', 'matchup', 'status', 'odds_home_ml', 'odds_home_spread', 'odds_total']
    with sqlite3.connect(db) as conn:
        pd.DataFrame(rows, columns=cols).to_sql('predictions_log_v2', conn, index=False)
    return db


def test_unpriced_games_are_pending_games_missing_any_line(tmp_path):
    db = _db(tmp_path, [
        ('1', '2026-10-23', 'A @ B', 'PENDING', -150.0, -3.5, 220.5),      # fully priced
        ('2', '2026-10-23', 'C @ D', 'PENDING', None, None, None),         # nothing posted
        ('3', '2026-10-23', 'E @ F', 'PENDING', -110.0, None, 221.0),      # moneyline only
        ('4', '2026-10-23', 'G @ H', 'COMPLETED', None, None, None),       # finished: not a candidate
        ('5', '2026-10-22', 'I @ J', 'PENDING', None, None, None),         # another day
    ])
    assert orr.unpriced_games(db, '2026-10-23')['game_id'].tolist() == ['2', '3']


def test_next_rerun_is_an_hour_out_while_the_first_unpriced_game_has_not_tipped_off():
    tip = dt.datetime(2026, 10, 23, 23, 0, tzinfo=UTC)
    assert orr.next_rerun_time(NOW, [tip], attempts=0) == NOW + dt.timedelta(hours=1)


def test_no_rerun_when_nothing_is_unpriced_or_the_attempts_are_used_up():
    tip = dt.datetime(2026, 10, 23, 23, 0, tzinfo=UTC)
    assert orr.next_rerun_time(NOW, [], attempts=0) is None
    assert orr.next_rerun_time(NOW, [tip], attempts=orr.MAX_RERUNS) is None


def test_no_rerun_that_would_start_at_or_after_tipoff():
    soon = NOW + dt.timedelta(minutes=70)                     # re-run would start 10 min before tip: inside the 15 min margin
    assert orr.next_rerun_time(NOW, [soon], attempts=0) is None
    assert orr.next_rerun_time(NOW, [soon + dt.timedelta(minutes=10)], attempts=0) is not None


def test_the_earliest_unpriced_tipoff_decides():
    early, late = NOW + dt.timedelta(minutes=30), NOW + dt.timedelta(hours=8)
    assert orr.next_rerun_time(NOW, [late, early], attempts=0) is None


def _status(next_rerun, n=2, game_date='2026-10-23'):
    missing = pd.DataFrame({'game_id': [str(i) for i in range(n)], 'matchup': ['x'] * n})
    return orr.build_status(game_date, missing, next_rerun, 0, NOW, 'waiting for lines')


def test_status_round_trips_through_the_file(tmp_path):
    path = tmp_path / 's.json'
    orr.write_status(path, _status(NOW + dt.timedelta(hours=1)))
    got = orr.read_status(path)
    assert got['next_rerun'] == '2026-10-23T17:15+00:00' and len(got['missing']) == 2
    assert orr.read_status(tmp_path / 'missing.json') is None


def test_message_names_the_next_check_in_pacific_time():
    msg = orr.rerun_message(_status(NOW + dt.timedelta(hours=1)), '2026-10-23', now=NOW)
    assert '2 games' in msg and '10:15 AM Pacific' in msg


def test_message_singular_overdue_and_gave_up():
    assert '1 game still has no odds' in orr.rerun_message(_status(NOW + dt.timedelta(hours=1), n=1), '2026-10-23', now=NOW)
    assert 'due now' in orr.rerun_message(_status(NOW - dt.timedelta(minutes=5)), '2026-10-23', now=NOW)
    assert 'no more automatic re-checks' in orr.rerun_message(_status(None), '2026-10-23', now=NOW)


def test_no_message_without_missing_games_or_for_another_day():
    assert orr.rerun_message(None, '2026-10-23') is None
    assert orr.rerun_message(_status(None, n=0), '2026-10-23') is None
    assert orr.rerun_message(_status(NOW + dt.timedelta(hours=1), game_date='2026-10-22'), '2026-10-23', now=NOW) is None
