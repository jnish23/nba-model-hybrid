import sqlite3

import numpy as np
import pandas as pd
import pytest

from core import prediction_logger as pl


def _pk_columns(conn):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({pl.TABLE})") if r[5]]


def _old_style_table(db):
    """The table as `to_sql(if_exists='replace')` leaves it: no PRIMARY KEY, duplicate games."""
    rows = pd.DataFrame({
        'game_id': ['001', '001', '002'], 'game_date': ['2026-01-01'] * 3,
        'prediction_date': ['2026-01-01 09:00:00', '2026-01-01 12:00:00', '2026-01-01 09:00:00'],
        'home_team': 'A', 'away_team': 'B', 'matchup': 'B @ A', 'ml_prob': [0.6, 0.7, 0.5],
        'spread_pred': 1.0, 'total_pred': 220.0, 'odds_home_ml': -150.0, 'odds_away_ml': 130.0,
        'odds_home_spread': -3.0, 'odds_total': 221.0,
        'home_score': [np.nan, 110.0, np.nan], 'away_score': [np.nan, 100.0, np.nan],
        'status': ['PENDING', 'COMPLETED', 'PENDING']})
    with sqlite3.connect(db) as conn:
        rows.to_sql(pl.TABLE, conn, index=False)


def test_old_table_without_primary_key_is_repaired_and_deduplicated(tmp_path):
    db = tmp_path / 'x.db'
    _old_style_table(db)
    with sqlite3.connect(db) as conn:
        assert _pk_columns(conn) == []
    pl.PredictionLogger(str(db), log_path=str(tmp_path / 'p.csv'))
    with sqlite3.connect(db) as conn:
        assert _pk_columns(conn) == ['game_id']
        df = pd.read_sql(f"SELECT * FROM {pl.TABLE} ORDER BY game_id", conn)
        assert len(df) == 2
        done = df[df.game_id == '001'].iloc[0]
        assert done.status == 'COMPLETED' and done.home_score == 110          # the completed duplicate survives
        assert 'model_version' in df.columns
        backups = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'predictions_log_v2_prepk_%'")]
        assert len(backups) == 1


def test_existing_primary_key_is_left_alone(tmp_path):
    db = tmp_path / 'x.db'
    pl.PredictionLogger(str(db), log_path=str(tmp_path / 'p.csv'))
    with sqlite3.connect(db) as conn:
        assert pl.ensure_primary_key(conn) is False


def test_replacing_a_season_keeps_the_primary_key(tmp_path):
    db = tmp_path / 'x.db'
    pl.PredictionLogger(str(db), log_path=str(tmp_path / 'p.csv'))
    old = pd.DataFrame({'game_id': ['a', 'b'], 'game_date': ['2025-10-21', '2026-02-01'], 'home_team': 'H',
                        'away_team': 'A', 'matchup': 'A @ H', 'status': 'COMPLETED', 'model_version': 'old'})
    new = pd.DataFrame({'game_id': ['b', 'c'], 'game_date': ['2026-02-01', '2026-02-02'], 'home_team': 'H',
                        'away_team': 'A', 'matchup': 'A @ H', 'status': 'PENDING', 'model_version': 'new'})
    with sqlite3.connect(db) as conn:
        old.to_sql(pl.TABLE, conn, if_exists='append', index=False)
        pl.replace_rows_in_date_range(conn, new)
        assert _pk_columns(conn) == ['game_id']
        got = pd.read_sql(f"SELECT game_id, model_version FROM {pl.TABLE} ORDER BY game_id", conn)
        assert got.to_dict('records') == [{'game_id': 'a', 'model_version': 'old'},       # outside the new date range
                                          {'game_id': 'b', 'model_version': 'new'},
                                          {'game_id': 'c', 'model_version': 'new'}]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"INSERT INTO {pl.TABLE} (game_id) VALUES ('b')")


def _entry(game_id='0022500001', prob=0.6):
    return {'gameId': game_id, 'gameDate': '2026-01-05', 'home_team': 'Home', 'away_team': 'Away',
            'pred_home_win_prob': prob, 'pred_spread': 2.0, 'pred_total': 220.0, 'line_spread': -3.0,
            'line_total': 221.0, 'line_ml_home': -150, 'line_ml_away': 130, 'model_version': 'v-test'}


def test_rerun_refreshes_the_prediction_but_keeps_scores_and_status(tmp_path):
    db, csv = tmp_path / 'x.db', tmp_path / 'p.csv'
    lg = pl.PredictionLogger(str(db), log_path=str(csv))
    lg.log_prediction(_entry(prob=0.60))
    with sqlite3.connect(db) as conn:       # the game finishes
        conn.execute(f"UPDATE {pl.TABLE} SET home_score = 110, away_score = 101, status = 'COMPLETED'")
    lg.log_prediction(_entry(prob=0.99))     # inference re-run after the result is known
    with sqlite3.connect(db) as conn:
        r = pd.read_sql(f"SELECT * FROM {pl.TABLE}", conn).iloc[0]
    assert (r.home_score, r.away_score, r.status) == (110, 101, 'COMPLETED')
    assert r.ml_prob == pytest.approx(0.60)       # history is not rewritten after the fact


def test_rerun_before_the_game_refreshes_prediction_and_stays_pending(tmp_path):
    db, csv = tmp_path / 'x.db', tmp_path / 'p.csv'
    lg = pl.PredictionLogger(str(db), log_path=str(csv))
    lg.log_prediction(_entry(prob=0.60))
    lg.log_prediction(_entry(prob=0.65))
    with sqlite3.connect(db) as conn:
        df = pd.read_sql(f"SELECT * FROM {pl.TABLE}", conn)
    assert len(df) == 1 and df.ml_prob.iloc[0] == pytest.approx(0.65) and df.status.iloc[0] == 'PENDING'
    assert df.model_version.iloc[0] == 'v-test'


def test_csv_mirror_keeps_completed_rows_and_column_alignment(tmp_path):
    db, csv = tmp_path / 'x.db', tmp_path / 'p.csv'
    lg = pl.PredictionLogger(str(db), log_path=str(csv))
    lg.log_prediction(_entry('0022500001', 0.6))
    lg.log_prediction(_entry('0022500002', 0.5))
    done = pd.read_csv(csv, dtype=str)
    done.loc[done.game_id == '0022500001', ['home_score', 'away_score', 'status']] = ['110', '101', 'COMPLETED']
    done.to_csv(csv, index=False)
    lg2 = pl.PredictionLogger(str(db), log_path=str(csv))
    lg2.log_prediction(_entry('0022500001', 0.9))        # completed: untouched
    lg2.log_prediction(_entry('0022500002', 0.4))        # pending: refreshed
    lg2.log_prediction(_entry('0022500003', 0.3))        # new
    out = pd.read_csv(csv, dtype=str).set_index('game_id')
    assert list(pd.read_csv(csv, nrows=0).columns) == pl.COLUMNS
    assert out.loc['0022500001', ['ml_prob', 'home_score', 'status']].tolist() == ['0.6', '110', 'COMPLETED']
    assert out.loc['0022500002', 'ml_prob'] == '0.4' and out.loc['0022500002', 'status'] == 'PENDING'
    assert out.loc['0022500003', 'status'] == 'PENDING' and len(out) == 3


def test_settling_a_game_updates_the_csv_as_well_as_the_db(tmp_path):
    db, csv = tmp_path / 'x.db', tmp_path / 'p.csv'
    lg = pl.PredictionLogger(str(db), log_path=str(csv))
    lg.log_prediction(_entry('0022500001', 0.6))
    lg.log_prediction(_entry('0022500002', 0.5))
    box = pd.DataFrame({
        'gameId': ['0022500001'] * 2, 'gameDate': ['2026-01-01'] * 2, 'teamId': [1, 2],
        'teamName': ['A', 'B'], 'points': [110, 101], 'matchup': ['A vs. B', 'B @ A'], 'WL': ['W', 'L']})
    with sqlite3.connect(db) as conn:
        box.to_sql('team_basic_stats', conn, index=False)
    lg.update_actual_scores()
    out = pd.read_csv(csv, dtype=str).set_index('game_id')
    assert out.loc['0022500001', ['home_score', 'away_score', 'status']].tolist() == ['110', '101', 'COMPLETED']
    assert out.loc['0022500002', 'status'] == 'PENDING'


# ------------------------------------------------------------------ bet_recommendations_log (append-only run record)

def _bet_row(game_id='0022500001', run_ts='2026-11-11T16:19:00.000001Z', **kw):
    row = {c: None for c in pl.BET_LOG_NAMES}
    row.update(game_id=game_id, game_date='2026-11-11', run_ts_utc=run_ts, model_set_hash='abc', odds_source='sbr',
               open_book='draftkings', curr_book='fanduel', odds_home_spread=-3.0, curr_odds_home_spread=-4.0,
               curr_price_home_spread_odds=-115.0, ml_prob=0.61, spread_pred=2.5, spread_std=13.4, spread_bet='home',
               spread_edge=0.07, spread_own_view=-0.5, spread_line_move=1.0, injury_cutoff_et='2026-11-11 11:19:00-05:00',
               injury_report_file='data/raw/injury_reports/x.pdf')
    row.update(kw)
    return row


def test_bet_log_table_is_created_with_a_composite_primary_key(tmp_path):
    db = tmp_path / 'x.db'
    assert pl.log_bet_recommendations(str(db), [_bet_row()]) == 1
    with sqlite3.connect(db) as conn:
        info = conn.execute(f"PRAGMA table_info({pl.BET_LOG_TABLE})").fetchall()
        assert [r[1] for r in info] == pl.BET_LOG_NAMES
        assert [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])] == ['game_id', 'run_ts_utc']


def test_bet_log_row_round_trips_and_nulls_stay_null(tmp_path):
    db = tmp_path / 'x.db'
    pl.log_bet_recommendations(str(db), [_bet_row()])
    with sqlite3.connect(db) as conn:
        r = pd.read_sql(f"SELECT * FROM {pl.BET_LOG_TABLE}", conn).iloc[0]
    assert (r.game_id, r.curr_book, r.spread_bet) == ('0022500001', 'fanduel', 'home')
    assert r.odds_home_spread == -3.0 and r.curr_odds_home_spread == -4.0 and r.curr_price_home_spread_odds == -115.0
    assert r.spread_edge == pytest.approx(0.07) and r.spread_own_view == -0.5 and r.spread_line_move == 1.0
    assert r.injury_report_file.endswith('x.pdf') and r.injury_cutoff_et.startswith('2026-11-11 11:19')
    assert pd.isna(r.ml_bet) and pd.isna(r.total_edge) and pd.isna(r.ml_prob_raw)       # not logged = NULL, never 0


def test_a_rerun_appends_and_never_overwrites(tmp_path):
    db = tmp_path / 'x.db'
    pl.log_bet_recommendations(str(db), [_bet_row(run_ts='2026-11-11T16:19:00.000001Z', spread_edge=0.07)])
    pl.log_bet_recommendations(str(db), [_bet_row(run_ts='2026-11-11T18:30:00.000001Z', spread_edge=0.02, spread_bet=None)])
    with sqlite3.connect(db) as conn:
        df = pd.read_sql(f"SELECT run_ts_utc, spread_bet, spread_edge FROM {pl.BET_LOG_TABLE} ORDER BY run_ts_utc", conn)
    assert len(df) == 2 and df.spread_edge.tolist() == [0.07, 0.02] and df.spread_bet.tolist()[0] == 'home'
    assert pd.isna(df.spread_bet.iloc[1])


def test_bet_log_does_not_touch_predictions_log_v2_or_its_csv(tmp_path):
    db, csv = tmp_path / 'x.db', tmp_path / 'p.csv'
    lg = pl.PredictionLogger(str(db), log_path=str(csv))
    lg.log_prediction(_entry())
    before = csv.read_bytes()
    pl.log_bet_recommendations(str(db), [_bet_row()])
    assert csv.read_bytes() == before
    with sqlite3.connect(db) as conn:
        assert [r[1] for r in conn.execute(f"PRAGMA table_info({pl.TABLE})")] == pl.COLUMNS


def test_a_logger_failure_never_raises_and_is_logged_at_error(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.ERROR):
        assert pl.log_bet_recommendations(str(tmp_path), [_bet_row()]) == 0                       # a directory is not a database
        assert pl.log_bet_recommendations(str(tmp_path / 'y.db'), [_bet_row(ml_prob={'bad': 1})]) == 0   # an unbindable value
        assert pl.log_bet_recommendations(str(tmp_path / 'z.db'), 5) == 0                         # not even rows
    assert sum(r.levelno == logging.ERROR for r in caplog.records) == 3
    with sqlite3.connect(tmp_path / 'y.db') as conn:               # a failed insert leaves nothing half-written
        assert conn.execute(f"SELECT COUNT(*) FROM {pl.BET_LOG_TABLE}").fetchone()[0] == 0


def test_the_same_game_and_run_cannot_be_logged_twice(tmp_path):
    db = tmp_path / 'x.db'
    assert pl.log_bet_recommendations(str(db), [_bet_row()]) == 1
    assert pl.log_bet_recommendations(str(db), [_bet_row(spread_edge=0.5)]) == 0         # primary key: refused, original kept
    with sqlite3.connect(db) as conn:
        assert conn.execute(f"SELECT spread_edge FROM {pl.BET_LOG_TABLE}").fetchall() == [(0.07,)]


# ------------------------------------------------------------------ the CSV mirror agrees with the DB (2026-10-26..28 stayed PENDING)

def _stale_csv_setup(tmp_path):
    db, csv = tmp_path / 'x.db', tmp_path / 'p.csv'
    lg = pl.PredictionLogger(str(db), log_path=str(csv))
    lg.log_prediction(_entry('0022500001', 0.6))
    lg.log_prediction(_entry('0022500002', 0.5))
    with sqlite3.connect(db) as conn:       # settled in the DB by a run that did not mirror (older code, locked file, restored CSV)
        conn.execute(f"UPDATE {pl.TABLE} SET home_score = 110, away_score = 101, status = 'COMPLETED' WHERE game_id = '0022500001'")
    return lg, db, csv


def test_update_actual_scores_repairs_csv_rows_the_db_already_settled(tmp_path):
    lg, db, csv = _stale_csv_setup(tmp_path)
    assert pd.read_csv(csv, dtype=str).set_index('game_id').loc['0022500001', 'status'] == 'PENDING'      # the drift
    lg.update_actual_scores()          # game 2 stays pending and unmatched (no box score table): the repair must still run
    out = pd.read_csv(csv, dtype=str).set_index('game_id')
    assert out.loc['0022500001', ['home_score', 'away_score', 'status']].tolist() == ['110', '101', 'COMPLETED']
    assert out.loc['0022500002', 'status'] == 'PENDING'


def test_reconcile_works_when_there_are_no_pending_games(tmp_path):
    lg, db, csv = _stale_csv_setup(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute(f"UPDATE {pl.TABLE} SET home_score = 99, away_score = 98, status = 'COMPLETED' WHERE game_id = '0022500002'")
    lg.update_actual_scores()          # nothing has home_score NULL any more: update_actual_scores returns early, then repairs
    assert pd.read_csv(csv, dtype=str)['status'].tolist() == ['COMPLETED', 'COMPLETED']


def test_reconcile_leaves_a_correct_csv_byte_for_byte_and_never_raises(tmp_path):
    lg, db, csv = _stale_csv_setup(tmp_path)
    lg.reconcile_csv_with_db()
    fixed = csv.read_bytes()
    lg.reconcile_csv_with_db()
    assert csv.read_bytes() == fixed                                    # nothing left to change: file not rewritten
    csv.write_bytes(b'')                                                # a broken CSV is logged, not raised
    lg.reconcile_csv_with_db()


# ------------------------------------------------------------------ bet_suppressed_reason (B3): new column at the END, old tables altered

def test_bet_suppressed_reason_is_the_last_column_and_an_existing_column_order_is_untouched():
    names = pl.BET_LOG_NAMES
    assert names[-1] == 'bet_suppressed_reason' and names.count('bet_suppressed_reason') == 1
    assert names[:-1][-2:] == ['injury_cutoff_et', 'injury_report_file']                   # what was there before keeps its place
    assert dict(pl.BET_LOG_COLUMNS)['bet_suppressed_reason'] == 'TEXT'


def test_a_new_database_gets_the_column_from_create_table(tmp_path):
    db = tmp_path / 'x.db'
    assert pl.log_bet_recommendations(str(db), [_bet_row(bet_suppressed_reason='injury_report_stale', spread_bet=None)]) == 1
    with sqlite3.connect(db) as conn:
        info = conn.execute(f"PRAGMA table_info({pl.BET_LOG_TABLE})").fetchall()
        assert [r[1] for r in info][-1] == 'bet_suppressed_reason' and info[-1][2] == 'TEXT' and info[-1][3] == 0     # nullable TEXT
        assert conn.execute(f"SELECT bet_suppressed_reason FROM {pl.BET_LOG_TABLE}").fetchone()[0] == 'injury_report_stale'


def test_an_existing_table_without_the_column_is_altered_and_old_rows_keep_their_data(tmp_path):
    db = tmp_path / 'x.db'
    old_cols = [(c, t) for c, t in pl.BET_LOG_COLUMNS if c != 'bet_suppressed_reason']
    with sqlite3.connect(db) as conn:                                                      # the table as A3 created it
        conn.execute(f"CREATE TABLE {pl.BET_LOG_TABLE} (" + ", ".join(f"{c} {t}" for c, t in old_cols) + ", PRIMARY KEY (game_id, run_ts_utc))")
        conn.execute(f"INSERT INTO {pl.BET_LOG_TABLE} (game_id, run_ts_utc, spread_bet, spread_edge) VALUES ('old', '2026-11-10T16:19:00.000001Z', 'home', 0.07)")
    assert pl.log_bet_recommendations(str(db), [_bet_row(bet_suppressed_reason='injury_report_missing', spread_bet=None)]) == 1
    with sqlite3.connect(db) as conn:
        info = conn.execute(f"PRAGMA table_info({pl.BET_LOG_TABLE})").fetchall()
        assert [r[1] for r in info] == pl.BET_LOG_NAMES                                    # same order as a new table
        assert [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])] == ['game_id', 'run_ts_utc']
        rows = pd.read_sql(f"SELECT * FROM {pl.BET_LOG_TABLE} ORDER BY run_ts_utc", conn)
    assert rows.loc[0, 'game_id'] == 'old' and rows.loc[0, 'spread_bet'] == 'home' and pd.isna(rows.loc[0, 'bet_suppressed_reason'])
    assert rows.loc[1, 'bet_suppressed_reason'] == 'injury_report_missing'


def test_ensure_bet_log_schema_is_idempotent(tmp_path):
    with sqlite3.connect(tmp_path / 'x.db') as conn:
        pl.ensure_bet_log_schema(conn)
        before = conn.execute(f"PRAGMA table_info({pl.BET_LOG_TABLE})").fetchall()
        pl.ensure_bet_log_schema(conn)
        assert conn.execute(f"PRAGMA table_info({pl.BET_LOG_TABLE})").fetchall() == before


# ------------------------------------------------------------------ inference_skipped_games (B3): every dropped game leaves a record

def _skip_row(**kw):
    row = {'game_date': '2026-11-11', 'game_id': '0022600004', 'home_team': 'UTA', 'away_team': 'DEN', 'reason': 'no_odds', 'detail': 'x'}
    row.update(kw)
    return row


def test_skipped_games_table_columns_and_primary_key(tmp_path):
    db = tmp_path / 'x.db'
    assert pl.log_skipped_games(str(db), [_skip_row()], '2026-11-11T16:19:00.000001Z') == 1
    with sqlite3.connect(db) as conn:
        info = conn.execute(f"PRAGMA table_info({pl.SKIPPED_TABLE})").fetchall()
        row = pd.read_sql(f"SELECT * FROM {pl.SKIPPED_TABLE}", conn).iloc[0]
    assert [r[1] for r in info] == ['game_date', 'run_ts_utc', 'game_id', 'home_team', 'away_team', 'reason', 'detail'] == pl.SKIPPED_COLUMNS
    assert [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])] == ['game_date', 'run_ts_utc', 'home_team', 'away_team', 'reason']
    assert (row.game_id, row.reason, row.run_ts_utc) == ('0022600004', 'no_odds', '2026-11-11T16:19:00.000001Z')


def test_skipped_games_game_id_may_be_unknown(tmp_path):
    db = tmp_path / 'x.db'
    pl.log_skipped_games(str(db), [_skip_row(game_id=None, reason='not_on_schedule', home_team='Dallas Mavericks')], 'ts')
    with sqlite3.connect(db) as conn:
        assert conn.execute(f"SELECT game_id FROM {pl.SKIPPED_TABLE}").fetchone()[0] is None


def test_skipped_games_are_appended_never_replaced_and_an_exact_repeat_is_ignored(tmp_path):
    db = tmp_path / 'x.db'
    pl.log_skipped_games(str(db), [_skip_row(), _skip_row(reason='missing_opening_line', home_team='ORL', away_team='MIA')], 'run1')
    pl.log_skipped_games(str(db), [_skip_row()], 'run2')                                   # a re-run adds its own rows
    pl.log_skipped_games(str(db), [_skip_row(detail='changed')], 'run2')                   # same key in the same run: the first one stays
    with sqlite3.connect(db) as conn:
        df = pd.read_sql(f"SELECT * FROM {pl.SKIPPED_TABLE} ORDER BY run_ts_utc, reason", conn)
    assert len(df) == 3 and df.run_ts_utc.tolist() == ['run1', 'run1', 'run2'] and df.detail.tolist().count('changed') == 0


def test_skipped_games_logger_never_raises(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.ERROR):
        assert pl.log_skipped_games(str(tmp_path), [_skip_row()], 'ts') == 0               # a directory is not a database
        assert pl.log_skipped_games(str(tmp_path / 'y.db'), 5, 'ts') == 0                  # not even rows
        assert pl.log_skipped_games(str(tmp_path / 'z.db'), [None], 'ts') == 0                                    # a row that is not a mapping
        assert pl.log_skipped_games(str(tmp_path / 'w.db'), [], 'ts') == 0                 # nothing to write is not an error
    assert sum(r.levelno == logging.ERROR for r in caplog.records) >= 2
    assert not (tmp_path / 'w.db').exists()


def test_skipped_games_do_not_touch_the_prediction_tables(tmp_path):
    db = tmp_path / 'x.db'
    pl.PredictionLogger(str(db), log_path=str(tmp_path / 'p.csv'))
    with sqlite3.connect(db) as conn:
        before = conn.execute(f"PRAGMA table_info({pl.TABLE})").fetchall()
    pl.log_skipped_games(str(db), [_skip_row()], 'ts')
    with sqlite3.connect(db) as conn:
        assert conn.execute(f"PRAGMA table_info({pl.TABLE})").fetchall() == before
        assert conn.execute(f"SELECT COUNT(*) FROM {pl.TABLE}").fetchone()[0] == 0


def test_prediction_date_is_still_machine_local_in_the_old_format(tmp_path, monkeypatch):
    # B3 left prediction_date alone on purpose (the published dashboard reads it): machine-local, 'YYYY-MM-DD HH:MM:SS'
    import datetime
    local = datetime.datetime(2026, 11, 10, 23, 30, 5)

    class FakeDT(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return local if tz is None else datetime.datetime.now(tz)
    monkeypatch.setattr(pl, 'datetime', type('M', (), {'datetime': FakeDT}))
    lg = pl.PredictionLogger(str(tmp_path / 'x.db'), log_path=str(tmp_path / 'p.csv'))
    lg.log_prediction({'gameId': '22600001', 'gameDate': '2026-11-11', 'home_team': 'A', 'away_team': 'B'})
    with sqlite3.connect(tmp_path / 'x.db') as conn:
        assert conn.execute(f"SELECT prediction_date FROM {pl.TABLE}").fetchone()[0] == '2026-11-10 23:30:05'
    assert pd.read_csv(tmp_path / 'p.csv')['prediction_date'][0] == '2026-11-10 23:30:05'
