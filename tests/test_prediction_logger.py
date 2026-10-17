import sqlite3

import numpy as np
import pandas as pd
import pytest

from models import prediction_logger as pl


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
