import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from core import prediction_logger as pl

ROOT = Path(__file__).resolve().parent.parent.parent


def _db(path):
    rows = pd.DataFrame({
        'game_id': ['0022500001', '0022500002'], 'game_date': ['2025-11-01', '2025-11-02'], 'prediction_date': ['2025-11-01 09:00:00', '2026-10-22 09:24:07'],
        'home_team': ['A', 'C'], 'away_team': ['B', 'D'], 'matchup': ['B @ A', 'D @ C'], 'ml_prob': [0.6, 0.4], 'spread_pred': [3.0, -2.0],
        'total_pred': [220.0, 225.0], 'odds_home_ml': [-150.0, 120.0], 'odds_away_ml': [130.0, -140.0], 'odds_home_spread': [-3.5, 2.5],
        'odds_total': [221.5, 224.5], 'home_score': [110, 100], 'away_score': [100, 104], 'status': ['COMPLETED', 'COMPLETED']})
    with sqlite3.connect(path) as conn:
        conn.execute(pl.CREATE_TABLE_SQL.format(table=pl.TABLE))
        rows.to_sql(pl.TABLE, conn, if_exists='append', index=False)


def test_ttls_are_short_and_the_sidebar_has_a_refresh_button_and_the_latest_run_time(tmp_path, monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    db = tmp_path / 'x.db'
    _db(db)
    monkeypatch.setenv('NBA_DASHBOARD_DB', str(db))
    at = AppTest.from_file(str(ROOT / 'dashboard' / 'app.py'), default_timeout=120).run()
    assert not at.exception
    assert [b for b in at.sidebar.button if 'Refresh data' in b.label]
    assert any('Latest prediction run: 2026-10-22 09:24' in c.value for c in at.sidebar.caption)
    at.sidebar.button[0].click().run()                                      # clearing the caches and rerunning works
    assert not at.exception


def test_data_loaders_use_the_short_ttl_constants():
    pytest.importorskip('streamlit')
    spec = importlib.util.spec_from_file_location('dashboard_app_ttl', ROOT / 'dashboard' / 'app.py')
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    assert app.DATA_TTL <= 300 < app.HISTORY_TTL
