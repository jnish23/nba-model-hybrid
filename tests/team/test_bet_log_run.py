"""InferenceEngine.run writes one bet_recommendations_log row per game per run, including --only-unpriced re-runs,
and a failure of that logging never stops the run. The model, the feature build and the network are replaced by stubs."""
import logging
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

run_inference_v2 = pytest.importorskip('run_inference_v2')


def _games():
    base = {'gameDate': '2026-11-11', 'gameStatus': 1, 'odds_source': 'sbr', 'open_book': 'draftkings', 'curr_book': 'fanduel',
            'curr_price_home_spread_odds': -110.0, 'curr_price_away_spread_odds': -110.0,
            'curr_price_over_odds': -110.0, 'curr_price_under_odds': -110.0}
    rows = [
        {'gameId': '0022600301', 'home_team': 'H1', 'away_team': 'A1', 'teamId_home': 1, 'teamId_away': 2,
         'odds_home_ml': -135.0, 'odds_away_ml': 115.0, 'odds_home_spread': -3.0, 'odds_total': 219.0,
         'curr_odds_home_ml': -150.0, 'curr_odds_away_ml': 130.0, 'curr_odds_home_spread': -4.0, 'curr_odds_total': 221.5},
        {'gameId': '0022600302', 'home_team': 'H2', 'away_team': 'A2', 'teamId_home': 3, 'teamId_away': 4,   # priced at the open only
         'odds_home_ml': -110.0, 'odds_away_ml': -110.0, 'odds_home_spread': -1.0, 'odds_total': 225.0,
         'curr_odds_home_ml': np.nan, 'curr_odds_away_ml': np.nan, 'curr_odds_home_spread': np.nan, 'curr_odds_total': np.nan},
    ]
    return pd.DataFrame([{**base, **r} for r in rows])


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = run_inference_v2
    games = _games()
    out = pd.DataFrame({'ml_prob_raw': [0.58, 0.51], 'ml_prob': [0.60, 0.52], 'spread_pred': [5.0, 1.0],
                        'total_pred': [225.0, 226.0], 'spread_std': [13.0, 13.0], 'total_std': [18.0, 18.0]})
    monkeypatch.setattr(mod, 'build_live_game_frame', lambda *a, **k: games.copy())
    monkeypatch.setattr(mod, 'predict_markets', lambda *a, **k: out)
    eng = object.__new__(mod.InferenceEngine)
    db = str(tmp_path / 'x.db')
    eng.db_path, eng.only_unpriced, eng.target_date = db, False, None
    eng.injury_cutoff, eng.injury_report_path = None, None
    eng.bundle = SimpleNamespace(model_set_hash='hash-xyz', training_set_hash='t', ml=None, spread=None, total=None,
                                 features={'ml': [], 'spread': [], 'total': []})
    eng.selected_features = eng.bundle.features
    eng.logger = SimpleNamespace(log_prediction=lambda entry: None)
    eng.get_todays_games = lambda: games.copy()
    eng.get_injury_features = lambda g: pd.DataFrame({'gameId': [], 'teamId': []}).set_index(['gameId', 'teamId'])
    eng.explain_games = lambda *a, **k: None
    eng.keep_unpriced = lambda g: g.iloc[:1].reset_index(drop=True)          # the re-run keeps only game 1
    return eng, db


def _rows(db):
    with sqlite3.connect(db) as conn:
        return pd.read_sql("SELECT * FROM bet_recommendations_log ORDER BY run_ts_utc, game_id", conn)


def test_a_run_logs_every_game_with_what_the_model_read_and_what_it_judged(engine):
    eng, db = engine
    eng.injury_cutoff, eng.injury_report_path = pd.Timestamp('2026-11-11 11:19'), 'data/raw/injury_reports/r.pdf'
    eng.run()
    df = _rows(db)
    assert df.game_id.tolist() == ['0022600301', '0022600302'] and df.run_ts_utc.nunique() == 1
    g1, g2 = df.iloc[0], df.iloc[1]
    assert (g1.odds_home_spread, g1.curr_odds_home_spread, g1.curr_book, g1.open_book) == (-3.0, -4.0, 'fanduel', 'draftkings')
    assert g1.model_set_hash == 'hash-xyz' and g1.ml_prob_raw == 0.58 and g1.spread_std == 13.0
    # +2 points against the opening line, but the line then moved 1 point toward the model: only +1 left at the current line, no bet
    assert pd.isna(g1.spread_bet) and g1.spread_own_view == 2.0 and g1.spread_line_move == 1.0 and 0 < g1.spread_edge < 0.05
    assert g1.total_bet == 'over' and g1.total_edge > 0.05 and g1.total_own_view == 6.0 and g1.total_line_move == -2.5
    assert g1.injury_report_file == 'data/raw/injury_reports/r.pdf' and g1.injury_cutoff_et.startswith('2026-11-11 11:19')
    assert pd.isna(g2.curr_odds_home_spread) and pd.isna(g2.spread_edge) and pd.isna(g2.spread_bet) and g2.spread_own_view == 0.0


def test_reruns_append_rows_and_an_only_unpriced_rerun_is_logged_too(engine):
    eng, db = engine
    eng.run()
    eng.only_unpriced = True
    eng.run()
    df = _rows(db)
    assert len(df) == 3 and df.run_ts_utc.nunique() == 2
    assert df.game_id.tolist().count('0022600301') == 2 and df.game_id.tolist().count('0022600302') == 1


def test_a_failing_bet_log_does_not_stop_the_run(engine, monkeypatch, caplog):
    eng, db = engine
    import core.prediction_logger as pl
    monkeypatch.setattr(pl.sqlite3, 'connect', lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError('database is locked')))
    with caplog.at_level(logging.ERROR):
        eng.run()                               # the real logger with a broken database: logged at ERROR, never raised
    assert any('bet_recommendations_log' in r.message and r.levelno == logging.ERROR for r in caplog.records)
    assert pd.read_csv('todays_picks.csv').shape[0] == 2          # the picks file is still written after the failure
