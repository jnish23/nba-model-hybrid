"""A model set cannot be silently overwritten, and a log row can be tied to the files that produced it."""
import json
import sqlite3

import joblib
import pandas as pd
import pytest

from core import prediction_logger as pl
from team import log_audit
from team import model_registry as mr


def write_set(d, suffix='_x', seed=0):
    """Smallest on-disk set: arbitrary picklable objects are enough for hashing."""
    for stem in ('model_ml', 'calibrator_ml', 'model_spread', 'model_total'):
        joblib.dump({'stem': stem, 'seed': seed}, d / f'{stem}{suffix}.pkl')
    (d / f'feature_config{suffix}.json').write_text(json.dumps({'ml': ['a']}))
    (d / f'calibration_meta{suffix}.json').write_text(json.dumps({'seed': seed}))


def test_hash_changes_when_any_file_is_rewritten(tmp_path):
    write_set(tmp_path)
    h = mr.model_set_hash('_x', tmp_path)
    assert h == mr.model_set_hash('_x', tmp_path)
    joblib.dump({'stem': 'model_total', 'seed': 1}, tmp_path / 'model_total_x.pkl')
    assert mr.model_set_hash('_x', tmp_path) != h


def test_existing_suffix_is_refused_until_overwrite_is_explicit(tmp_path):
    mr.assert_suffix_free('_x', tmp_path)                       # nothing there yet
    write_set(tmp_path)
    with pytest.raises(FileExistsError, match='already exists'):
        mr.assert_suffix_free('_x', tmp_path)
    mr.assert_suffix_free('_y', tmp_path)                       # another suffix is fine


def test_manifest_records_file_and_training_set_hashes(tmp_path):
    write_set(tmp_path)
    csv = tmp_path / 'train.csv'
    csv.write_text('a,b\n1,2\n')
    m = mr.write_manifest('_x', tmp_path, training_set=csv, extra={'rows': 1})
    assert m['model_set_hash'] == mr.model_set_hash('_x', tmp_path)
    assert m['training_set_hash'] == mr.file_sha256(csv) and m['training_set'] == 'train.csv' and m['rows'] == 1
    assert mr.read_manifest('_x', tmp_path)['model_set_hash'] == m['model_set_hash']
    assert mr.read_manifest('_nope', tmp_path) is None


def test_log_audit_flags_rows_without_a_hash_and_hashes_with_changed_files(tmp_path):
    write_set(tmp_path)
    m = mr.write_manifest('_x', tmp_path)
    log = pd.DataFrame({'game_id': ['1', '2', '3'], 'model_set_hash': [m['model_set_hash'], m['model_set_hash'], None]})
    rep = log_audit.provenance_report(log, tmp_path)
    assert rep['rows_without_hash'] == 1 and rep['unknown_hashes'] == []
    assert rep['hashes'][m['model_set_hash']] == {'rows': 2, 'suffix': '_x', 'files_unchanged': True}
    joblib.dump({'overwritten': True}, tmp_path / 'calibrator_ml_x.pkl')                 # the 2025-26 failure mode
    rep = log_audit.provenance_report(log, tmp_path)
    assert rep['hashes'][m['model_set_hash']]['files_unchanged'] is False
    assert log_audit.provenance_report(log.assign(model_set_hash='deadbeef'), tmp_path)['unknown_hashes'] == ['deadbeef']
    assert log_audit.provenance_report(pd.DataFrame({'game_id': ['1']}), tmp_path)['rows_without_hash'] == 1      # old log with no column


@pytest.fixture
def trained_set(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from team.train_baseline import BaselineModel
    from synthetic_history import training_csv
    csv = tmp_path / 'train.csv'
    training_csv(csv, per_season=400)
    m = BaselineModel(csv, model_dir=str(tmp_path / 'saved'))
    m.train_and_save_all(use_odds=True, save_suffix='_t', time_budget=1, calib_holdout_days=90)
    with pytest.raises(FileExistsError):                                                # retraining into the same suffix is refused
        m.train_and_save_all(use_odds=True, save_suffix='_t', time_budget=1, calib_holdout_days=90)
    return tmp_path / 'saved', '_t'


def test_load_bundle_refuses_a_set_changed_after_its_manifest(trained_set):
    d, suffix = trained_set
    b = mr.load_bundle(suffix, d)                                                       # intact: loads, carries its hashes
    assert b.model_set_hash == mr.read_manifest(suffix, d)['model_set_hash'] and b.training_set_hash
    meta = d / f'calibration_meta{suffix}.json'
    meta.write_text(meta.read_text().replace('{', '{"edited": 1,', 1))
    with pytest.raises(mr.ModelSetHashMismatch, match='changed after it was trained'):
        mr.load_bundle(suffix, d)


def test_overwrite_replaces_the_set_and_its_manifest(trained_set):
    d, suffix = trained_set
    old = mr.read_manifest(suffix, d)
    from team.train_baseline import BaselineModel
    m = BaselineModel(d.parent / 'train.csv', model_dir=str(d))
    m.train_and_save_all(use_odds=True, save_suffix=suffix, time_budget=1, calib_holdout_days=90, overwrite=True, refit_on_all=False)
    new = mr.read_manifest(suffix, d)
    assert new['model_set_hash'] != old['model_set_hash'] and new['refit_on_all'] is False
    mr.load_bundle(suffix, d)                                                           # and the new manifest matches the new files


# ------------------------------------------------------------------ provenance columns in the prediction log

def test_logger_stores_the_hashes_and_upgrades_old_tables(tmp_path):
    db = tmp_path / 'x.db'
    with sqlite3.connect(db) as conn:                                                   # a table from before the hash columns existed
        conn.execute("CREATE TABLE predictions_log_v2 (game_id TEXT PRIMARY KEY, game_date TEXT, prediction_date TEXT, home_team TEXT, "
                     "away_team TEXT, matchup TEXT, ml_prob REAL, spread_pred REAL, total_pred REAL, odds_home_ml REAL, odds_away_ml REAL, "
                     "odds_home_spread REAL, odds_total REAL, home_score INTEGER, away_score INTEGER, status TEXT DEFAULT 'PENDING', model_version TEXT)")
    logger = pl.PredictionLogger(str(db), log_path=str(tmp_path / 'p.csv'))
    logger.log_prediction({'gameId': '22600001', 'gameDate': '2026-10-21', 'home_team': 'H', 'away_team': 'A', 'pred_home_win_prob': 0.6,
                           'model_version': 'v', 'model_set_hash': 'abc123', 'training_set_hash': 'def456'})
    with sqlite3.connect(db) as conn:
        row = pd.read_sql("SELECT * FROM predictions_log_v2", conn).iloc[0]
    assert row['model_set_hash'] == 'abc123' and row['training_set_hash'] == 'def456'
    assert pd.read_csv(tmp_path / 'p.csv', dtype=str).loc[0, 'model_set_hash'] == 'abc123'


# ------------------------------------------------------------------ line endings (Windows CRLF vs Linux CI LF)

def _convert_json_line_endings(d, suffix, to):
    for p in d.glob(f'*{suffix}.json'):
        if p.name.startswith('model_manifest'):
            continue
        data = p.read_bytes().replace(b'\r\n', b'\n')
        p.write_bytes(data.replace(b'\n', b'\r\n') if to == 'crlf' else data)


def test_hash_is_the_same_for_crlf_and_lf_checkouts(tmp_path):
    write_set(tmp_path)
    (tmp_path / 'calibration_meta_x.json').write_bytes(b'{\n  "a": 1,\n  "b": 2\n}')
    _convert_json_line_endings(tmp_path, '_x', 'lf')
    lf = mr.model_set_hash('_x', tmp_path)
    _convert_json_line_endings(tmp_path, '_x', 'crlf')
    assert b'\r\n' in (tmp_path / 'calibration_meta_x.json').read_bytes()
    assert mr.model_set_hash('_x', tmp_path) == lf                      # git converting the JSON must not break the manifest
    joblib.dump({'stem': 'model_ml', 'seed': 9}, tmp_path / 'model_ml_x.pkl')
    assert mr.model_set_hash('_x', tmp_path) != lf                      # but a real change still does


def test_manifest_written_on_windows_before_hash_version_existed_still_verifies_on_linux(trained_set):
    """The first manifests hashed JSON with CRLF (written on Windows). CI checks the files out with LF and must still accept them."""
    d, suffix = trained_set
    _convert_json_line_endings(d, suffix, 'crlf')
    legacy = mr.read_manifest(suffix, d)
    legacy['model_set_hash'] = mr.model_set_hash(suffix, d, version=1)
    legacy.pop('hash_version')
    mr.manifest_path(suffix, d).write_text(json.dumps(legacy))
    b = mr.load_bundle(suffix, d)                                        # CRLF files, legacy manifest
    assert b.model_set_hash == legacy['model_set_hash']                  # the value already logged on rows is unchanged
    _convert_json_line_endings(d, suffix, 'lf')                          # what the Linux runner checks out
    assert mr.load_bundle(suffix, d).model_set_hash == legacy['model_set_hash']
    assert log_audit.provenance_report(pd.DataFrame({'model_set_hash': [legacy['model_set_hash']]}), d)['hashes'][
        legacy['model_set_hash']]['files_unchanged'] is True
