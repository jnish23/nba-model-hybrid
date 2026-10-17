import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from models import model_registry as mr


def _fit(cols, classifier=False):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(cols))), columns=cols)
    y = (X.iloc[:, 0] > 0).astype(int) if classifier else X.iloc[:, 0] * 2
    m = (lgb.LGBMClassifier if classifier else lgb.LGBMRegressor)(n_estimators=5, verbosity=-1)
    return m.fit(X, y)


def _save_set(d, suffix, features, calibrator=True):
    joblib.dump(_fit(features['ml'], True), d / f'model_ml{suffix}.pkl')
    joblib.dump(_fit(features['spread']), d / f'model_spread{suffix}.pkl')
    joblib.dump(_fit(features['total']), d / f'model_total{suffix}.pkl')
    joblib.dump(object(), d / f'calibrator_ml{suffix}.pkl')
    (d / f'feature_config{suffix}.json').write_text(json.dumps(features))
    (d / f'calibration_meta{suffix}.json').write_text(json.dumps({'spread_std': 13.0, 'total_std': 18.0}))


FEATS = {'ml': ['a', 'b', 'c'], 'spread': ['a', 'd'], 'total': ['e', 'f', 'g']}


def test_consistent_set_loads(tmp_path):
    _save_set(tmp_path, '_x', FEATS)
    b = mr.load_bundle('_x', tmp_path)
    assert b.features == FEATS and b.calib_meta['spread_std'] == 13.0


def test_config_from_another_model_is_refused(tmp_path):
    _save_set(tmp_path, '_x', FEATS)
    other = {**FEATS, 'ml': ['a', 'b', 'z']}
    (tmp_path / 'feature_config_x.json').write_text(json.dumps(other))
    with pytest.raises(mr.ModelConfigMismatch, match='ml'):
        mr.load_bundle('_x', tmp_path)


def test_reordered_config_is_refused(tmp_path):
    _save_set(tmp_path, '_x', FEATS)
    (tmp_path / 'feature_config_x.json').write_text(json.dumps({**FEATS, 'total': ['g', 'f', 'e']}))
    with pytest.raises(mr.ModelConfigMismatch, match='total'):
        mr.load_bundle('_x', tmp_path)


def test_incomplete_set_is_an_error_not_a_warning(tmp_path):
    _save_set(tmp_path, '_x', FEATS)
    (tmp_path / 'model_total_x.pkl').unlink()
    with pytest.raises(FileNotFoundError, match='model_total_x.pkl'):
        mr.load_bundle('_x', tmp_path)


def test_deployed_suffix_is_what_the_entry_points_use():
    import inspect
    import run_inference_v2
    from models import train_baseline
    assert inspect.signature(run_inference_v2.InferenceEngine).parameters['suffix'].default == mr.DEPLOYED_SUFFIX
    assert 'DEPLOYED_SUFFIX' in inspect.getsource(train_baseline.main)


def test_train_baseline_runs_as_a_script():
    """README: `python models/train_baseline.py ...` - the script folder is on sys.path, not the project root."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, str(root / 'models' / 'train_baseline.py'), '--help'], cwd=root,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert '--deploy' in out.stdout
