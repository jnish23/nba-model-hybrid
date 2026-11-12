import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from team import model_registry as mr


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
    from team import train_baseline
    assert inspect.signature(run_inference_v2.InferenceEngine).parameters['suffix'].default == mr.DEPLOYED_SUFFIX
    assert 'DEPLOYED_SUFFIX' in inspect.getsource(train_baseline.main)


def test_train_baseline_runs_as_a_script():
    """README: `python team/train_baseline.py ...` - the script folder is on sys.path, not the project root."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, str(root / 'team' / 'train_baseline.py'), '--help'], cwd=root,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert '--deploy' in out.stdout


def test_deployed_model_set_loads_in_a_fresh_process():
    """Regression: the v6 pickles record their classes as models.market_models; after the core/ team/ move that path must still resolve."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    code = ("from team.model_registry import load_bundle\n"
            "from team.market_models import ResidualSpreadModel\n"
            "b = load_bundle()\n"
            "assert isinstance(b.spread, ResidualSpreadModel), type(b.spread)\n"
            "print('ok', type(b.calibrator).__module__)")
    r = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith('ok'), r.stderr[-800:]


# ------------------------------------------------------------------ optional calibrator_total (E18 fix c): part of the hash only when present

PRE_CALIBRATOR_STEMS = [('model_ml', 'pkl'), ('calibrator_ml', 'pkl'), ('model_spread', 'pkl'), ('model_total', 'pkl'),
                        ('sigma_total', 'pkl'), ('feature_config', 'json'), ('calibration_meta', 'json')]


def _reference_hash_before_the_calibrator_existed(suffix, model_dir, version=mr.HASH_VERSION):
    """The model_set_hash algorithm as it was before calibrator_total was added (seven stems, in this order)."""
    import hashlib
    h = hashlib.sha256()
    for stem, ext in PRE_CALIBRATOR_STEMS:
        p = model_dir / f'{stem}{suffix}.{ext}'
        if p.exists():
            h.update(p.name.encode())
            h.update(mr.file_sha256(p, version).encode())
    return h.hexdigest()


def test_a_set_without_the_optional_files_hashes_exactly_as_before(tmp_path):
    _save_set(tmp_path, '_x', FEATS)
    for version in (1, 2):
        assert mr.model_set_hash('_x', tmp_path, version) == _reference_hash_before_the_calibrator_existed('_x', tmp_path, version)
    joblib.dump({'sigma': 1}, tmp_path / 'sigma_total_x.pkl')                           # with the E4 sigma file too
    assert mr.model_set_hash('_x', tmp_path) == _reference_hash_before_the_calibrator_existed('_x', tmp_path)
    assert mr.REQUIRED_STEMS == [s for s in PRE_CALIBRATOR_STEMS if s[0] != 'sigma_total']          # nothing new became mandatory


def test_calibrator_total_joins_the_hash_when_present_and_a_set_lacking_it_still_loads(tmp_path):
    _save_set(tmp_path, '_x', FEATS)
    h0 = mr.model_set_hash('_x', tmp_path)
    assert mr.load_bundle('_x', tmp_path).calibrator_total is None
    joblib.dump({'w0': 0.3}, tmp_path / 'calibrator_total_x.pkl')
    h1 = mr.model_set_hash('_x', tmp_path)
    assert h1 != h0
    assert 'calibrator_total_x.pkl' in [p.name for p in mr.model_set_paths('_x', tmp_path) if p.exists()]
    joblib.dump({'w0': 0.4}, tmp_path / 'calibrator_total_x.pkl')
    assert mr.model_set_hash('_x', tmp_path) not in (h0, h1)                            # changing the file changes the hash
    assert mr.load_bundle('_x', tmp_path).calibrator_total == {'w0': 0.4}


def test_the_deployed_set_still_loads_and_its_hash_equals_its_manifest():
    """The deployed set predates calibrator_total: loading and hashing it must give the value recorded when it was trained."""
    from pathlib import Path
    manifest = mr.read_manifest(mr.DEPLOYED_SUFFIX)
    if manifest is None or not (mr.MODEL_DIR / f'model_total{mr.DEPLOYED_SUFFIX}.pkl').exists():
        pytest.skip('deployed model set files are not in this checkout')
    assert not (mr.MODEL_DIR / f'calibrator_total{mr.DEPLOYED_SUFFIX}.pkl').exists()
    b = mr.load_bundle()                                                                # raises ModelSetHashMismatch if the hash moved
    assert b.calibrator_total is None
    assert b.model_set_hash == manifest['model_set_hash'] == mr.model_set_hash(mr.DEPLOYED_SUFFIX, mr.MODEL_DIR, manifest.get('hash_version', 1))
    assert sorted(manifest['files']) == sorted(p.name for p in mr.model_set_paths(mr.DEPLOYED_SUFFIX) if p.exists())
