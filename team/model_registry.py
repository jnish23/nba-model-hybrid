"""
The one place that says which trained models production uses, and how a model set is loaded.

A "model set" is the files that share a suffix in models/saved/:
    model_ml<suffix>.pkl  calibrator_ml<suffix>.pkl  model_spread<suffix>.pkl  model_total<suffix>.pkl
    sigma_total<suffix>.pkl (optional: per-game total sigma, see team/market_models.py)
    feature_config<suffix>.json  calibration_meta<suffix>.json  best_config_<target><suffix>.json
`train_baseline.py --deploy` writes the set named DEPLOYED_SUFFIX; `scripts/run_inference_v2.py`
(and therefore run_daily_inference.bat) and scripts/maintenance/backfill_inference.py load it.
Loading refuses a set whose feature_config does not match the features stored inside each pickle,
so a model can never be scored with another model's feature list.
"""
import datetime as dt
import hashlib
import importlib
import json
import logging
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import joblib

# Pickles written before the core/ team/ layout record their classes under the old module path. Loading them needs these names to resolve,
# without putting code back into models/ (which holds artifacts only). New pickles record the team.* path and need no alias.
LEGACY_PICKLE_MODULES = {'models.market_models': 'team.market_models'}


def alias_legacy_pickle_modules():
    for old, new in LEGACY_PICKLE_MODULES.items():
        if old in sys.modules:
            continue
        top = old.split('.')[0]
        try:
            importlib.import_module(top)
        except ModuleNotFoundError:                      # pickle.find_class imports the top-level package first
            stub = types.ModuleType(top)
            stub.__path__ = []
            sys.modules[top] = stub
        sys.modules[old] = importlib.import_module(new)

# Model artifacts stay in <repo>/models/saved (code lives in team/; models/ now holds data only)
MODEL_DIR = Path(__file__).resolve().parent.parent / 'models' / 'saved'

# Production model set: trained WITH opening odds as inputs, on the parity-fixed training set, with the
# market-aware components of team/market_models.py (residual spread, margin-stack ML, per-game total sigma).
DEPLOYED_SUFFIX = '_with_odds_cand_20261019'

TARGETS = {'ml': 'model_ml', 'spread': 'model_spread', 'total': 'model_total'}


class ModelConfigMismatch(RuntimeError):
    pass


class ModelSetHashMismatch(RuntimeError):
    """The files of a model set no longer match the manifest written when it was trained (overwritten or edited in place)."""


# Files whose bytes define a model set. best_config_* are the tuner's output (already reflected in the models) and the manifest is
# the record of the hash itself, so neither is hashed.
HASHED_STEMS = [('model_ml', 'pkl'), ('calibrator_ml', 'pkl'), ('model_spread', 'pkl'), ('model_total', 'pkl'),
                ('sigma_total', 'pkl'), ('feature_config', 'json'), ('calibration_meta', 'json')]
REQUIRED_STEMS = HASHED_STEMS[:4] + HASHED_STEMS[5:]      # sigma_total is optional


def model_set_paths(suffix, model_dir=MODEL_DIR):
    return [Path(model_dir) / f'{stem}{suffix}.{ext}' for stem, ext in HASHED_STEMS]


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def model_set_hash(suffix, model_dir=MODEL_DIR):
    """sha256 over the (file name, content hash) pairs of the model set's files that exist. Changes if any file is rewritten."""
    h = hashlib.sha256()
    for p in model_set_paths(suffix, model_dir):
        if p.exists():
            h.update(p.name.encode())
            h.update(file_sha256(p).encode())
    return h.hexdigest()


def manifest_path(suffix, model_dir=MODEL_DIR):
    return Path(model_dir) / f'model_manifest{suffix}.json'


def assert_suffix_free(suffix, model_dir=MODEL_DIR):
    """Refuse to train into a suffix that already holds a model set: a log written by the old files could no longer be tied to them."""
    existing = [p.name for p in model_set_paths(suffix, model_dir) + [manifest_path(suffix, model_dir)] if p.exists()]
    if existing:
        raise FileExistsError(
            f"Model set '{suffix}' already exists in {model_dir} ({len(existing)} files, e.g. {existing[0]}). Train under a new suffix, "
            "or pass overwrite=True / --overwrite to replace it deliberately (anything logged with the old files will no longer match).")


def _git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except Exception:
        return None


def write_manifest(suffix, model_dir=MODEL_DIR, training_set=None, extra=None):
    """Record what a model set was trained from: the hash of its files and of the training set CSV. Returns the manifest dict."""
    manifest = {
        'suffix': suffix,
        'created_at': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_set_hash': model_set_hash(suffix, model_dir),
        'files': {p.name: file_sha256(p) for p in model_set_paths(suffix, model_dir) if p.exists()},
        'training_set': Path(training_set).name if training_set else None,
        'training_set_hash': file_sha256(training_set) if training_set else None,
        'git_commit': _git_commit(),
        **(extra or {}),
    }
    manifest_path(suffix, model_dir).write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(suffix, model_dir=MODEL_DIR):
    p = manifest_path(suffix, model_dir)
    return json.loads(p.read_text()) if p.exists() else None


def model_feature_names(model):
    """Ordered feature names stored in a fitted LightGBM sklearn model (None if it has none)."""
    if hasattr(model, 'feature_name_') and model.feature_name_:
        return list(model.feature_name_)
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    booster = getattr(model, 'booster_', None)
    if booster is not None:
        return list(booster.feature_name())
    return None


def assert_config_matches_model(target, model, configured):
    """Raise unless `configured` (feature_config[target]) is exactly the model's feature list."""
    actual = model_feature_names(model)
    if actual is None:
        raise ModelConfigMismatch(f"{target}: model carries no feature names to check against the config")
    if list(configured) != actual:
        missing = [f for f in configured if f not in actual]
        extra = [f for f in actual if f not in configured]
        raise ModelConfigMismatch(
            f"{target}: feature_config does not match the pickled model "
            f"({len(configured)} configured vs {len(actual)} in model; "
            f"in config only: {missing[:5]}; in model only: {extra[:5]}; same set, different order: "
            f"{sorted(configured) == sorted(actual)})")


@dataclass
class ModelBundle:
    suffix: str
    ml: object
    calibrator: object
    spread: object
    total: object
    features: dict
    calib_meta: dict = field(default_factory=dict)
    sigma_total: object = None          # models/market_models.HeteroSigma; None for sets trained before E4 (global std)
    model_set_hash: str = None          # sha256 of the set's files (computed on load)
    training_set_hash: str = None       # sha256 of the training CSV, from the manifest (None for sets trained before manifests existed)


def load_bundle(suffix=DEPLOYED_SUFFIX, model_dir=MODEL_DIR):
    """Load a complete, consistent model set; any missing piece or config mismatch is an error."""
    model_dir = Path(model_dir)
    alias_legacy_pickle_modules()

    def path(stem, ext):
        return model_dir / f'{stem}{suffix}.{ext}'

    needed = [path('model_ml', 'pkl'), path('calibrator_ml', 'pkl'), path('model_spread', 'pkl'),
              path('model_total', 'pkl'), path('feature_config', 'json'), path('calibration_meta', 'json')]
    absent = [p.name for p in needed if not p.exists()]
    if absent:
        raise FileNotFoundError(f"Model set '{suffix}' in {model_dir} is incomplete; missing {absent}")

    features = json.loads(path('feature_config', 'json').read_text())
    bundle = ModelBundle(
        suffix=suffix,
        ml=joblib.load(path('model_ml', 'pkl')),
        calibrator=joblib.load(path('calibrator_ml', 'pkl')),
        spread=joblib.load(path('model_spread', 'pkl')),
        total=joblib.load(path('model_total', 'pkl')),
        features=features,
        calib_meta=json.loads(path('calibration_meta', 'json').read_text()),
    )
    sigma_path = path('sigma_total', 'pkl')
    if sigma_path.exists():
        bundle.sigma_total = joblib.load(sigma_path)
    elif bundle.calib_meta.get('total_sigma_model'):
        raise FileNotFoundError(f"Model set '{suffix}' says it uses a per-game total sigma but {sigma_path.name} is missing")
    for target in TARGETS:
        assert_config_matches_model(target, getattr(bundle, target), features[target])
    bundle.model_set_hash = model_set_hash(suffix, model_dir)
    manifest = read_manifest(suffix, model_dir)
    if manifest is not None:
        if manifest['model_set_hash'] != bundle.model_set_hash:
            raise ModelSetHashMismatch(
                f"Model set '{suffix}' was changed after it was trained: files hash to {bundle.model_set_hash[:12]} but the manifest "
                f"recorded {manifest['model_set_hash'][:12]} ({manifest['created_at']}). Do not reuse a suffix for a new model set.")
        bundle.training_set_hash = manifest.get('training_set_hash')
    logging.info(f"Loaded model set '{suffix}' ({', '.join(f'{t}: {len(features[t])} features' for t in TARGETS)})")
    return bundle
