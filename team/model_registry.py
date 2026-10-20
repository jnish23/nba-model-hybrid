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
import importlib
import json
import logging
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
DEPLOYED_SUFFIX = '_with_odds_v6'

TARGETS = {'ml': 'model_ml', 'spread': 'model_spread', 'total': 'model_total'}


class ModelConfigMismatch(RuntimeError):
    pass


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
    logging.info(f"Loaded model set '{suffix}' ({', '.join(f'{t}: {len(features[t])} features' for t in TARGETS)})")
    return bundle
