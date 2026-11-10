"""
Print the top features (share of total gain) of a saved model set, for the ML, spread and total models.

    .venv\Scripts\python.exe scripts\analyze_feature_importance.py                       # the deployed set
    .venv\Scripts\python.exe scripts\analyze_feature_importance.py --suffix _with_odds_cand_20261019 --top 30

Loads the set through team.model_registry.load_bundle, so the manifest hash and the feature configs are checked and the
market-aware wrappers (ResidualSpreadModel) are handled. Gain is read from the underlying LightGBM booster.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from team.model_registry import DEPLOYED_SUFFIX, MODEL_DIR, TARGETS, load_bundle


def gain_importance(model):
    """Gain per feature as a Series, from the LightGBM booster inside `model` (or inside its .model wrapper)."""
    inner = getattr(model, 'model', model)
    booster = inner.booster_
    return pd.Series(booster.feature_importance(importance_type='gain'), index=booster.feature_name())


def print_feature_importances(name, model, top_n=20):
    imp = gain_importance(model).sort_values(ascending=False)
    total = imp.sum()
    pct = imp / total * 100 if total > 0 else imp * 0.0
    print(f"\n{'=' * 50}\n{name} model ({type(model).__name__})\n{'=' * 50}")
    print(f"Top {top_n} features by gain:\n")
    for feature, share in pct.head(top_n).items():
        print(f"  {share:5.1f}%  |  {feature}")
    print(f"\nFeatures with gain > 0: {(imp > 0).sum()} of {len(imp)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suffix', default=DEPLOYED_SUFFIX, help='model set suffix (default: the deployed set)')
    ap.add_argument('--model-dir', default=str(MODEL_DIR))
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args()

    bundle = load_bundle(args.suffix, args.model_dir)
    print(f"Model set {bundle.suffix}")
    for target in TARGETS:
        print_feature_importances(target, getattr(bundle, target), args.top)
