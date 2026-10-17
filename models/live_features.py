"""Helpers for building model input rows at inference time."""
import logging

import numpy as np
import pandas as pd


class MissingFeaturesError(RuntimeError):
    """A model's required features are mostly absent from the live row (wrong config or broken builder)."""


def align_features(row_df, expected_cols, model_name='model', max_missing_frac=0.10):
    """
    Return row_df[expected_cols]. Features the live builder did not produce are filled with NaN
    (LightGBM's own missing-value path - the same thing training sees for an unavailable stat,
    unlike 0.0, which is a real value for most features) and reported at ERROR level.
    If more than `max_missing_frac` of the model's features are absent the row is not scored at
    all: that means a feature-name drift or a mismatched model/config pair, not a stray gap.
    """
    expected_cols = list(expected_cols)
    missing = [c for c in expected_cols if c not in row_df.columns]
    if missing:
        frac = len(missing) / max(len(expected_cols), 1)
        logging.error(f"{model_name}: {len(missing)}/{len(expected_cols)} model features were NOT built live "
                      f"and are set to NaN: {missing}")
        if frac > max_missing_frac:
            raise MissingFeaturesError(
                f"{model_name}: {len(missing)}/{len(expected_cols)} features missing ({frac:.0%} > "
                f"{max_missing_frac:.0%}); refusing to score. Missing: {missing[:10]}")
    out = row_df.reindex(columns=expected_cols)
    return out.astype(float) if len(out) else out
