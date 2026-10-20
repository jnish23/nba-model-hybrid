"""Checks on the training frame that tell a human when the model may be learning the calendar instead of basketball."""
import numpy as np
import pandas as pd


def missingness_era_report(df, features, era_col='seasonYear', min_missing=0.02, threshold=0.5):
    """
    A feature that is empty for the early seasons (moneylines before 2019-20, injury reports before 2019-20, 12-24% of injury
    rows later) lets a tree learn "missing = old era", and null-importance selection then rewards it for separating eras rather
    than for predicting games. For each feature with at least `min_missing` missing values, the rank correlation between "is
    missing" and the season (oldest = lowest). Returns the features with |corr| >= `threshold`, strongest first, as a frame
    with columns feature, missing_share, corr_with_season.
    """
    eras = pd.Series(pd.Categorical(df[era_col].astype(str), ordered=True).codes, index=df.index)
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        miss = df[f].isna()
        share = float(miss.mean())
        if share < min_missing or share > 1 - 1e-9:
            continue
        c = float(pd.Series(miss.astype(float)).corr(eras, method='spearman'))
        if np.isfinite(c) and abs(c) >= threshold:
            rows.append({'feature': f, 'missing_share': round(share, 4), 'corr_with_season': round(c, 3)})
    return pd.DataFrame(rows, columns=['feature', 'missing_share', 'corr_with_season']).sort_values(
        'corr_with_season', key=lambda s: -s.abs()).reset_index(drop=True)
