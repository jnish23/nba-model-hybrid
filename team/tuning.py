"""Deterministic hyperparameter search shared by the walk-forward backtests and the final-model trainer.

A fixed grid of regularised LightGBM configurations scored on a time-ordered validation block. It takes seconds, does not depend on the
machine's speed (FLAML with a wall-clock limit does) and is exactly what the walk-forward evaluation (experiments/lib.py) validated, so
a model set trained with tuner='grid' is the pipeline that was backtested.
"""
import lightgbm as lgb
import numpy as np
from scipy.special import expit

SEED = 42

# Candidate hyperparameters, searched inside each fold (deterministic: same grid, same seed, same split)
GRID = [
    dict(learning_rate=0.02, n_estimators=400, num_leaves=4, min_child_samples=70, reg_lambda=20.0, colsample_bytree=0.7),
    dict(learning_rate=0.02, n_estimators=400, num_leaves=8, min_child_samples=50, reg_lambda=10.0, colsample_bytree=0.7),
    dict(learning_rate=0.01, n_estimators=800, num_leaves=4, min_child_samples=70, reg_lambda=20.0, colsample_bytree=0.7),
    dict(learning_rate=0.01, n_estimators=800, num_leaves=8, min_child_samples=50, reg_lambda=10.0, colsample_bytree=0.7),
    dict(learning_rate=0.05, n_estimators=150, num_leaves=4, min_child_samples=70, reg_lambda=20.0, colsample_bytree=0.7),
    dict(learning_rate=0.03, n_estimators=300, num_leaves=16, min_child_samples=40, reg_lambda=5.0, colsample_bytree=0.7),
    dict(learning_rate=0.01, n_estimators=1000, num_leaves=31, min_child_samples=30, reg_lambda=0.5, colsample_bytree=1.0),
    dict(learning_rate=0.02, n_estimators=600, num_leaves=6, min_child_samples=100, reg_lambda=30.0, colsample_bytree=0.7),
]


def make_est(kind, cfg):
    cls = lgb.LGBMClassifier if kind == 'clf' else lgb.LGBMRegressor
    return cls(**cfg, random_state=SEED, deterministic=True, force_row_wise=True, verbosity=-1, n_jobs=-1)


def raw_predict(est, kind, X, init=None):
    """Link-scale prediction (+ init_score offset)."""
    r = est.predict(X, raw_score=True) if kind == 'clf' else est.predict(X)
    return r + (0.0 if init is None else init)


def pick_config(kind, X, y, n_fit, grid=GRID):
    """(config, validation score): the grid entry that scores best on rows [n_fit:] after fitting on rows [:n_fit]."""
    Xa, ya, Xv, yv = X.iloc[:n_fit], y.iloc[:n_fit], X.iloc[n_fit:], y.iloc[n_fit:]
    best, best_score = None, np.inf
    for cfg in grid:
        est = make_est(kind, cfg)
        est.fit(Xa, ya)
        r = raw_predict(est, kind, Xv)
        if kind == 'clf':
            p = np.clip(expit(r), 1e-6, 1 - 1e-6)
            score = float(-(yv.to_numpy() * np.log(p) + (1 - yv.to_numpy()) * np.log(1 - p)).mean())
        else:
            score = float(np.mean((yv.to_numpy() - r) ** 2))
        if score < best_score:
            best, best_score = dict(cfg), score
    return best, best_score
