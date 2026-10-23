"""
Shared training utilities: deterministic LightGBM boosters with early stopping on a time-ordered validation block, blocked out-of-fold predictions
(used only INSIDE a training window, to get honest residual sizes), and the configuration log that feeds the multiple-testing count.
"""
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np

DEFAULT_PARAMS = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=10.0, verbosity=-1, num_threads=8, seed=7, deterministic=True, force_row_wise=True)
VAL_FRAC = 0.15                       # share of the fit window (by date, the latest part) used only to pick the number of boosting rounds
MAX_ROUNDS = 1500
EARLY_STOP = 50

CONFIG_LOG = Path('data/props_models/config_log.jsonl')


def log_config(kind, **info):
    """Every model configuration tried is appended here, so the number of looks at the data can be counted and corrected for."""
    CONFIG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'kind': kind, **info}, default=str) + '\n')


def split_by_date(dates, frac=VAL_FRAC):
    """Positional indices of the earlier (1 - frac) and the later (frac) part of the rows, split on a date boundary."""
    d = np.asarray(dates)
    u = np.unique(d)
    cut = u[int(np.floor(len(u) * (1 - frac)))]
    return np.flatnonzero(d < cut), np.flatnonzero(d >= cut)


def _dataset(X, y, w, reference=None):
    return lgb.Dataset(X, label=y, weight=w, reference=reference, free_raw_data=False)


def fit_booster(objective, X, y, w, dates, params=None, extra=None):
    """
    Early-stops on the latest VAL_FRAC of the rows (by date) and returns (booster trained on the rest, best number of rounds).
    objective: 'binary', 'regression', 'poisson', 'cross_entropy'.
    """
    p = {**DEFAULT_PARAMS, **(params or {}), 'objective': objective, **(extra or {})}
    tr, va = split_by_date(dates)
    dtr, dva = _dataset(X.iloc[tr], y[tr], w[tr]), _dataset(X.iloc[va], y[va], w[va])
    b = lgb.train(p, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return b, max(int(b.best_iteration), 20)


def refit(objective, X, y, w, n_rounds, params=None, extra=None):
    p = {**DEFAULT_PARAMS, **(params or {}), 'objective': objective, **(extra or {})}
    return lgb.train(p, _dataset(X, y, w), num_boost_round=n_rounds)


def blocked_oof(objective, X, y, w, dates, n_rounds, n_blocks=4, params=None, extra=None):
    """
    Out-of-fold predictions inside one training window: rows are cut into n_blocks contiguous date blocks; block k is predicted by a model trained on
    the other blocks. This is not walk-forward (a block is predicted using later blocks) but it never leaves the training window, and it is used only to
    size residuals (the spread model and the residual banks), where an honest out-of-sample error is what matters.
    """
    d = np.asarray(dates)
    u = np.unique(d)
    edges = [u[int(len(u) * k / n_blocks)] for k in range(n_blocks)] + [u[-1] + 'z']
    out = np.full(len(d), np.nan)
    for k in range(n_blocks):
        te = np.flatnonzero((d >= edges[k]) & (d < edges[k + 1]))
        tr = np.flatnonzero((d < edges[k]) | (d >= edges[k + 1]))
        if len(te) == 0 or len(tr) == 0:
            continue
        m = refit(objective, X.iloc[tr], y[tr], w[tr], n_rounds, params, extra)
        out[te] = m.predict(X.iloc[te])
    return out


def logloss(y, p, w=None, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(ll, weights=w))


def forward_oof(objective, X, y, w, dates, n_rounds, n_blocks=5, params=None, extra=None, usable=None, min_train=100):
    """
    Forward-chaining out-of-fold predictions inside one training window: rows are cut into n_blocks contiguous date blocks and block k (k >= 1) is predicted
    by a model trained ONLY on earlier blocks (rows flagged `usable`). Block 0 is never predicted. Unlike blocked_oof nothing in the future helps predict the past,
    which is what makes the residuals honest for a forecaster; use it for the spread of an error as well (train it on the earlier blocks' residuals).
    """
    d = np.asarray(dates)
    u = np.unique(d)
    edges = [u[int(len(u) * k / n_blocks)] for k in range(n_blocks)] + [u[-1] + 'z']
    usable = np.ones(len(d), bool) if usable is None else np.asarray(usable, bool)
    out = np.full(len(d), np.nan)
    for k in range(1, n_blocks):
        te = np.flatnonzero((d >= edges[k]) & (d < edges[k + 1]))
        tr = np.flatnonzero((d < edges[k]) & usable)
        if len(te) == 0 or len(tr) < min_train:
            continue
        m = refit(objective, X.iloc[tr], y[tr], w[tr], n_rounds, params, extra)
        out[te] = m.predict(X.iloc[te])
    return out
