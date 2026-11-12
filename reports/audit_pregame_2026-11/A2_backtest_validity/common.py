"""Shared helpers for the A2 backtest-validity audit. Written independently of experiments/metrics.py and core/*: only plain pandas/numpy.
Run everything from the worktree root with the repo venv python."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('C:/Users/jorda/nba-pregame-audit')
import os
OUT = ROOT / 'reports/audit_pregame_2026-11/A2_backtest_validity'
# A2_RUN=<run name under reports/experiments> reruns the prediction-based checks (7, 8, 9) on another saved walk-forward run;
# outputs then go to OUT/run_<name>/ so the E25_with_odds files are not overwritten.
RUN = os.environ.get('A2_RUN', 'E25_with_odds')
if RUN != 'E25_with_odds':
    OUT = OUT / f'run_{RUN}'
    OUT.mkdir(exist_ok=True)
DB = ROOT / 'data/nba_stats.db'
TRAIN = ROOT / 'data/processed/training_set_v1_reduced_2026-11-01.csv'
# E25 artifacts are untracked, so they exist only in the production checkout; READ-ONLY use.
E25 = Path('C:/Users/jorda/nba-model-hybrid/reports/experiments') / RUN
SEED = 42


def conn():
    return sqlite3.connect(str(DB))


def load_train():
    t = pd.read_csv(TRAIN, dtype={'gameId': str})
    t['gameId'] = t['gameId'].str.zfill(10)
    t['gameDate'] = pd.to_datetime(t['gameDate'])
    return t


def load_e25(name='predictions.csv'):
    p = pd.read_csv(E25 / name, dtype={'gameId': str})
    p['gameId'] = p['gameId'].str.zfill(10)
    p['gameDate'] = pd.to_datetime(p['gameDate'])
    return p


def day_bootstrap(df, stat_fn, n=1000, seed=SEED, day_col='gameDate'):
    """Resample game DATES with replacement (1,000 draws, seed 42). stat_fn(frame) -> float or tuple/array.
    Returns (point, lo, hi) arrays (2.5/97.5 percentiles). Uses cluster sums for speed when stat_fn is None-free."""
    rng = np.random.default_rng(seed)
    days = df[day_col].to_numpy()
    uniq, inv = np.unique(days, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    point = np.atleast_1d(np.asarray(stat_fn(df), dtype=float))
    draws = np.empty((n, point.size))
    for b in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([groups[i] for i in pick])
        draws[b] = np.atleast_1d(np.asarray(stat_fn(df.iloc[idx]), dtype=float))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return point, lo, hi, draws


def fmt_ci(point, lo, hi, d=4):
    return f'{point:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'
