"""Shared helpers for workstream A1 (season bets). Run every script from the worktree root:
   C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe reports/audit_pregame_2026-11/A1_season_bets/<script>.py
"""
import os
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[3]          # worktree root
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DB = ROOT / 'data' / 'nba_stats.db'
LOG = ROOT / 'data' / 'raw' / 'daily_inference.log'
SEED, NBOOT = 42, 1000
NORM = NormalDist()


def con():
    return sqlite3.connect(DB)


# ---------------------------------------------------------------- odds math
def implied(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))


def payout(a, stake=1.0):
    """profit on a win of `stake` at American odds a"""
    a = np.asarray(a, float)
    return np.where(a > 0, stake * a / 100, stake * 100 / -a)


def devig2(a_home, a_away):
    """two-way proportional de-vig -> P(home)"""
    ph, pa = implied(a_home), implied(a_away)
    return ph / (ph + pa)


# ---------------------------------------------------------------- bootstrap over days
def day_boot(days, num, den=None, stat='ratio', nboot=NBOOT, seed=SEED):
    """Resample game DATES with replacement. For stat='ratio' returns the bootstrap distribution of sum(num)/sum(den)
    (den=None -> count of rows; use den array for stakes).  For stat='mean' same as ratio with den = 1 per row.
    Returns (point, lo, hi, draws)."""
    days = np.asarray(days)
    num = np.asarray(num, float)
    den = np.ones_like(num) if den is None else np.asarray(den, float)
    ud, inv = np.unique(days, return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(ud))
    sd = np.bincount(inv, weights=den, minlength=len(ud))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ud), size=(nboot, len(ud)))
    bn, bd = sn[idx].sum(1), sd[idx].sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        draws = bn / bd
    point = sn.sum() / sd.sum() if sd.sum() else np.nan
    lo, hi = np.nanpercentile(draws, [2.5, 97.5]) if np.isfinite(draws).any() else (np.nan, np.nan)
    return point, lo, hi, draws


def boot_p(draws, point=None):
    """two-sided bootstrap p-value for H0: stat = 0 (share of draws on the other side of 0, doubled)"""
    d = draws[np.isfinite(draws)]
    if not len(d):
        return np.nan
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return min(1.0, max(p, 1.0 / (len(d) + 1)))


def holm(pvals):
    """Holm-adjusted p-values (step-down), NaNs ignored."""
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m = len(order)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def fmt_ci(p, lo, hi, pct=False, nd=2):
    if pct:
        return f"{100*p:+.{nd}f}% [{100*lo:+.{nd}f}, {100*hi:+.{nd}f}]"
    return f"{p:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def joint_boot(days, series, nboot=NBOOT, seed=SEED):
    """Joint day-bootstrap of several ratio statistics with the SAME resampled days (needed for paired differences).
    series: {name: (num, den)} arrays aligned with `days` (den None -> 1 per row). Returns {name: (point, draws)}."""
    days = np.asarray(days)
    ud, inv = np.unique(days, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ud), size=(nboot, len(ud)))
    out = {}
    for k, (num, den) in series.items():
        num = np.asarray(num, float)
        den = np.ones_like(num) if den is None else np.asarray(den, float)
        sn = np.bincount(inv, weights=np.nan_to_num(num), minlength=len(ud))
        sd = np.bincount(inv, weights=np.nan_to_num(den), minlength=len(ud))
        with np.errstate(invalid='ignore', divide='ignore'):
            draws = sn[idx].sum(1) / sd[idx].sum(1)
            point = sn.sum() / sd.sum() if sd.sum() else np.nan
        out[k] = (point, draws)
    return out


def ci(draws):
    d = draws[np.isfinite(draws)]
    return (np.percentile(d, 2.5), np.percentile(d, 97.5)) if len(d) else (np.nan, np.nan)
