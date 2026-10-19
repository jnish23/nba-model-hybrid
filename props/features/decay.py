"""
Point-in-time exponentially decayed sums.

decayed_before(values, groups, played, alpha) returns, for EVERY row, the decayed sum of `values` over the same group's earlier PLAYED rows:

    S_before[t] = sum over played rows k < t (in the group's order) of alpha ** (number of played rows between k and t) * values[k]

so the row's own value is never included (strictly before), a DNP row neither adds to nor decays the sum, and a row inherits the state left by
the group's most recent played game. Half-life h in games: alpha = 0.5 ** (1 / h).

Rows must be sorted by (group, time). The implementation is a linear filter over each group's played rows; no row ever looks forward.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter


def alpha_for(half_life_games):
    return 0.5 ** (1.0 / half_life_games)


def decayed_before(values, groups, played, alpha):
    """
    values: array (NaN counts as 0 on played rows); groups: group ids (sorted, contiguous); played: boolean array.
    Returns an array of the same length.
    """
    values = np.asarray(values, dtype=float)
    played = np.asarray(played, dtype=bool)
    groups = np.asarray(groups)
    n = len(values)
    out = np.zeros(n)
    # group boundaries (rows are contiguous per group)
    change = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1], True])
    for lo, hi in zip(change[:-1], change[1:]):
        v = np.where(played[lo:hi], np.nan_to_num(values[lo:hi]), 0.0)
        p = played[lo:hi]
        idx = np.flatnonzero(p)
        state_after = np.zeros(hi - lo)                       # decayed sum including the row itself, defined at played rows
        if len(idx):
            filt = lfilter([1.0], [1.0, -alpha], v[idx])      # S_after over the played rows only
            state_after[idx] = filt
            # state before row i = S_after of the most recent played row strictly before i, decayed by one step per played row
            # (already applied inside filt), so it is just forward-filled from the previous played row's S_after.
            last = np.full(hi - lo, -1)
            last[idx] = idx
            last = np.maximum.accumulate(last)                # most recent played row at or before i
            prev = np.r_[-1, last[:-1]]                       # most recent played row strictly before i
            before = np.where(prev >= 0, state_after[np.maximum(prev, 0)], 0.0)
            out[lo:hi] = before
    return out


def decayed_count_before(groups, played, alpha):
    """Decayed number of earlier played games (the denominator for a decayed mean per game)."""
    return decayed_before(np.ones(len(played)), groups, played, alpha)


def effective_games(alpha, n):
    """Sum of weights after n played games (for tests)."""
    return (1 - alpha ** n) / (1 - alpha)
