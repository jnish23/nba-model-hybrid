"""
Stage F0: dispersion calibration of the predictive pmfs, by stat and predicted-minutes bucket.

The development-fold diagnostics showed the raw pmfs a little too narrow, most for bench players and for the combos (the stats share game-flow noise the model
does not carry). On the calibration window, per (stat, bucket): c = mean squared error / mean predicted variance, floored at 1 and capped. A pmf is then
widened around its own mean by sqrt(c): the pmf is read as a continuous distribution that is piecewise linear in its CDF (one bin per integer, from y-0.5 to
y+0.5), stretched about the mean, and discretized again. The location is nudged so the mean is preserved where the stretch runs into zero.
"""
import numpy as np

from props.models import calibrate, outcome as oc

BUCKET_EDGES = (15.0, 28.0)
MAX_C = 1.6
MIN_C = 0.7                                           # the maps can also NARROW a pmf: the forward-chaining minutes bank and the fit-only calibration models make raw pmfs slightly too wide
MIN_SCALE = 0.5
MIN_ROWS = 300


def bucket_of(minutes_mean):
    return np.digitize(np.asarray(minutes_mean, float), BUCKET_EDGES)


def widen(pmf, scale, iterations=5):
    """Mean-preserving rescaling of each row's pmf's spread by `scale` (one value per row; above 1 widens, below 1 narrows)."""
    pmf = np.asarray(pmf, np.float64)
    n, width = pmf.shape
    scale = np.maximum(np.asarray(scale, float), MIN_SCALE)
    if np.all(scale == 1.0):
        return pmf.astype(np.float32)
    mean = oc.pmf_mean(pmf)
    grid_cdf = np.concatenate([np.zeros((n, 1)), np.cumsum(pmf, axis=1)], axis=1)       # grid_cdf[k] = P(X <= k - 1): the CDF at the bin edge k - 0.5
    grid_cdf[:, -1] = 1.0
    ys = np.arange(width)[None, :]
    loc = mean.copy()
    out = pmf
    for _ in range(iterations):
        pos = loc[:, None] + (ys + 0.5 - loc[:, None]) / scale[:, None] + 0.5         # where the stretched upper edge of bin y falls on the original grid
        pos = np.clip(pos, 0.0, width)
        lo = np.floor(pos).astype(int)
        lo = np.minimum(lo, width - 1)
        frac = pos - lo
        cdf = np.take_along_axis(grid_cdf, lo, axis=1) * (1 - frac) + np.take_along_axis(grid_cdf, lo + 1, axis=1) * frac
        cdf[:, -1] = 1.0
        out = np.diff(np.concatenate([np.zeros((n, 1)), cdf], axis=1), axis=1)
        out = np.maximum(out, 0.0)
        # the stretch is cut at zero, which lifts the mean; a higher centre raises the CDF (moves mass down) at rate 1 - 1/scale per unit, so step by the gap over that
        loc = loc + np.where(scale > 1.0, (oc.pmf_mean(out) - mean) / np.maximum(1 - 1 / scale, 0.05), 0.0)       # compressing never runs into zero: no correction needed
    return (out / out.sum(axis=1, keepdims=True)).astype(np.float32)


class DispersionScaler:
    def __init__(self):
        self.c = {}                                                                    # (stat, bucket) -> variance inflation

    def fit(self, stat, pmf, obs, minutes_mean):
        mean, var = oc.pmf_mean(pmf), calibrate.pmf_sd(pmf) ** 2
        b = bucket_of(minutes_mean)
        for k in range(len(BUCKET_EDGES) + 1):
            sel = b == k
            if sel.sum() < MIN_ROWS:
                self.c[(stat, k)] = 1.0
                continue
            self.c[(stat, k)] = float(np.clip(np.mean((obs[sel] - mean[sel]) ** 2) / np.mean(var[sel]), MIN_C, MAX_C))
        return self

    def scale_for(self, stat, minutes_mean):
        b = bucket_of(minutes_mean)
        return np.sqrt(np.array([self.c.get((stat, k), 1.0) for k in b]))

    def transform(self, stat, pmf, minutes_mean):
        return widen(pmf, self.scale_for(stat, minutes_mean))
