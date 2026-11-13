"""
Stage F0: dispersion calibration of the predictive pmfs, by stat and predicted-minutes bucket.

The development-fold diagnostics showed the raw pmfs a little too narrow, most for bench players and for the combos (the stats share game-flow noise the model
does not carry). On the calibration window, per (stat, bucket): c = mean squared error / mean predicted variance, floored at 1 and capped. A pmf is then
widened around its own mean by sqrt(c): the pmf is read as a continuous distribution that is piecewise linear in its CDF (one bin per integer, from y-0.5 to
y+0.5), stretched about the mean, and discretized again. The location is nudged so the mean is preserved where the stretch runs into zero.
"""
import numpy as np
from scipy.special import gammaln

from props.models import calibrate, outcome as oc

METHODS = ('widen', 'nb')
NB_MIN_MEAN = 1e-9                                    # a row whose mean is below this has no usable negative binomial (the fallback to widen applies)
NB_K_MAX = 1e9                                        # size cap: beyond it the negative binomial is the Poisson to within 1e-9 of the variance, and gammaln(k) loses digits

BUCKET_EDGES = (15.0, 28.0)
MAX_C = 1.6
MIN_C = 1.0                                           # the maps CAN narrow (DispersionScaler(min_c=0.7)), but on the development folds allowing it changed almost nothing and was slightly worse
                                                      # for rebounds, so the frozen pipeline keeps the floor at 1 (reports/props/phase5_dev_gate.md)
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


def nb_moments(pmf):
    """Mean and variance of each row's pmf (rows renormalised first; the variance is the centred one, so it does not lose digits when the mean is large)."""
    p = np.asarray(pmf, np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    ys = np.arange(p.shape[1], dtype=np.float64)[None, :]
    mu = (p * ys).sum(axis=1)
    s2 = (p * (ys - mu[:, None]) ** 2).sum(axis=1)
    return mu, s2


def _nb_match(pmf, c, widened=None):
    """nb_match and its diagnostics: returns (pmf float32, info dict). info: mu, s2 (the raw pmf's moments), exists (bool, the row got a negative binomial),
    tail (mass of that negative binomial above the last bin, before folding; 0 where it does not exist). `widened`, if given, must be widen(pmf, sqrt(c)) and saves computing it again."""
    pmf = np.asarray(pmf)
    n, width = pmf.shape
    c = np.broadcast_to(np.asarray(c, np.float64), (n,))
    if not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError('nb_match: c must be finite and positive')
    mu, s2 = nb_moments(pmf)
    exists = (c > 1.0) & (mu > NB_MIN_MEAN) & (c * s2 > mu)
    tail = np.zeros(n)
    if exists.all():
        out = np.empty((n, width))
    else:
        out = (widen(pmf, np.sqrt(c)) if widened is None else np.asarray(widened)).astype(np.float64)      # the whole batch, so a fallback row is exactly what widen(pmf, sqrt(c)) returns for it
    if exists.any():
        m, v = mu[exists], c[exists] * s2[exists]
        k = np.minimum(m * m / (v - m), NB_K_MAX)                        # variance m + m^2 / k = v
        ys = np.arange(width, dtype=np.float64)[None, :]
        km, mm = k[:, None], m[:, None]
        logp = gammaln(ys + km) - gammaln(km) - gammaln(ys + 1.0) - km * np.log1p(mm / km) + ys * (np.log(mm) - np.log(km + mm))
        p = np.exp(logp)
        head = p[:, :-1]
        beyond = np.maximum(1.0 - p.sum(axis=1), 0.0)                    # P(Y > cap) of the negative binomial itself
        last = np.maximum(1.0 - head.sum(axis=1), 0.0)                   # P(Y >= cap): the last bin absorbs the whole tail, as in every pmf of the pipeline
        q = np.concatenate([head, last[:, None]], axis=1)
        out[exists] = q / q.sum(axis=1, keepdims=True)
        tail[exists] = beyond
    return out.astype(np.float32), {'mu': mu, 's2': s2, 'exists': exists, 'tail': tail}


def nb_match(pmf, c):
    """
    Negative-binomial replacement for `widen(pmf, sqrt(c))` (experiment 7e). pmf: (n, cap + 1) raw pmfs; c: variance inflation, a scalar or one value per row.

    Per row: mu and s2 are the raw pmf's mean and variance; the target variance is c * s2. Where c > 1 and c * s2 > mu a negative binomial exists with that mean and variance
    (size k = mu^2 / (c * s2 - mu)); its pmf on 0..cap (stable gammaln form, as outcome.nb_pmf) is returned with all mass above cap added to the last bin. Where c <= 1, or
    c * s2 <= mu (no negative binomial has that variance), the row is exactly what `widen(pmf, sqrt(c))` gives for it. Float32, rows sum to 1.

    What the replacement does and does not keep:
      * mean and variance of the returned row equal mu and c * s2 up to the cap fold: the fold moves the negative binomial's mass above cap onto cap, which lowers the mean and the variance a
        little. Negligible where mu + 6 sd is far below cap (threes: cap 15, rebounds and assists: 30, ra: 60); it grows as the mean approaches the cap.
      * a row with mu below NB_MIN_MEAN (a pmf concentrated on 0) has no usable negative binomial and takes the widen output.
      * the SHAPE is replaced by a two-parameter family. A raw pmf that is a mixture over the minutes distribution (bimodal, skewed, a spike at low counts) loses that structure
        wherever c > 1, however close c is to 1: at c = 1.0001 widen returns the raw pmf almost unchanged and nb_match returns a negative binomial with the same mean and (almost) the
        same variance. Only c = 1 (the floor MIN_C = 1) leaves a row alone. The distance between the two is measured in tests/props/test_a11_nb_dispersion.py.
    """
    return _nb_match(pmf, c)[0]


class DispersionScaler:
    def __init__(self, min_c=None, method='widen'):
        if method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}, got {method!r}')
        self.min_c = MIN_C if min_c is None else min_c
        self.method = method                                                           # 'widen' (the frozen pipeline) or 'nb' (experiment 7e): how transform() applies c
        self.c = {}                                                                    # (stat, bucket) -> variance inflation

    def fit(self, stat, pmf, obs, minutes_mean):
        mean, var = oc.pmf_mean(pmf), calibrate.pmf_sd(pmf) ** 2
        b = bucket_of(minutes_mean)
        for k in range(len(BUCKET_EDGES) + 1):
            sel = b == k
            if sel.sum() < MIN_ROWS:
                self.c[(stat, k)] = 1.0
                continue
            self.c[(stat, k)] = float(np.clip(np.mean((obs[sel] - mean[sel]) ** 2) / np.mean(var[sel]), self.min_c, MAX_C))
        return self

    def scale_for(self, stat, minutes_mean):
        b = bucket_of(minutes_mean)
        return np.sqrt(np.array([self.c.get((stat, k), 1.0) for k in b]))

    def c_for(self, stat, minutes_mean):
        b = bucket_of(minutes_mean)
        return np.array([self.c.get((stat, k), 1.0) for k in b])

    def transform(self, stat, pmf, minutes_mean, method=None):
        method = getattr(self, 'method', 'widen') if method is None else method          # scalers pickled before the option existed have no attribute: they widen
        if method == 'widen':
            return widen(pmf, self.scale_for(stat, minutes_mean))
        if method == 'nb':
            return nb_match(pmf, self.c_for(stat, minutes_mean))
        raise ValueError(f'method must be one of {METHODS}, got {method!r}')
