"""
Prices -> probabilities for two-way markets (over/under, spread sides, moneyline).

american_to_prob           raw implied probability (includes the book's margin)
devig_multiplicative       divide by the booksum
devig_power                p_i ** k with k chosen so the two probabilities sum to 1 (shrinks long shots more)
devig_shin                 Shin (1993) insider-trading model, two outcomes

All functions take array-likes (or scalars) and return numpy arrays; a NaN price on either side gives NaN
(a one-sided line cannot be de-vigged, and is flagged by quality_flags rather than guessed at).
Pure numpy/pandas: this module must stay importable without scipy.
"""
import numpy as np
import pandas as pd

METHODS = ('multiplicative', 'power', 'shin')


def american_to_prob(odds):
    o = np.asarray(odds, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))
    return np.where(np.isnan(o) | (o == 0), np.nan, p)


def prob_to_american(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(p >= 0.5, -100.0 * p / (1.0 - p), 100.0 * (1.0 - p) / p)


def _pair(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return a.copy(), b.copy()


def devig_multiplicative(q_a, q_b):
    """Fair probabilities from implied probabilities q_a, q_b: q / (q_a + q_b)."""
    a, b = _pair(q_a, q_b)
    s = a + b
    return a / s, b / s          # NaN in, NaN out


def devig_power(q_a, q_b, iters=80):
    """Find k with q_a**k + q_b**k = 1 (bisection; k >= 1 when the book has margin); return q**k."""
    a, b = _pair(q_a, q_b)
    lo, hi = np.zeros_like(a), np.full_like(a, 20.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_big = a ** mid + b ** mid > 1.0       # the sum falls as k grows
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    k = (lo + hi) / 2
    missing = np.isnan(a) | np.isnan(b)
    return np.where(missing, np.nan, a ** k), np.where(missing, np.nan, b ** k)


def devig_shin(q_a, q_b, iters=80):
    """
    Shin two-outcome de-vig. With booksum S = q_a + q_b, find the insider share z in [0, 1) such that
    sum_i (sqrt(z^2 + 4 (1 - z) q_i^2 / S) - z) / (2 (1 - z)) = 1, and return those terms.
    A book with no margin (S <= 1) gets the multiplicative answer.
    """
    a, b = _pair(q_a, q_b)
    s = a + b

    def terms(z):
        def f(q):
            return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s) - z) / (2 * (1 - z))
        return f(a), f(b)

    lo, hi = np.zeros_like(a), np.full_like(a, 0.999)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ta, tb = terms(mid)
        too_big = ta + tb > 1.0                   # the sum falls as z grows
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    ta, tb = terms((lo + hi) / 2)
    ma, mb = devig_multiplicative(a, b)
    missing = np.isnan(s)
    return (np.where(missing, np.nan, np.where(s <= 1.0, ma, ta)),
            np.where(missing, np.nan, np.where(s <= 1.0, mb, tb)))


def devig_prices(price_a, price_b, method='multiplicative'):
    """American price pair -> (fair prob a, fair prob b, booksum)."""
    qa, qb = american_to_prob(price_a), american_to_prob(price_b)
    fn = {'multiplicative': devig_multiplicative, 'power': devig_power, 'shin': devig_shin}[method]
    fa, fb = fn(qa, qb)
    return fa, fb, qa + qb


def quality_flags(over_price, under_price, line_ts=None, snapshot_ts=None, stale_after_minutes=60,
                  max_booksum=1.25):
    """
    Row-level flags for an over/under quote (arrays or Series). Returns a DataFrame of booleans:
      one_sided   at least one price missing   -> cannot be de-vigged
      no_margin   booksum <= 1                 -> crossed / arbitrage-looking quote, often stale or mis-parsed
      wide        booksum > max_booksum        -> abnormally high margin (thin or alt-style market)
      stale       the book's own last_update is more than stale_after_minutes before the snapshot
    """
    o = pd.Series(over_price, dtype=float).reset_index(drop=True)
    u = pd.Series(under_price, dtype=float).reset_index(drop=True)
    qs = pd.Series(american_to_prob(o) + american_to_prob(u))
    out = pd.DataFrame({'one_sided': o.isna() | u.isna(), 'no_margin': qs <= 1.0, 'wide': qs > max_booksum})
    if line_ts is not None and snapshot_ts is not None:
        lag = (pd.to_datetime(pd.Series(snapshot_ts), utc=True).reset_index(drop=True)
               - pd.to_datetime(pd.Series(line_ts), utc=True).reset_index(drop=True))
        out['stale'] = (lag > pd.Timedelta(minutes=stale_after_minutes)).fillna(False)
    else:
        out['stale'] = False
    return out


def consensus_probability(fair_over, groups, weights=None):
    """Mean (optionally weighted) of per-book fair over-probabilities within `groups`; NaNs are skipped."""
    s = pd.Series(np.asarray(fair_over, dtype=float))
    g = pd.Series(np.asarray(groups))
    if weights is None:
        return s.groupby(g).transform('mean').to_numpy()
    w = pd.Series(np.asarray(weights, dtype=float)).where(s.notna())
    return ((s * w).groupby(g).transform('sum') / w.groupby(g).transform('sum')).to_numpy()
