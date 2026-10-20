"""
Stages D and E (decision D7): the full discrete distribution of each stat, given that the player plays.

  singles  rebounds and assists: an exact weighted mixture over the minutes support points, each a negative binomial with mean rate x minutes (no simulation noise).
  joint    points, threes and the combos (PRA, P+R, P+A, R+A): one simulation per player-game with SHARED minutes drawn from the minutes distribution;
           attempts ~ negative binomial (Poisson when k is large), makes ~ binomial, rebounds and assists ~ negative binomial. Points = 2 x two-point makes +
           3 x three-point makes + free-throw makes, so their shape (and the threes inside them) comes from the structure. Given minutes the counts are drawn
           independently, which the exploration supports (residual correlations of 0.07, 0.006 and 0.06 once minutes are fixed); a copula is added only if the
           diagnostic in tests/diagnostics says otherwise.

Everything is a probability mass function on 0..cap (the last bin absorbs anything larger) stored as float32. Simulations use a seed derived from the
rows' position, so a rerun on the same input gives the same numbers.
"""
import numpy as np
from scipy.special import gammaln

CAPS = {'points': 60, 'rebounds': 30, 'assists': 30, 'threes': 15, 'pra': 100, 'pr': 80, 'pa': 80, 'ra': 60}
SINGLES = ('rebounds', 'assists')
JOINT = ('points', 'threes', 'pra', 'pr', 'pa', 'ra')
POISSON_K = 1000.0                       # k at or above this is treated as Poisson


def nb_pmf(y, mu, k):
    """Negative-binomial pmf with mean mu and size k, broadcasting; k >= POISSON_K uses the Poisson limit."""
    mu = np.maximum(mu, 1e-9)
    if k >= POISSON_K:
        return np.exp(y * np.log(mu) - mu - gammaln(y + 1))
    return np.exp(gammaln(y + k) - gammaln(k) - gammaln(y + 1) + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))


def mixture_pmf(vals, wts, rate, k, cap, chunk=2000):
    """Exact pmf of a count with mean rate x minutes over the weighted minutes support. vals, wts: (n, M); rate: (n,). Returns (n, cap + 1) float32."""
    n = len(rate)
    ys = np.arange(cap + 1)[None, None, :]
    out = np.empty((n, cap + 1), dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        mu = (rate[lo:hi, None] * vals[lo:hi])[:, :, None]                      # (c, M, 1)
        p = nb_pmf(ys, mu, k)                                                    # (c, M, cap + 1)
        pm = (p * wts[lo:hi, :, None]).sum(axis=1)
        pm[:, -1] += np.maximum(1.0 - pm.sum(axis=1), 0.0)                       # everything above the cap goes into the last bin
        out[lo:hi] = pm
    return out


def _draw_minutes(vals, wts, n_sims, rng):
    n, m = vals.shape
    cw = np.cumsum(wts, axis=1)
    cw /= cw[:, -1:]
    rows = np.arange(n)[:, None]
    u = rng.random((n, n_sims))
    flat = (cw + rows).ravel()                                                   # strictly increasing across rows because each cw lies in (0, 1]
    idx = np.searchsorted(flat, (u + rows).ravel(), side='left').reshape(n, n_sims) - rows * m
    idx = np.clip(idx, 0, m - 1)
    return np.take_along_axis(vals, idx, axis=1)


def _nb_draw(mu, k, rng):
    if k >= POISSON_K:
        return rng.poisson(mu)
    return rng.poisson(rng.gamma(shape=k, scale=np.maximum(mu, 1e-9) / k))


def _hist(x, cap):
    n = x.shape[0]
    x = np.minimum(x, cap).astype(np.int64)
    flat = (x + (cap + 1) * np.arange(n)[:, None]).ravel()
    h = np.bincount(flat, minlength=n * (cap + 1)).reshape(n, cap + 1)
    return (h / x.shape[1]).astype(np.float32)


def simulate_joint(vals, wts, rates, nb_k, n_sims=4000, seed=0, chunk=400):
    """
    rates: dict of per-minute arrays (n,) for fg2a, fg3a, fta, reb, ast, p2, p3, pft. Returns {stat: pmf (n, cap + 1)} for JOINT stats, plus 'sim_means'
    (the simulated means of points, threes, rebounds, assists for diagnostics).
    """
    n = len(vals)
    out = {s: np.empty((n, CAPS[s] + 1), dtype=np.float32) for s in JOINT}
    means = np.empty((n, 4))
    for ci, lo in enumerate(range(0, n, chunk)):
        hi = min(lo + chunk, n)
        rng = np.random.default_rng([seed, lo])
        m = _draw_minutes(vals[lo:hi], wts[lo:hi], n_sims, rng)                  # (c, N) shared by every stat of the same simulated game
        r = {k: v[lo:hi, None] for k, v in rates.items()}
        a2 = _nb_draw(r['fg2a'] * m, nb_k['fg2a'], rng)
        a3 = _nb_draw(r['fg3a'] * m, nb_k['fg3a'], rng)
        af = _nb_draw(r['fta'] * m, nb_k['fta'], rng)
        m2 = rng.binomial(a2, np.broadcast_to(r['p2'], a2.shape))
        m3 = rng.binomial(a3, np.broadcast_to(r['p3'], a3.shape))
        mf = rng.binomial(af, np.broadcast_to(r['pft'], af.shape))
        reb = _nb_draw(r['reb'] * m, nb_k['reb'], rng)
        ast = _nb_draw(r['ast'] * m, nb_k['ast'], rng)
        pts = 2 * m2 + 3 * m3 + mf
        vals_by = {'points': pts, 'threes': m3, 'pra': pts + reb + ast, 'pr': pts + reb, 'pa': pts + ast, 'ra': reb + ast}
        for s in JOINT:
            out[s][lo:hi] = _hist(vals_by[s], CAPS[s])
        means[lo:hi] = np.c_[pts.mean(axis=1), m3.mean(axis=1), reb.mean(axis=1), ast.mean(axis=1)]
    out['sim_means'] = means
    return out


def outcome_distributions(vals, wts, rates, nb_k, n_sims=4000, seed=0):
    """All eight stats for the rows. Rebounds and assists are exact mixtures; the others come from the joint simulation."""
    res = simulate_joint(vals, wts, rates, nb_k, n_sims, seed)
    for s, key in (('rebounds', 'reb'), ('assists', 'ast')):
        res[s] = mixture_pmf(vals, wts, rates[key], nb_k[key], CAPS[s])
    return res


# ---- scoring helpers on pmfs
def pmf_mean(pmf):
    return (pmf * np.arange(pmf.shape[1])[None, :]).sum(axis=1)


def crps(pmf, obs):
    """Discrete CRPS: sum over y of (F(y) - 1{y >= obs})^2."""
    cdf = np.cumsum(pmf.astype(np.float64), axis=1)
    ys = np.arange(pmf.shape[1])[None, :]
    ind = (ys >= np.minimum(obs, pmf.shape[1] - 1)[:, None]).astype(float)
    return ((cdf - ind) ** 2).sum(axis=1)


def prob_over(pmf, line):
    """P(X > line) for a half-integer or integer line (X > 24.5 is X >= 25; X > 24 is X >= 25 too: a push is not an over)."""
    k = np.floor(np.asarray(line, dtype=float)).astype(int)
    cdf = np.cumsum(pmf.astype(np.float64), axis=1)
    k = np.clip(k, 0, pmf.shape[1] - 1)
    return 1.0 - np.take_along_axis(cdf, k[:, None], axis=1)[:, 0]
