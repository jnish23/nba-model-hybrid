import numpy as np
from scipy.stats import nbinom, poisson

from props.models import calibrate, dispersion, outcome as oc


def _pmf(mu, k=None, cap=40):
    ys = np.arange(cap + 1)[None, :]
    p = poisson.pmf(ys, np.asarray(mu)[:, None]) if k is None else nbinom.pmf(ys, k, k / (k + np.asarray(mu)[:, None]))
    p[:, -1] += 1 - p.sum(axis=1)
    return p.astype(np.float32)


def test_widen_keeps_the_mean_and_grows_the_sd():
    for mu in (1.2, 5.0, 18.0):
        pm = _pmf([mu])
        w = dispersion.widen(pm, [1.3])
        assert abs(w.sum() - 1) < 1e-5
        assert abs(oc.pmf_mean(w)[0] - oc.pmf_mean(pm)[0]) < 0.02 * max(mu, 1)
        assert calibrate.pmf_sd(w)[0] > calibrate.pmf_sd(pm)[0] * 1.15


def test_widen_with_scale_one_is_the_identity():
    pm = _pmf([3.0, 9.0])
    assert np.allclose(dispersion.widen(pm, [1.0, 1.0]), pm)


def test_scaler_repairs_underdispersion_out_of_sample():
    rng = np.random.default_rng(1)
    n = 8000
    mu = rng.uniform(3, 20, n)
    minutes = rng.uniform(5, 38, n)
    pm = _pmf(mu)                                                            # the model says Poisson ...
    obs = rng.negative_binomial(8, 8 / (8 + mu))                              # ... the truth is overdispersed
    half = n // 2
    sc = dispersion.DispersionScaler().fit('points', pm[:half], obs[:half], minutes[:half])
    assert all(sc.c[('points', k)] > 1.1 for k in range(3))
    w = sc.transform('points', pm[half:], minutes[half:])
    before = np.mean((obs[half:] - oc.pmf_mean(pm[half:])) ** 2) / np.mean(calibrate.pmf_sd(pm[half:]) ** 2)
    after = np.mean((obs[half:] - oc.pmf_mean(w)) ** 2) / np.mean(calibrate.pmf_sd(w) ** 2)
    assert abs(after - 1) < abs(before - 1) * 0.5
    assert oc.crps(w, obs[half:]).mean() < oc.crps(pm[half:], obs[half:]).mean()


def test_scaler_narrows_a_pmf_that_is_too_wide_down_to_the_floor():
    rng = np.random.default_rng(2)
    n = 6000
    mu = rng.uniform(3, 20, n)
    minutes = rng.uniform(5, 38, n)
    pm = _pmf(mu, k=4.0)                                                     # the model is wider than the truth (negative binomial with k = 4)
    obs = rng.poisson(mu)
    half = n // 2
    sc = dispersion.DispersionScaler().fit('points', pm[:half], obs[:half], minutes[:half])
    assert all(dispersion.MIN_C <= v < 1.0 for v in sc.c.values())
    w = sc.transform('points', pm[half:], minutes[half:])
    assert np.all(np.abs(oc.pmf_mean(w) - oc.pmf_mean(pm[half:])) < 0.05 * np.maximum(oc.pmf_mean(pm[half:]), 1))        # the mean is preserved
    assert calibrate.pmf_sd(w).mean() < calibrate.pmf_sd(pm[half:]).mean()
    assert oc.crps(w, obs[half:]).mean() < oc.crps(pm[half:], obs[half:]).mean()
    ratio = lambda p: np.mean((obs[half:] - oc.pmf_mean(p)) ** 2) / np.mean(calibrate.pmf_sd(p) ** 2)
    assert abs(ratio(w) - 1) < abs(ratio(pm[half:]) - 1)


def test_scaler_floor_and_cap():
    rng = np.random.default_rng(3)
    mu = rng.uniform(5, 15, 4000)
    pm = _pmf(mu, k=1.0)                                                     # absurdly wide
    sc = dispersion.DispersionScaler().fit('points', pm, rng.poisson(mu), rng.uniform(5, 38, 4000))
    assert all(v == dispersion.MIN_C for v in sc.c.values())
