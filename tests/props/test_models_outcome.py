import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from props.models import outcome as oc


def toy(n=6, m=8, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.sort(rng.uniform(5, 38, (n, m)), axis=1)
    wts = rng.dirichlet(np.ones(m), n)
    rates = {'fg2a': rng.uniform(0.15, 0.4, n), 'fg3a': rng.uniform(0.05, 0.25, n), 'fta': rng.uniform(0.05, 0.2, n),
             'reb': rng.uniform(0.05, 0.4, n), 'ast': rng.uniform(0.03, 0.3, n),
             'p2': rng.uniform(0.4, 0.6, n), 'p3': rng.uniform(0.3, 0.42, n), 'pft': rng.uniform(0.7, 0.9, n)}
    k = {'fg2a': 80.0, 'fg3a': 5000.0, 'fta': 3.5, 'reb': 115.0, 'ast': 37.0}
    return vals, wts, rates, k


def test_nb_pmf_matches_scipy_and_poisson_limit():
    y = np.arange(0, 12)
    assert oc.nb_pmf(y, 3.2, 8.0) == pytest.approx(nbinom.pmf(y, 8.0, 8.0 / (8.0 + 3.2)))
    assert oc.nb_pmf(y, 3.2, 5000.0) == pytest.approx(poisson.pmf(y, 3.2))


def test_mixture_pmf_is_a_distribution_with_the_right_mean():
    vals, wts, rates, k = toy()
    pm = oc.mixture_pmf(vals, wts, rates['reb'], k['reb'], 30)
    assert pm.shape == (6, 31) and pm.sum(axis=1) == pytest.approx(np.ones(6), abs=1e-5) and (pm >= 0).all()
    assert oc.pmf_mean(pm) == pytest.approx((wts * vals).sum(axis=1) * rates['reb'], rel=0.01)
    # a single minutes support point reduces to a plain negative binomial
    one = oc.mixture_pmf(np.full((1, 1), 30.0), np.ones((1, 1)), np.array([0.2]), 10.0, 30)[0]
    assert one == pytest.approx(nbinom.pmf(np.arange(31), 10.0, 10.0 / (10.0 + 6.0)), abs=1e-6)


def test_joint_simulation_is_a_distribution_and_matches_analytic_means():
    vals, wts, rates, k = toy()
    res = oc.simulate_joint(vals, wts, rates, k, n_sims=20000, seed=3)
    exp_min = (wts * vals).sum(axis=1)
    for s in oc.JOINT:
        assert res[s].sum(axis=1) == pytest.approx(np.ones(6), abs=1e-4)
    mean_pts = exp_min * (2 * rates['fg2a'] * rates['p2'] + 3 * rates['fg3a'] * rates['p3'] + rates['fta'] * rates['pft'])
    assert oc.pmf_mean(res['points']) == pytest.approx(mean_pts, rel=0.03)
    assert oc.pmf_mean(res['threes']) == pytest.approx(exp_min * rates['fg3a'] * rates['p3'], rel=0.04)
    # a combo's mean is the sum of the singles' means (expectation is additive even with shared minutes)
    singles = oc.pmf_mean(res['points']) + exp_min * rates['reb'] + exp_min * rates['ast']
    assert oc.pmf_mean(res['pra']) == pytest.approx(singles, rel=0.03)


def test_shared_minutes_make_combos_more_variable_than_independent_sums():
    vals, wts, rates, k = toy(n=3)
    res = oc.simulate_joint(vals, wts, rates, k, n_sims=20000, seed=1)
    ys = np.arange(oc.CAPS['pra'] + 1)
    var_pra = (res['pra'] * ys ** 2).sum(axis=1) - oc.pmf_mean(res['pra']) ** 2
    pts_v = (res['points'] * np.arange(61) ** 2).sum(axis=1) - oc.pmf_mean(res['points']) ** 2
    reb_v = (oc.mixture_pmf(vals, wts, rates['reb'], k['reb'], 30) * np.arange(31) ** 2).sum(axis=1) - \
        oc.pmf_mean(oc.mixture_pmf(vals, wts, rates['reb'], k['reb'], 30)) ** 2
    ast_v = (oc.mixture_pmf(vals, wts, rates['ast'], k['ast'], 30) * np.arange(31) ** 2).sum(axis=1) - \
        oc.pmf_mean(oc.mixture_pmf(vals, wts, rates['ast'], k['ast'], 30)) ** 2
    assert (var_pra > pts_v + reb_v + ast_v).all()                    # the covariance through minutes is positive


def test_simulation_is_reproducible_by_seed():
    vals, wts, rates, k = toy()
    a = oc.simulate_joint(vals, wts, rates, k, n_sims=2000, seed=11)
    b = oc.simulate_joint(vals, wts, rates, k, n_sims=2000, seed=11)
    c = oc.simulate_joint(vals, wts, rates, k, n_sims=2000, seed=12)
    assert np.array_equal(a['points'], b['points']) and not np.array_equal(a['points'], c['points'])


def test_scoring_helpers():
    pm = np.array([[0.2, 0.5, 0.3]], dtype=np.float32)
    assert oc.prob_over(pm, np.array([0.5]))[0] == pytest.approx(0.8)          # X > 0.5 means X >= 1
    assert oc.prob_over(pm, np.array([1.0]))[0] == pytest.approx(0.3)          # X > 1 means X >= 2 (a push at 1 is not an over)
    assert oc.prob_over(pm, np.array([1.5]))[0] == pytest.approx(0.3)
    assert oc.crps(pm, np.array([1]))[0] == pytest.approx(0.2 ** 2 + 0.3 ** 2)  # cdf = .2, .7, 1.0 vs indicator 0, 1, 1
    assert oc.crps(pm, np.array([1]))[0] < oc.crps(pm, np.array([2]))[0]
