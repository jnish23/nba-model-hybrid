"""
Experiment 7e (plan reports/props/audit_2026-11/WP6_WP7_experiment_plan.md, D "7e"), step S8: dispersion.nb_match, DispersionScaler(method=...) and scripts/a11_nb_dispersion.py.
Everything here runs on small synthetic pmfs; no model is fitted and nothing under data/ is read.
"""
import hashlib
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import a11_nb_dispersion as a11  # noqa: E402
from props.models import calibrate, dispersion, evaluate as ev, outcome as oc  # noqa: E402


# ---------------------------------------------------------------------------------------------------- the frozen copy of today's code (props/models/dispersion.py at HEAD before 7e)

class _Frozen:
    BUCKET_EDGES = (15.0, 28.0)
    MAX_C = 1.6
    MIN_C = 1.0
    MIN_SCALE = 0.5
    MIN_ROWS = 300

    @staticmethod
    def bucket_of(minutes_mean):
        return np.digitize(np.asarray(minutes_mean, float), _Frozen.BUCKET_EDGES)

    @staticmethod
    def widen(pmf, scale, iterations=5):
        pmf = np.asarray(pmf, np.float64)
        n, width = pmf.shape
        scale = np.maximum(np.asarray(scale, float), _Frozen.MIN_SCALE)
        if np.all(scale == 1.0):
            return pmf.astype(np.float32)
        mean = oc.pmf_mean(pmf)
        grid_cdf = np.concatenate([np.zeros((n, 1)), np.cumsum(pmf, axis=1)], axis=1)
        grid_cdf[:, -1] = 1.0
        ys = np.arange(width)[None, :]
        loc = mean.copy()
        out = pmf
        for _ in range(iterations):
            pos = loc[:, None] + (ys + 0.5 - loc[:, None]) / scale[:, None] + 0.5
            pos = np.clip(pos, 0.0, width)
            lo = np.floor(pos).astype(int)
            lo = np.minimum(lo, width - 1)
            frac = pos - lo
            cdf = np.take_along_axis(grid_cdf, lo, axis=1) * (1 - frac) + np.take_along_axis(grid_cdf, lo + 1, axis=1) * frac
            cdf[:, -1] = 1.0
            out = np.diff(np.concatenate([np.zeros((n, 1)), cdf], axis=1), axis=1)
            out = np.maximum(out, 0.0)
            loc = loc + np.where(scale > 1.0, (oc.pmf_mean(out) - mean) / np.maximum(1 - 1 / scale, 0.05), 0.0)
        return (out / out.sum(axis=1, keepdims=True)).astype(np.float32)

    class Scaler:
        def __init__(self, min_c=None):
            self.min_c = _Frozen.MIN_C if min_c is None else min_c
            self.c = {}

        def fit(self, stat, pmf, obs, minutes_mean):
            mean, var = oc.pmf_mean(pmf), calibrate.pmf_sd(pmf) ** 2
            b = _Frozen.bucket_of(minutes_mean)
            for k in range(len(_Frozen.BUCKET_EDGES) + 1):
                sel = b == k
                if sel.sum() < _Frozen.MIN_ROWS:
                    self.c[(stat, k)] = 1.0
                    continue
                self.c[(stat, k)] = float(np.clip(np.mean((obs[sel] - mean[sel]) ** 2) / np.mean(var[sel]), self.min_c, _Frozen.MAX_C))
            return self

        def scale_for(self, stat, minutes_mean):
            b = _Frozen.bucket_of(minutes_mean)
            return np.sqrt(np.array([self.c.get((stat, k), 1.0) for k in b]))

        def transform(self, stat, pmf, minutes_mean):
            return _Frozen.widen(pmf, self.scale_for(stat, minutes_mean))


# ---------------------------------------------------------------------------------------------------- helpers

def _poisson_pmf(mu, cap):
    ys = np.arange(cap + 1)[None, :]
    p = poisson.pmf(ys, np.asarray(mu, float)[:, None])
    p[:, -1] += np.maximum(1 - p.sum(axis=1), 0.0)
    return p.astype(np.float32)


def _nb_exact(mu, k, cap):
    ys = np.arange(cap + 1)[None, :]
    mu = np.asarray(mu, float)[:, None]
    p = nbinom.pmf(ys, k, k / (k + mu))
    p[:, -1] += np.maximum(1 - p.sum(axis=1), 0.0)
    return p.astype(np.float32)


def _mixture(vals, wts, rate, k, cap):
    return oc.mixture_pmf(np.array([vals], float), np.array([wts], float), np.array([rate], float), k, cap)


def _tv(a, b):
    return 0.5 * np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).sum(axis=1)


def _sample(pmf, rng):
    cdf = np.cumsum(pmf.astype(np.float64), axis=1)
    return (cdf < rng.random(len(pmf))[:, None]).sum(axis=1)


# ---------------------------------------------------------------------------------------------------- default path unchanged

def test_default_widen_path_is_byte_identical_to_the_frozen_copy():
    rng = np.random.default_rng(0)
    pm = np.concatenate([_poisson_pmf(rng.uniform(0.3, 20, 60), 40), _nb_exact(rng.uniform(1, 15, 60), 4.0, 40),
                         _mixture([8, 20, 33], [0.2, 0.3, 0.5], 0.3, 6.0, 40).repeat(10, axis=0)])
    for scale in (rng.uniform(0.5, 1.4, len(pm)), np.full(len(pm), 1.0), np.full(len(pm), 1.2), np.where(rng.random(len(pm)) < 0.5, 1.0, 1.15)):
        assert dispersion.widen(pm, scale).tobytes() == _Frozen.widen(pm, scale).tobytes()
    assert dispersion.widen(pm, 1.2 * np.ones(len(pm)), iterations=3).tobytes() == _Frozen.widen(pm, 1.2 * np.ones(len(pm)), iterations=3).tobytes()


@pytest.mark.parametrize('min_c', [None, 0.7])
def test_default_scaler_fit_and_transform_are_byte_identical(min_c):
    rng = np.random.default_rng(1)
    n = 3000
    mu, minutes = rng.uniform(2, 20, n), rng.uniform(4, 38, n)
    pm = _poisson_pmf(mu, 40)
    obs = rng.negative_binomial(8, 8 / (8 + mu))
    new = dispersion.DispersionScaler(min_c=min_c).fit('points', pm, obs, minutes)
    old = _Frozen.Scaler(min_c=min_c).fit('points', pm, obs, minutes)
    assert new.c == old.c and new.min_c == old.min_c
    assert new.method == 'widen'
    assert np.array_equal(new.scale_for('points', minutes), old.scale_for('points', minutes))
    ref = old.transform('points', pm, minutes).tobytes()
    assert new.transform('points', pm, minutes).tobytes() == ref
    assert new.transform('points', pm, minutes, method='widen').tobytes() == ref
    revived = pickle.loads(pickle.dumps(new))
    del revived.__dict__['method']                                                   # a scaler pickled before the option existed (every deployed bundle)
    assert revived.transform('points', pm, minutes).tobytes() == ref
    with pytest.raises(ValueError):
        dispersion.DispersionScaler(method='gamma')
    with pytest.raises(ValueError):
        new.transform('points', pm, minutes, method='gamma')


# ---------------------------------------------------------------------------------------------------- nb_match

def test_nb_match_hits_the_target_mean_and_variance_when_the_cap_is_far():
    rng = np.random.default_rng(2)
    n = 400
    mu = rng.uniform(0.4, 8.0, n)
    raw = np.concatenate([_poisson_pmf(mu[:200], 60), _nb_exact(mu[200:], 12.0, 60)])
    c = rng.uniform(1.01, 1.6, n)
    out = dispersion.nb_match(raw, c)
    m0, v0 = dispersion.nb_moments(raw)
    m1, v1 = dispersion.nb_moments(out)
    assert np.allclose(m1, m0, rtol=1e-4, atol=1e-6)
    assert np.allclose(v1, c * v0, rtol=1e-3, atol=1e-5)
    p, info = dispersion._nb_match(raw, c)
    assert info['exists'].all() and info['tail'].max() < 1e-6
    # it is the negative binomial of that mean and variance: compare with scipy
    k = m0 / (c * v0 / m0 - 1.0)
    ref = nbinom.pmf(np.arange(61)[None, :], k[:, None], (k / (k + m0))[:, None])
    assert np.abs(out[:, :-1] - ref[:, :-1]).max() < 1e-5


def test_poisson_2_5_with_c_1_09_the_plan_example():
    raw = _poisson_pmf([2.5], 15)
    nb = dispersion.nb_match(raw, [1.09])
    wd = dispersion.widen(raw, np.sqrt([1.09]))
    k = 2.5 ** 2 / (1.09 * 2.5 - 2.5)
    exact = (k / (k + 2.5)) ** k                                                       # P(0) of the negative binomial with mean 2.5, variance 2.725
    assert abs(nb[0, 0] - exact) < 2e-4 and 0.0905 < nb[0, 0] < 0.0920
    assert 0.1005 < wd[0, 0] < 0.1015                                                  # the plan's 0.101 for widen, 0.091 for the variance-matched negative binomial
    print(f'Poisson(2.5), c = 1.09: P(0) raw {raw[0, 0]:.4f}, widen {wd[0, 0]:.4f}, nb {nb[0, 0]:.4f} (exact NB {exact:.4f}); variance widen {dispersion.nb_moments(wd)[1][0]:.4f}, '
          f'nb {dispersion.nb_moments(nb)[1][0]:.4f}, target {1.09 * 2.5:.4f}')


def test_fallback_rows_are_exactly_the_widen_output():
    rng = np.random.default_rng(3)
    binom = np.zeros((1, 21))
    binom[0, :11] = [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]
    binom /= 1024.0                                                                    # Binomial(10, 0.5): mean 5, variance 2.5
    delta = np.zeros((1, 21))
    delta[0, 0] = 1.0                                                                  # a pmf that sits on 0: mean 0
    pois = _poisson_pmf([4.0], 20)
    raw = np.concatenate([binom, pois, delta, pois, binom, pois]).astype(np.float32)
    c = np.array([1.2, 1.2, 1.2, 1.0, 0.9, 1.3])                                       # 0: c * s2 = 3 <= mu = 5; 1: exists; 2: mu = 0; 3: c = 1; 4: c < 1; 5: exists
    out, info = dispersion._nb_match(raw, c)
    ref = dispersion.widen(raw, np.sqrt(c))
    assert info['exists'].tolist() == [False, True, False, False, False, True]
    for i in (0, 2, 3, 4):
        assert out[i].tobytes() == ref[i].tobytes()
    assert not np.array_equal(out[1], ref[1]) and not np.array_equal(out[5], ref[5])
    assert np.isfinite(out).all()
    assert dispersion.nb_match(raw, c).tobytes() == out.tobytes()


def test_rows_sum_to_one_are_non_negative_float32_even_in_extreme_rows():
    rng = np.random.default_rng(4)
    mu = np.concatenate([[1e-4, 1e-6, 0.01, 0.05], rng.uniform(0.1, 25, 200), [29.0, 40.0]])
    raw = _poisson_pmf(mu, 30)
    for c in (1.0001, 1.3, 1.6):
        out = dispersion.nb_match(raw, np.full(len(mu), c))
        assert out.dtype == np.float32 and out.shape == raw.shape
        assert np.isfinite(out).all() and (out >= 0).all()
        assert np.abs(out.astype(np.float64).sum(axis=1) - 1).max() < 1e-5
    assert dispersion.nb_match(raw, 1.2).shape == raw.shape                             # a scalar c broadcasts
    with pytest.raises(ValueError):
        dispersion.nb_match(raw, np.zeros(len(mu)))


def test_the_cap_fold_lowers_mean_and_variance_when_the_mean_is_near_the_cap():
    raw = _nb_exact([20.0], 30.0, 30)
    c = 1.5
    out, info = dispersion._nb_match(raw, [c])
    m0, v0 = dispersion.nb_moments(raw)
    m1, v1 = dispersion.nb_moments(out)
    assert info['exists'][0] and info['tail'][0] > 0.01                                # real mass above the cap is folded onto it
    assert m1[0] < m0[0] - 0.05 and v1[0] < c * v0[0] - 1.0                           # so the returned mean and variance fall short of the targets
    assert abs(out.astype(np.float64).sum() - 1) < 1e-6
    far = _nb_exact([4.0], 30.0, 30)
    assert dispersion._nb_match(far, [c])[1]['tail'][0] < 1e-6


def test_minutes_mixture_is_far_from_the_raw_pmf_at_c_barely_above_one_for_nb_but_not_for_widen():
    # a bimodal minutes mixture: bench minutes (a spike at low counts) and starter minutes
    for name, raw in (('threes', _mixture([4, 12, 22, 30, 36], [0.15, 0.2, 0.25, 0.25, 0.15], 0.06, 12.0, 15)), ('rebounds', _mixture([6, 14, 24, 32], [0.1, 0.2, 0.4, 0.3], 0.25, 8.0, 30))):
        c = 1.0001
        wd, nb = dispersion.widen(raw, np.sqrt([c])), dispersion.nb_match(raw, [c])
        tv_w, tv_n = _tv(wd, raw)[0], _tv(nb, raw)[0]
        p0_w, p0_n = wd[0, 0] - raw[0, 0], nb[0, 0] - raw[0, 0]
        print(f'{name} mixture, c = {c}: TV to raw widen {tv_w:.5f} nb {tv_n:.5f}; P(0) shift widen {p0_w:+.5f} nb {p0_n:+.5f}; P(Y>1.5) shift widen '
              f'{(1 - wd[0, :2].sum()) - (1 - raw[0, :2].sum()):+.5f} nb {(1 - nb[0, :2].sum()) - (1 - raw[0, :2].sum()):+.5f}')
        assert tv_w < 1e-3                                                             # widen at c = 1.0001 leaves the raw pmf alone
        assert tv_n > 0.01 and tv_n > 20 * tv_w                                        # nb at c = 1.0001 replaces its shape by a two-parameter family: the DESIGN CONCERN
        assert abs(p0_n) > 0.005 and abs(p0_w) < 1e-3
    exact_c1 = dispersion.nb_match(raw, [1.0])
    assert exact_c1.tobytes() == dispersion.widen(raw, np.sqrt([1.0])).tobytes()        # only c = 1 exactly leaves a row alone


def test_scaler_nb_method_uses_c_per_bucket():
    rng = np.random.default_rng(5)
    mu, minutes = rng.uniform(1, 8, 500), rng.uniform(4, 38, 500)
    raw = _poisson_pmf(mu, 30)
    sc = dispersion.DispersionScaler(method='nb')
    sc.c = {('threes', 0): 1.0, ('threes', 1): 1.2, ('threes', 2): 1.5}
    out = sc.transform('threes', raw, minutes)
    assert out.tobytes() == dispersion.nb_match(raw, sc.c_for('threes', minutes)).tobytes()
    assert sc.transform('threes', raw, minutes, method='widen').tobytes() == dispersion.widen(raw, sc.scale_for('threes', minutes)).tobytes()
    low = dispersion.bucket_of(minutes) == 0
    assert out[low].tobytes() == dispersion.widen(raw, sc.scale_for('threes', minutes))[low].tobytes()      # the c = 1 bucket is widen's output, i.e. the raw pmf


# ---------------------------------------------------------------------------------------------------- the script: arithmetic

def _fake_stat(n, crps_ratio=1.0, nb_closer=True, p0_nb_bias=0.0, p0_w_bias=0.0):
    """Frames with constant per-class differences, so every paired interval collapses to a point (exact arithmetic)."""
    y = np.tile([0, 1], n // 2)                                                         # half zeros, half ones
    hit = (y > 0.5).astype(float)
    p_w = np.full(n, 0.5)
    p_n = np.where(hit == 1, 0.9, 0.1) if nb_closer else np.where(hit == 1, 0.1, 0.9)
    # y is 0 or 1, so P(Y > 1.5) never happens: the nb arm's p_over15 is 0.1 (closer) or 0.9 (further), widen's 0.5
    return pd.DataFrame({'game_id': (np.arange(n) // 2).astype(str), 'y': y, 'crps_w': np.full(n, 1.0), 'crps_n': np.full(n, crps_ratio),
                         'p_over05_w': p_w, 'p_over05_n': p_n, 'p_over15_w': p_w, 'p_over15_n': np.full(n, 0.1 if nb_closer else 0.9), 'p0_w': np.full(n, 0.5 + p0_w_bias), 'p0_n': np.full(n, 0.5 + p0_nb_bias)})


def test_log_loss_arithmetic():
    assert np.allclose(a11.ll([0.9, 0.2], [1, 0]), [-np.log(0.9), -np.log(0.8)])
    assert np.allclose(a11.ll([0.0, 1.0], [1, 0]), [-np.log(1e-4), -np.log(1 - (1 - 1e-4))])        # clipped at 1e-4
    df = _fake_stat(200)
    r = a11.logscore_row(df, 'threes', 0.5, 'dev_pooled', 'all_played', False, n_boot=100)
    assert abs(r['logscore_widen'] - np.log(2)) < 1e-12 and abs(r['logscore_nb'] - (-np.log(0.9))) < 1e-12
    assert abs(r['diff_nb_minus_widen'] - (-np.log(0.9) - np.log(2))) < 1e-12 and r['hi95'] < 0


def test_keep_rule_clause_by_clause_arithmetic():
    def pooled(**kw):
        return {s: _fake_stat(400, **kw.get(s, {})) for s in a11.BET_STATS}
    ok, verdict = a11.keep_rule(pooled(**{s: {'crps_ratio': 1.0004} for s in a11.BET_STATS}), n_boot=100, n_boot_bonf=100)
    assert verdict and ok['passed'].all() and len(ok) == 7                              # +0.04% CRPS is inside the +0.05% allowance
    bad_crps = a11.keep_rule(pooled(ra={'crps_ratio': 1.0006}), n_boot=100, n_boot_bonf=100)[0]
    assert not bad_crps.loc[(bad_crps['clause'] == 3) & bad_crps['what'].str.startswith('ra'), 'passed'].item() and bad_crps['passed'].sum() == 6
    worse_ll = a11.keep_rule(pooled(threes={'nb_closer': False}), n_boot=100, n_boot_bonf=100)[0]
    assert not worse_ll.loc[worse_ll['clause'].isin([1, 2]), 'passed'].any()
    p0 = a11.keep_rule(pooled(threes={'p0_nb_bias': 0.05}), n_boot=100, n_boot_bonf=100)[0]       # observed P(0) is 0.5; nb says 0.55 where widen says 0.5
    assert not p0.loc[p0['clause'] == 4, 'passed'].item() and p0.loc[p0['clause'] == 1, 'passed'].item()
    tie = a11.keep_rule(pooled(threes={'p0_nb_bias': 0.05, 'p0_w_bias': -0.05}), n_boot=100, n_boot_bonf=100)[0]
    assert tie.loc[tie['clause'] == 4, 'passed'].item()                                  # equal absolute bias: "no larger" holds
    missing = {s: _fake_stat(400) for s in a11.BET_STATS if s != 'ra'}
    assert not a11.keep_rule(missing, n_boot=100, n_boot_bonf=100)[1]


def _simulated_frames(truth, c=1.6, seed=7):
    """Raw Poisson pmfs, a common c, and counts drawn from a true negative binomial (nb is right) or from the widen pmf itself (widen is right)."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in a11.BET_STATS:
        n = 30000 if s == 'threes' else 12000
        mu = rng.uniform(1.0, 2.5, n) if s == 'threes' else rng.uniform(2.0, 8.0, n)
        raw = _poisson_pmf(mu, oc.CAPS[s])
        cc = np.full(n, c)
        if truth == 'nb':
            k = mu / (c - 1.0)
            y = rng.negative_binomial(k, k / (k + mu))
        else:
            y = _sample(dispersion.widen(raw, np.sqrt(cc)), rng)
        f = a11.row_metrics(raw, cc, y)
        f.insert(0, 'game_id', (np.arange(n) // 2).astype(str))
        out[s] = f
    return out


def test_keep_rule_passes_on_counts_from_a_true_negative_binomial_and_fails_where_widen_is_right():
    keep_nb, adopt_nb = a11.keep_rule(_simulated_frames('nb'), n_boot=500, n_boot_bonf=500)
    assert adopt_nb, keep_nb[['clause', 'what', 'passed', 'value']].to_string()
    keep_w, adopt_w = a11.keep_rule(_simulated_frames('widen'), n_boot=500, n_boot_bonf=500)
    assert not adopt_w
    assert not keep_w.loc[keep_w['clause'] == 3, 'passed'].any()                       # nb is worse in CRPS beyond the allowance in every bet stat
    assert not keep_w.loc[keep_w['clause'] == 1, 'passed'].item() and not keep_w.loc[keep_w['clause'] == 4, 'passed'].item()


def test_paired_crps_difference_matches_evaluate_compare_versions():
    rng = np.random.default_rng(8)
    n = 600
    mu = rng.uniform(1.0, 3.0, n)
    raw = _poisson_pmf(mu, 15)
    c = np.full(n, 1.3)
    y = rng.negative_binomial(mu / 0.3, (mu / 0.3) / (mu / 0.3 + mu))
    f = a11.row_metrics(raw, c, y)
    rows = pd.DataFrame({'game_id': (np.arange(n) // 3).astype(str), 'player_id': np.arange(n), 'y_threes': y.astype(float)})
    cmp = ev.compare_versions(rows, {'threes': dispersion.widen(raw, np.sqrt(c))}, rows, {'threes': dispersion.nb_match(raw, c)}, stats=['threes'], n_boot=300).iloc[0]
    r = a11.paired(f['crps_n'] - f['crps_w'], rows['game_id'].to_numpy(), bonf=False, n_boot=300)
    assert abs(cmp['B_minus_A'] - r['diff']) < 1e-12 and abs(cmp['lo'] - r['lo']) < 1e-12 and abs(cmp['hi'] - r['hi']) < 1e-12


def test_bonferroni_interval_is_wider_than_95():
    rng = np.random.default_rng(9)
    d = rng.normal(0.0, 1.0, 3000)
    g = np.repeat(np.arange(1000), 3)
    r = a11.paired(d, g, bonf=True, n_boot=500, n_boot_bonf=2000)
    assert r['bonf_lo'] < r['lo'] < r['hi'] < r['bonf_hi'] and abs(a11.ALPHA_BONF - 0.05 / 23) < 1e-15


# ---------------------------------------------------------------------------------------------------- the script: refusals and an end-to-end run on a synthetic registry

def test_refuse_holdout_season():
    a11.refuse_holdout(['2021-22', '2024-25'])
    for bad in (['2025-26'], ['2023-24', '2026-27'], ['2030-31']):
        with pytest.raises(SystemExit, match='refuses'):
            a11.refuse_holdout(bad, '--folds')
    with pytest.raises(SystemExit):
        a11.refuse_holdout(['2025'])


def _args(tmp, **kw):
    base = ['--raw-version', 'raw', '--base-version', 'base', '--registry-root', str(tmp / 'reg'), '--prices-dir', str(tmp / 'prices'), '--features-db', str(tmp / 'f.db'),
            '--frozen-rule', str(tmp / 'rule.json'), '--out', str(tmp / 'out'), '--report', str(tmp / 'rpt' / 'report.md'), '--n-boot', '100', '--n-boot-bonf', '200']
    for k, v in kw.items():
        base += [f'--{k}'] + (v if isinstance(v, list) else [v])
    return base


def _snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(path).rglob('*')) if p.is_file()}


def _make_world(tmp, late_date=False, seasons=('2021-22', '2022-23', '2023-24', '2024-25')):
    """A tiny registry: raw pmfs (Poisson, 4 bet stats) + index per season, base dispersion factors (a json for three seasons, a bundle.pkl for the last), labels, a 2023-24 price file."""
    rng = np.random.default_rng(11)
    reg = tmp / 'reg'
    (tmp / 'prices').mkdir()
    labels = []
    cons_rows, bet_rows = [], []
    for si, season in enumerate(seasons):
        n_games, per_game = 40, 8
        gids = [f'00{21 + si}{g:05d}' for g in range(n_games)]
        gdate = lambda g: f'{2021 + si}-1{si % 3}-{1 + g % 28:02d}' if not late_date else '2025-07-15'
        rows = []
        for g in range(n_games):
            for p in range(per_game):
                rows += [{'game_id': gids[g], 'player_id': 1000 * (g + 1) + p, 'minutes_before': t, 'season': season, 'game_date': gdate(g), 'minutes_mean': float(rng.uniform(5, 36))} for t in (240, 60, 5)]
        idx = pd.DataFrame(rows)
        d = reg / 'raw' / season
        d.mkdir(parents=True)
        idx.to_csv(d / 'index.csv', index=False)
        mu = {'threes': rng.uniform(0.8, 2.6, len(idx)), 'rebounds': rng.uniform(2, 9, len(idx)), 'assists': rng.uniform(1, 7, len(idx))}
        for s in ('threes', 'rebounds', 'assists'):
            np.save(d / f'raw_{s}.npy', _poisson_pmf(mu[s], oc.CAPS[s]))
        np.save(d / 'raw_ra.npy', _poisson_pmf(mu['rebounds'] + mu['assists'], oc.CAPS['ra']))
        pl = idx.drop_duplicates(['game_id', 'player_id']).reset_index(drop=True)
        lab = pl[['game_id', 'player_id']].copy()
        pm = idx.groupby(['game_id', 'player_id']).first().reset_index()
        lab['lbl_played'], lab['lbl_minutes'] = 1, 20.0
        lab['y_points'] = rng.poisson(10, len(lab)).astype(float)
        lab['y_rebounds'] = rng.poisson(5, len(lab)).astype(float)
        lab['y_assists'] = rng.poisson(3, len(lab)).astype(float)
        lab['y_threes'] = rng.poisson(1.5, len(lab)).astype(float)
        lab['y_pra'] = lab['y_points'] + lab['y_rebounds'] + lab['y_assists']
        labels.append(lab)
        sc = dispersion.DispersionScaler()
        sc.c = {(s, b): 1.0 + 0.1 * b + 0.05 * si for s in ('threes', 'rebounds', 'assists', 'ra') for b in range(3)}
        sc.c[('threes', 0)] = 1.0                                                      # one bucket at the floor
        bd = reg / 'base' / season
        bd.mkdir(parents=True)
        if season == seasons[-1]:
            joblib.dump({'dispersion': sc}, bd / 'bundle.pkl')
        else:
            (bd / 'dispersion_c.json').write_text(json.dumps({f'{k[0]}:{k[1]}': v for k, v in sc.c.items()}))
        if season == '2023-24':                                                          # quotes: threes at 0.5 / 1.5, three books
            th = idx[idx['minutes_before'] == 60].reset_index(drop=True)
            for i, r in th.iterrows():
                line = 0.5 if i % 2 == 0 else 1.5
                cons_rows.append({'game_id': r['game_id'], 'player_id': r['player_id'], 'minutes_before': 60, 'stat': 'threes', 'consensus_line': line, 'consensus_fair_over': 0.55 if line == 0.5 else 0.3, 'n_at_line': 5})
                for book, (o, u) in zip(('draftkings', 'fanduel', 'betmgm'), ((1.8, 2.0), (1.85, 1.95), (1.9, 1.9))):
                    bet_rows.append({'game_id': r['game_id'], 'player_id': r['player_id'], 'minutes_before': 60, 'stat': 'threes', 'book': book, 'line': line, 'over_dec': o, 'under_dec': u,
                                     'consensus_fair_over': 0.55 if line == 0.5 else 0.3, 'n_at_line': 5})
    lab = pd.concat(labels, ignore_index=True)
    with sqlite3.connect(tmp / 'f.db') as conn:
        lab.to_sql('player_game_labels', conn, index=False)
    with open(tmp / 'prices' / 'prices_2023-24.pkl', 'wb') as f:
        pickle.dump((pd.DataFrame(cons_rows), pd.DataFrame(bet_rows)), f)
    (tmp / 'rule.json').write_text(json.dumps({'k_by_stat': {'threes': 0.3, 'rebounds': 0.26, 'assists': 0.29, 'ra': 0.23}, 'primary': {'e': 0.03, 'time': 'earliest', 'min_books': 3}}))


def test_end_to_end_on_a_synthetic_registry_reads_only_and_writes_only_out_and_report(tmp_path):
    _make_world(tmp_path)
    before = {k: _snapshot(tmp_path / k) for k in ('reg', 'prices')} | {'f.db': hashlib.sha256((tmp_path / 'f.db').read_bytes()).hexdigest(), 'rule': (tmp_path / 'rule.json').read_text()}
    a11.main(_args(tmp_path, folds=['2021-22', '2022-23', '2023-24'], seen=['2024-25']))
    assert _snapshot(tmp_path / 'reg') == before['reg'] and _snapshot(tmp_path / 'prices') == before['prices']
    assert hashlib.sha256((tmp_path / 'f.db').read_bytes()).hexdigest() == before['f.db'] and (tmp_path / 'rule.json').read_text() == before['rule']
    assert {p.name for p in tmp_path.iterdir()} == {'reg', 'prices', 'f.db', 'rule.json', 'out', 'rpt'}
    out = tmp_path / 'out'
    for name in ('keep_rule.csv', 'keep_rule.json', 'logscore.csv', 'booklines.csv', 'p0.csv', 'crps.csv', 'existence.csv', 'c.csv', 'rule_bets.csv', 'p0_rel.csv', 'rows_2023-24_threes.csv.gz'):
        assert (out / name).exists(), name
    ls = pd.read_csv(out / 'logscore.csv')
    assert set(ls['scope']) == {'dev_pooled', 'dev_2021-22', 'dev_2022-23', 'dev_2023-24', 'seen_2024-25'}
    pooled = ls[(ls['scope'] == 'dev_pooled') & (ls['stat'] == 'threes') & (ls['line'] == 0.5)].iloc[0]
    assert pooled['n'] == 3 * 40 * 8 * 3 and np.isfinite(pooled['bonf_lo']) and pooled['bonf_lo'] <= pooled['lo95']
    assert np.isnan(ls[ls['scope'] == 'dev_2021-22']['bonf_lo']).all()                     # Bonferroni intervals for the pooled and seen scopes only
    bl = pd.read_csv(out / 'booklines.csv')
    assert set(bl['scope']) == {'dev_2023-24'} and set(bl['line']) == {0.5, 1.5} and (bl['stat'] == 'threes').all()
    cr = pd.read_csv(out / 'crps.csv')
    assert set(cr['stat']) == set(a11.BET_STATS)
    rb = pd.read_csv(out / 'rule_bets.csv')
    assert len(rb) and {'n_bets_widen', 'n_bets_nb', 'n_side_flipped'} <= set(rb.columns)
    ex = pd.read_csv(out / 'existence.csv')
    assert (ex['share_nb_exists'] > 0).all() and (ex['share_c_at_floor_(widen_used)'] >= 0).all()
    kr = json.loads((out / 'keep_rule.json').read_text())
    assert kr['adopt'] in (True, False) and len(kr['clauses']) == 7
    report = (tmp_path / 'rpt' / 'report.md').read_text(encoding='utf-8')
    assert '## Keep rule' in report and 'Verdict' in report and 'Bonferroni' in report
    # the rows the script scored are the rows of the index: widen of the raw pmf, row for row
    rows = pd.read_csv(out / 'rows_2023-24_threes.csv.gz')
    raw = np.load(tmp_path / 'reg' / 'raw' / '2023-24' / 'raw_threes.npy')
    assert len(rows) == len(raw)
    p_over05 = oc.prob_over(dispersion.widen(raw[:50], np.sqrt(rows['c'].to_numpy()[:50])), np.full(50, 0.5))
    assert np.allclose(rows['p_over05_w'].to_numpy()[:50], p_over05)


def test_script_refuses_the_holdout_before_touching_anything(tmp_path):
    for kw in ({'folds': ['2023-24', '2025-26'], 'seen': ['2024-25']}, {'folds': ['2023-24'], 'seen': ['2025-26']}):
        with pytest.raises(SystemExit, match='refuses'):
            a11.main(_args(tmp_path, **kw))
    assert not (tmp_path / 'out').exists() and not (tmp_path / 'rpt').exists()


def test_script_refuses_an_index_with_dates_after_the_last_seen_season(tmp_path):
    _make_world(tmp_path, late_date=True)
    with pytest.raises(SystemExit, match='2025-06-30'):
        a11.main(_args(tmp_path, folds=['2021-22'], seen='none'))


def test_script_refuses_outputs_inside_an_input(tmp_path):
    _make_world(tmp_path)
    bad = _args(tmp_path, folds=['2021-22'], seen='none')
    bad[bad.index('--out') + 1] = str(tmp_path / 'reg' / 'raw' / 'results')
    with pytest.raises(SystemExit, match='overlaps'):
        a11.main(bad)
    assert not (tmp_path / 'reg' / 'raw' / 'results').exists()


def test_script_complains_about_a_missing_raw_file(tmp_path):
    _make_world(tmp_path)
    (tmp_path / 'reg' / 'raw' / '2021-22' / 'raw_ra.npy').unlink()
    with pytest.raises(FileNotFoundError, match='raw_ra.npy'):
        a11.main(_args(tmp_path, folds=['2021-22'], seen='none'))
