"""The E28 screen's statistics on simulated data: a null family is rarely 'significant', a planted effect is found and replicated."""
import numpy as np
import pandas as pd
import pytest

from experiments import e28_screen as sc
from experiments import features_tracking as ft

FIND = ['2019-20', '2020-21']
CONF = ['2022-23']


def sim_frame(effect=0.0, win_effect=0.0, n_per=900, seed=0, family='F2'):
    rng = np.random.default_rng(seed)
    rows = []
    for si, s in enumerate(FIND + CONF):
        dates = pd.Timestamp(f'{2019 + si}-11-01') + pd.to_timedelta(np.sort(rng.integers(0, 150, n_per)), unit='D')
        rows.append(pd.DataFrame({'gameId': [f'002{si}{i:06d}' for i in range(n_per)], 'gameDate': dates, 'seasonYear': s}))
    f = pd.concat(rows, ignore_index=True)
    names = ft.FAMILIES[family]
    Z = rng.normal(size=(len(f), len(names)))
    for i, n in enumerate(names):
        f[f'diff_trk_{n}'] = Z[:, i]
        f[f'trk_{n}_home'] = Z[:, i] * 0.5
        f[f'trk_{n}_away'] = -Z[:, i] * 0.5 + Z[:, i]
    f['trk_one_side_tired'] = Z[:, 0]
    f['y_total'] = effect * Z[:, 0] + rng.normal(0, 18, len(f))
    f['y_margin'] = effect * Z[:, 0] + rng.normal(0, 12, len(f))
    off = rng.normal(0, 1, len(f))
    f['off_win'] = off
    f['y_win'] = (rng.random(len(f)) < 1 / (1 + np.exp(-(off + win_effect * Z[:, 0])))).astype(float)
    return f


@pytest.fixture(autouse=True)
def small_boot(monkeypatch):
    monkeypatch.setattr(sc, 'N_BOOT', 300)


def test_planted_effect_is_found_and_replicates():
    f = sim_frame(effect=2.0)
    r = sc.run_test(f, 'F2', 'margin', FIND, CONF, conf_extra=True)
    assert r['p_find'] < 0.001
    assert r['gamma'] > 0.4 and r['gamma_lo'] > 0 and r['p_one'] < 0.01
    assert r['wald_conf_p'] < 0.01
    assert r['coefs']['rot_speed_z_last']['beta'] == pytest.approx(2.0, abs=0.9)


def test_planted_logit_effect_is_found():
    f = sim_frame(win_effect=0.25, n_per=2500, seed=2)
    r = sc.run_test(f, 'F2', 'win', FIND, CONF)
    assert r['p_find'] < 0.01 and r['gamma'] > 0 and r['p_one'] < 0.05


def test_null_is_rarely_significant():
    ps, gs = [], []
    for seed in range(30):
        f = sim_frame(effect=0.0, seed=100 + seed, n_per=500)
        r = sc.run_test(f, 'F2', 'total', FIND, CONF, conf_extra=False)
        ps.append(r['p_find'])
    ps = np.array(ps)
    assert (ps < 0.10).mean() < 0.30           # nominal 10%; 30 reps, so the bound is loose
    assert 0.25 < ps.mean() < 0.75             # roughly uniform


def test_wrong_sign_in_confirmation_is_negative_gamma():
    f = sim_frame(effect=2.0)
    late = f['seasonYear'].isin(CONF)
    f.loc[late, 'y_margin'] = -2.0 * f.loc[late, 'diff_trk_rot_speed_z_last'] + np.random.default_rng(5).normal(0, 12, late.sum())
    r = sc.run_test(f, 'F2', 'margin', FIND, CONF, conf_extra=False)
    assert r['gamma'] < 0 and r['p_one'] > 0.9


def test_bh_and_tiers():
    q = sc.bh({'a': 0.001, 'b': 0.02, 'c': 0.5, 'd': 1.0})
    assert q['a'] == pytest.approx(0.004) and q['b'] == pytest.approx(0.04) and q['c'] == pytest.approx(2 / 3, abs=1e-9) and q['d'] == 1.0
    res = {f'k{i}': {'p_find': 0.01, 'gamma': 0.5, 'p_one': 0.001 if i == 0 else 0.4} for i in range(15)}
    res['k_fail'] = {'p_find': 0.5, 'gamma': 1.0, 'p_one': 0.0001}
    res['k_neg'] = {'p_find': 0.01, 'gamma': -0.2, 'p_one': 0.9}
    out = sc.tiers(res)
    assert out['k0']['tier'] == 'CANDIDATE'
    assert out['k1']['tier'].startswith('Reject (not confirmed')
    assert out['k_fail']['tier'] == 'Reject (fails find step)' and out['k_neg']['tier'].startswith('Reject (opposite')


def test_wald_matches_known_chi_square():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(2000, 3)) * np.array([1.0, 2.0, 0.5])
    beta = np.array([0.0, 0.0, 0.0])
    assert sc.wald(beta, draws)[2] == pytest.approx(1.0)
    stat, df, p = sc.wald(np.array([3.0, 0.0, 0.0]), draws)
    assert df == 3 and p < 0.05
