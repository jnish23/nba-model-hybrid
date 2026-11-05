"""E30 forward-test pieces: the frozen signal is a pure function of the game's own regressors, the slope test finds a planted slope, Holm."""
import numpy as np
import pandas as pd
import pytest

from experiments import e30_score as sc30
from experiments import e30_signals as sg


def spec_and_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({'diff_trk_g3in4': rng.integers(0, 2, n) - rng.integers(0, 2, n), 'diff_trk_g4in6': rng.normal(size=n),
                      'diff_trk_games_last7': rng.normal(size=n), 'diff_trk_trip_last_game': rng.normal(size=n),
                      'diff_trk_homecoming': rng.normal(size=n), 'diff_trk_prev_altitude': rng.normal(size=n)})
    cols = ['g3in4', 'g4in6']
    spec = {'family': 'F4', 'target': 'margin', 'columns': cols, 'mu': {'g3in4': 0.1, 'g4in6': 0.0}, 'sd': {'g3in4': 0.5, 'g4in6': 2.0},
            'beta': {'g3in4': 0.3, 'g4in6': -0.2}, 'centre': 0.05}
    return spec, f


def test_signal_is_standardised_weighted_sum_minus_centre():
    spec, f = spec_and_frame()
    s = sg.signal_points(f, spec)
    want = ((f['diff_trk_g3in4'] - 0.1) / 0.5) * 0.3 + ((f['diff_trk_g4in6'] - 0.0) / 2.0) * -0.2 - 0.05
    np.testing.assert_allclose(s.to_numpy(), want.to_numpy())
    f.loc[3, 'diff_trk_g4in6'] = np.nan
    assert np.isnan(sg.signal_points(f, spec).iloc[3]) and not np.isnan(sg.signal_points(f, spec).iloc[4])


def test_slope_finds_a_planted_slope_and_not_a_null(monkeypatch):
    monkeypatch.setattr(sc30.sc, 'N_BOOT', 300)
    rng = np.random.default_rng(1)
    n = 3000
    dates = pd.Timestamp('2026-11-01') + pd.to_timedelta(np.sort(rng.integers(0, 150, n)), unit='D')
    s = rng.normal(0, 0.6, n)
    y = 1.0 * s + rng.normal(0, 13.5, n) + 2.0
    r = sc30.slope(y, s, dates)
    assert abs(r['slope'] - 1.0) < 4 * r['se'] and r['se'] == pytest.approx(13.5 / (0.6 * np.sqrt(n)), rel=0.25)
    null = sc30.slope(rng.normal(0, 13.5, n), s, dates)
    assert null['p_one_sided'] > 0.01
    assert sc30.slope(y[:10], s[:10], dates[:10])['note'] == 'too few games'


def test_holm_adjustment():
    h = sc30.holm({'a': 0.001, 'b': 0.01, 'c': 0.2, 'd': 0.5})
    assert h['a'] == pytest.approx(0.004) and h['b'] == pytest.approx(0.03) and h['c'] == pytest.approx(0.4) and h['d'] == pytest.approx(0.5)


def test_betting_readout_signs():
    y = np.array([3.0, -3.0, 0.0, 2.0])
    s = np.array([1.0, 1.0, 1.0, -1.0])
    move = np.array([0.5, 0.5, 0.5, 0.5])
    r = sc30.betting(y, s, move, 0.5, True)
    assert r['bets'] == 4 and r['hit_rate_excl_pushes'] == pytest.approx(1 / 3)
    assert r['roi_per_bet'] == pytest.approx((sc30.WIN - 1.0 + 0.0 - 1.0) / 4)
    assert r['mean_clv_points'] == pytest.approx(np.mean([0.5, 0.5, 0.5, -0.5]))
