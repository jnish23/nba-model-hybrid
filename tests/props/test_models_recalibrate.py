import numpy as np
import pandas as pd
from scipy.stats import poisson

from props.models import outcome as oc, recalibrate as rc


def _pmf(mu, cap=40):
    ys = np.arange(cap + 1)[None, :]
    p = poisson.pmf(ys, np.asarray(mu, float)[:, None])
    p[:, -1] += 1 - p.sum(axis=1)
    return p.astype(np.float32)


def test_scale_pmf_scales_the_mean():
    pm = _pmf([3.0, 8.0, 15.0])
    for r in (0.8, 1.0, 1.15):
        out = rc.scale_pmf(pm, [r, r, r])
        assert np.allclose(out.sum(axis=1), 1, atol=1e-5)
        assert np.allclose(oc.pmf_mean(out), r * oc.pmf_mean(pm), rtol=0.02, atol=0.05)


def test_ratio_uses_only_strictly_earlier_dates():
    dates = np.repeat(pd.date_range('2025-01-01', periods=40).strftime('%Y-%m-%d'), 30)
    rng = np.random.default_rng(0)
    mu = rng.uniform(5, 15, len(dates))
    obs = rng.poisson(mu * 0.9).astype(float)
    base = rc.rolling_ratio(dates, obs, mu)
    changed = obs.copy()
    changed[dates == '2025-02-05'] += 50                                      # change one date's outcomes
    after = rc.rolling_ratio(dates, changed, mu)
    cut = np.where(dates == '2025-02-05')[0]
    assert np.array_equal(base[dates <= '2025-02-05'], after[dates <= '2025-02-05'])        # that date and everything before it: unchanged
    assert (after[dates > '2025-02-05'][:60] != base[dates > '2025-02-05'][:60]).any()       # later dates do see it
    assert base[0] == 1.0                                                      # empty window: no adjustment


def test_ratio_recovers_a_persistent_bias_and_shrinks_early():
    dates = np.repeat(pd.date_range('2025-01-01', periods=90).strftime('%Y-%m-%d'), 1500)      # 500 player-games a day, about 10,500 in a 21-day window
    rng = np.random.default_rng(1)
    mu = rng.uniform(5, 15, len(dates))
    obs = rng.poisson(mu * 0.9).astype(float)                                 # the model is 10% too high
    r = rc.rolling_ratio(dates, obs, mu)
    late = r[dates >= '2025-03-01']
    shrink = rc.PRIOR_PLAYER_GAMES / (rc.PRIOR_PLAYER_GAMES + 21 * 500)        # the prior's share of the window
    assert abs(late.mean() - (0.9 * (1 - shrink) + shrink)) < 0.004           # the bias is recovered, less the pre-registered shrinkage
    assert 0.9 < r[dates == '2025-01-05'].mean() < 1.0                         # early in the window: pulled toward 1
    assert r[dates == '2025-01-05'].mean() > late.mean()


def test_gameday_window_uses_only_earlier_dates_and_reaches_across_the_offseason():
    rng = np.random.default_rng(3)
    old_dates = np.repeat(pd.date_range('2024-03-01', periods=30).strftime('%Y-%m-%d'), 600)          # the previous season's last game-days
    new_dates = np.repeat(pd.date_range('2024-10-22', periods=30).strftime('%Y-%m-%d'), 600)
    mu_o, mu_n = rng.uniform(5, 15, len(old_dates)), rng.uniform(5, 15, len(new_dates))
    obs_o, obs_n = rng.poisson(mu_o * 0.85).astype(float), rng.poisson(mu_n).astype(float)
    carry = (old_dates, obs_o, mu_o)
    r_with = rc.rolling_ratio_gamedays(new_dates, obs_n, mu_n, carry)
    r_without = rc.rolling_ratio_gamedays(new_dates, obs_n, mu_n, None)
    assert r_without[0] == 1.0                                                                          # no earlier game-day at all
    first = new_dates == '2024-10-22'
    assert r_with[first].mean() < 0.97                                                                  # the old season's bias carries over the off-season gap
    changed = obs_n.copy()
    changed[new_dates == '2024-11-10'] += 50
    after = rc.rolling_ratio_gamedays(new_dates, changed, mu_n, carry)
    assert np.array_equal(r_with[new_dates <= '2024-11-10'], after[new_dates <= '2024-11-10'])           # outcomes of a date never move that date's or earlier ratios
    assert (after[new_dates > '2024-11-10'] != r_with[new_dates > '2024-11-10']).any()
    # the window is 21 game dates: a bias that ended more than 21 game dates ago no longer matters
    assert abs(r_with[new_dates == '2024-11-20'].mean() - 1.0) < 0.02


def test_game_groups_keep_regular_season_and_postseason_ratios_apart():
    rng = np.random.default_rng(4)
    old_dates = np.repeat(pd.date_range('2024-04-01', periods=40).strftime('%Y-%m-%d'), 400)
    old_ids = np.where(old_dates < '2024-04-21', '0022300001', '0042300101')                       # 20 regular dates, then 20 playoff dates
    mu_o = rng.uniform(5, 15, len(old_dates))
    obs_o = np.where(old_ids == '0022300001', rng.poisson(mu_o * 1.0), rng.poisson(mu_o * 0.85)).astype(float)       # regular season unbiased, playoffs 15% below the forecasts
    new_dates = np.repeat(pd.date_range('2024-10-22', periods=25).strftime('%Y-%m-%d'), 400)
    new_ids = np.array(['0022400001'] * len(new_dates))
    mu_n = rng.uniform(5, 15, len(new_dates))
    obs_n = rng.poisson(mu_n).astype(float)
    carry = (old_dates, obs_o, mu_o, rc.game_group(old_ids))
    grouped = rc.rolling_ratio_gamedays(new_dates, obs_n, mu_n, carry, groups=rc.game_group(new_ids))
    naive = rc.rolling_ratio_gamedays(new_dates, obs_n, mu_n, (old_dates, obs_o, mu_o))                # the old behaviour: the window is the last 21 dates = playoffs
    first = new_dates == '2024-10-22'
    assert naive[first].mean() < 0.95 and abs(grouped[first].mean() - 1.0) < 0.03                      # the playoff bias no longer leaks into opening night
    # a playoff game never takes a regular-season ratio, and has no ratio before its own postseason starts
    post = rc.rolling_ratio_gamedays(new_dates[:400], obs_n[:400], mu_n[:400], carry, groups=np.array(['post'] * 400))
    assert np.all(post == 1.0)
    assert set(rc.game_group(['0022300001', '0042300101', '0052300101'])) == {'regular', 'post'}
