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
