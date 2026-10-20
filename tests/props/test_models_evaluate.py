import numpy as np
import pandas as pd

from props.models import evaluate as ev, outcome as oc


def test_cluster_bootstrap_mean_and_width():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(300), 10)
    x = rng.normal(2.0, 1.0, len(clusters)) + np.repeat(rng.normal(0, 0.5, 300), 10)            # a cluster effect: clustering must widen the interval
    p, lo, hi = ev.cluster_bootstrap(x, clusters, n_boot=800)
    assert lo < p < hi and abs(p - x.mean()) < 1e-9
    iid_se = x.std() / np.sqrt(len(x))
    assert (hi - lo) / 3.92 > 1.5 * iid_se                                                    # the cluster-aware interval is wider than the naive one
    p0, lo0, hi0 = ev.cluster_bootstrap(np.full(50, 3.0), np.arange(50) % 5, n_boot=200)
    assert p0 == lo0 == hi0 == 3.0


def test_cluster_bootstrap_ratio():
    clusters = np.repeat(np.arange(100), 5)
    num, den = np.full(500, 0.9), np.ones(500)
    p, lo, hi = ev.cluster_bootstrap(num, clusters, den=den, n_boot=200)
    assert abs(p - 0.9) < 1e-12 and abs(lo - 0.9) < 1e-12


def _history_frame():
    rows = []
    for i, d in enumerate(pd.date_range('2025-01-01', periods=30)):
        rows.append({'game_id': f'{i:010d}', 'player_id': 7, 'minutes_before': 60, 'game_date': d.strftime('%Y-%m-%d'), 'lbl_played': 1, 'lbl_minutes': 20.0,
                     'y_points': float(10 + (i % 3)), 'y_rebounds': 4.0, 'y_assists': 2.0, 'y_threes': 1.0, 'y_pra': 16.0})
    return pd.DataFrame(rows)


def test_empirical_baseline_uses_only_strictly_earlier_games():
    frame = _history_frame()
    target = pd.DataFrame({'player_id': [7, 7, 7], 'game_date': ['2025-01-03', '2025-01-20', '2025-01-20'], 'm_min_h15': [20.0, 20.0, 20.0]})
    pm = ev.empirical_pmfs(target, frame, 'points', oc.CAPS['points'])
    assert np.isnan(pm[0]).all()                                                              # only 2 prior games: not enough, falls back
    assert abs(pm[1].sum() - 1) < 1e-6 and pm[1][10] > 0 and pm[1].argmax() in (10, 11, 12)
    changed = frame.copy()
    changed.loc[changed['game_date'] >= '2025-01-20', 'y_points'] = 40.0                      # outcomes on or after the target date must not matter
    pm2 = ev.empirical_pmfs(target, changed, 'points', oc.CAPS['points'])
    assert np.array_equal(np.nan_to_num(pm), np.nan_to_num(pm2))
    assert pm[1][40] == 0 and pm2[1][40] == 0
