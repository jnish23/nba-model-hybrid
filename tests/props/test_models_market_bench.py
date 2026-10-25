import numpy as np
import pandas as pd
from scipy.stats import poisson

from props.models import market_bench as mb


def _setup(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(8, 20, n)
    ys = np.arange(61)[None, :]
    pm = poisson.pmf(ys, mu[:, None]).astype(np.float32)
    pm[:, -1] += 1 - pm.sum(axis=1)
    y = rng.poisson(mu)
    rows = pd.DataFrame({'game_id': [f'{i // 10:010d}' for i in range(n)], 'player_id': np.arange(n) % 10, 'minutes_before': 60,
                         'y_points': y.astype(float), 'y_rebounds': y.astype(float), 'y_assists': y.astype(float), 'y_threes': y.astype(float), 'y_pra': y.astype(float)})
    line = np.floor(mu) + 0.5
    return rows, {'points': pm}, mu, line, y


def test_compare_drops_pushes_and_uses_no_push_probability():
    rows, pmfs, mu, line, y = _setup()
    line[:200] = np.floor(mu[:200])                                                      # integer lines: some outcomes push
    cons = pd.DataFrame({'game_id': rows['game_id'], 'player_id': rows['player_id'], 'minutes_before': 60, 'market': 'points', 'consensus_line': line,
                         'consensus_fair_over': 0.5, 'n_at_line': 4, 'book_spread': 0.0})
    cmp = mb.compare(rows, pmfs, cons)
    assert len(cmp) == len(rows) - int((y[:200] == line[:200]).sum())                      # pushes dropped, nothing else
    ints = cmp['line'] == np.floor(cmp['line'])
    assert (cmp.loc[ints, 'push_share_at_line'] > 0).all() and (cmp.loc[~ints, 'push_share_at_line'] == 0).all()
    assert cmp['p_model'].between(0, 1).all()


def test_a_calibrated_model_ties_the_market_when_the_market_is_the_truth_and_loses_when_not():
    rows, pmfs, mu, line, y = _setup(n=6000, seed=1)
    true_p = 1 - poisson.cdf(np.floor(line), mu)                                          # P(over) at the half line
    base = pd.DataFrame({'game_id': rows['game_id'], 'player_id': rows['player_id'], 'minutes_before': 60, 'market': 'points', 'consensus_line': line,
                         'n_at_line': 4, 'book_spread': 0.0})
    good = mb.compare(rows, pmfs, base.assign(consensus_fair_over=true_p))
    t = mb.score_table(good, n_boot=300).set_index('stat').loc['all']
    assert abs(t['diff']) < 1e-9 and t['lo'] <= 0 <= t['hi'] + 1e-12                      # identical probabilities: zero difference
    noisy = mb.compare(rows, pmfs, base.assign(consensus_fair_over=np.clip(true_p + np.random.default_rng(2).normal(0, 0.08, len(rows)), 0.02, 0.98)))
    t2 = mb.score_table(noisy, n_boot=300).set_index('stat').loc['all']
    assert t2['diff'] < 0 and t2['hi'] < 0                                                # a noisy "market" is beaten by the true model: model minus market is negative
    dt = mb.disagreement_table(noisy, n_boot=200)
    assert dt['n'].sum() == len(noisy) and (dt['hit - model'].abs() < 0.08).all()          # the true model is calibrated in every disagreement band
