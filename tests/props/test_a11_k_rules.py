"""Steps S5-S7/S16 (props/betting/a11_rules.py, scripts/a11_k_estimation.py, scripts/a11_rule_eval.py): simulated data only, no data/ files, seconds."""
import hashlib
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from props.betting import a11_rules as ar, rules
from props.live import monitor as mon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
import a11_k_estimation as ke  # noqa: E402
import a11_rule_eval as re_  # noqa: E402


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# simulated market rows (C.1 rows: stat, game_date, minutes_before, p_model, p_market, hit)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _sim_cmp(n, a, b, g_mean, g_sd, stats=('assists',), seed=0, n_dates=200, a_by_side=None, c=None, continuous=False, noise=0.05):
    rng = np.random.default_rng(seed)
    pm = rng.uniform(0.3, 0.7, n)
    g = rng.normal(g_mean, g_sd, n) if g_sd > 0 else rng.uniform(-0.15, 0.15, n)
    x = np.clip(g, -c, c) if c is not None else g
    a_eff = np.where(g >= 0, a_by_side[0], a_by_side[1]) if a_by_side else a
    p_true = pm + a_eff + b * x
    hit = p_true + rng.normal(0, noise, n) if continuous else (rng.random(n) < np.clip(p_true, 0, 1)).astype(float)
    dates = pd.date_range('2023-10-24', periods=n_dates).strftime('%Y-%m-%d').to_numpy()
    return pd.DataFrame({'stat': np.asarray(stats)[rng.integers(0, len(stats), n)], 'game_date': dates[rng.integers(0, n_dates, n)],
                         'minutes_before': np.array([240, 60, 5])[rng.integers(0, 3, n)], 'p_model': pm + g, 'p_market': pm, 'hit': hit})


def test_e0_biased_e1_recovers_true_slope_with_over_bias_and_negative_mean_gap():
    a, b, gm, gs = -0.02, 0.25, -0.03, 0.05
    xy = ar.xy_frame(_sim_cmp(300_000, a, b, gm, gs, seed=1))
    e0 = ar.fit_estimator(xy, 'E0', n_boot=200)['by_stat'].iloc[0]
    e1 = ar.fit_estimator(xy, 'E1', n_boot=200)['by_stat'].iloc[0]
    expected_e0 = b + a * gm / (gs ** 2 + gm ** 2)                 # plan section 0 item 5: b + a E[g] / E[g^2], here 0.426 (overstated)
    assert abs(e0['b_raw'] - expected_e0) < 0.04
    assert e0['b_raw'] > b + 0.12
    assert abs(e1['b_raw'] - b) < 0.05
    assert abs(e1['a_raw'] - a) < 0.004
    assert e1['b_lo'] < b < e1['b_hi']


def test_e1_matches_monitor_information_slope_point_and_interval():
    cmp = _sim_cmp(40_000, -0.02, 0.3, -0.03, 0.06, stats=('assists', 'rebounds', 'threes'), seed=2, n_dates=80)
    xy = ar.xy_frame(cmp)
    f = ar.fit_estimator(xy, 'E1', n_boot=300, seed=7)
    m = mon.information_slope(cmp.rename(columns={'hit': 'hit_over'}), n_boot=300, seed=7)
    for _, r in f['by_stat'].iterrows():
        mr = m[m['stat'] == r['stat']].iloc[0]
        assert r['b_raw'] == pytest.approx(mr['slope_icpt'], abs=1e-10)
        assert r['a_raw'] == pytest.approx(mr['intercept'], abs=1e-10)
        assert r['b_lo'] == pytest.approx(mr['si_lo'], abs=1e-10) and r['b_hi'] == pytest.approx(mr['si_hi'], abs=1e-10)
    pm = m[m['stat'] == 'pooled'].iloc[0]
    assert f['pooled']['b'] == pytest.approx(pm['slope_icpt'], abs=1e-10)
    assert f['pooled']['lo'] == pytest.approx(pm['si_lo'], abs=1e-10)
    e0 = ar.fit_estimator(xy, 'E0', n_boot=300, seed=7)
    assert e0['by_stat'].set_index('stat')['b_raw'].to_dict() == pytest.approx(m.set_index('stat')['slope_origin'].drop('pooled').to_dict(), abs=1e-10)


def test_e2_recovers_side_specific_intercepts():
    xy = ar.xy_frame(_sim_cmp(300_000, None, 0.2, 0.0, 0.06, stats=('assists', 'threes'), seed=3, a_by_side=(-0.01, -0.035)))
    t = ar.fit_estimator(xy, 'E2', n_boot=100)['by_stat']
    for _, r in t.iterrows():
        assert abs(r['b_raw'] - 0.2) < 0.06
        assert abs(r['a_pos_raw'] + 0.01) < 0.008
        assert abs(r['a_neg_raw'] + 0.035) < 0.008


def test_e3_recovers_saturation_point_on_grid():
    xy = ar.xy_frame(_sim_cmp(120_000, -0.01, 0.3, 0.0, 0.0, stats=ar.BET_STATS, seed=4, c=0.06, continuous=True))
    f = ar.fit_estimator(xy, 'E3', n_boot=50)
    assert f['c'] == pytest.approx(0.06)
    assert list(f['profile']['c']) == list(ar.C_GRID)
    assert (f['by_stat']['b_raw'] - 0.3).abs().max() < 0.03
    assert (f['by_stat']['a_raw'] + 0.01).abs().max() < 0.003


def test_clip_refits_intercept_at_clipped_slope():
    for b_true, b_exp in ((0.8, 0.5), (-0.3, 0.0)):
        xy = ar.xy_frame(_sim_cmp(50_000, -0.02, b_true, -0.03, 0.05, seed=5, continuous=True))
        r = ar.fit_estimator(xy, 'E1', n_boot=50)['by_stat'].iloc[0]
        assert r['b'] == b_exp and r['clipped']
        assert r['b'] == rules.clip_k(r['b_raw'])
        assert r['a'] == pytest.approx(xy['r'].mean() - b_exp * xy['g'].mean(), abs=1e-12)    # least-squares intercept with the slope fixed at the clipped value
        assert r['a'] != pytest.approx(r['a_raw'], abs=1e-6)
    xy = ar.xy_frame(_sim_cmp(50_000, -0.02, 0.25, -0.03, 0.05, seed=6, continuous=True))
    r = ar.fit_estimator(xy, 'E1', n_boot=50)['by_stat'].iloc[0]
    assert not r['clipped'] and r['a'] == pytest.approx(r['a_raw'])


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# p*, EV, side
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

PARAMS = {'C0': {'b': {'assists': 0.3}}, 'C1': {'b': {'assists': 0.3}}, 'C2': {'b': {'assists': 0.3}, 'a': {'assists': -0.02}},
          'C3': {'b': {'assists': 0.3}, 'a_pos': {'assists': -0.01}, 'a_neg': {'assists': -0.04}},
          'C4': {'b': {'assists': 0.3}, 'a': {'assists': -0.02}, 'c': 0.05}, 'P0': {'a': {'assists': -0.02}}, 'P00': {}}


def _tiny(p_mod=0.6, p_mkt=0.5):
    return pd.DataFrame({'stat': ['assists'], 'consensus_fair_over': [p_mkt], 'p_mod': [p_mod], 'push': [0.1], 'over_dec': [2.0], 'under_dec': [1.9], 'minutes_before': [60]})


@pytest.mark.parametrize('cand, p, ev_o, ev_u, side', [
    ('C0', 0.53, 0.054, -0.0963, 'over'), ('C1', 0.53, 0.054, -0.0963, 'over'), ('C2', 0.51, 0.018, -0.0621, 'over'), ('C3', 0.52, 0.036, -0.0792, 'over'),
    ('C4', 0.495, -0.009, -0.03645, 'over'), ('P0', 0.48, -0.036, -0.0108, 'under'), ('P00', 0.5, 0.0, -0.045, 'over')])
def test_candidate_pstar_ev_and_side_by_hand(cand, p, ev_o, ev_u, side):
    q = ar.add_ev_pstar(_tiny(), ar.p_star(cand, _tiny(), PARAMS[cand]))
    assert q['p_star'].iloc[0] == pytest.approx(p)
    assert q['ev_over'].iloc[0] == pytest.approx(ev_o)
    assert q['ev_under'].iloc[0] == pytest.approx(ev_u)
    assert q['best_side'].iloc[0] == side
    assert q['best_ev'].iloc[0] == pytest.approx(max(ev_o, ev_u))


def test_c3_negative_gap_and_clip():
    assert ar.p_star('C3', _tiny(p_mod=0.4), PARAMS['C3'])[0] == pytest.approx(0.5 - 0.04 - 0.03)
    up = {'b': {'assists': 0.3}, 'a': {'assists': 0.02}}
    assert ar.p_star('C2', _tiny(p_mod=0.999, p_mkt=0.995), up)[0] == 0.99
    assert ar.p_star('P0', _tiny(p_mkt=0.005), {'a': {'assists': -0.02}})[0] == 0.01
    assert ar.p_star('C1', _tiny(p_mod=0.999, p_mkt=0.995), {'b': {'assists': 0.5}})[0] == pytest.approx(0.997)   # C0/C1 are not clipped (plan C.4)


def _quotes(n_markets=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_markets):
        pm = rng.uniform(0.35, 0.65)
        for t in (240, 60, 5):
            pmod = np.clip(pm + rng.normal(0, 0.08), 0.02, 0.98)
            for book in ('draftkings', 'fanduel', 'betmgm'):
                rows.append({'game_id': f'g{m // 6}', 'player_id': m % 6, 'stat': ('assists', 'rebounds', 'threes', 'ra')[m % 4], 'minutes_before': t, 'book': book, 'line': 4.5,
                             'consensus_fair_over': pm, 'p_mod': pmod, 'push': 0.0, 'over_dec': 1 / (pm * 1.05) + rng.normal(0, 0.04), 'under_dec': 1 / ((1 - pm) * 1.05) + rng.normal(0, 0.04),
                             'n_at_line': int(rng.integers(2, 6)), 'game_date': f'2024-01-{1 + (m // 6) % 28:02d}'})
    return pd.DataFrame(rows)


def test_c0_selection_equals_rules_select_with_frozen_k():
    k = json.loads((ROOT / 'config' / 'props_frozen_rule.json').read_text(encoding='utf-8'))['k_by_stat']
    q = _quotes()
    mine = ar.select_candidate(q, 'C0', {'b': k})
    ref = rules.select(rules.add_ev(q, k), 0.03, 'earliest', 3)
    pd.testing.assert_frame_equal(mine, ref)
    assert len(mine) > 20


def test_candidates_recompute_selection_not_patch():
    q = _quotes(seed=1)
    p = {'b': {s: 0.3 for s in ar.BET_STATS}, 'a': {s: -0.03 for s in ar.BET_STATS}}
    sel = ar.select_candidate(q, 'C2', p)
    full = ar.add_ev_pstar(q, ar.p_star('C2', q, p))
    pd.testing.assert_frame_equal(sel, rules.select(full, 0.03, 'earliest', 3))
    assert (sel['best_ev'] >= 0.03).all()
    assert (sel['best_side'] == 'under').mean() > (ar.select_candidate(q, 'C1', {'b': p['b']})['best_side'] == 'under').mean()   # the intercept tilts toward unders


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# prices, settlement, bootstrap
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _settle_frame():
    base = {'player_id': 1, 'stat': 'assists', 'minutes_before': 60, 'line': 4.5, 'consensus_fair_over': 0.5, 'p_mod': 0.6, 'push': 0.0, 'under_dec': 1.9, 'n_at_line': 3,
            'lbl_minutes': 30.0, 'expected_minutes': 30.0, 'played': True}
    q = []
    for gid, y, played in (('g1', 6.0, True), ('g2', 3.0, True), ('g3', np.nan, False)):
        for book, od in (('draftkings', 1.9), ('fanduel', 2.0), ('betmgm', 2.1)):
            q.append({**base, 'game_id': gid, 'book': book, 'over_dec': od, 'y': y, 'played': played, 'game_date': '2024-01-0' + gid[1]})
    return pd.DataFrame(q)


def test_median_reprice_uses_the_best_price_selection_and_voids_are_not_stakes():
    q = _settle_frame()
    sel = ar.select_candidate(q, 'C1', {'b': {'assists': 1.0}})
    b = ar.price_and_settle(sel, ar.market_prices(q))
    assert len(b) == 3 and (b['book'] == 'betmgm').all() and (b['best_side'] == 'over').all()
    assert (b['dec_best'] == 2.1).all() and (b['dec_med'] == 2.0).all()
    assert list(b['status_stands']) == ['win', 'loss', 'void']
    assert list(b['stake_stands']) == [1.0, 1.0, 0.0]
    assert list(b['profit_med_stands']) == pytest.approx([1.0, -1.0, 0.0])
    assert ar.roi(b, 'med') == pytest.approx(0.0) and ar.roi(b, 'best') == pytest.approx(0.05)
    import props_phase5_dev as p5d
    lvl = p5d.market_level(q)
    mp = ar.market_prices(q)
    pd.testing.assert_frame_equal(mp, lvl[mp.columns])


def test_push_is_not_a_stake():
    q = _settle_frame()
    q['line'] = 6.0
    q['push'] = 0.1
    b = ar.price_and_settle(ar.select_candidate(q, 'C1', {'b': {'assists': 1.0}}), ar.market_prices(q))
    assert b.loc[b['game_id'] == 'g1', 'status_stands'].iloc[0] == 'push'
    assert b.loc[b['game_id'] == 'g1', 'stake_stands'].iloc[0] == 0.0


def test_paired_bootstrap_is_ratio_of_sums_deterministic_and_covers():
    # ratio of sums, not mean of per-bet differences: A = {+1, -1, -1} on 3 dates -> -1/3; B = {+0.9 x 1 stake} -> 0.9
    pb = ar.PairedBoot(['d1', 'd2', 'd3'], n_boot=200, seed=1)
    pb.add('A', ['d1', 'd2', 'd3'], [1.0, -1.0, -1.0], [1, 1, 1])
    pb.add('B', ['d1'], [0.9], [1])
    assert pb.diff('A', 'B')['point'] == pytest.approx(-1 / 3 - 0.9)
    rng = np.random.default_rng(0)
    dates = pd.date_range('2024-01-01', periods=200).strftime('%Y-%m-%d').to_numpy()
    da, db = dates[rng.integers(0, 200, 8000)], dates[rng.integers(0, 200, 6000)]
    pa = np.where(rng.random(8000) < 0.55, 0.909, -1.0) + 0.0
    pbb = np.where(rng.random(6000) < 0.524, 0.909, -1.0)
    out = []
    for _ in range(2):
        b = ar.PairedBoot(dates, n_boot=500, seed=ar.SEED)
        b.add('a', da, pa, np.ones(8000))
        b.add('b', db, pbb, np.ones(6000))
        out.append(b.diff('a', 'b'))
    assert out[0] == out[1]
    true = 0.55 * 1.909 - 1 - (0.524 * 1.909 - 1)
    assert out[0]['lo'] < true < out[0]['hi']
    assert out[0]['point'] == pytest.approx(pa.mean() - pbb.mean())
    with pytest.raises(ValueError):
        b.add('c', ['1999-01-01'], [1.0], [1.0])


def test_same_resamples_across_rules():
    b = ar.PairedBoot(['d1', 'd2', 'd3', 'd4'], n_boot=100, seed=3)
    b.add('a', ['d1', 'd2', 'd3', 'd4'], [1, -1, 1, -1], [1, 1, 1, 1])
    b.add('a2', ['d1', 'd2', 'd3', 'd4'], [1, -1, 1, -1], [1, 1, 1, 1])
    assert np.allclose(b.draws('a'), b.draws('a2'))
    d = b.diff('a', 'a2')
    assert d['lo'] == 0 and d['hi'] == 0


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# caps, Kelly, metrics
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_kelly_units_and_per_game_cap_cuts_lower_ev_first():
    sel = pd.DataFrame({'game_id': ['g1'] * 4 + ['g2'], 'best_ev': [0.06, 0.10, 0.04, 0.08, 0.05], 'dec_price': [2.0, 2.0, 2.0, 1.8, 3.0]})
    u = ar.kelly_units(sel)
    # f* = EV/(d-1): 0.06, 0.10, 0.04, 0.10, 0.025 -> min(0.25 f*, 0.02) x 100 = 1.5, 2, 1, 2, 0.625; game g1 capped at 5: 0.10 (2) + 0.08 (2) + 0.06 trimmed to 1 + 0.04 -> 0
    assert list(u) == pytest.approx([1.0, 2.0, 0.0, 2.0, 0.625])


def test_cap_per_player_game_keeps_highest_ev():
    sel = pd.DataFrame({'game_id': ['g'] * 3, 'player_id': [1, 1, 2], 'stat': ['ra', 'assists', 'ra'], 'best_ev': [0.05, 0.08, 0.04]})
    c1 = rules.cap_per_player_game(sel, 1)
    assert sorted(c1['best_ev']) == [0.04, 0.08]


def test_daily_metrics_sharpe_and_drawdown():
    b = pd.DataFrame({'game_date': ['d1', 'd1', 'd2', 'd3'], 'units': [2.0, 2.0, 3.0, 2.0], 'stake_stands': [1, 1, 1, 1.0], 'profit_med_stands': [1.0, 0.0, -1.0, 1.0]})
    m = ar.daily_metrics(b)
    rd = np.array([2 / 4, -1.0, 1.0])
    assert m['sharpe'] == pytest.approx(rd.mean() / rd.std(ddof=1))
    assert m['max_drawdown'] == pytest.approx(3.0) and m['dd_over_stake'] == pytest.approx(3 / 9)
    assert m['roi'] == pytest.approx(1 / 9) and m['n_stakes'] == 4


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# keep rules (amended thresholds), just inside and just outside
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _d(point, lo=-1.0, hi=1.0):
    return {'point': point, 'lo': lo, 'hi': hi}


def _base_inputs():
    pairs = {(x, y): _d(0.0) for x in ar.CANDIDATES for y in ar.CANDIDATES if x != y}
    cal = {'over': {'n': 100, 'diff': 1.0, 'diff_se': 1.0, 'within_2se': True}, 'under': {'n': 100, 'diff': 1.0, 'diff_se': 1.0, 'within_2se': True}}
    return pairs, cal, {c: 0.0 for c in ar.CANDIDATES}, {e: {'b': 0.2, 'lo': 0.1, 'hi': 0.3} for e in ('E1', 'E2', 'E3')}


def test_c5_item1_c1_replaces_c0_unless_clearly_worse():
    d, cal, b7, ss = _base_inputs()
    d[('C1', 'C0')] = _d(-0.0051, hi=-0.0001)
    assert ar.c5_decide(d, cal, b7, ss)['pick'] == 'C0'
    for pt, hi in ((-0.0049, -0.0001), (-0.0051, 0.0001)):
        d[('C1', 'C0')] = _d(pt, hi=hi)
        assert ar.c5_decide(d, cal, b7, ss)['pick'] == 'C1'


def test_c5_item2_amended_margin_and_calibration():
    d, cal, b7, ss = _base_inputs()
    d[('C2', 'C1')], d[('C2', 'P0')] = _d(0.0026), _d(0.0026)
    assert ar.c5_decide(d, cal, b7, ss)['pick'] == 'C2'
    for key in (('C2', 'C1'), ('C2', 'P0')):
        dd = dict(d)
        dd[key] = _d(0.0024)                                       # +0.24 points: below the amended +0.25
        assert ar.c5_decide(dd, cal, b7, ss)['pick'] == 'C1'
    bad = {**cal, 'under': {'n': 100, 'diff': 2.01, 'diff_se': 1.0, 'within_2se': False}}
    assert ar.c5_decide(d, bad, b7, ss)['pick'] == 'C1'
    calc = ar.calibration_by_side  # within_2se is computed as |diff| <= 2 SE
    assert callable(calc)
    d[('C1', 'C0')] = _d(-0.01, hi=-0.001)                        # C0 is the pick: item 2 is not reached
    res = ar.c5_decide(d, cal, b7, ss)
    assert res['pick'] == 'C0' and any(c['item'] == 2 and not c['reached'] for c in res['clauses'])


def test_c5_items3_4_and_bucket():
    d, cal, b7, ss = _base_inputs()
    d[('C3', 'C1')] = _d(0.0026, lo=-0.0099)
    assert ar.c5_decide(d, cal, b7, ss)['pick'] == 'C3'
    d[('C3', 'C1')] = _d(0.0026, lo=-0.0101)
    assert ar.c5_decide(d, cal, b7, ss)['pick'] == 'C1'
    d[('C3', 'C1')] = _d(0.0024, lo=0.0)
    assert ar.c5_decide(d, cal, b7, ss)['pick'] == 'C1'
    d[('C4', 'C1')] = _d(0.003, lo=-0.005)
    b7 = {**b7, 'C4': 0.05, 'C1': 0.04}
    assert ar.c5_decide(d, cal, b7, ss)['pick'] == 'C4'
    b7['C4'] = 0.039
    assert ar.c5_decide(d, cal, b7, ss)['pick'] == 'C1'
    d[('C3', 'C1')] = _d(0.003, lo=0.0)                          # C3 first, then C4 is judged against C3
    d[('C4', 'C3')] = _d(0.001, lo=0.0)
    b7['C4'] = 0.05
    assert ar.c5_decide(d, cal, b7, ss)['pick'] == 'C3'


def test_c5_stop_condition():
    d, cal, b7, ss = _base_inputs()
    assert not ar.c5_decide(d, cal, b7, ss)['stop']
    ss['E1'] = {'b': 0.05, 'lo': -0.001, 'hi': 0.1}
    res = ar.c5_decide(d, cal, b7, ss)
    assert res['pick'] == 'C1' and res['stop']
    d[('C1', 'C0')] = _d(-0.01, hi=-0.001)                        # E0 pick: judged on E1's slope with intercept
    assert ar.c5_decide(d, cal, b7, ss)['stop']
    ss['E1'] = {'b': 0.05, 'lo': 0.0001, 'hi': 0.1}
    assert not ar.c5_decide(d, cal, b7, ss)['stop']


def _m(sharpe, roi, dd=0.1):
    return {'sharpe': sharpe, 'roi': roi, 'dd_over_stake': dd}


def test_7c_cap_rule_thresholds():
    u = _m(0.20, 0.03)
    assert ar.c7_cap_decide('cap1', _m(0.20, 0.0251), u, _m(0.1, 0), _m(0.1, 0))['adopt']
    assert not ar.c7_cap_decide('cap1', _m(0.20, 0.0249), u, _m(0.1, 0), _m(0.1, 0))['adopt']
    assert not ar.c7_cap_decide('cap1', _m(0.1999, 0.03), u, _m(0.1, 0), _m(0.1, 0))['adopt']
    assert not ar.c7_cap_decide('cap1', _m(0.21, 0.03), u, _m(0.0999, 0), _m(0.1, 0))['adopt']


def test_7c_kelly_rule_thresholds():
    f = _m(0.20, 0.03, 0.10)
    assert ar.c7_kelly_decide(_m(0.2201, 0.0251, 0.10), f, _m(0.11, 0), _m(0.1, 0))['adopt']
    assert not ar.c7_kelly_decide(_m(0.2199, 0.03, 0.10), f, _m(0.11, 0), _m(0.1, 0))['adopt']
    assert not ar.c7_kelly_decide(_m(0.25, 0.03, 0.1001), f, _m(0.11, 0), _m(0.1, 0))['adopt']
    assert not ar.c7_kelly_decide(_m(0.25, 0.0249, 0.10), f, _m(0.11, 0), _m(0.1, 0))['adopt']
    assert not ar.c7_kelly_decide(_m(0.25, 0.03, 0.10), f, _m(0.09, 0), _m(0.1, 0))['adopt']
    assert ar.kelly_sharpe_bar(-0.2) == pytest.approx(-0.18)          # negative flat Sharpe: the bar stays above flat


def test_7d_amended_unadjusted_bound():
    ok = ar.d7_decide({'point': 0.03, 'lo': 0.0001, 'hi': 0.06}, {'2023-24': {'point': 0.02}, '2024-25': {'point': 0.04}})
    assert ok['keep']
    assert not ar.d7_decide({'point': 0.03, 'lo': -0.0001, 'hi': 0.06}, {'2023-24': {'point': 0.02}, '2024-25': {'point': 0.04}})['keep']
    assert not ar.d7_decide({'point': 0.03, 'lo': 0.01, 'hi': 0.06}, {'2023-24': {'point': -0.001}, '2024-25': {'point': 0.04}})['keep']


def test_subset_diff_minutes_split():
    cl = np.array(['d1', 'd1', 'd2', 'd2'])
    _, d = ar.subset_diff(cl, cl, np.array([1.0, -1.0, 0.5, -1.0]), np.ones(4), np.array([True, False, True, False]), np.array([False, True, False, True]), n_boot=50)
    assert d['point'] == pytest.approx(0.75 - (-1.0))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# refusals and the candidate rule file
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_refuses_2025_26_everywhere():
    for s in ('2025-26', '2026-27'):
        with pytest.raises(SystemExit):
            ar.check_season(s)
    assert ar.check_season('2024-25') == '2024-25'
    with pytest.raises(SystemExit):
        ar.check_dates(['2025-06-30', '2025-07-01'])
    base = ['--p5-version', 'x', '--registry-root', 'nowhere', '--prices-dir', 'nowhere', '--features-db', 'nowhere.db']
    for extra in (['--mode', 'walkforward', '--score', '2025-26', '--out', 'o'], ['--mode', 'walkforward', '--train', '2025-26', '--out', 'o'],
                  ['--mode', 'reproduce', '--seasons', '2024-25', '2025-26', '--out', 'o.json'], ['--mode', 'final', '--train', '2023-24', '2025-26', '--write-rule', 'r.json']):
        with pytest.raises(SystemExit, match='2025-26|2026'):
            ke.main(base + extra)
    with pytest.raises(SystemExit, match='2025-26'):
        re_.main(base + ['--k-file', 'k.json', '--seasons', '2024-25', '2025-26', '--out', 'o'])


def test_rule_file_keys_sidecar_and_config_guard(tmp_path):
    frozen_path = ROOT / 'config' / 'props_frozen_rule.json'
    before = _sha(frozen_path)
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    p = {'b': {s: 0.2 for s in ar.BET_STATS}, 'a': {s: -0.02 for s in ar.BET_STATS}, 'c': 0.06}
    rule = ar.candidate_rule('C4', p, ar.BET_STATS, ['2023-24', '2024-25'], {'all': [0.1, 0.0, 0.2]}, 'abc', 'f' * 64, 1000, 'a11_r0_p5', ar.CURRENT_PMARKET)
    for k in list(frozen) + ['estimator', 'intercept_by_stat', 'c', 'developed_on', 'tried_rules', 'features_db_sha256', 'model_version', 'code_commit']:
        assert k in rule, k
    assert rule['tried_rules'] == 60 and rule['estimator'] == 'E3'
    path, side, h = ar.write_rule_file(tmp_path / 'candidate_rule_a11.json', rule, ROOT)
    assert side.read_text(encoding='utf-8').split()[0] == h == _sha(path)
    with pytest.raises(FileExistsError):
        ar.write_rule_file(tmp_path / 'candidate_rule_a11.json', rule, ROOT)
    for bad in (ROOT / 'config' / 'candidate.json', ROOT / 'config' / 'sub' / 'x.json', tmp_path / 'props_frozen_rule.json'):
        with pytest.raises(SystemExit):
            ar.write_rule_file(bad, rule, ROOT)
    assert not (ROOT / 'config' / 'candidate.json').exists()
    c1 = ar.candidate_rule('C1', {'b': p['b']}, ar.BET_STATS, ['2023-24'], {}, 'abc', 'f', 1, 'v', ar.CURRENT_PMARKET)
    assert 'intercept_by_stat' not in c1 and 'c' not in c1
    c3 = ar.candidate_rule('C3', {'b': p['b'], 'a_pos': p['a'], 'a_neg': p['a']}, ar.BET_STATS, ['2023-24'], {}, 'abc', 'f', 1, 'v', ar.CURRENT_PMARKET)
    assert set(c3['intercept_by_stat']['ra']) == {'a_pos', 'a_neg'}
    assert _sha(frozen_path) == before


def test_pmarket_choice_and_variant_substitution():
    assert ar.pmarket_from_choice(None) == ar.CURRENT_PMARKET
    assert ar.pmarket_from_choice({'devig_method': 'shin', 'consensus_set': 'all_median', 'passed': False, 'reason': 'x'}) == ar.CURRENT_PMARKET
    assert ar.pmarket_from_choice({'devig_method': 'shin', 'consensus_set': 'all_median', 'passed': True, 'reason': 'x'}) == ('shin', 'all_median')
    key = {'game_id': 'g1', 'player_id': 1, 'minutes_before': 60, 'stat': 'assists', 'snapshot_ts_utc': 's1'}
    cons = pd.DataFrame([{**key, 'consensus_line': 4.5, 'consensus_fair_over': 0.5, 'n_at_line': 4}])
    bets = pd.DataFrame([{**key, 'book': b, 'line': ln, 'consensus_fair_over': 0.5, 'n_at_line': 4, 'over_dec': 1.9} for b, ln in (('draftkings', 4.5), ('fanduel', 4.5))])
    var = pd.DataFrame([{**key, 'consensus_line': 4.5, 'consensus_fair_over': 0.47, 'n_at_line': 3}])
    c2, b2, dropped = ar.apply_pmarket(cons, bets, var)
    assert dropped == 0 and (b2['consensus_fair_over'] == 0.47).all() and (b2['n_at_line'] == 3).all() and list(b2.columns) == list(bets.columns)
    var['consensus_line'] = 5.5
    _, b3, dropped = ar.apply_pmarket(cons, bets, var)
    assert dropped == 2 and b3.empty


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# end to end on a simulated registry, price files and labels DB (the on-disk formats the scripts read)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

STATS = ('rebounds', 'assists', 'threes', 'ra')
CAPS = {'rebounds': 30, 'assists': 30, 'threes': 15, 'ra': 60}
BOOKS = ('draftkings', 'fanduel', 'betmgm', 'williamhill_us')


def _write_world(root, seasons=('2023-24', '2024-25'), n_dates=30, games_per_date=4, players=6, seed=0):
    rng = np.random.default_rng(seed)
    reg, prices = root / 'models', root / 'prices'
    prices.mkdir(parents=True)
    labels = []
    for si, season in enumerate(seasons):
        start = pd.Timestamp(f'{season[:4]}-11-01')
        idx_rows, pm = [], {s: [] for s in STATS}
        cons_rows, bet_rows = [], []
        for di in range(n_dates):
            date = (start + pd.Timedelta(days=di)).strftime('%Y-%m-%d')
            for gi in range(games_per_date):
                gid = f'00{si}{di:03d}{gi}'
                for pid in range(players):
                    pl = 1000 * si + 10 * gi + pid
                    mu = {'rebounds': rng.uniform(2, 9), 'assists': rng.uniform(1, 7), 'threes': rng.uniform(0.5, 3)}
                    y = {s: rng.poisson(mu[s]) for s in mu}
                    labels.append({'game_id': gid, 'player_id': pl, 'lbl_played': 1, 'lbl_minutes': float(rng.uniform(10, 36)), 'y_points': 10, 'y_rebounds': y['rebounds'],
                                   'y_assists': y['assists'], 'y_threes': y['threes'], 'y_pra': 10 + y['rebounds'] + y['assists']})
                    mu['ra'] = mu['rebounds'] + mu['assists']
                    for t in (240, 60, 5):
                        idx_rows.append({'game_id': gid, 'player_id': pl, 'minutes_before': t, 'season': season, 'game_date': date, 'minutes_mean': float(rng.uniform(12, 36))})
                        for s in STATS:
                            m_model = mu[s] * np.exp(rng.normal(0, 0.08))
                            k = np.arange(CAPS[s] + 1)
                            p = poisson.pmf(k, m_model)
                            pm[s].append(p / p.sum())
                            line = np.floor(mu[s]) + 0.5
                            m_mkt = mu[s] * np.exp(rng.normal(0, 0.25))
                            fair = float(np.clip(poisson.sf(line, m_mkt) + 0.02, 0.03, 0.97))       # market over-bias: overs priced 2 points too high
                            snap = f'{date}T0{t % 10}:00:00Z'
                            cons_rows.append({'game_id': gid, 'player_id': pl, 'minutes_before': t, 'stat': s, 'snapshot_ts_utc': snap, 'event_id_vendor': 'e' + gid,
                                              'player_name_raw': f'p{pl}', 'consensus_line': line, 'consensus_fair_over': fair, 'n_at_line': 4, 'book_spread': 0.0})
                            for book in BOOKS:
                                fo = float(np.clip(fair + rng.normal(0, 0.015), 0.02, 0.98))
                                od, ud = 1 / (fo * 1.045), 1 / ((1 - fo) * 1.045)
                                bet_rows.append({'game_id': gid, 'player_id': pl, 'minutes_before': t, 'stat': s, 'book': book, 'line': line, 'over_price': np.nan, 'under_price': np.nan,
                                                 'over_dec': od, 'under_dec': ud, 'fair_over': fo, 'consensus_fair_over': fair, 'n_at_line': 4, 'snapshot_ts_utc': snap,
                                                 'quote_age_min': 3.0})
        d = reg / 'a11_test_p5' / season
        d.mkdir(parents=True)
        pd.DataFrame(idx_rows).to_csv(d / 'index.csv', index=False)
        for s in STATS:
            np.save(d / f'pred_{s}.npy', np.asarray(pm[s], dtype=np.float32))
        with open(prices / f'prices_{season}.pkl', 'wb') as f:
            pickle.dump((pd.DataFrame(cons_rows), pd.DataFrame(bet_rows)), f)
    (reg / 'a11_test_p5' / 'manifest.json').write_text(json.dumps({'features_db_sha256': 'x' * 64}), encoding='utf-8')
    db = root / 'labels.db'
    with sqlite3.connect(db) as conn:
        pd.DataFrame(labels).to_sql('player_game_labels', conn, index=False)
    return reg, prices, db


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp('a11world')
    return (root,) + _write_world(root)


def test_end_to_end_walkforward_rule_eval_final(world):
    root, reg, prices, db = world
    common = ['--p5-version', 'a11_test_p5', '--registry-root', str(reg), '--prices-dir', str(prices), '--features-db', str(db), '--n-boot', '60']
    inputs = {p: _sha(p) for p in list(prices.glob('*.pkl')) + [db] + list(reg.rglob('*.npy'))}
    kw = ke.main(common + ['--mode', 'walkforward', '--train', '2023-24', '--score', '2024-25', '--out', str(root / 'res6b'), '--report', str(root / 'rpt' / '6b.md')])
    kf = json.loads((root / 'res6b' / 'k_walkforward.json').read_text(encoding='utf-8'))
    assert kf['pick'] in ar.CANDIDATES[:5] and kf['estimator'] == ar.CAND_ESTIMATOR[kf['pick']]
    assert {c['item'] for c in kf['clauses']} >= {1, 2, 3, 4, 6}
    assert kf['thresholds']['adopt_margin'] == 0.0025
    assert kf['params_by_estimator']['E1']['b']['assists'] > 0.05         # the simulated model is informative
    assert 'PASS' in (root / 'rpt' / '6b.md').read_text(encoding='utf-8') or 'FAIL' in (root / 'rpt' / '6b.md').read_text(encoding='utf-8')
    assert kw['stop'] is False
    res = re_.main(common + ['--k-file', str(root / 'res6b' / 'k_walkforward.json'), '--experiments', '7c', '7d', '--out', str(root / 'res7cd'), '--report', str(root / 'rpt' / '7cd.md')])
    assert set(res['7c']['decisions']) == {'cap1', 'cap2', 'kelly'}
    assert res['7d']['decision'] is not None and len(res['7d']['decision']['clauses']) == 3
    met = res['7c']['metrics']
    assert set(met['variant']) == {'flat', 'cap1', 'cap2', 'kelly'} and set(met['season']) == {'2023-24', '2024-25'}
    rp = root / 'rpt' / 'cand.json'
    rule = ke.main(common + ['--mode', 'final', '--train', '2023-24', '2024-25', '--k-file', str(root / 'res6b' / 'k_walkforward.json'), '--write-rule', str(rp)])
    assert rule['candidate'] == kf['pick'] and rule['developed_on'] == ['2023-24', '2024-25'] and rule['tried_rules'] == 60
    assert (rp.parent / (rp.name + '.sha256')).exists()
    assert {p: _sha(p) for p in inputs} == inputs                       # inputs untouched
    re_.main(common + ['--rule-file', str(rp), '--experiments', '7d', '--out', str(root / 'res_final')])


def test_end_to_end_reproduce(world):
    root, reg, prices, db = world
    out = ke.main(['--mode', 'reproduce', '--p5-version', 'a11_test_p5', '--registry-root', str(reg), '--prices-dir', str(prices), '--features-db', str(db),
                   '--n-boot', '50', '--out', str(root / 'repro.json')])
    assert out['gate'] in ('PASS', 'FAIL')
    t = out['table'].set_index('stat')
    assert set(ar.BET_STATS) <= set(t.index)
    assert t.loc['assists', 'within_0.05'] == (abs(t.loc['assists', 'k_diff']) <= 0.05 and abs(t.loc['assists', 'slope_diff']) <= 0.05)
    assert set(out['baseline_frozen_k']) == {'2023-24', '2024-25'}
    assert (root / 'repro.md').exists()
