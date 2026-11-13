"""Forward-test monitor (props/live/monitor.py, scripts/props_forward_monitor.py): synthetic data only, no data/ files."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from props.live import monitor as mon
from props.live import store

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _bets_results(rows):
    """rows: dicts with the fields that matter; fills the rest with defaults. Returns (bets, results)."""
    bets, results = [], []
    for i, r in enumerate(rows):
        b = {'bet_id': f'b{i}', 'game_id': f'g{i}', 'player_id': i, 'game_date': '2026-11-10', 'stat': 'assists', 'side': 'under', 'line': 5.5, 'minutes_before': 240, 'model_version': 'v1',
             'dec_best': 2.0, 'dec_median': 1.9, 'dec_flat': 1.909, 'ev_best': 0.04, 'p_over_model': 0.45, 'p_over_star': 0.47, 'p_over_market': 0.5, 'expected_minutes': 30.0, 'book': 'fanduel'}
        res = {'bet_id': f'b{i}', 'status_stands': 'win', 'profit_stands_best': 1.0, 'profit_stands_median': 0.9, 'profit_stands_flat': 0.909, 'clv_prob': 0.01, 'y': 4.0, 'played': 1}
        for k, v in r.items():
            (b if k in b or k not in res else res)[k] = v
        bets.append(b)
        results.append(res)
    return pd.DataFrame(bets), pd.DataFrame(results)


def _roi_data(n_dates=20, per_date=30, roi=0.02, date_sd=0.10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2026-11-13', periods=n_dates).strftime('%Y-%m-%d')
    cl, pr = [], []
    for d in dates:
        eff = rng.normal(0, date_sd)
        cl += [d] * per_date
        pr += list(roi + eff + rng.normal(0, 1.0, per_date))
    return np.array(pr), np.ones(len(pr)), np.array(cl)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# cluster bootstrap
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_roi_point_is_ratio_of_sums_and_voids_without_stake_do_not_count():
    profit = np.array([1.0, -1.0, 1.0, 0.0])
    stake = np.array([1.0, 1.0, 1.0, 0.0])
    r = mon.cluster_bootstrap_roi(profit, stake, ['a', 'a', 'b', 'b'], n_boot=200, seed=1)
    assert r['point'] == pytest.approx(1 / 3)
    assert r['n'] == 3 and r['n_clusters'] == 2


def test_bootstrap_is_deterministic_for_a_seed_and_changes_with_it():
    p, s, c = _roi_data()
    a = mon.cluster_bootstrap_roi(p, s, c, n_boot=500, seed=7)
    b = mon.cluster_bootstrap_roi(p, s, c, n_boot=500, seed=7)
    d = mon.cluster_bootstrap_roi(p, s, c, n_boot=500, seed=8)
    assert a == b
    assert (a['lo'], a['hi']) != (d['lo'], d['hi'])
    assert a['lo'] < a['point'] < a['hi'] and 0 <= a['p_le0'] <= 1


def test_bootstrap_coverage_is_sane_on_clustered_data():
    true_roi, hits, reps = 0.02, 0, 120
    for rep in range(reps):
        p, s, c = _roi_data(n_dates=25, per_date=20, roi=true_roi, date_sd=0.15, seed=100 + rep)
        r = mon.cluster_bootstrap_roi(p, s, c, n_boot=300, seed=rep)
        hits += r['lo'] <= true_roi <= r['hi']
    assert 0.85 <= hits / reps <= 1.0                               # nominal 95%; 25 clusters under-cover a little


def test_clustering_widens_the_interval_when_dates_share_an_effect():
    p0, s0, c0 = _roi_data(date_sd=0.0, seed=3)
    p1, s1, c1 = _roi_data(date_sd=0.5, seed=3)
    r0 = mon.cluster_bootstrap_roi(p0, s0, c0, n_boot=800, seed=1)
    r1 = mon.cluster_bootstrap_roi(p1, s1, c1, n_boot=800, seed=1)
    assert (r1['hi'] - r1['lo']) > 2 * (r0['hi'] - r0['lo'])


def test_clustered_mean_ignores_nan_and_reports_se():
    vals = np.array([1.0, 3.0, np.nan, 5.0, 7.0])
    r = mon.cluster_bootstrap_mean(vals, ['a', 'a', 'a', 'b', 'b'], n_boot=300, seed=2)
    assert r['point'] == pytest.approx(4.0) and r['n'] == 4 and r['n_clusters'] == 2
    # cluster sums of (v - mean): a: -2 + 0... = (1-4)+(3-4) = -4 ; b: (5-4)+(7-4) = 4 -> SE = sqrt(32)/4
    assert r['se'] == pytest.approx(np.sqrt(32) / 4)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# segment table
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_segment_table_arithmetic_with_voids_and_pushes_not_stakes():
    rows = [
        {'side': 'under', 'status_stands': 'win', 'profit_stands_median': 0.9, 'profit_stands_best': 1.0, 'profit_stands_flat': 0.909, 'game_date': '2026-11-10'},
        {'side': 'under', 'status_stands': 'loss', 'profit_stands_median': -1.0, 'profit_stands_best': -1.0, 'profit_stands_flat': -1.0, 'game_date': '2026-11-11'},
        {'side': 'under', 'status_stands': 'void', 'profit_stands_median': 0.0, 'profit_stands_best': 0.0, 'profit_stands_flat': 0.0, 'game_date': '2026-11-11', 'played': 0, 'y': np.nan},
        {'side': 'over', 'status_stands': 'win', 'profit_stands_median': 0.9, 'profit_stands_best': 1.0, 'profit_stands_flat': 0.909, 'game_date': '2026-11-10', 'p_over_market': 0.4},
        {'side': 'over', 'status_stands': 'push', 'profit_stands_median': 0.0, 'profit_stands_best': 0.0, 'profit_stands_flat': 0.0, 'game_date': '2026-11-12'},
    ]
    bets, res = _bets_results(rows)
    prep = mon.prepare_bets(bets, res)
    seg = mon.segment_table(prep, 'side', n_boot=200, seed=1).set_index('side')
    u, o = seg.loc['under'], seg.loc['over']
    assert u['n_bets'] == 3 and u['n_stakes'] == 2 and u['wins'] == 1                                  # the void is a bet but not a stake
    assert u['win_rate'] == pytest.approx(0.5)
    assert u['roi_median'] == pytest.approx(-0.1 / 2) and u['roi_best'] == pytest.approx(0.0)
    assert u['roi_flat'] == pytest.approx((0.909 - 1.0) / 2)
    assert o['n_bets'] == 2 and o['n_stakes'] == 1 and o['roi_median'] == pytest.approx(0.9)         # the push is not a stake
    assert u['low_n'] and o['low_n']                                                                  # n < 50 flagged
    assert o['mean_p_market'] == pytest.approx(0.4) and u['mean_p_market'] == pytest.approx(0.5)      # fair probability of the bet side (over at 0.4; under at 1 - 0.5)
    assert u['clv_mean'] == pytest.approx(1.0) and u['clv_n'] == 3


def test_over_side_probability_is_the_over_probability_and_under_is_its_complement():
    bets, res = _bets_results([{'side': 'over', 'p_over_market': 0.4, 'p_over_star': 0.45, 'p_over_model': 0.6}, {'side': 'under', 'p_over_market': 0.4, 'p_over_star': 0.45, 'p_over_model': 0.6}])
    prep = mon.prepare_bets(bets, res)
    assert prep['p_market_side'].tolist() == pytest.approx([0.4, 0.6])
    assert prep['p_star_side'].tolist() == pytest.approx([0.45, 0.55])
    assert prep['p_model_side'].tolist() == pytest.approx([0.6, 0.4])


def test_segment_buckets_ev_minutes_slot_week_and_cross_columns():
    rows = [{'ev_best': 0.031, 'expected_minutes': 15.0, 'minutes_before': 240, 'game_date': '2026-11-09'},     # Monday
            {'ev_best': 0.045, 'expected_minutes': 20.0, 'minutes_before': 60, 'game_date': '2026-11-15'},      # Sunday of that week
            {'ev_best': 0.06, 'expected_minutes': 30.0, 'minutes_before': 5, 'game_date': '2026-11-16'},
            {'ev_best': 0.09, 'expected_minutes': 40.0, 'minutes_before': 5, 'game_date': '2026-11-16', 'stat': 'threes', 'side': 'over'}]
    bets, res = _bets_results(rows)
    d = mon.add_segments(mon.prepare_bets(bets, res))
    assert d['ev_bucket'].astype(str).tolist() == ['3-4%', '4-5%', '5-7%', '7%+']
    assert d['minutes_bucket'].astype(str).tolist() == ['<20', '20-28', '28-34', '34+']
    assert d['slot'].tolist() == ['T-240', 'T-60', 'T-5', 'T-5']
    assert d['week'].tolist() == ['2026-11-09', '2026-11-09', '2026-11-16', '2026-11-16']
    seg = mon.segment_table(mon.prepare_bets(bets, res), ['stat', 'side'], n_boot=50)
    assert set(map(tuple, seg[['stat', 'side']].values)) == {('assists', 'under'), ('threes', 'over')}
    allrow = mon.segment_table(mon.prepare_bets(bets, res), 'slot', n_boot=50, include_all=True).iloc[0]
    assert allrow['slot'] == 'ALL' and allrow['n_stakes'] == 4


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# information slope
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _simulate_markets(n, beta, biases, x_mean=0.02, x_sd=0.05, seed=0, n_dates=40):
    rng = np.random.default_rng(seed)
    stats = list(biases)
    stat = rng.choice(stats, n)
    p_mkt = rng.uniform(0.3, 0.7, n)
    p_mod = np.clip(p_mkt + rng.normal(x_mean, x_sd, n), 0.01, 0.99)
    x = p_mod - p_mkt
    prob = np.clip(p_mkt + np.array([biases[s] for s in stat]) + beta * x, 0, 1)
    hit = (rng.random(n) < prob).astype(float)
    dates = pd.date_range('2026-11-13', periods=n_dates).strftime('%Y-%m-%d')
    return pd.DataFrame({'stat': stat, 'game_date': rng.choice(dates, n), 'hit_over': hit, 'p_market': p_mkt, 'p_model': p_mod})


def test_slope_with_intercept_recovers_the_truth_and_through_origin_is_biased_by_a_market_bias():
    df = _simulate_markets(150_000, beta=0.30, biases={'rebounds': -0.03})
    sl = mon.information_slope(df, n_boot=200, seed=3).set_index('stat')
    r = sl.loc['rebounds']
    # E[through-origin] = beta + bias * E[x] / E[x^2] = 0.30 - 0.03 * 0.02 / (0.0004 + 0.0025) = 0.093
    assert 0.0 < r['slope_origin'] < 0.20
    assert abs(r['slope_icpt'] - 0.30) < 0.08
    assert r['si_lo'] < 0.30 < r['si_hi']
    assert not (r['so_lo'] < 0.30 < r['so_hi'])                       # the origin interval excludes the truth
    assert r['intercept'] == pytest.approx(-0.03, abs=0.01)
    assert r['over_bias'] == pytest.approx(-0.03 + 0.30 * 0.02, abs=0.01)


def test_pooled_slope_uses_stat_specific_intercepts():
    df = _simulate_markets(200_000, beta=0.30, biases={'rebounds': -0.04, 'threes': 0.04}, x_mean=0.0, seed=5)
    sl = mon.information_slope(df, n_boot=200, seed=2).set_index('stat')
    assert set(sl.index) == {'rebounds', 'threes', 'pooled'}
    assert abs(sl.loc['pooled', 'slope_icpt'] - 0.30) < 0.06
    assert sl.loc['pooled', 'n'] == 200_000
    assert sl.loc['rebounds', 'intercept'] < 0 < sl.loc['threes', 'intercept']


def test_slope_without_signal_is_near_zero_and_empty_input_is_handled():
    df = _simulate_markets(100_000, beta=0.0, biases={'assists': 0.0}, x_mean=0.0, seed=9)
    r = mon.information_slope(df, n_boot=200, seed=1).set_index('stat').loc['pooled']
    assert r['si_lo'] < 0 < r['si_hi']
    assert mon.information_slope(df.iloc[0:0]).empty


def test_market_frame_takes_earliest_slot_and_drops_pushes_and_non_players():
    sig = pd.DataFrame({'game_id': ['g1'] * 4 + ['g2', 'g3'], 'player_id': [1, 1, 2, 3, 4, 5], 'stat': ['assists'] * 6, 'slot_minutes': [60, 240, 240, 240, 240, 240],
                        'line': [5.5, 6.5, 4.5, 3.5, 2.5, 8.5], 'p_market': [0.5] * 6, 'p_model': [0.6] * 6, 'game_date': ['2026-11-12'] * 6})
    out = pd.DataFrame({'game_id': ['g1', 'g1', 'g1', 'g2'], 'player_id': [1, 2, 3, 4], 'stat': ['assists'] * 4, 'y': [7.0, 4.5 - 0.5, 3.5, np.nan], 'played': [1, 1, 1, 0]})
    m = mon.market_frame(sig, out).set_index('player_id')
    assert m.loc[1, 'slot_minutes'] == 240 and m.loc[1, 'line'] == 6.5 and m.loc[1, 'hit_over'] == 1.0          # earliest slot, y 7 > 6.5
    assert m.loc[2, 'hit_over'] == 0.0                                                                           # 4.0 < 4.5
    assert np.isnan(m.loc[3, 'hit_over'])                                                                        # push
    assert np.isnan(m.loc[4, 'hit_over']) and np.isnan(m.loc[5, 'hit_over'])                                    # did not play / outcome unknown


def test_outcomes_from_box_resolves_ra_and_marks_absent_players_as_not_played_but_skips_unknown_games():
    box = pd.DataFrame({'game_id': ['g1', 'g1'], 'player_id': [1, 2], 'minutes': [30.0, 0.0], 'points': [10, 0], 'rebounds': [5, 0], 'assists': [3, 0], 'threes': [2, 0]})
    keys = pd.DataFrame({'game_id': ['g1', 'g1', 'g1', 'g1', 'g9'], 'player_id': [1, 1, 2, 7, 1], 'stat': ['ra', 'threes', 'assists', 'assists', 'ra']})
    o = mon.outcomes_from_box(box, keys).set_index(['player_id', 'stat'])
    assert len(o) == 4 and (1, 'ra') in o.index
    assert o.loc[(1, 'ra'), 'y'] == 8 and o.loc[(1, 'threes'), 'y'] == 2
    assert o.loc[(2, 'assists'), 'played'] == 0 and o.loc[(7, 'assists'), 'played'] == 0                         # zero minutes / not on the box score


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# calibration, CLV break-even, sample size, fills
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_calibration_expected_versus_actual_wins_and_brier():
    rows = [{'status_stands': 'win', 'p_over_star': 0.4, 'game_date': 'd1'}, {'status_stands': 'loss', 'p_over_star': 0.4, 'game_date': 'd1'}, {'status_stands': 'win', 'p_over_star': 0.4, 'game_date': 'd2'},
            {'status_stands': 'void', 'p_over_star': 0.4, 'game_date': 'd2'}]
    bets, res = _bets_results(rows)
    cal = mon.calibration(mon.prepare_bets(bets, res))                                   # under bets: p* of the bet side = 0.6
    assert cal['n'] == 3 and cal['actual_wins'] == 2 and cal['expected_wins_pstar'] == pytest.approx(1.8)
    assert cal['diff'] == pytest.approx(0.2)
    # cluster sums of residuals: d1 = (0.4) + (-0.6) = -0.2 ; d2 = 0.4  ->  SE = sqrt(0.04 + 0.16)
    assert cal['diff_se'] == pytest.approx(np.sqrt(0.2))
    assert cal['brier_pstar'] == pytest.approx(np.mean([0.16, 0.36, 0.16]))


def test_clv_breakeven_arithmetic():
    bets, res = _bets_results([{'side': 'over', 'p_over_market': 0.5, 'dec_median': 1.9, 'dec_best': 2.0, 'clv_prob': 0.01}])
    be = mon.clv_breakeven(mon.prepare_bets(bets, res))
    assert be['breakeven_clv_median_points'] == pytest.approx(100 * (1 / 1.9 - 0.5))
    assert be['breakeven_clv_best_points'] == pytest.approx(100 * (1 / 2.0 - 0.5))
    assert be['clv_mean_points'] == pytest.approx(1.0)
    assert be['clv_implied_roi_median'] == pytest.approx(0.51 * 1.9 - 1)
    assert be['clv_implied_roi_best'] == pytest.approx(0.51 * 2.0 - 1)


def test_required_dates_scales_with_the_inverse_square_of_the_half_width():
    p, s, c = _roi_data(n_dates=30, per_date=40, roi=0.0, date_sd=0.2, seed=4)
    now = mon.required_dates(p, s, c, 0.02)
    same = mon.required_dates(p, s, c, now['half_width_now'])
    assert same['dates'] == pytest.approx(30, rel=1e-6)
    half = mon.required_dates(p, s, c, now['half_width_now'] / 2)
    assert half['dates'] == pytest.approx(4 * 30, rel=1e-6)
    assert now['mean_stakes_per_date'] == 40 and now['stakes'] == pytest.approx(now['dates'] * 40)


def test_config_segments_shows_a_second_configuration():
    bets, _ = _bets_results([{'model_sha256': 'a', 'rule_sha256': 'r', 'roster_rule': 'v2'}, {'model_sha256': 'a', 'rule_sha256': 'r', 'roster_rule': 'v2', 'game_date': '2026-11-12', 'minutes_before': 5},
                             {'model_sha256': 'b', 'rule_sha256': 'r', 'roster_rule': 'v2'}])
    seg = mon.config_segments(bets)
    assert len(seg) == 2
    first = seg[seg['model_sha256'] == 'a'].iloc[0]
    assert first['n_bets'] == 2 and first['last_date'] == '2026-11-12' and first['slots'] == 'T-240,T-5'


def test_fill_summary_reports_shares_and_best_price_roi_by_fill_flag():
    rows = [{'status_stands': 'win'}, {'status_stands': 'loss', 'profit_stands_best': -1.0}, {'status_stands': 'win'}]
    bets, res = _bets_results(rows)
    prep = mon.prepare_bets(bets, res)
    fills = pd.DataFrame({'bet_id': ['b0', 'b1', 'b2'], 'book_still_at_or_better_next': [1, 0, 1], 'best_still_at_or_better_next': [1, 0, 0], 'minutes_price_available': [10.0, 0.0, 30.0],
                          'max_best_price_before_tip': [2.1, 1.9, 2.0]})
    fs = mon.fill_summary(prep, fills)
    assert fs['n'] == 3 and fs['share_best_still_at_or_better_next'] == pytest.approx(1 / 3)
    assert fs['share_max_best_price_at_or_above_bet'] == pytest.approx(2 / 3)
    assert fs['best_roi_by_best_still_next'][1] == {'n_stakes': 1, 'roi_best': 1.0}
    assert fs['best_roi_by_best_still_next'][0]['n_stakes'] == 2 and fs['best_roi_by_best_still_next'][0]['roi_best'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# sequential rule
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _stream(n_dates, per_date, roi, seed=0, start='2026-11-13', noise=1.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_dates).strftime('%Y-%m-%d')
    cl = np.repeat(dates, per_date)
    pr = roi + rng.normal(0, noise, len(cl))
    return pr, np.ones(len(cl)), cl


def test_sequential_continue_before_the_first_look():
    p, s, c = _stream(10, 30, 0.0)
    r = mon.sequential_status(p, s, c, horizon_stakes=3000, n_boot=300)
    assert r['status'] == 'CONTINUE' and r['looks'] == [] and r['n_stakes'] == 300


def test_sequential_pass_and_fail_at_the_horizon():
    p, s, c = _stream(40, 30, 0.20, seed=1)
    ok = mon.sequential_status(p, s, c, horizon_stakes=1200, n_boot=500)
    assert ok['status'] == 'PASS' and ok['lo'] > 0
    p0, s0, c0 = _stream(40, 30, 0.0, seed=1)
    no = mon.sequential_status(p0, s0, c0, horizon_stakes=1200, n_boot=500)
    assert no['status'] == 'FAIL' and no['lo'] <= 0
    assert no['n_stakes'] == 1200


def test_sequential_final_uses_only_data_through_the_horizon_date():
    p, s, c = _stream(40, 30, 0.20, seed=1)
    p_extra = np.concatenate([p, np.full(300, -5.0)])                      # 10 more terrible dates after the horizon was reached
    s_extra = np.concatenate([s, np.ones(300)])
    c_extra = np.concatenate([c, np.repeat(pd.date_range('2027-01-01', periods=10).strftime('%Y-%m-%d'), 30)])
    assert mon.sequential_status(p_extra, s_extra, c_extra, horizon_stakes=1200, n_boot=500)['status'] == 'PASS'


def test_sequential_needs_a_minimum_number_of_dates_for_a_pass():
    p, s, c = _stream(10, 200, 0.20, seed=2)                               # plenty of stakes, few dates
    r = mon.sequential_status(p, s, c, horizon_stakes=2000, n_boot=300, min_dates=30)
    assert r['status'] == 'FAIL' and 'game dates' in r['reason']


def test_sequential_calendar_horizon_applies_when_it_comes_first():
    p, s, c = _stream(40, 10, 0.25, seed=3)                                # 400 stakes by 2026-12-22
    pending = mon.sequential_status(p, s, c, horizon_stakes=5000, horizon_date='2027-04-12', today='2026-12-23', n_boot=300)
    assert pending['status'] in ('CONTINUE', 'STOP-FUTILITY')
    done = mon.sequential_status(p, s, c, horizon_stakes=5000, horizon_date='2026-12-22', today='2026-12-23', n_boot=300, min_dates=30)
    assert done['status'] in ('PASS', 'FAIL') and 'horizon reached' in done['reason']


def test_sequential_futility_stops_a_clearly_losing_stream_and_not_a_winning_one():
    p, s, c = _stream(40, 25, -0.15, seed=4)                               # 1000 stakes of a 3000 horizon = past the 1/3 look
    bad = mon.sequential_status(p, s, c, horizon_stakes=3000, n_boot=300)
    assert bad['status'] == 'STOP-FUTILITY' and bad['cp'] < 0.10 and bad['looks'][0]['fraction'] == pytest.approx(1 / 3)
    pg, sg, cg = _stream(40, 25, 0.10, seed=4)
    good = mon.sequential_status(pg, sg, cg, horizon_stakes=3000, n_boot=300)
    assert good['status'] == 'CONTINUE' and good['looks'][0]['cp'] > 0.10


def test_futility_is_evaluated_at_the_look_and_not_on_later_data():
    p1, s1, c1 = _stream(34, 30, 0.10, seed=6)                              # 1020 stakes (>= 1/3 of 3000) with a good start
    p2, s2, c2 = _stream(10, 30, -0.9, seed=7, start='2027-01-05')          # then a collapse, still before the 2/3 look
    r = mon.sequential_status(np.r_[p1, p2], np.r_[s1, s2], np.r_[c1, c2], horizon_stakes=3000, n_boot=300)
    assert r['looks'][0]['cutoff'] <= c1[-1]                                # the first look is taken on the data through its own cutoff date
    assert r['looks'][0]['cp'] > 0.10


def test_conditional_power_properties():
    cp = mon.conditional_power
    assert cp(0.03, 0.01, 0.5, 0.03) > cp(0.0, 0.01, 0.5, 0.03) > cp(-0.03, 0.01, 0.5, 0.03)
    assert cp(0.03, 0.01, 0.5, 0.05) > cp(0.03, 0.01, 0.5, 0.0)
    assert cp(0.03, 0.01, 1.0, 0.03) == 1.0 and cp(0.01, 0.01, 1.0, 0.03) == 0.0                # at the horizon it is the pass indicator
    assert 0.0 <= cp(0.0, 0.02, 0.2, 0.03) <= 1.0


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the script
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _fake_live_db(path):
    conn = store.connect(path)
    bets, res = _bets_results([{'game_date': '2026-11-10', 'status_stands': 'win'}, {'game_date': '2026-11-11', 'status_stands': 'loss', 'profit_stands_best': -1.0, 'profit_stands_median': -1.0,
                                                                                         'profit_stands_flat': -1.0},
                               {'game_date': '2026-11-14', 'status_stands': 'win'}])
    bets = bets.assign(season='2026-27', created_utc='2026-11-10T00:00:00Z', tip_ref_utc='2026-11-10T01:00:00Z')
    res = res.assign(settled_utc='2026-11-12T00:00:00Z')
    store._insert(conn, 'prop_bets', bets)
    store._insert(conn, 'prop_results', res)
    conn.commit()
    conn.close()


def test_script_is_read_only_and_reports_a_forward_split(tmp_path):
    import props_forward_monitor as cli
    db = tmp_path / 'live.db'
    _fake_live_db(db)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    text = cli.build_report(str(db), nba_db=None, since='2026-11-13', today='2026-11-15')
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert 'Forward test: game_date >= 2026-11-13' in text and 'descriptive only' in text and 'Decision rule' in text
    assert 'MEDIAN price' in text
    conn = cli.open_ro(str(db))
    with pytest.raises(Exception):
        conn.execute('CREATE TABLE x (a)')
    conn.close()


def test_script_main_writes_a_dated_file_only_where_asked(tmp_path, capsys):
    import props_forward_monitor as cli
    db = tmp_path / 'live.db'
    _fake_live_db(db)
    out = tmp_path / 'out'
    assert cli.main(['--db', str(db), '--out', str(out), '--today', '2026-11-15']) == 0
    assert (out / 'props_forward_monitor_2026-11-15.md').exists()
    assert not (tmp_path / 'data').exists()
