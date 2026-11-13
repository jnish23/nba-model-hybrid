"""
Segment 2 decision rule of the forward-test monitor (reports/props/forward_test_preregistration_segment2.md, sections 1, 6, 7, 11) and the segment 1 regression.
Synthetic data only, no data/ files. The simulations use the analytic clustered SE (the bootstrap would be far too slow for thousands of seasons); the decision function itself uses the
analytic SE, so the simulation exercises exactly the production decision path. The bootstrap only serves the secondary intervals and the paired difference.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from props.betting import rules
from props.live import monitor as mon
from props.live import store

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

GOLDEN = Path(__file__).resolve().parent / 'golden'
BOUNDS = (3.471, 2.454, 2.004)
LOOKS = (50, 100)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# synthetic store shared by the segment 1 regression and the segment 2 report tests
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def regression_store(path, n_dates=14, start='2026-11-08', per_date=6, seed=11, rule_sha='r1', roster='v2', late_rule_sha=None, p_win=None, p_win_switch=None, unsettled_dates=()):
    """
    A deterministic synthetic props_live.db: n_dates game dates from `start`, per_date bets each (three slots, four stats, both sides, a few voids and pushes), two model versions.
    late_rule_sha: rule sha of the last four dates. p_win: win probability with no voids or pushes (default: the mixed outcomes of the segment 1 golden files).
    unsettled_dates: dates whose bets get no result row. p_win_switch: (date index, p) from which that win probability applies. The random draws do not depend on these options, so the default store never changes.
    """
    rng = np.random.default_rng(seed)
    bets, results = [], []
    stats = ['rebounds', 'assists', 'threes', 'ra']
    slots = [240, 60, 5]
    dates = pd.date_range(start, periods=n_dates).strftime('%Y-%m-%d')
    i = 0
    for di, d in enumerate(dates):
        for k in range(per_date):
            dec_best = float(np.round(rng.uniform(1.85, 2.15), 3))
            dec_med = float(np.round(dec_best - rng.uniform(0.02, 0.1), 3))
            p_mkt = float(np.round(rng.uniform(0.40, 0.60), 3))
            late = di >= n_dates - 4
            b = {'bet_id': f'b{i}', 'season': '2026-27', 'game_id': f'g{di}_{k // 2}', 'game_date': d, 'tip_ref_utc': f'{d}T23:00:00Z', 'created_utc': f'{d}T12:00:00Z', 'minutes_before': slots[k % 3],
                 'player_id': 1000 + i, 'stat': stats[(k + di) % 4], 'side': 'over' if (k + di) % 2 else 'under', 'line': 5.5, 'book': ['fanduel', 'draftkings', 'betmgm'][k % 3],
                 'dec_best': dec_best, 'dec_median': dec_med, 'dec_flat': 1.909, 'ev_best': float(np.round(rng.uniform(0.03, 0.10), 4)), 'p_over_model': float(np.round(p_mkt + rng.normal(0.02, 0.05), 3)),
                 'p_over_star': float(np.round(p_mkt + rng.normal(0.01, 0.02), 3)), 'p_over_market': p_mkt, 'expected_minutes': float(np.round(rng.uniform(12, 38), 1)),
                 'model_version': 'v1b' if late else 'v1', 'model_sha256': 'b' * 16 if late else 'a' * 16, 'rule_sha256': late_rule_sha if (late and late_rule_sha) else rule_sha, 'roster_rule': roster}
            u = rng.random()
            pw = p_win_switch[1] if (p_win_switch and di >= p_win_switch[0]) else p_win
            if pw is None:
                status = 'void' if u < 0.06 else ('push' if u < 0.10 else ('win' if u < 0.55 else 'loss'))
            else:
                status = 'win' if u < pw else 'loss'
            profit = {'win': (dec_best - 1, dec_med - 1, 0.909), 'loss': (-1.0, -1.0, -1.0)}.get(status, (0.0, 0.0, 0.0))
            r = {'bet_id': f'b{i}', 'settled_utc': f'{d}T23:59:00Z', 'status_stands': status, 'profit_stands_best': profit[0], 'profit_stands_median': profit[1], 'profit_stands_flat': profit[2],
                 'clv_prob': float(np.round(rng.normal(0.005, 0.02), 4)) if k % 4 else None, 'y': 4.0 if status != 'void' else None, 'played': 0 if status == 'void' else 1}
            bets.append(b)
            if d not in unsettled_dates:
                results.append(r)
            i += 1
    conn = store.connect(path)
    store._insert(conn, 'prop_bets', pd.DataFrame(bets))
    store._insert(conn, 'prop_results', pd.DataFrame(results))
    conn.commit()
    conn.close()
    return dates


def render_segment1(db, since, today):
    import props_forward_monitor as cli
    return cli.build_report(str(db), nba_db=None, since=since, today=today).replace(str(db), '{DB}')


def render_segment2(db, since, today, expected_dates=150):
    import props_forward_monitor as cli
    return cli.build_report(str(db), nba_db=None, since=since, today=today, segment=2, expected_dates=expected_dates).replace(str(db), '{DB}')


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# per-date test data with an exactly known Z
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _dates(n, start='2026-11-14'):
    return pd.date_range(start, periods=n).strftime('%Y-%m-%d').to_numpy()


def _z_stream(n, z, per_date=10, c=0.12, start='2026-11-14', shift=0.0):
    """
    Per-bet arrays over n game dates (n even) whose ROI is a = z c / sqrt(n) + shift and whose per-date ROI deviations are +-c, alternating: the analytic clustered SE is exactly c / sqrt(n),
    so Z = z when shift = 0 (c = 0.12 is about the per-date ROI SD at 72 stakes of unit variance per date).
    """
    e = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    roi_d = z * c / np.sqrt(n) + shift + c * e
    cluster = np.repeat(_dates(n, start), per_date)
    return np.repeat(roi_d, per_date), np.ones(n * per_date), cluster


def _concat(*streams):
    return tuple(np.concatenate(parts) for parts in zip(*streams))


def _gs(profit, stake, cluster, **kw):
    kw.setdefault('expected_dates', 150)
    return mon.group_sequential_status(profit, stake, cluster, LOOKS, BOUNDS, **kw)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. boundaries
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_z_stream_helper_has_the_stated_z():
    p, s, c = _z_stream(50, 3.0)
    r = mon.cluster_bootstrap_roi(p, s, c, n_boot=50, seed=1)
    assert r['point'] / r['se'] == pytest.approx(3.0, abs=1e-9)


def test_look_1_boundary_3_471():
    below = _gs(*_z_stream(50, 3.470))
    above = _gs(*_z_stream(50, 3.472))
    assert below['status'] == 'CONTINUE' and not below['looks'][0]['passed']
    assert above['status'] == 'PASS' and above['decided_at'] == 'look 1' and above['looks'][0]['z'] == pytest.approx(3.472, abs=1e-9)
    assert not mon.look_passes(3.471, 50, 3.471) and mon.look_passes(3.4711, 50, 3.471)               # strictly above


def test_look_2_boundary_2_454():
    below = _gs(*_z_stream(100, 2.453))
    above = _gs(*_z_stream(100, 2.455))
    assert below['status'] == 'CONTINUE' and [lk['passed'] for lk in below['looks']] == [False, False]
    assert above['status'] == 'PASS' and above['decided_at'] == 'look 2' and above['looks'][0]['passed'] is False       # look 1 saw Z = 2.455 / sqrt(2) = 1.74


def test_final_boundary_2_004_and_a_fail_below_it():
    dates = _dates(150)
    kw = dict(horizon_date=dates[-1], today='2027-04-13')
    below = _gs(*_z_stream(150, 2.003), **kw)
    above = _gs(*_z_stream(150, 2.005), **kw)
    assert below['status'] == 'FAIL' and below['decided_at'] == 'final' and below['looks'][-1]['look'] == 'final'
    assert above['status'] == 'PASS' and above['decided_at'] == 'final'
    assert [lk['passed'] for lk in above['looks']] == [False, False, True]                              # no interim look passed or stopped it
    assert below['n_dates'] == 150
    # before the horizon date the final look is not taken
    assert _gs(*_z_stream(150, 2.5), horizon_date=dates[-1], today='2027-04-11')['status'] == 'CONTINUE'


def test_fewer_than_30_game_dates_never_passes():
    assert not mon.look_passes(50.0, 29, 2.004) and mon.look_passes(50.0, 30, 2.004)
    p, s, c = _z_stream(20, 12.0)                                                                       # enormous Z, 20 dates, the calendar horizon reached
    r = _gs(p, s, c, horizon_date=_dates(20)[-1], today='2027-04-13')
    assert r['status'] == 'FAIL' and 'only 20 game dates' in r['reason']
    assert _gs(p, s, c)['status'] == 'CONTINUE'                                                          # and no look is reached below 50 dates


def test_looks_count_dates_with_a_settled_stake_not_bets_or_calendar_days():
    p, s, c = _z_stream(60, 0.0, per_date=3)
    s_void = s.copy()
    s_void[c == np.unique(c)[10]] = 0.0                                                                  # one date has only voids / pushes: no stake, not a date
    r = _gs(p, s_void, c)
    assert r['n_dates'] == 59 and r['looks'][0]['cutoff'] == np.unique(c)[50]                            # the 50th date with a stake is the 51st calendar date


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. each look uses only the data through its own cut-off
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_later_dates_do_not_change_look_1_and_a_pass_at_look_1_stays():
    base = _z_stream(50, 4.2)
    terrible = _z_stream(40, 0.0, start='2027-01-05', shift=-0.5)
    alone = _gs(*base)
    with_later = _gs(*_concat(base, terrible))
    assert alone['status'] == 'PASS' and with_later['status'] == 'PASS' and with_later['decided_at'] == 'look 1'
    assert with_later['looks'] == alone['looks'] and with_later['reason'] == alone['reason']


def test_later_dates_do_not_change_look_2_and_a_pass_at_look_2_stays():
    base = _z_stream(100, 2.6)
    terrible = _z_stream(60, 0.0, start='2027-02-23', shift=-1.0)
    alone = _gs(*base)
    with_later = _gs(*_concat(base, terrible))
    assert alone['status'] == 'PASS' and alone['decided_at'] == 'look 2'
    assert with_later['status'] == 'PASS' and with_later['decided_at'] == 'look 2' and with_later['looks'] == alone['looks']


def test_extreme_later_dates_leave_the_earlier_looks_unchanged_when_nothing_passed():
    base = _z_stream(100, 1.5)
    great = _z_stream(40, 0.0, start='2027-02-23', shift=3.0)
    alone = _gs(*base)
    later = _gs(*_concat(base, great))
    assert alone['status'] == 'CONTINUE' and later['looks'] == alone['looks']                            # the later dates matter only at the final look
    assert later['n_dates'] == 140


def test_the_per_bet_and_per_date_entry_points_agree_and_input_order_is_irrelevant():
    p, s, c = _z_stream(100, 2.455)
    perm = np.random.default_rng(0).permutation(len(p))
    a = _gs(p, s, c)
    b = _gs(p[perm], s[perm], c[perm])
    assert a == b
    g = pd.DataFrame({'c': c, 'p': p, 's': s}).groupby('c').agg(num=('p', 'sum'), den=('s', 'sum'))
    d = mon.group_sequential_dates(g.index.to_numpy(), g['num'], g['den'], LOOKS, BOUNDS, 150)
    assert d['status'] == a['status'] and d['looks'] == a['looks']


def test_a_look_waits_while_a_date_through_its_cutoff_is_not_fully_settled():
    p, s, c = _z_stream(50, 4.2)
    cut = np.unique(c)[-1]
    held = _gs(p, s, c, pending_dates=[cut])
    assert held['status'] == 'CONTINUE' and held['waiting'] == [cut] and 'not fully settled' in held['reason']
    later_pending = _gs(p, s, c, pending_dates=['2027-02-15'])                                         # a pending date AFTER the cut-off does not matter
    assert later_pending['status'] == 'PASS'
    release = _gs(p, s, c, pending_dates=[])
    assert release['status'] == 'PASS' and release['looks'][0]['z'] == later_pending['looks'][0]['z']


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. futility
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_conditional_power_matches_a_hand_computed_value_with_the_final_boundary_2_004():
    # ROI 3.0%, SE 2.0 pts (Z 1.5) at t = 0.5, true ROI 3.0% for the rest:
    # B_t = 1.5 sqrt(.5) = 1.0607; SE_final = .02 sqrt(.5) = .014142; mu = .03 / .014142 = 2.1213; E[Z_final] = 1.0607 + 2.1213 (.5) = 2.1213; sd = sqrt(.5) = .70711
    # CP = 1 - Phi((2.004 - 2.1213) / .70711) = 1 - Phi(-0.16592) = 0.56589
    assert mon.conditional_power(0.03, 0.02, 0.5, 0.03, z_crit=2.004) == pytest.approx(0.565888, abs=1e-5)
    # ROI 0, SE 2.0 pts at t = 1/3: B_t = 0; mu = .03 / (.02 sqrt(1/3)) = 2.5981; E = 2.5981 (2/3) = 1.7321; sd = sqrt(2/3) = .81650; CP = 1 - Phi((2.004 - 1.7321) / .8165) = 0.36954
    assert mon.conditional_power(0.0, 0.02, 1 / 3, 0.03, z_crit=2.004) == pytest.approx(0.369541, abs=1e-5)


def test_futility_stops_at_look_1_and_look_2_and_uses_the_information_fraction_in_dates():
    r1 = _gs(*_z_stream(50, -2.5))
    assert r1['status'] == 'STOP-FUTILITY' and r1['decided_at'] == 'look 1' and r1['looks'][0]['cp'] < 0.10 and r1['looks'][0]['t'] == pytest.approx(50 / 150)
    ok = _gs(*_z_stream(50, 1.0))
    assert ok['status'] == 'CONTINUE' and ok['looks'][0]['cp'] > 0.10
    r2 = _gs(*_concat(_z_stream(50, 1.0), _z_stream(50, 0.0, start='2027-01-03', shift=-0.1)))          # fine at look 1, collapses by date 100
    assert r2['status'] == 'STOP-FUTILITY' and r2['decided_at'] == 'look 2' and r2['looks'][1]['t'] == pytest.approx(100 / 150)
    # a smaller expected total makes the same look later in the season, hence less room to recover: conditional power falls
    assert _gs(*_z_stream(50, 1.0), expected_dates=100)['looks'][0]['cp'] < ok['looks'][0]['cp']


def test_futility_cannot_fire_at_the_final_look():
    first = _z_stream(100, 0.0, shift=0.02)                                                               # modest, never futile or passing at looks 1 and 2
    crash = _z_stream(50, 0.0, start='2027-02-23', shift=-0.5)
    r = _gs(*_concat(first, crash), horizon_date=_dates(150)[-1], today='2027-04-13')
    assert r['status'] == 'FAIL' and r['decided_at'] == 'final' and np.isnan(r['looks'][-1]['cp'])
    assert all(lk['cp'] > 0.10 for lk in r['looks'][:2])


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. false-PASS rate and power (analytic SE, vectorised draws, one pure-function call per season)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _simulate(true_roi, n_seasons, seed, n_dates=150, mean_stakes=72, stake_sd=25, per_bet_sd=0.969, date_sd=0.02):
    """
    Seasons of per-date sums: stakes per date ~ N(72, 25) (at least 10), per-bet SD 0.969 (the segment 1 record's), a date-level ROI effect of SD 2 pts. The rule is the production function
    on the per-date sums with the analytic SE; the horizon is the last date. Returns the share of seasons ending in PASS and the counts by outcome.
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n_dates)
    st = np.maximum(10.0, np.round(rng.normal(mean_stakes, stake_sd, (n_seasons, n_dates))))
    pr = true_roi * st + per_bet_sd * np.sqrt(st) * rng.standard_normal(st.shape) + st * date_sd * rng.standard_normal(st.shape)
    counts = {}
    for i in range(n_seasons):
        s = mon.group_sequential_dates(dates, pr[i], st[i], LOOKS, BOUNDS, 150, horizon_date=dates[-1], today='2027-12-31')
        key = (s['status'], s['decided_at'])
        counts[key] = counts.get(key, 0) + 1
    return sum(v for (status, _), v in counts.items() if status == 'PASS') / n_seasons, counts


def test_false_pass_rate_under_the_null_is_about_the_nominal_level():
    # Measured when written: 2.77% for this test's 12,000 seasons (of which 4 PASS at look 1, 102 at look 2, 226 at the final look) and 2.7% over 60,000 null seasons (2.6% to 2.8% per 20,000),
    # against the nominal 2.5% of the O'Brien-Fleming boundaries. The small excess is the estimated SE (the analytic cluster SE on 50 to 150 dates has t-like tails); futility stopping removes a little.
    rate, counts = _simulate(0.0, 12000, seed=20261113)
    print(f'null false-PASS rate {rate:.4f}  {counts}')
    assert rate <= 0.03, (rate, counts)
    assert rate >= 0.015, (rate, counts)                                                                  # the simulation is not trivially dead


def test_power_at_a_true_roi_of_3_percent():
    # Measured when written: 87.7% for this test's 3,000 seasons (207 PASS at look 1, 1,466 at look 2, 958 at the final look) and 88.6% over 4,000 other seasons, at 72 stakes per date
    # (the segment 2 document quotes about 80% at live volume and 91% at backtest volume, 93 to 104 stakes per date; the share passing by look 2 here is 56%, the document says about 44%).
    power, counts = _simulate(0.03, 3000, seed=7)
    print(f'power at +3% {power:.4f}  {counts}')
    assert 0.80 <= power <= 0.95, (power, counts)
    power_low, _ = _simulate(0.0, 500, seed=8)
    assert power > 10 * power_low


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 5. capped subset and the paired difference
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _rec(i, game, player, stat, slot, ev, date='2026-11-20'):
    return {'bet_id': f'c{i}', 'game_id': game, 'player_id': player, 'stat': stat, 'minutes_before': slot, 'ev_best': ev, 'game_date': date}


def test_capped_subset_equals_the_rules_function_on_a_hand_built_record():
    recs = [_rec(0, 'g1', 1, 'assists', 240, 0.05), _rec(1, 'g1', 1, 'rebounds', 240, 0.08),         # same player, same slot: the higher EV (rebounds) wins
            _rec(2, 'g1', 1, 'threes', 60, 0.09),                                                      # a later slot of a player-game that already has a bet: dropped although its EV is higher
            _rec(3, 'g1', 2, 'threes', 60, 0.04), _rec(4, 'g1', 2, 'assists', 60, 0.04),              # tie in EV: the stat that sorts first (assists) wins
            _rec(5, 'g2', 3, 'ra', 5, 0.06),                                                           # alone: kept
            _rec(6, 'g2', 4, 'assists', 240, 0.05), _rec(7, 'g2', 4, 'rebounds', 60, 0.07),            # the earlier slot blocks the later one even though the later bet has the higher EV
            _rec(8, 'g3', 1, 'assists', 240, 0.05)]                                                    # player 1 in ANOTHER game: independent
    bets = pd.DataFrame(recs)
    ids = mon.capped_bet_ids(bets)
    assert ids == {'c1', 'c4', 'c5', 'c6', 'c8'}
    sel = bets.rename(columns={'ev_best': 'best_ev'})
    assert ids == set(rules.cap_per_player_game_causal(sel, 1)['bet_id'])
    assert mon.capped_bet_ids(bets, 2) == set(rules.cap_per_player_game_causal(sel, 2)['bet_id']) and len(mon.capped_bet_ids(bets, 2)) > len(ids)
    assert mon.CAP_COLUMNS == {'ev_best': 'best_ev'}


def test_capped_subset_is_taken_over_all_recorded_bets_including_unsettled_ones():
    bets = pd.DataFrame([_rec(0, 'g1', 1, 'assists', 240, 0.05), _rec(1, 'g1', 1, 'rebounds', 60, 0.09)])   # c0 never settles, c1 does
    assert mon.capped_bet_ids(bets) == {'c0'}                                                                 # the settled later bet is NOT in the subset


def _paired_frames(n_dates=40, per_date=12, seed=3):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({'game_date': np.repeat(_dates(n_dates), per_date), 'stake': 1.0})
    d['profit_median'] = rng.normal(0.01, 1.0, len(d)) + np.repeat(rng.normal(0, 0.15, n_dates), per_date)
    d['profit_best'] = d['profit_median'] + 0.04
    d['bet_id'] = np.arange(len(d))
    sub = d[d['bet_id'] % 3 != 0]
    return d, sub


def test_paired_difference_uses_the_same_resamples_as_the_uncapped_interval():
    a, b = _paired_frames()
    r = mon.paired_roi_comparison(a, b, n_boot=500, seed=5)
    prim = mon.cluster_bootstrap_roi(a['profit_median'], a['stake'], a['game_date'], n_boot=500, seed=5)
    assert (r['roi_a'], r['lo_a'], r['hi_a']) == pytest.approx((prim['point'], prim['lo'], prim['hi']))        # same resamples as the primary interval
    # by hand: the same multinomial counts applied to both records' per-date sums
    ga = a.groupby('game_date')[['profit_median', 'stake']].sum()
    gb = b.groupby('game_date')[['profit_median', 'stake']].sum().reindex(ga.index)
    counts = mon._boot_counts(len(ga), 500, 5)
    diff = (counts @ gb['profit_median'].to_numpy()) / (counts @ gb['stake'].to_numpy()) - (counts @ ga['profit_median'].to_numpy()) / (counts @ ga['stake'].to_numpy())
    assert r['diff_lo'] == pytest.approx(np.percentile(diff, 2.5)) and r['diff_hi'] == pytest.approx(np.percentile(diff, 97.5))
    assert r['diff'] == pytest.approx(b['profit_median'].sum() / b['stake'].sum() - a['profit_median'].sum() / a['stake'].sum())
    # the subset is positively correlated with the full record, so the paired SE is far below the independent combination
    se_a = mon.cluster_bootstrap_roi(a['profit_median'], a['stake'], a['game_date'], n_boot=50)['se']
    se_b = mon.cluster_bootstrap_roi(b['profit_median'], b['stake'], b['game_date'], n_boot=50)['se']
    assert r['diff_se_boot'] < 0.8 * np.sqrt(se_a ** 2 + se_b ** 2)
    best = mon.paired_roi_comparison(a, b, n_boot=500, seed=5, profit='profit_best')
    assert best['roi_a'] == pytest.approx(r['roi_a'] + 0.04)


def test_a_date_without_a_capped_stake_stays_on_the_axis():
    a, b = _paired_frames(n_dates=10)
    b = b[b['game_date'] != a['game_date'].iloc[0]]
    r = mon.paired_roi_comparison(a, b, n_boot=100, seed=1)
    assert r['n_dates'] == 10 and np.isfinite(r['diff_lo'])


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. segment 1 regression
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('since,name', [('2026-11-13', 'monitor_segment1_since.md'), (None, 'monitor_segment1_full.md')])
def test_segment1_report_is_byte_identical_to_the_pre_change_output(tmp_path, since, name):
    """The golden files were rendered by the code before segment 2 was added (same synthetic store, same arguments)."""
    db = tmp_path / 'live.db'
    regression_store(db)
    text = render_segment1(db, since, '2026-11-22')
    expected = (GOLDEN / name).read_text(encoding='utf-8')
    assert text == expected


def _stream(n_dates, per_date, roi, seed=0, start='2026-11-13', noise=1.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_dates).strftime('%Y-%m-%d')
    cl = np.repeat(dates, per_date)
    return roi + rng.normal(0, noise, len(cl)), np.ones(len(cl)), cl


def test_segment1_sequential_status_fields_are_unchanged():
    """Every field of `sequential_status` on five streams, recorded from the code before segment 2 was added."""
    r = mon.sequential_status(*_stream(40, 30, 0.2, 1), horizon_stakes=1200, n_boot=500)
    assert (r['status'], r['n_stakes'], r['n_dates'], r['cutoff'], r['t']) == ('PASS', 1200, 40, '2026-12-22', 1.0) and r['looks'] == []
    assert r['reason'] == 'horizon reached (1200 stakes through 2026-12-22): lower bound +0.0918 > 0'
    assert (r['roi'], r['lo'], r['hi'], r['p_le0']) == pytest.approx((0.1482049424, 0.0917528681, 0.2130581146, 0.0), abs=1e-9)
    r = mon.sequential_status(*_stream(40, 30, 0.0, 1), horizon_stakes=1200, n_boot=500)
    assert r['status'] == 'FAIL' and r['reason'] == 'horizon reached (1200 stakes through 2026-12-22): lower bound -0.1082 <= 0'
    assert (r['roi'], r['lo'], r['hi'], r['p_le0']) == pytest.approx((-0.0517950576, -0.1082471319, 0.0130581146, 0.946), abs=1e-9)
    r = mon.sequential_status(*_stream(40, 25, -0.15, 4), horizon_stakes=3000, n_boot=300)
    assert r['status'] == 'STOP-FUTILITY' and r['reason'] == 'look at 0.33 of the horizon: conditional power 0.000 < 0.10 (true ROI +0.030 assumed for the rest)'
    assert (r['roi'], r['lo'], r['hi'], r['t'], r['cp']) == pytest.approx((-0.1597929061, -0.2199562596, -0.1061843372, 1 / 3, 1.335e-06), abs=1e-9)
    assert r['looks'][0]['se'] == pytest.approx(0.030757743, abs=1e-9) and r['looks'][0]['cutoff'] == '2026-12-22'
    r = mon.sequential_status(*_stream(40, 25, 0.10, 4), horizon_stakes=3000, n_boot=300)
    assert r['status'] == 'CONTINUE' and r['reason'] == 'horizon not reached, no futility trigger'
    assert (r['roi'], r['lo'], r['hi'], r['p_le0'], r['cp']) == pytest.approx((0.0902070939, 0.0300437404, 0.1438156628, 0.0033333333, 0.8537593896), abs=1e-9)
    r = mon.sequential_status(*_stream(40, 10, 0.25, 3), horizon_stakes=5000, horizon_date='2026-12-22', today='2026-12-23', n_boot=300)
    assert r['status'] == 'PASS' and r['horizon_date'] == '2026-12-22' and r['t'] == pytest.approx(0.08)
    assert (r['roi'], r['lo'], r['hi']) == pytest.approx((0.2839362552, 0.2026092927, 0.3627249619), abs=1e-9)


def test_segment_1_is_the_default_and_the_segment_table_holds_its_numbers():
    import props_forward_monitor as cli
    assert cli.SEGMENTS[1]['horizon_stakes'] == 8800 and cli.SEGMENTS[1]['horizon_date'] == '2027-04-12' and cli.SEGMENTS[1]['looks'] == (1 / 3, 2 / 3)
    assert (cli.HORIZON_STAKES, cli.HORIZON_DATE, cli.ALT_ROI, cli.CP_FLOOR, cli.SECONDARY_MIN_STAKES, cli.N_BOOT) == (8800, '2027-04-12', 0.03, 0.10, 300, 2000)
    s2 = cli.SEGMENTS[2]
    assert s2['look_dates'] == LOOKS and s2['boundaries'] == BOUNDS and s2['horizon_date'] == '2027-04-12' and (s2['n_boot'], s2['seed']) == (4000, 20261112)
    assert s2['registered']['roster_rule'] == 'pregame_all' and s2['registered']['rule_sha256'].startswith('fd026afd')
    with pytest.raises(SystemExit):
        cli.main(['--segment', '1', '--expected-dates', '150'])                                          # segment 2 options are refused for segment 1
    with pytest.raises(SystemExit):
        cli.build_report('unused.db', since=None, segment=2)                                              # segment 2 needs an explicit E2


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 7. configuration integrity (section 11)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _cfg(rows):
    return pd.DataFrame([{'bet_id': f'x{i}', 'game_date': r.get('date', '2026-11-14'), 'minutes_before': 240, 'model_version': r.get('mv', 'v4'), 'model_sha256': r.get('msha', 'm1'),
                          'rule_sha256': r.get('rule', 'fd02'), 'roster_rule': r.get('roster', 'pregame_all')} for i, r in enumerate(rows)])


def test_a_second_model_version_with_the_same_rule_and_roster_does_not_end_segment_2():
    bets = _cfg([{}, {'date': '2026-12-20', 'mv': 'v4r1', 'msha': 'm2'}, {'date': '2027-01-10', 'mv': 'v4r2', 'msha': 'm3'}])
    r = mon.segment_integrity(bets)
    assert not r['ended'] and r['end_date'] is None and r['n_model_rows'] == 3 and len(r['table']) == 3
    assert 'segment continues' in r['reason'] and 'per model version' in r['reason']
    assert not mon.segment_integrity(bets, 'fd02', 'pregame_all')['ended']                                  # with the registered values given
    assert mon.config_segments(bets).shape[0] == 3                                                          # segment 1's own helper still lists every row


def test_a_different_rule_sha_or_roster_rule_ends_segment_2_on_its_first_date():
    rule = mon.segment_integrity(_cfg([{}, {'date': '2026-12-20', 'rule': 'ab99'}, {'date': '2027-01-10', 'rule': 'ab99'}]))
    assert rule['ended'] and rule['end_date'] == '2026-12-20' and 'SEGMENT ENDED' in rule['reason']
    roster = mon.segment_integrity(_cfg([{}, {'date': '2026-12-20', 'mv': 'v4r1', 'msha': 'm2', 'roster': 'v2'}]))
    assert roster['ended'] and roster['end_date'] == '2026-12-20'
    wrong_from_start = mon.segment_integrity(_cfg([{}, {'date': '2026-12-20'}]), expected_rule_sha='zzzz', expected_roster='pregame_all')
    assert wrong_from_start['ended'] and wrong_from_start['end_date'] == '2026-11-14'                        # bets never carried the registered rule


def test_report_says_what_the_monitor_prints_in_each_case(tmp_path, monkeypatch):
    import props_forward_monitor as cli
    monkeypatch.setitem(cli.SEGMENTS[2], 'registered', {**cli.SEGMENTS[2]['registered'], 'rule_sha256': 'r1', 'roster_rule': 'v2'})
    same = tmp_path / 'same.db'
    regression_store(same)                                                                                   # two model versions, one rule sha and roster rule
    before = hashlib.sha256(same.read_bytes()).hexdigest()
    text = render_segment2(same, '2026-11-13', '2026-11-22')
    assert hashlib.sha256(same.read_bytes()).hexdigest() == before                                           # read-only
    assert '2 (model_version, model sha256) rows under one rule sha256 and roster rule: the segment continues' in text
    assert 'SEGMENT ENDED' not in text and '**CONTINUE**' in text and 'By model version' in text
    changed = tmp_path / 'changed.db'
    regression_store(changed, late_rule_sha='r2')
    text2 = render_segment2(changed, '2026-11-13', '2026-11-22')
    assert 'SEGMENT ENDED: from 2026-11-18' in text2 and '**SEGMENT-ENDED**' in text2


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the script, segment 2
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_segment_2_report_latches_a_look_1_pass_and_ignores_later_dates(tmp_path, monkeypatch):
    import props_forward_monitor as cli
    monkeypatch.setitem(cli.SEGMENTS[2], 'registered', {**cli.SEGMENTS[2]['registered'], 'rule_sha256': 'r1', 'roster_rule': 'v2'})
    short, long = tmp_path / 'short.db', tmp_path / 'long.db'
    regression_store(short, n_dates=60, start='2026-11-14', p_win=0.75)
    regression_store(long, n_dates=90, start='2026-11-14', p_win=0.75, p_win_switch=(60, 0.05))             # 30 more dates, every bet losing
    t_short = render_segment2(short, '2026-11-14', '2027-03-01')
    t_long = render_segment2(long, '2026-11-14', '2027-03-01')
    for t in (t_short, t_long):
        assert '| Status | **PASS** |' in t and '| Decided at | look 1 |' in t and 'The confirmatory test ended at look 1' in t
    row = lambda t: next(line for line in t.splitlines() if line.startswith('| 1 | 3.471 |'))
    assert row(t_short) == row(t_long)                                                                       # look 1 was computed on its own 50 dates only
    assert 'capped subset' in t_long and 'diff vs uncapped' in t_long and 'to date (descriptive)' in t_long
    assert 'Source of the expected dates | --expected-dates 150' in t_long


def test_segment_2_report_waits_for_unsettled_dates_and_lists_stale_ones(tmp_path, monkeypatch):
    import props_forward_monitor as cli
    monkeypatch.setitem(cli.SEGMENTS[2], 'registered', {**cli.SEGMENTS[2]['registered'], 'rule_sha256': 'r1', 'roster_rule': 'v2'})
    db = tmp_path / 'w.db'
    regression_store(db, n_dates=60, start='2026-11-14', p_win=0.75, unsettled_dates=('2027-01-02',))        # the 50th date has no settled bet yet
    held = render_segment2(db, '2026-11-14', '2027-01-03')                                                  # unsettled date is yesterday: inside the grace period
    assert '**CONTINUE**' in held and 'not fully settled (2027-01-02)' in held
    stale = render_segment2(db, '2026-11-14', '2027-02-01')                                                 # a month later: stale, excluded and listed, the look is taken
    assert 'Unsettled bets older than 3 days' in stale and '2027-01-02' in stale
    assert '| Status | **PASS** |' in stale
