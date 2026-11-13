"""
WP2: a missing stat on a PLAYED row is excluded from that stat's rate (numerator and exposure), and nothing else changes.
Synthetic arrays only.
"""
import numpy as np
import pytest
from scipy.signal import lfilter

from props.features import decay


def old_decayed_before(values, groups, played, alpha):
    """Reference: the implementation of props/features/decay.py before WP2, copied verbatim."""
    values = np.asarray(values, dtype=float)
    played = np.asarray(played, dtype=bool)
    groups = np.asarray(groups)
    n = len(values)
    out = np.zeros(n)
    change = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1], True])
    for lo, hi in zip(change[:-1], change[1:]):
        v = np.where(played[lo:hi], np.nan_to_num(values[lo:hi]), 0.0)
        p = played[lo:hi]
        idx = np.flatnonzero(p)
        state_after = np.zeros(hi - lo)
        if len(idx):
            filt = lfilter([1.0], [1.0, -alpha], v[idx])
            state_after[idx] = filt
            last = np.full(hi - lo, -1)
            last[idx] = idx
            last = np.maximum.accumulate(last)
            prev = np.r_[-1, last[:-1]]
            before = np.where(prev >= 0, state_after[np.maximum(prev, 0)], 0.0)
            out[lo:hi] = before
    return out


def synthetic(rng, n=600, n_groups=25, p_played=0.8):
    groups = np.sort(rng.integers(0, n_groups, n))
    played = rng.random(n) < p_played
    minutes = np.where(played, rng.uniform(4, 42, n), np.nan)          # as in the real frame: NaN when not played
    stat = np.where(played, rng.poisson(8, n).astype(float), np.nan)   # NaN on DNP rows, finite on played rows
    return groups, played, minutes, stat


HALF_LIVES = (1, 5, 15, 40, 200)


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('hl', HALF_LIVES)
def test_no_nan_on_played_rows_is_identical_to_the_old_code(seed, hl):
    rng = np.random.default_rng(seed)
    groups, played, minutes, stat = synthetic(rng)
    a = decay.alpha_for(hl)
    old_num = old_decayed_before(stat, groups, played, a)
    old_den = old_decayed_before(minutes, groups, played, a)
    # decayed_before itself is unchanged
    assert np.array_equal(decay.decayed_before(stat, groups, played, a), old_num)
    new_num, new_den = decay.decayed_rate_parts_before(stat, minutes, groups, played, a)
    assert np.max(np.abs(new_num - old_num)) == 0.0
    assert np.max(np.abs(new_den - old_den)) == 0.0
    # the rate itself, as history.py forms it
    k, prior = 200.0, 0.4
    assert np.array_equal((new_num + k * prior) / (new_den + k), (old_num + k * prior) / (old_den + k))


def test_regression_covers_dnp_rows_multiple_groups_and_singleton_groups():
    rng = np.random.default_rng(99)
    groups = np.array([0, 1, 1, 2, 2, 2, 3, 4, 4])                  # a group of one, groups with all-DNP rows
    played = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1], dtype=bool)
    minutes = np.where(played, rng.uniform(5, 40, len(groups)), np.nan)
    stat = np.where(played, rng.uniform(0, 10, len(groups)), np.nan)
    a = decay.alpha_for(5)
    num, den = decay.decayed_rate_parts_before(stat, minutes, groups, played, a)
    assert np.array_equal(num, old_decayed_before(stat, groups, played, a))
    assert np.array_equal(den, old_decayed_before(minutes, groups, played, a))


@pytest.mark.parametrize('hl', (5, 15, 40))
def test_a_stat_on_a_played_row_never_leaks_forward(hl):
    """Strictly-before: the output at row i does not change when ANY value (or the played/NaN status) at row i or later rows of the same group changes."""
    rng = np.random.default_rng(3)
    groups, played, minutes, stat = synthetic(rng, n=400, n_groups=6)
    stat = stat.copy()
    stat[np.flatnonzero(played)[rng.random(played.sum()) < 0.08]] = np.nan         # NaN on some played rows too
    a = decay.alpha_for(hl)
    base_num, base_den = decay.decayed_rate_parts_before(stat, minutes, groups, played, a)
    for i in rng.integers(0, len(groups), 25):
        later = (groups == groups[i]) & (np.arange(len(groups)) >= i)
        s2, m2, p2 = stat.copy(), minutes.copy(), played.copy()
        s2[later] = rng.normal(1000, 100, later.sum())                              # alter the row and everything after it for that player
        m2[later] = rng.uniform(1, 48, later.sum())
        p2[later] = rng.random(later.sum()) < 0.5
        s2[later & (rng.random(len(groups)) < 0.3)] = np.nan                        # including NaNs appearing there
        num, den = decay.decayed_rate_parts_before(s2, m2, groups, p2, a)
        upto = np.arange(len(groups)) <= i                                         # row i itself and all earlier rows (any group)
        assert np.array_equal(num[upto], base_num[upto])
        assert np.array_equal(den[upto], base_den[upto])
        other = groups != groups[i]
        assert np.array_equal(num[other], base_num[other])                          # other players are untouched entirely


@pytest.mark.parametrize('hl', (5, 15, 40))
def test_a_played_row_with_a_nan_stat_equals_the_game_removed_for_that_stat(hl):
    rng = np.random.default_rng(11)
    groups, played, minutes, stat = synthetic(rng, n=500, n_groups=8)
    other = np.where(played, rng.poisson(3, len(groups)).astype(float), np.nan)    # a second stat of the same games
    a = decay.alpha_for(hl)
    played_idx = np.flatnonzero(played)
    for j in rng.choice(played_idx, 12, replace=False):
        # stat A is missing for game j; stat B is fine
        stat_nan = stat.copy()
        stat_nan[j] = np.nan
        num, den = decay.decayed_rate_parts_before(stat_nan, minutes, groups, played, a)
        # reference: the same data with game j deleted outright (for stat A)
        keep = np.arange(len(groups)) != j
        r_num, r_den = decay.decayed_rate_parts_before(stat[keep], minutes[keep], groups[keep], played[keep], a)
        assert num[keep] == pytest.approx(r_num, abs=1e-12)
        assert den[keep] == pytest.approx(r_den, abs=1e-12)
        # the other stat of the same game is unaffected: identical to the series without any NaN
        o_num, o_den = decay.decayed_rate_parts_before(other, minutes, groups, played, a)
        assert np.array_equal(o_num, old_decayed_before(other, groups, played, a))
        assert np.array_equal(o_den, old_decayed_before(minutes, groups, played, a))


def test_the_old_code_was_biased_low_and_the_new_code_is_not():
    """One player, four games, tracking stat missing in the third: the old rate counts its minutes with a zero."""
    g = np.zeros(5, int)
    played = np.ones(5, dtype=bool)
    minutes = np.array([30.0, 30.0, 30.0, 30.0, 30.0])
    touches = np.array([60.0, 60.0, np.nan, 60.0, 60.0])                           # 2 per minute whenever it exists
    a = 1.0 - 1e-9                                                                 # (almost) no decay
    old_rate = old_decayed_before(touches, g, played, a)[4] / old_decayed_before(minutes, g, played, a)[4]
    num, den = decay.decayed_rate_parts_before(touches, minutes, g, played, a)
    assert old_rate == pytest.approx(1.5)
    assert num[4] / den[4] == pytest.approx(2.0)


def test_all_games_missing_gives_zero_exposure_not_a_crash():
    g = np.zeros(3, int)
    num, den = decay.decayed_rate_parts_before(np.full(3, np.nan), np.array([30.0, 31.0, 32.0]), g, np.ones(3, bool), 0.9)
    assert num.tolist() == [0.0, 0.0, 0.0] and den.tolist() == [0.0, 0.0, 0.0]    # the prior carries the rate (s_stat + k*prior) / (0 + k)


def test_valid_rows():
    ok = decay.valid_rows([True, True, False, True], np.array([1.0, np.nan, 2.0, 3.0]), np.array([1.0, 1.0, 1.0, np.nan]))
    assert ok.tolist() == [True, False, False, False]
