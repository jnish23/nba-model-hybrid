import numpy as np
import pandas as pd
import pytest

from core import devig as dv
from core.timeutil import et_game_date, minutes_to_tip, to_utc_iso


def test_american_to_prob_basics():
    assert dv.american_to_prob(-110) == pytest.approx(110 / 210)
    assert dv.american_to_prob(+150) == pytest.approx(0.4)
    assert dv.american_to_prob(-100) == pytest.approx(0.5) and dv.american_to_prob(100) == pytest.approx(0.5)
    assert np.isnan(dv.american_to_prob(np.nan)) and np.isnan(dv.american_to_prob(0))
    p = np.array([0.2, 0.5, 0.8])
    assert dv.american_to_prob(dv.prob_to_american(p)) == pytest.approx(p)


@pytest.mark.parametrize('method', dv.METHODS)
def test_fair_probabilities_sum_to_one_and_keep_order(method):
    over = np.array([-110, -130, +120, -250, +400, -105])
    under = np.array([-110, -100, -150, +190, -600, -125])
    fo, fu, booksum = dv.devig_prices(over, under, method)
    assert fo + fu == pytest.approx(np.ones(len(over)))
    assert (booksum > 1).all()
    assert (np.sign(np.round(fo - fu, 9)) == np.sign(np.round(dv.american_to_prob(over) - dv.american_to_prob(under), 9))).all()
    if method != 'shin':        # multiplicative and power only ever shrink a probability (Shin can raise a favourite's)
        assert (fo <= dv.american_to_prob(over) + 1e-12).all() and (fu <= dv.american_to_prob(under) + 1e-12).all()


def test_symmetric_market_is_fifty_fifty():
    for m in dv.METHODS:
        fo, fu, _ = dv.devig_prices(-110, -110, m)
        assert fo == pytest.approx(0.5) and fu == pytest.approx(0.5)


def test_power_and_shin_take_less_from_the_favourite_than_multiplicative():
    """Both methods put the margin disproportionately on the long shot, so the favourite's fair probability is lower
    under multiplicative than under power/shin."""
    res = {m: dv.devig_prices(-500, +350, m)[0] for m in dv.METHODS}
    assert res['multiplicative'] < res['power'] and res['multiplicative'] < res['shin']


def test_one_sided_is_nan_and_flagged():
    fo, fu, bs = dv.devig_prices(np.array([-110, np.nan]), np.array([-110, -120]), 'power')
    assert np.isnan(fo[1]) and np.isnan(fu[1]) and not np.isnan(fo[0])
    fl = dv.quality_flags([-110, np.nan, -150], [-110, -120, +160])
    assert list(fl['one_sided']) == [False, True, False]
    assert list(fl['no_margin']) == [False, False, True]       # -150 / +160 is a crossed quote


def test_stale_flag_uses_the_books_own_update_time():
    snap = ['2026-10-21T20:00:00Z'] * 2
    fl = dv.quality_flags([-110, -110], [-110, -110], line_ts=['2026-10-21T19:50:00Z', '2026-10-21T17:00:00Z'],
                          snapshot_ts=snap, stale_after_minutes=60)
    assert list(fl['stale']) == [False, True]


def test_consensus_skips_nans():
    c = dv.consensus_probability([0.5, 0.6, np.nan, 0.4], ['a', 'a', 'a', 'b'])
    assert c.tolist()[:3] == pytest.approx([0.55] * 3) and c[3] == pytest.approx(0.4)


def test_et_game_date_uses_the_eastern_calendar_date():
    assert et_game_date('2026-10-21T01:00:00Z') == '2026-10-20'      # 9 PM ET on the 20th (EDT)
    assert et_game_date('2026-12-01T01:30:00Z') == '2026-11-30'      # EST
    assert et_game_date('2026-10-20T23:30:00Z') == '2026-10-20'
    assert to_utc_iso('2026-10-20T19:30:00-04:00') == '2026-10-20T23:30:00Z'
    assert minutes_to_tip('2026-10-20T22:00:00Z', '2026-10-20T23:30:00Z') == 90
