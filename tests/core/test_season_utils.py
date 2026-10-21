import datetime as dt

from core.season_utils import season_date_range, season_for_date, season_options

D = dt.date


def test_season_for_date_rolls_over_in_october():
    assert season_for_date(D(2026, 10, 20)) == '2026-27'
    assert season_for_date(D(2026, 6, 13)) == '2025-26'
    assert season_for_date(D(2026, 9, 30)) == '2025-26'


def test_current_season_is_offered_even_before_it_has_games_and_listed_first():
    dates = [D(2025, 10, 21), D(2026, 6, 13)]
    assert season_options(dates, today=D(2026, 10, 19)) == ['2026-27', '2025-26']
    assert season_options(dates + [D(2026, 10, 20)], today=D(2026, 10, 20)) == ['2026-27', '2025-26']


def test_current_season_starts_on_opening_night_and_ends_today():
    dates = [D(2025, 10, 21), D(2026, 6, 13), D(2026, 10, 20), D(2026, 10, 21)]       # tomorrow's games are in the data too
    start, end, lo, hi = season_date_range(dates, '2026-27', today=D(2026, 10, 20))
    assert start == D(2026, 10, 20) and end == D(2026, 10, 21) and (lo, hi) == (D(2026, 10, 20), D(2026, 10, 21))
    start, end, lo, hi = season_date_range(dates, '2026-27', today=D(2026, 11, 3))
    assert (start, end) == (D(2026, 10, 20), D(2026, 11, 3)) and hi == D(2026, 11, 3)       # the current date, even with no games yet


def test_past_season_spans_its_own_games():
    dates = [D(2025, 10, 21), D(2026, 1, 5), D(2026, 6, 13), D(2026, 10, 20)]
    assert season_date_range(dates, '2025-26', today=D(2026, 10, 20)) == (D(2025, 10, 21), D(2026, 6, 13), D(2025, 10, 21), D(2026, 6, 13))


def test_season_without_any_data_still_gives_a_valid_range():
    assert season_date_range([], '2026-27', today=D(2026, 10, 19)) == (D(2026, 10, 20), D(2026, 10, 20), D(2026, 10, 20), D(2026, 10, 20))
    start, end, lo, hi = season_date_range([], '2024-25', today=D(2026, 10, 19))
    assert lo <= start <= end <= hi


def test_date_bounds_span_every_season_so_a_range_can_cross_seasons():
    from core.season_utils import date_bounds
    dates = [D(2025, 10, 21), D(2026, 6, 13), D(2026, 10, 20)]
    assert date_bounds(dates, today=D(2026, 10, 25)) == (D(2025, 10, 21), D(2026, 10, 25))
    assert date_bounds(dates, today=D(2026, 10, 20)) == (D(2025, 10, 21), D(2026, 10, 20))
    assert date_bounds([], today=D(2026, 10, 20)) == (D(2026, 10, 20), D(2026, 10, 20))
