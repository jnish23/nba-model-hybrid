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


def test_nba_today_follows_eastern_time_with_a_5am_rollover_whatever_the_server_clock():
    from core.season_utils import nba_today
    utc = dt.timezone.utc
    # 7:44 PM Pacific on Oct 21 = 02:44 UTC on Oct 22: a UTC server's date.today() says Oct 22, but tonight's games are still Oct 21
    assert nba_today(dt.datetime(2026, 10, 22, 2, 44, tzinfo=utc)) == D(2026, 10, 21)
    assert nba_today(dt.datetime(2026, 10, 22, 8, 30, tzinfo=utc)) == D(2026, 10, 21)    # 4:30 AM ET: late games are still last night
    assert nba_today(dt.datetime(2026, 10, 22, 9, 30, tzinfo=utc)) == D(2026, 10, 22)    # 5:30 AM ET: the new day
    assert nba_today(dt.datetime(2026, 10, 21, 16, 15, tzinfo=utc)) == D(2026, 10, 21)   # the 9:15 AM Pacific job
