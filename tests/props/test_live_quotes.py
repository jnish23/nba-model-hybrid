import pandas as pd
import pytest

from props.live import quotes


def snaps(*minutes):
    return pd.Series(list(minutes), index=[f's{m}' for m in minutes], dtype=float)


def test_the_slot_waits_before_its_nominal_time():
    assert quotes.choose_snapshot(snaps(250, 220), 240, 245)[0] == 'wait'


def test_a_snapshot_up_to_ten_minutes_after_the_nominal_time_is_preferred_to_an_older_one():
    st, ts, m = quotes.choose_snapshot(snaps(264, 234), 240, 231)
    assert (st, ts, m) == ('use', 's234', 234.0)


def test_the_slot_waits_for_a_fresher_snapshot_then_falls_back_to_the_freshest_older_one():
    s = snaps(264)                                                  # nothing between T and T-10 yet
    assert quotes.choose_snapshot(s, 240, 238)[0] == 'wait'
    st, ts, m = quotes.choose_snapshot(s, 240, 229)                 # the grace period is over
    assert (st, ts) == ('use', 's264') and m == 264.0


def test_a_too_old_snapshot_is_a_missed_slot_and_so_is_a_game_under_way():
    assert quotes.choose_snapshot(snaps(300), 240, 229)[0] == 'missed'
    assert quotes.choose_snapshot(snaps(60, 20), 5, 0.2)[0] == 'missed'


def test_the_last_slot_uses_the_closing_snapshot_without_waiting():
    assert quotes.choose_snapshot(snaps(30, 8), 5, 4.5) == ('use', 's8', 8.0)           # the closing snapshot is older than T-5 but the freshest there is
    assert quotes.choose_snapshot(snaps(30, 8, 3), 5, 2.5) == ('use', 's3', 3.0)


def test_a_snapshot_not_yet_taken_is_never_chosen():
    st, ts, _ = quotes.choose_snapshot(snaps(264, 231), 240, 236)       # the 231 snapshot lies in the future of 'now' (236 minutes before tip)
    assert ts == 's264' or st == 'wait'


def test_a_slot_worked_more_than_25_minutes_late_is_missed():
    assert quotes.choose_snapshot(snaps(250, 235), 240, 212)[0] == 'missed'
    assert quotes.choose_snapshot(snaps(250, 235), 240, 216)[0] in ('use', 'wait')
