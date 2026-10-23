import pandas as pd

from props.live import freshness


def test_only_earlier_dates_and_games_that_began_long_ago_must_be_in_the_history():
    sched = pd.DataFrame({'game_id': ['0022600001', '0022600002', '0022600003', '0012600009'], 'et_game_date': ['2026-10-21', '2026-10-22', '2026-10-23', '2026-10-21'],
                          'nba_tip_utc': ['2026-10-22T02:00:00Z', '2026-10-23T02:00:00Z', '2026-10-24T02:00:00Z', '2026-10-22T02:00:00Z']})
    hist = pd.DataFrame({'game_id': ['0022600001']})
    now = pd.Timestamp('2026-10-23T13:00:00Z')
    assert freshness.stale_history(hist, sched, now, '2026-10-23') == ['0022600002']        # the 22nd is missing; the preseason game is ignored; the 23rd is the slot's own date
    assert freshness.stale_history(hist, sched, pd.Timestamp('2026-10-23T05:00:00Z'), '2026-10-23') == []     # it began only 3 hours ago
    assert freshness.stale_history(hist, sched, now, '2026-10-22') == []
