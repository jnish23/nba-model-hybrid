import sqlite3

import numpy as np
import pandas as pd
import pytest

from core.travel_builder import TravelFeatureBuilder

ARENAS = [  # tricode, teamId, lat, long, altitude, tz_offset
    ('AAA', 1, 40.0, -74.0, 10, -5),
    ('BBB', 2, 41.0, -87.0, 600, -6),
    ('CCC', 3, 34.0, -118.0, 300, -8),
    ('DDD', 4, 39.7, -105.0, 5280, -7),
]
# Team 1 (AAA): home, away at BBB, away at CCC, away at DDD (the "live" game), then home
SCHEDULE = [
    # date, home tricode, away tricode (game ids are 00<index>)
    ('2025-10-30', 'DDD', 'CCC'),   # gives DDD a history so it exists in team_basic_stats
    ('2025-11-01', 'AAA', 'BBB'),
    ('2025-11-03', 'BBB', 'AAA'),
    ('2025-11-05', 'CCC', 'AAA'),
    ('2025-11-08', 'DDD', 'AAA'),
    ('2025-11-10', 'AAA', 'CCC'),
]


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / 'mini.db'
    names = {t: f'Team {t}' for t, *_ in ARENAS}
    ids = {t: i for t, i, *_ in ARENAS}
    rows = []
    for n, (d, h, a) in enumerate(SCHEDULE):
        rows.append((ids[h], names[h], h, f'00{n}', d, f'{h} vs. {a}'))
        rows.append((ids[a], names[a], a, f'00{n}', d, f'{a} @ {h}'))
    with sqlite3.connect(path) as conn:
        pd.DataFrame(rows, columns=['teamId', 'teamName', 'teamTricode', 'gameId', 'gameDate', 'MATCHUP']) \
            .to_sql('team_basic_stats', conn, index=False)
        pd.DataFrame(ARENAS, columns=['teamTricode', 'teamId', 'lat', 'long', 'altitude', 'tz_offset']) \
            .to_sql('nba_arenas', conn, index=False)
    return str(path)


def _hist(db):
    b = TravelFeatureBuilder(db)
    return b.build_features().set_index(['teamId', 'gameId'])


def test_road_trip_and_body_clock_are_real_values_in_batch(db):
    f = _hist(db)
    assert f.loc[(1, '004'), 'road_trip_count'] == 3      # 3rd straight road game
    assert f.loc[(1, '004'), 'body_clock_lag'] != f.loc[(1, '004'), 'tz_delta']
    assert f.loc[(1, '005'), 'road_trip_count'] == 0


def test_live_travel_equals_batch_for_the_same_game(db):
    """Game 004 is 'upcoming' for the live path: it must equal the batch row for that game."""
    batch = _hist(db)
    # live run on 2025-11-08 sees only the first three games in the DB
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM team_basic_stats WHERE gameDate >= '2025-11-08'")
    live = TravelFeatureBuilder(db).get_travel_features(pd.DataFrame(
        {'gameDate': ['2025-11-08'], 'home_team': ['Team DDD'], 'away_team': ['Team AAA']}))
    assert not live.empty
    aaa = live[live['team'] == 'Team AAA'].iloc[0]
    for col in ['dist_miles', 'tz_delta', 'body_clock_lag', 'road_trip_count', 'game_altitude_diff']:
        np.testing.assert_allclose(aaa[col], batch.loc[(1, '004'), col], err_msg=col)
    assert aaa['road_trip_count'] == 3
    ddd = live[live['team'] == 'Team DDD'].iloc[0]
    np.testing.assert_allclose(ddd['road_trip_count'], batch.loc[(4, '004'), 'road_trip_count'])


def test_live_run_ignores_games_already_in_db_on_or_after_the_target_date(db):
    """If the DB already holds game 004 (e.g. a re-run after tip-off) the features must not change."""
    batch = _hist(db)
    live = TravelFeatureBuilder(db).get_travel_features(pd.DataFrame(
        {'gameDate': ['2025-11-08'], 'home_team': ['Team DDD'], 'away_team': ['Team AAA']}))
    aaa = live[live['team'] == 'Team AAA'].iloc[0]
    for col in ['dist_miles', 'body_clock_lag', 'road_trip_count']:
        np.testing.assert_allclose(aaa[col], batch.loc[(1, '004'), col], err_msg=col)
