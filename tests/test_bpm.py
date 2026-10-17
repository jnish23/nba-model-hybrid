import sqlite3

import numpy as np
import pandas as pd
import pytest

from models.bpm_builder import BPM_HISTORY_START, BPMBuilder
from models.player_features import PlayerFeatureBuilder

SEASONS = ['2013-14', '2014-15', '2015-16']
STARTS = {'2013-14': '2013-10-29', '2014-15': '2014-10-28', '2015-16': '2015-10-27'}


@pytest.fixture()
def db(tmp_path):
    rng = np.random.default_rng(1)
    rows, gid = [], 0
    for season in SEASONS:
        for g in range(30):
            date = (pd.Timestamp(STARTS[season]) + pd.Timedelta(days=2 * g)).strftime('%Y-%m-%dT00:00:00')
            gid += 1
            for team in (1, 2):
                for k in range(8):
                    pid = team * 100 + k
                    mins = float(rng.integers(8, 36))
                    rows.append({'gameId': f'{gid:010d}', 'teamId': team, 'personId': pid, 'playerName': f'P{pid}',
                                 'gameDate': date, 'seasonYear': season, 'minutes': mins,
                                 'fieldGoalsMade': float(rng.integers(0, 10)), 'fieldGoalsAttempted': float(rng.integers(3, 18)),
                                 'threePointersMade': float(rng.integers(0, 4)), 'threePointersAttempted': float(rng.integers(0, 8)),
                                 'freeThrowsMade': float(rng.integers(0, 6)), 'freeThrowsAttempted': float(rng.integers(0, 8)),
                                 'offensiveRebounds': float(rng.integers(0, 4)), 'defensiveRebounds': float(rng.integers(0, 8)),
                                 'assists': float(rng.integers(0, 9)), 'steals': float(rng.integers(0, 3)),
                                 'blocks': float(rng.integers(0, 3)), 'turnovers': float(rng.integers(0, 5)),
                                 'personalFouls': float(rng.integers(0, 5)), 'points': float(rng.integers(0, 30))})
    path = tmp_path / 'p.db'
    with sqlite3.connect(path) as conn:
        pd.DataFrame(rows).to_sql('player_basic_stats', conn, index=False)
        pd.DataFrame({'PERSON_ID': sorted({r['personId'] for r in rows}),
                      'POSITION': ['Guard', 'Forward', 'Center', 'Guard-Forward'] * 4}).to_sql('player_info', conn, index=False)
    return str(path)


def _bpm(db, seasons):
    b = BPMBuilder(db)
    b.load_data(seasons)
    return b.calculate_rolling_bpm()


def test_bpm_does_not_depend_on_which_season_the_load_starts_at(db):
    full = _bpm(db, SEASONS)
    last_only = _bpm(db, ['2015-16'])
    both = full.merge(last_only, on=['gameId', 'personId'], suffixes=('_full', '_one'))
    assert len(both) == len(last_only) > 0
    np.testing.assert_allclose(both['bpm_full'], both['bpm_one'])
    np.testing.assert_allclose(both['e_minutes_full'], both['e_minutes_one'])
    assert BPM_HISTORY_START == '2013-14'


def test_roster_features_do_not_depend_on_the_seasons_requested(db):
    a = PlayerFeatureBuilder(db).build_roster_features(SEASONS)
    b = PlayerFeatureBuilder(db).build_roster_features(['2015-16'])
    m = a.merge(b, on=['gameId', 'teamId'], suffixes=('_a', '_b'))
    assert len(b) == len(m) == 60
    np.testing.assert_allclose(m['roster_bpm_a'], m['roster_bpm_b'])
    np.testing.assert_allclose(m['roster_bpm_ewma_a'], m['roster_bpm_ewma_b'])


def test_cutoff_date_changes_nothing_about_earlier_rows(db):
    full = _bpm(db, SEASONS)
    cut = BPMBuilder(db)
    cut.load_data(SEASONS, before='2015-12-01')
    early = cut.calculate_rolling_bpm()
    m = full.merge(early, on=['gameId', 'personId'], suffixes=('_f', '_c'))
    assert len(m) == len(early)
    np.testing.assert_allclose(m['bpm_f'], m['bpm_c'])


def test_player_with_a_prior_season_is_fully_reliable_from_the_first_game(db):
    """Cumulative minutes carry over from earlier seasons (the 2019-20-start variant reset them)."""
    b = BPMBuilder(db)
    b.load_data(['2015-16'])
    b.calculate_rolling_bpm()
    first = b.df[b.df['seasonYear'] == '2015-16'].sort_values('gameDate').groupby('personId').head(1)
    assert (first['reliability'] == 1.0).all()
