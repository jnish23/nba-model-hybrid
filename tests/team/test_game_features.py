import numpy as np
import pandas as pd

from team.game_features_v2 import GameFeatureBuilder


def _games():
    return pd.DataFrame({
        'gameId': ['g1', 'g2'],
        'teamId_home': [1, 3],
        'teamId_away': [2, 4],
        'off_offensiveRating_5g_home': [110.0, 100.0],
        'def_defensiveRating_5g_away': [105.0, 108.0],
        'off_pts_home': [100.0, 90.0],
        'off_pts_away': [95.0, 99.0],
        'is_home_home': [1, 1],          # excluded: not a feature
    })


def _players():
    return pd.DataFrame({
        'gameId': ['g1', 'g1', 'g2', 'g2'],
        'teamId': [1, 2, 3, 4],
        'roster_bpm': [1.0, 0.5, -1.0, 2.0],
        'roster_bpm_ewma': [0.8, 0.2, -0.5, 1.5],
    })


def test_roster_columns_are_home_away_not_x_y():
    out = GameFeatureBuilder.add_roster_features(_games(), _players())
    assert not [c for c in out.columns if c.endswith(('_x', '_y'))]
    assert out['roster_bpm_ewma_home'].tolist() == [0.8, -0.5]
    assert out['roster_bpm_ewma_away'].tolist() == [0.2, 1.5]
    assert out['roster_bpm_home'].tolist() == [1.0, -1.0]


def test_roster_diffs_exist_when_merged_before_diff_step():
    g = GameFeatureBuilder.add_roster_features(_games(), _players())
    out = GameFeatureBuilder.add_derived_features(g)
    np.testing.assert_allclose(out['diff_roster_bpm'], [0.5, -3.0])
    np.testing.assert_allclose(out['diff_roster_bpm_ewma'], [0.6, -2.0])
    assert out['diff_off_pts'].tolist() == [5.0, -9.0]
    np.testing.assert_allclose(out['matchup_off_vs_def_rating_5g'], [5.0, -8.0])
    assert 'diff_is_home' not in out.columns


def test_diff_step_is_idempotent_in_column_names():
    g = GameFeatureBuilder.add_roster_features(_games(), _players())
    out = GameFeatureBuilder.add_derived_features(g)
    assert out.columns.is_unique
