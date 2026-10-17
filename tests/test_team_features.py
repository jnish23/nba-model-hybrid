import numpy as np
import pandas as pd
import pytest

from models import team_features_v2 as tf
from models.team_features_v2 import TeamFeatureBuilder


def _batch(team_db):
    b = TeamFeatureBuilder(team_db)
    return b, b.build_entire_history()


def test_momentum_features_exist_in_batch_build(team_db):
    """Regression: WL/MATCHUP are upper-case in team_game_full, so the lower-case checks used to skip them."""
    _, full = _batch(team_db)
    for col in ('win_pct_last10', 'win_streak', 'home_win_pct_20g'):
        assert col in full.columns
        assert full[col].notna().any()


def test_batch_and_live_features_agree_for_every_game(team_db):
    """For each historical game, build_features() on the history before that date == the batch row."""
    builder, full = _batch(team_db)
    checked = 0
    for _, row in full.iterrows():
        live = builder.build_features(int(row['teamId']), row['gameDate'].strftime('%Y-%m-%d'),
                                      row['seasonYear'], game_id=row['gameId'])
        if not live:                      # first game of the log: no history
            continue
        for k, v in live.items():
            assert k in full.columns, f"live-only feature {k}"
            b = row[k]
            if pd.isna(v) and pd.isna(b):
                continue
            np.testing.assert_allclose(float(v), float(b), rtol=1e-9, atol=1e-9,
                                       err_msg=f"{k} team {row['teamId']} {row['gameDate']}")
        checked += 1
    assert checked > 40


def test_live_feature_set_covers_batch_feature_set(team_db):
    builder, full = _batch(team_db)
    r = full.iloc[-1]
    live = builder.build_features(int(r['teamId']), r['gameDate'].strftime('%Y-%m-%d'), r['seasonYear'], r['gameId'])
    meta = {'teamId', 'gameDate', 'seasonYear', 'gameId'}
    missing = set(full.columns) - meta - set(live)
    assert not missing, f"batch features missing from the live path: {sorted(missing)[:5]}"


def test_days_rest_is_not_capped_after_offseason(team_db):
    builder, full = _batch(team_db)
    first_of_new_season = full[(full['seasonYear'] == '2025-26')].groupby('teamId').head(1).iloc[0]
    assert first_of_new_season['days_rest'] > 7
    live = builder.build_features(int(first_of_new_season['teamId']),
                                  first_of_new_season['gameDate'].strftime('%Y-%m-%d'), '2025-26')
    assert live['days_rest'] == first_of_new_season['days_rest']


def test_get_latest_team_features_uses_one_history_load_and_matches_per_team(team_db):
    builder = TeamFeatureBuilder(team_db)
    date = '2025-12-20'
    latest = builder.get_latest_team_features(date)
    for tid in (1, 2):
        single = builder.build_features(tid, date, '2025-26')
        assert set(latest[tid]) == set(single)
        for k, v in single.items():
            np.testing.assert_allclose(latest[tid][k], v, equal_nan=True)


def test_momentum_helpers_known_values():
    win = pd.Series([1, 1, 0, 0, 0, 1.0])
    assert tf.win_streak(win).tolist() == [0, 1, 2, -1, -2, -3]
    pct = tf.win_pct_last10(win)
    assert np.isnan(pct.iloc[2]) and pct.iloc[3] == pytest.approx(2 / 3)
