"""Matchup features (GameFeatureBuilder.MATCHUP_PAIRS): every pair must be built, and none may touch the deployed model.

Before 2026-11 two of the three pairs named `def_defensiveRating_*_away`, a column that never exists (defensiveRating is built
under the `off_` prefix), so only matchup_3pa_vs_opp_3pa_5g was ever produced and the other two were skipped at debug level.
"""
import json
import logging

import numpy as np
import pandas as pd
import pytest

from team.game_features_v2 import GameFeatureBuilder
from team.team_features_v2 import TeamFeatureBuilder


def _game_frame_from_team_features(team_db):
    """One game (team 1 home, team 2 away) with the real team-form columns, named *_home / *_away as in the training set."""
    feats = TeamFeatureBuilder(team_db).get_latest_team_features('2026-01-15')
    home, away = pd.DataFrame([feats[1]]).add_suffix('_home'), pd.DataFrame([feats[2]]).add_suffix('_away')
    return pd.concat([pd.DataFrame({'gameId': ['g1'], 'teamId_home': [1], 'teamId_away': [2]}), home, away], axis=1)


def test_every_matchup_pair_is_produced_on_a_frame_with_the_team_features(team_db):
    frame = _game_frame_from_team_features(team_db)
    out = GameFeatureBuilder.add_derived_features(frame)
    for home_col, away_col, name in GameFeatureBuilder.MATCHUP_PAIRS:
        assert home_col in frame.columns and away_col in frame.columns, f'{name}: inputs {home_col} / {away_col} are not built'
        assert name in out.columns, name
        np.testing.assert_allclose(out[name], frame[home_col] - frame[away_col])
    assert out.columns.is_unique


def test_the_rating_pairs_read_defensive_rating_from_the_off_prefix():
    named = {n: (h, a) for h, a, n in GameFeatureBuilder.MATCHUP_PAIRS}
    assert named['matchup_off_vs_def_rating_5g'] == ('off_offensiveRating_5g_home', 'off_defensiveRating_5g_away')
    assert named['matchup_off_vs_def_rating_20g'] == ('off_offensiveRating_20g_home', 'off_defensiveRating_20g_away')


def test_a_missing_matchup_input_is_a_warning_not_a_debug_line(caplog):
    frame = pd.DataFrame({'gameId': ['g1'], 'off_offensiveRating_5g_home': [110.0]})     # no away columns at all
    with caplog.at_level(logging.DEBUG):
        out = GameFeatureBuilder.add_derived_features(frame)
    assert 'matchup_off_vs_def_rating_5g' not in out.columns
    warned = [r for r in caplog.records if r.levelno == logging.WARNING and 'matchup_off_vs_def_rating_5g' in r.message]
    assert warned and 'off_defensiveRating_5g_away' in warned[0].message


def test_the_deployed_model_set_uses_no_matchup_column():
    """Live predictions cannot change when matchup columns appear: the deployed feature lists contain none of them."""
    from team.model_registry import DEPLOYED_SUFFIX, MODEL_DIR
    path = MODEL_DIR / f'feature_config{DEPLOYED_SUFFIX}.json'
    if not path.exists():
        pytest.skip(f'{path.name} not available')
    cfg = json.loads(path.read_text())
    built = {name for _, _, name in GameFeatureBuilder.MATCHUP_PAIRS}
    used = {f for feats in cfg.values() for f in feats}
    assert not [f for f in used if f.startswith('matchup_')], 'the deployed model reads a matchup_* column'
    assert not used & built
