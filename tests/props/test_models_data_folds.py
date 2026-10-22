import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from props.features import build
from props.models import data, folds

DATASET = ROOT / 'data' / 'props_features.db'


def synthetic_frame():
    rows = []
    for season, start in (('2019-20', '2019-10-22'), ('2020-21', '2020-12-22'), ('2021-22', '2021-10-19'), ('2022-23', '2022-10-18')):
        for k in range(60):
            d = (pd.Timestamp(start) + pd.Timedelta(days=2 * k)).strftime('%Y-%m-%d')
            rows.append({'season': season, 'game_date': d, 'player_id': k})
    # bubble games of 2019-20
    rows += [{'season': '2019-20', 'game_date': '2020-08-05', 'player_id': 999}]
    return pd.DataFrame(rows)


def test_fold_is_walk_forward_and_excludes_bubble_from_training_only():
    f = synthetic_frame()
    fold = folds.make_fold(f, '2021-22')
    assert fold.describe()['test_rows'] == 60
    assert set(f.loc[fold.fit_idx, 'season']) | set(f.loc[fold.calib_idx, 'season']) <= {'2019-20', '2020-21'}
    assert 999 not in set(f.loc[np.r_[fold.fit_idx, fold.calib_idx], 'player_id'])                # the bubble game is not trained on
    d = f['game_date']
    assert d[fold.fit_idx].max() < d[fold.calib_idx].min() and d[fold.calib_idx].max() < d[fold.test_idx].min()
    assert len(fold.calib_idx) == pytest.approx(0.2 * (len(fold.fit_idx) + len(fold.calib_idx)), abs=4)


def test_integrity_check_catches_leaky_folds():
    f = synthetic_frame()
    fold = folds.make_fold(f, '2022-23')
    leaky = folds.Fold(fold.test_season, np.r_[fold.fit_idx, fold.test_idx[:1]], fold.calib_idx, fold.test_idx, fold.cutoff_date)
    with pytest.raises(AssertionError):
        folds.assert_fold_integrity(f, leaky)
    with pytest.raises(ValueError):
        folds.make_fold(f, '2019-20')                                                                # no training data before the first season


def test_feature_spec_groups_and_outcome_guard():
    cols = ['game_id', 'player_id', 'minutes_before', 'm_min_h15', 'r_points_h15', 'tm_off_rtg', 'opp_pts', 'inj_p_play', 'inj_status',
            'av_miss_minutes', 'env_team_spread', 'y_points', 'lbl_minutes', 'pos_group', 'season']
    base = data.FeatureSpec('base').columns(cols)
    assert 'env_team_spread' not in base and 'm_min_h15' in base and 'inj_p_play' in base and 'minutes_before' in base
    assert not {'y_points', 'lbl_minutes', 'game_id', 'inj_status', 'season'} & set(base)
    assert 'env_team_spread' in data.FeatureSpec('with_env').columns(cols)
    no_tm = data.FeatureSpec('base', exclude_groups=frozenset({'teammates'})).columns(cols)
    assert 'av_miss_minutes' not in no_tm and 'inj_p_play' in no_tm
    assert 'minutes_before' not in data.FeatureSpec('base', include_decision_time=False).columns(cols)
    with pytest.raises(ValueError):
        data.FeatureSpec().check(['m_min_h15', 'y_points'])


@pytest.mark.db
@pytest.mark.skipif(not DATASET.exists(), reason='dataset not built')
def test_real_frame_folds_and_feature_spec():
    df = data.load_frame(DATASET)
    assert len(df) == 698346 and df['weight'].groupby([df['game_id'], df['player_id']]).sum().round(6).eq(1.0).all()
    for s in folds.DEV_TEST_SEASONS + (folds.VALIDATION_SEASON,):
        fold = folds.make_fold(df, s)
        assert df.loc[fold.test_idx, 'season'].nunique() == 1
    cols = data.FeatureSpec('base').columns(df.columns)
    data.FeatureSpec().check(cols)
    assert not [c for c in cols if c.startswith('env_')] and len(cols) > 150
    assert str(df['pos_group'].dtype) == 'category'


def test_exclude_features_drops_only_the_named_columns():
    cols = ['m_min_h15', 'ctx_team_game_number', 'rest_games_played_season', 'r_points_h15', 'minutes_before']
    full = data.FeatureSpec('base').columns(cols)
    r1 = data.FeatureSpec('base', exclude_features=frozenset({'ctx_team_game_number', 'rest_games_played_season'})).columns(cols)
    assert set(full) - set(r1) == {'ctx_team_game_number', 'rest_games_played_season'} and set(r1) <= set(full)
    assert 'minutes_before' in r1                                              # the decision time is not a "feature" that R1 touches


def test_live_roster_places_a_traded_player_on_one_team_only():
    import pandas as pd
    from props.live import roster
    hist = pd.DataFrame({'game_id': ['g1', 'g2', 'g3', 'g4'], 'player_id': [7, 7, 7, 8], 'team_id': [1, 1, 2, 1],
                         'game_date': ['2026-01-01', '2026-01-03', '2026-01-05', '2026-01-03']})
    rosters = roster.recent_rosters(hist, '2026-01-10')
    assert 7 in rosters[1] and 7 in rosters[2]                                           # the raw rule puts him on both teams
    one = roster.current_team_only(rosters, roster.last_team(hist, '2026-01-10'))
    assert 7 not in one[1] and 7 in one[2] and 8 in one[1]


def test_live_team_resolution_never_uses_last_seasons_team_or_guesses():
    import pandas as pd
    from props.live import roster
    hist = pd.DataFrame({'game_id': ['a', 'b', 'c'], 'player_id': [5, 6, 6], 'team_id': [1, 1, 3],
                         'game_date': ['2025-04-01', '2025-11-01', '2025-11-02'], 'season': ['2024-25', '2025-26', '2025-26']})
    games = pd.DataFrame({'game_id': ['g'], 'home_team_id': [1], 'away_team_id': [2], 'season': ['2025-26'], 'game_date': ['2025-11-10']})
    extra = pd.DataFrame({'game_id': ['g'] * 4, 'player_id': [5, 6, 7, 8], 'team_id': [2, None, None, None]})
    placed, unresolved = roster.resolve_extra_players(extra, games, hist, '2025-11-10')
    assert 5 in placed['g'][2]                                    # the injury report's team wins over last season's team
    assert sorted(p for _, p in unresolved) == [6, 7, 8]          # 6: this season's team is not in the game; 7 and 8 have no listing at all


def test_live_placeholder_rows_put_a_player_on_one_team_per_game():
    import pandas as pd
    from props.live import roster
    games = pd.DataFrame({'game_id': ['g'], 'game_date': ['2025-11-10'], 'season': ['2025-26'], 'home_team_id': [1], 'away_team_id': [2]})
    rows = roster.placeholder_rows(['game_id', 'player_id', 'team_id', 'opp_team_id', 'game_date', 'season', 'game_type', 'is_home', 'lbl_played', 'lbl_started'],
                                   games, {1: {10, 11}, 2: {20}}, {'g': {2: {11}}})
    assert not rows.duplicated(['game_id', 'player_id']).any()
    assert rows.loc[rows['player_id'] == 11, 'team_id'].tolist() == [2]
