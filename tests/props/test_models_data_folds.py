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
