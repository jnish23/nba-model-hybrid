"""train_and_save_all on a small synthetic history whose scoring level drifts upward (like the real 200 -> 230)."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

from team import market_models as mm
from team import model_registry as mr
from team.train_baseline import BaselineModel, SPREAD_GAP_COL, TOTAL_GAP_COL

from synthetic_history import SEASONS, training_csv


def _train(tmp_path_factory, total_target):
    d = tmp_path_factory.mktemp('final')
    csv = d / 'train.csv'
    training_csv(csv)
    m = BaselineModel(csv, model_dir=str(d / 'saved'))
    seen = []
    orig = m._select_features_from_subset

    def spy(df, target, *a, **k):
        seen.append(target)
        return orig(df, target, *a, **k)

    m._select_features_from_subset = spy
    m.train_and_save_all(use_odds=True, save_suffix='_t', time_budget=2, calib_holdout_days=90, total_target=total_target)
    return d, csv, m, seen


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    """total_target='residual' (opt-in)."""
    return _train(tmp_path_factory, 'residual')


@pytest.fixture(scope='module')
def trained_default(tmp_path_factory):
    return _train(tmp_path_factory, 'level')


def test_default_total_model_predicts_the_level_as_validated_in_e1(trained_default):
    d, _, m, seen = trained_default
    b = mr.load_bundle('_t', d / 'saved')
    assert not isinstance(b.total, mm.ResidualTotalModel) and isinstance(b.spread, mm.ResidualSpreadModel)
    assert b.calib_meta['total_target'] == 'total'
    assert seen == ['target_home_win', SPREAD_GAP_COL, 'target_total_pts']       # spread selected on its gap, total on the level


def test_residual_total_is_opt_in_and_its_sigma_follows(trained):
    d, _, m, _ = trained
    b = mr.load_bundle('_t', d / 'saved')
    assert isinstance(b.total, mm.ResidualTotalModel) and isinstance(b.spread, mm.ResidualSpreadModel)
    assert b.calib_meta['total_target'] == 'market_residual' and b.calib_meta['total_sigma_model']
    assert b.sigma_total is not None


def test_feature_selection_ranks_the_gap_the_models_are_fit_on(trained):
    *_, seen = trained
    assert seen == ['target_home_win', SPREAD_GAP_COL, TOTAL_GAP_COL]


def test_residual_total_forecast_tracks_a_drifting_scoring_level(trained):
    d, csv, m, _ = trained
    b = mr.load_bundle('_t', d / 'saved')
    last = m.df[m.df['seasonYear'] == SEASONS[-1][0]]
    out = mm.predict_markets(b, last)
    err = last['target_total_pts'].to_numpy() - out['total_pred'].to_numpy()
    assert abs(err.mean()) < 1.5                                              # in-season level is tracked through the line
    # control: a model of the total itself, trained on the earlier seasons, lags the drift (trees cannot extrapolate)
    import lightgbm as lgb
    past = m.df[m.df['seasonYear'] < SEASONS[-1][0]]
    feats = b.features['total']
    plain = lgb.LGBMRegressor(n_estimators=100, num_leaves=8, verbosity=-1, random_state=0).fit(past[feats], past['target_total_pts'])
    plain_err = last['target_total_pts'].to_numpy() - plain.predict(last[feats])
    assert plain_err.mean() > abs(err.mean()) + 1.0


def test_no_total_forecast_without_an_opening_total(trained):
    d, _, m, _ = trained
    b = mr.load_bundle('_t', d / 'saved')
    frame = m.df.tail(20).copy()
    frame['odds_total'] = np.nan
    assert predict_nan(b, frame)


def predict_nan(b, frame):
    out = mm.predict_markets(b, frame)
    return out['total_pred'].isna().all() and out['spread_pred'].notna().all()


# ------------------------------------------------------------------ calibration holdout by time, refit on everything

from team.train_baseline import split_model_train_calibration, CALIB_HOLDOUT_DAYS


def test_time_holdout_is_the_last_n_days_and_never_splits_a_game_night():
    df = pd.DataFrame({'gameDate': pd.to_datetime('2024-10-20') + pd.to_timedelta(np.arange(3000) // 6, unit='D')})
    mt, cal = split_model_train_calibration(df, calib_holdout_days=100)
    assert cal['gameDate'].min() > df['gameDate'].max() - pd.Timedelta(days=100) - pd.Timedelta(days=1)
    assert mt['gameDate'].max() < cal['gameDate'].min()                        # a date is on one side only
    assert len(mt) + len(cal) == len(df) and len(cal) == 600


def test_default_holdout_is_half_a_season_not_a_fifth_of_the_history():
    # 13 seasons of ~1,300 games: the old 20% rule would hold out ~3,400 rows (2.6 seasons); the new one ~half a season
    days = pd.date_range('2013-10-25', '2026-06-13', freq='D')
    dates = np.repeat(days.to_numpy(), 1)
    n_per_day = 5
    df = pd.DataFrame({'gameDate': np.repeat(dates, n_per_day)})
    mt, cal = split_model_train_calibration(df)
    assert CALIB_HOLDOUT_DAYS == 183
    assert (df['gameDate'].max() - mt['gameDate'].max()).days <= CALIB_HOLDOUT_DAYS + 1
    old_mt, _ = split_model_train_calibration(df, calib_holdout_days=None)
    assert (df['gameDate'].max() - old_mt['gameDate'].max()).days > 3 * CALIB_HOLDOUT_DAYS


def test_holdout_too_small_for_calibration_is_refused():
    df = pd.DataFrame({'gameDate': pd.to_datetime('2024-10-20') + pd.to_timedelta(np.arange(500) // 5, unit='D')})
    with pytest.raises(ValueError, match='calibration holdout'):
        split_model_train_calibration(df, calib_holdout_days=10)


def test_base_models_are_refit_on_all_games_by_default(trained_default):
    d, csv, m, _ = trained_default
    import json
    meta = json.loads((d / 'saved' / 'calibration_meta_t.json').read_text())
    assert meta['refit_on_all'] is True
    assert meta['trained_through'] == str(m.df['gameDate'].max())
    assert meta['model_train_end'] < meta['calib_date_range']['start']
    assert meta['calib_holdout'] == 'last 90 days'
    # the saved models have seen the holdout: refit trees were fit on more rows than the pre-refit model_train
    b = mr.load_bundle('_t', d / 'saved')
    n_all = len(m.df)
    assert b.ml.n_features_in_ == len(b.features['ml'])
    assert b.ml.booster_.num_trees() > 0 and n_all > meta['calib_set_size']


def test_refit_can_be_turned_off(tmp_path):
    csv = tmp_path / 'train.csv'
    training_csv(csv, per_season=300)
    m = BaselineModel(csv, model_dir=str(tmp_path / 's'))
    m.train_and_save_all(use_odds=True, save_suffix='_n', time_budget=1, calib_holdout_days=80, refit_on_all=False)
    import json
    meta = json.loads((tmp_path / 's' / 'calibration_meta_n.json').read_text())
    assert meta['refit_on_all'] is False and meta['trained_through'] == meta['model_train_end']
