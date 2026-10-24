"""Experiment harness (experiments/lib.py): calibration-block cap, refit on model_train + calibration, fold labels."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments import lib
from team.train_baseline import BaselineModel


def _games(per_season=300, seasons=('2020-21', '2021-22', '2022-23'), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k, s in enumerate(seasons):
        d = pd.DataFrame({'seasonYear': s, 'gameDate': pd.Timestamp(2020 + k, 10, 20) + pd.to_timedelta(np.arange(per_season) // 3, unit='D')})
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    n = len(df)
    X = rng.normal(size=(n, 8))
    for j in range(8):
        df[f'diff_f{j}'] = X[:, j]
    df['odds_home_spread'] = -2.0 * X[:, 0]
    df['odds_total'] = 215 + 2 * X[:, 1]
    df['target_point_diff'] = 2.0 * X[:, 0] + 1.5 * X[:, 2] + rng.normal(scale=8, size=n)
    df['target_total_pts'] = 215 + 2 * X[:, 1] + 2 * X[:, 3] + rng.normal(scale=12, size=n)
    df['target_home_win'] = (df['target_point_diff'] > 0).astype(int)
    df['gameId'] = [f'{i:010d}' for i in range(n)]
    m = BaselineModel('unused.csv')
    m.features_diff = [c for c in df.columns if c.startswith('diff_')]
    m.features_sep = list(m.features_diff)
    return m, df


# ------------------------------------------------------------------ calibration block

def test_calibration_block_is_capped_at_about_one_season():
    df = pd.DataFrame({'seasonYear': np.repeat([f'{y}-{y + 1 - 2000}' for y in range(2013, 2026)], 1300), 'gameDate': pd.Timestamp('2013-10-25')})
    df['gameDate'] = df['gameDate'] + pd.to_timedelta(np.arange(len(df)) // 6, unit='D')
    mt, calib, test, n_fl = lib.split_fold(df, '2025-26')
    assert len(calib) == lib.CALIB_MAX_ROWS == 1300                                 # the old rule held out 20% of 15,600 = 3,120 games
    assert len(mt) + len(calib) == 12 * 1300 and mt['gameDate'].max() <= calib['gameDate'].min()
    old_mt, old_calib, _, _ = lib.split_fold(df, '2025-26', calib_cap=None)
    assert len(old_calib) == 15600 - int(15600 * 0.8) and len(old_mt) == int(15600 * 0.8)
    assert n_fl == int(len(mt) * lib.FLAML_TRAIN_FRAC)


def test_short_history_keeps_the_twenty_percent_block():
    df = pd.DataFrame({'seasonYear': np.repeat(['2018-19', '2019-20'], 1000), 'gameDate': pd.Timestamp('2018-10-20')})
    df['gameDate'] = df['gameDate'] + pd.to_timedelta(np.arange(len(df)) // 5, unit='D')
    _, calib, _, _ = lib.split_fold(df, '2019-20')
    assert len(calib) == 200


def test_fold_labels_follow_the_scheme_that_made_the_run():
    df = pd.DataFrame({'seasonYear': np.repeat(['2018-19', '2019-20', '2020-21'], 4000)})
    seasons = ['2019-20', '2020-21']
    capped = lib.label_calib_folds(df, pd.DataFrame(index=range(800 + 1300)), seasons=seasons)       # 20% of 4,000; 20% of 8,000 capped
    assert list(capped[:800]) == ['2019-20'] * 800 and list(capped[800:]) == ['2020-21'] * 1300
    legacy = lib.label_calib_folds(df, pd.DataFrame(index=range(800 + 1600)), seasons=seasons)      # the uncapped 20% blocks
    assert list(legacy[:800]) == ['2019-20'] * 800 and list(legacy[800:]) == ['2020-21'] * 1600
    tagged = lib.label_calib_folds(df, pd.DataFrame({'fold': ['2020-21', '2019-20', '2020-21']}), seasons=seasons)
    assert list(tagged) == ['2020-21', '2019-20', '2020-21']                       # an explicit fold column wins, in any row order
    with pytest.raises(AssertionError):
        lib.label_calib_folds(df, pd.DataFrame(index=range(7)), seasons=seasons)


def test_runs_without_split_json_are_read_as_the_old_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, 'OUT', tmp_path)
    (tmp_path / 'old').mkdir()
    assert lib.run_split_params('old') == {'calib_cap': None, 'refit_on_all': False}
    (tmp_path / 'new').mkdir()
    (tmp_path / 'new' / 'split.json').write_text(json.dumps({'calib_cap': 1300, 'refit_on_all': True}))
    assert lib.run_split_params('new') == {'calib_cap': 1300, 'refit_on_all': True}


# ------------------------------------------------------------------ refit on model_train + calibration

@pytest.fixture(scope='module')
def two_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp('wf')
    mp = pytest.MonkeyPatch()
    mp.setattr(lib, 'OUT', out)
    m, df = _games()
    runs = {}
    for name, refit in (('refit', True), ('plain', False)):
        runs[name] = lib.run_walk_forward(name, df_override=(m, df), seasons=['2022-23'], calib_cap=100, refit_on_all=refit,
                                          selection_seed=1)
    mp.undo()
    return out, m, df, runs


def test_refit_changes_only_the_test_predictions_not_the_calibration_set(two_runs):
    out, m, df, runs = two_runs
    (t_r, c_r, i_r), (t_p, c_p, i_p) = runs['refit'], runs['plain']
    pd.testing.assert_frame_equal(c_r, c_p)                                      # calibrators and sigma see the same out-of-sample predictions
    assert (t_r['sigma_spread'] == t_p['sigma_spread']).all() and (t_r['sigma_total'] == t_p['sigma_total']).all()
    for col in ('ml_prob_raw', 'spread_pred', 'total_pred'):
        assert not np.allclose(t_r[col], t_p[col])                               # the refit model has seen the calibration block
    assert i_r['2022-23']['spread']['refit_rows'] == 600 and 'refit_rows' not in i_p['2022-23']['spread']


def test_refit_model_equals_the_tuned_config_trained_on_all_earlier_games(two_runs):
    out, m, df, runs = two_runs
    t_r, _, info = runs['refit']
    feats = json.loads((out / 'refit' / 'features.json').read_text())['2022-23']['spread']
    train = df[df['seasonYear'] < '2022-23'].sort_values('gameDate', kind='stable').reset_index(drop=True)
    test = df[df['seasonYear'] == '2022-23'].reset_index(drop=True)
    est = lib.make_est('reg', lib.GRID[info['2022-23']['spread']['cfg']]).fit(train[feats], train['target_point_diff'])
    np.testing.assert_allclose(t_r['spread_pred'].to_numpy(), est.predict(test[feats]))


def test_saved_run_records_its_split_and_tags_calibration_rows_with_their_fold(two_runs):
    out, *_ = two_runs
    assert json.loads((out / 'refit' / 'split.json').read_text()) == {'calib_cap': 100, 'refit_on_all': True}
    cal = pd.read_csv(out / 'refit' / 'calibration_set_predictions.csv')
    assert len(cal) == 100 and set(cal['fold']) == {'2022-23'}


# ------------------------------------------------------------------ residual runs (E1 / E25): selection and missing lines

def _with_market_columns(df, seed=1):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df['odds_home_ml'] = np.where(df['odds_home_spread'] < 0, -150.0, 130.0)
    df['odds_away_ml'] = np.where(df['odds_home_spread'] < 0, 130.0, -150.0)
    gone = rng.random(len(df)) < 0.04
    gone[[0, 1, len(df) - 1, len(df) - 2]] = True                                  # the first training rows and the last test rows too
    df.loc[gone, ['odds_home_spread', 'odds_total', 'odds_home_ml', 'odds_away_ml']] = np.nan
    return df


@pytest.fixture(scope='module')
def residual_run(tmp_path_factory):
    from experiments import e01_residual as e1
    out = tmp_path_factory.mktemp('resid')
    mp = pytest.MonkeyPatch()
    mp.setattr(lib, 'OUT', out)
    m, df = _games(per_season=1000, seasons=('2019-20', '2020-21', '2021-22', '2022-23'))
    df = _with_market_columns(df)
    before = df.copy()
    hooks = {'init_scores': e1.init_scores, 'target': e1.target, 'pred_offset': e1.pred_offset, 'require_columns': e1.LINE_COLUMNS}
    res = lib.run_walk_forward('resid', df_override=(m, df), selection='residual_no_odds', hooks=hooks, seasons=['2022-23'], calib_cap=100)
    mp.undo()
    return out, m, df, before, res


def test_residual_selection_leaves_the_market_out_of_the_feature_lists(residual_run):
    out, *_ = residual_run
    feats = json.loads((out / 'resid' / 'features.json').read_text())['2022-23']
    assert all(not f.startswith('odds_') for fs in feats.values() for f in fs)
    assert json.loads((out / 'resid' / 'selection.json').read_text()) == {'selection': 'residual_no_odds'}


def test_missing_lines_are_dropped_inside_the_fold_not_from_the_frame(residual_run):
    out, m, df, before, (test, calib, info) = residual_run
    pd.testing.assert_frame_equal(df, before)                                       # the caller's frame is untouched
    train = df[df['seasonYear'] < '2022-23'].sort_values('gameDate', kind='stable').reset_index(drop=True)
    boundary = train.iloc[len(train) - 100:]                                         # the calibration block, set on the FULL history
    ok = lambda f: f.dropna(subset=['odds_home_spread', 'odds_total'])
    assert set(calib['gameId']) == set(ok(boundary)['gameId'])
    assert set(test['gameId']) == set(ok(df[df['seasonYear'] == '2022-23'])['gameId'])
    assert info['2022-23']['spread']['refit_rows'] == len(ok(train))
    assert test['spread_pred'].notna().all() and test['total_pred'].notna().all()


def test_cached_features_from_another_selection_rule_are_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, 'OUT', tmp_path)
    m, df = _games()
    calls = []
    real = lib.select_fold_features
    monkeypatch.setattr(lib, 'select_fold_features', lambda *a: calls.append(a[2]) or real(*a))
    kw = dict(df_override=(m, df), seasons=['2022-23'], calib_cap=100, refit_on_all=False)
    lib.run_walk_forward('c', selection='raw', **kw)
    lib.run_walk_forward('c', selection='raw', **kw)               # same rule: the cache is reused
    lib.run_walk_forward('c', selection='legacy', **kw)            # a different rule: selected afresh
    assert calls == ['raw', 'legacy']
    with pytest.raises(ValueError):
        real(m, df, 'bogus', 0)
