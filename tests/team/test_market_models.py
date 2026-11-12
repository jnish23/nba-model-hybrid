import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from team import market_models as mm
from team import model_registry as mr


def _games(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    line = np.round(rng.normal(-2, 6, n) * 2) / 2                 # opening home spread
    margin = -line + 2.0 * x + rng.normal(0, 12, n)               # x is information the line misses
    total_line = np.round(rng.normal(225, 8, n))
    return pd.DataFrame({
        'gameDate': pd.date_range('2020-01-01', periods=n, freq='D'),
        'x': x, 'odds_home_spread': line, 'odds_total': total_line,
        'target_point_diff': margin, 'target_total_pts': total_line + rng.normal(0, 18, n) * (1 + 0.3 * np.abs(x)),
        'target_home_win': (margin > 0).astype(int)})


def _reg(df, target, cols=('x',)):
    return lgb.LGBMRegressor(n_estimators=30, num_leaves=4, verbosity=-1, random_state=0).fit(df[list(cols)], target)


def test_residual_spread_model_adds_back_the_opening_line():
    df = _games()
    y = mm.ResidualSpreadModel.make_target(df)
    assert np.allclose(y, df['target_point_diff'] + df['odds_home_spread'])
    m = mm.ResidualSpreadModel(_reg(df, y))
    pred = m.predict(df[['x']], line=df['odds_home_spread'].to_numpy())
    assert np.allclose(pred, -df['odds_home_spread'] + m.predict_residual(df[['x']]))
    assert np.mean(np.abs(df['target_point_diff'] - pred)) < np.mean(np.abs(df['target_point_diff'] + df['odds_home_spread']))


def test_residual_spread_model_without_a_line_is_nan_not_a_guess():
    df = _games(200)
    m = mm.ResidualSpreadModel(_reg(df, mm.ResidualSpreadModel.make_target(df)))
    assert np.isnan(m.predict(df[['x']], line=np.full(len(df), np.nan))).all()
    with pytest.raises(ValueError):
        m.predict(df[['x']])                                      # no line given and not a column of X
    m2 = mm.ResidualSpreadModel(_reg(df, mm.ResidualSpreadModel.make_target(df), cols=('x', 'odds_home_spread')))
    X2 = df[['x', 'odds_home_spread']]                            # line among the model's features: read from X
    assert np.allclose(m2.predict(X2), m2.predict(X2, line=df['odds_home_spread']))


def test_margin_stack_beats_platt_when_the_margin_adds_information_and_falls_back_without_it():
    rng = np.random.default_rng(1)
    n = 6000
    margin = rng.normal(0, 6, n)
    y = (margin + rng.normal(0, 12, n) > 0).astype(int)
    raw = np.clip(0.5 + 0.02 * rng.normal(size=n), 0.05, 0.95)    # uninformative raw classifier
    cal = mm.MarginStackCalibrator().fit(raw[:4000], margin[:4000], 12.0, y[:4000])
    p = cal.predict(raw[4000:], margin=margin[4000:])
    p_platt = cal.predict(raw[4000:])                              # no margin -> Platt on the raw score only
    assert log_loss(y[4000:], p) < log_loss(y[4000:], p_platt) - 0.01
    assert ((p > 0) & (p < 1)).all()
    mixed = margin[4000:].copy()
    mixed[:5] = np.nan
    pm = cal.predict(raw[4000:], margin=mixed)
    assert np.allclose(pm[:5], p_platt[:5]) and np.allclose(pm[5:], p[5:])


def test_oof_errors_never_use_future_rows():
    df = _games(1000)
    feats = ['x', 'odds_home_spread']
    params = dict(n_estimators=20, num_leaves=4)
    idx, err = mm.oof_abs_errors(df, feats, 'target_total_pts', params)
    assert idx.min() >= len(df) // mm.SIGMA_FOLDS                  # the first part is only ever a training block
    corrupted = df.copy()
    cut = 600
    corrupted.loc[cut:, ['x', 'odds_home_spread', 'target_total_pts']] += 1000.0     # rewrite the future
    idx2, err2 = mm.oof_abs_errors(corrupted, feats, 'target_total_pts', params)
    before = idx < cut
    assert (idx == idx2).all() and np.allclose(err[before & (idx < 400)], err2[before & (idx < 400)])


def test_hetero_sigma_is_positive_finite_and_tracks_noise():
    df = _games(3000)
    model_train = df.iloc[:2400]
    sig = mm.fit_hetero_sigma(model_train, ['x', 'odds_home_spread'], 'target_total_pts', dict(n_estimators=20, num_leaves=4),
                              fallback_std=18.0, sigma_features=['x', 'odds_home_spread', 'not_a_column'])
    s = sig.predict(df.iloc[2400:])
    assert np.isfinite(s).all() and (s > 0).all()
    hi, lo = df.iloc[2400:]['x'].abs() > 1.5, df.iloc[2400:]['x'].abs() < 0.3
    assert s[hi.to_numpy()].mean() > s[lo.to_numpy()].mean()       # noise grows with |x| in the synthetic data
    nan_frame = df.iloc[2400:2410].assign(x=np.nan, odds_home_spread=np.nan)
    assert np.isfinite(sig.predict(nan_frame)).all()


# ------------------------------------------------------------------ registry + predict_markets

FEATS = {'ml': ['x'], 'spread': ['x'], 'total': ['x', 'odds_total']}


def _save_set(d, suffix, market_aware):
    df = _games(800)
    clf = lgb.LGBMClassifier(n_estimators=10, num_leaves=4, verbosity=-1).fit(df[['x']], df['target_home_win'])
    raw = clf.predict_proba(df[['x']])[:, 1]
    spread = _reg(df, mm.ResidualSpreadModel.make_target(df) if market_aware else df['target_point_diff'])
    total = lgb.LGBMRegressor(n_estimators=10, num_leaves=4, verbosity=-1).fit(df[['x', 'odds_total']], df['target_total_pts'])
    joblib.dump(clf, d / f'model_ml{suffix}.pkl')
    joblib.dump(total, d / f'model_total{suffix}.pkl')
    meta = {'spread_std': 13.0, 'total_std': 18.0}
    if market_aware:
        rs = mm.ResidualSpreadModel(spread)
        joblib.dump(rs, d / f'model_spread{suffix}.pkl')
        margin = rs.predict(df[['x']], line=df['odds_home_spread'])
        joblib.dump(mm.MarginStackCalibrator().fit(raw, margin, 13.0, df['target_home_win']), d / f'calibrator_ml{suffix}.pkl')
        joblib.dump(mm.fit_hetero_sigma(df, ['x', 'odds_total'], 'target_total_pts', dict(n_estimators=10, num_leaves=4), 18.0,
                                        sigma_features=['x', 'odds_total']), d / f'sigma_total{suffix}.pkl')
        meta['total_sigma_model'] = True
    else:
        joblib.dump(spread, d / f'model_spread{suffix}.pkl')
        joblib.dump(IsotonicRegression(out_of_bounds='clip').fit(raw, df['target_home_win']), d / f'calibrator_ml{suffix}.pkl')
    (d / f'feature_config{suffix}.json').write_text(json.dumps(FEATS))
    (d / f'calibration_meta{suffix}.json').write_text(json.dumps(meta))
    return df


def test_market_aware_set_loads_and_predicts_with_per_game_sigma(tmp_path):
    df = _save_set(tmp_path, '_m', True)
    b = mr.load_bundle('_m', tmp_path)
    assert isinstance(b.spread, mm.ResidualSpreadModel) and b.sigma_total is not None
    out = mm.predict_markets(b, df.iloc[:50])
    assert list(out.columns) == ['ml_prob_raw', 'ml_prob', 'spread_pred', 'total_pred', 'spread_std', 'total_std']
    assert out['total_std'].nunique() > 1 and (out['spread_std'] == 13.0).all()
    assert out['ml_prob'].between(0, 1).all()
    # without the opening spread: no spread forecast, but the ML probability still comes out (Platt fallback)
    no_line = df.iloc[:50].assign(odds_home_spread=np.nan)
    out2 = mm.predict_markets(b, no_line)
    assert out2['spread_pred'].isna().all() and out2['ml_prob'].notna().all()


def test_legacy_set_still_works_with_a_global_sigma(tmp_path):
    df = _save_set(tmp_path, '_old', False)
    b = mr.load_bundle('_old', tmp_path)
    assert b.sigma_total is None
    out = mm.predict_markets(b, df.iloc[:20])
    assert (out['total_std'] == 18.0).all() and out['spread_pred'].notna().all()


def test_set_that_declares_a_sigma_model_but_lacks_the_file_is_refused(tmp_path):
    _save_set(tmp_path, '_m', True)
    (tmp_path / 'sigma_total_m.pkl').unlink()
    with pytest.raises(FileNotFoundError, match='sigma_total'):
        mr.load_bundle('_m', tmp_path)


def test_residual_total_model_adds_back_the_opening_total():
    df = _games()
    y = mm.ResidualTotalModel.make_target(df)
    assert np.allclose(y, df['target_total_pts'] - df['odds_total'])
    m = mm.ResidualTotalModel(_reg(df, y))
    pred = m.predict(df[['x']], line=df['odds_total'].to_numpy())
    assert np.allclose(pred, df['odds_total'] + m.predict_residual(df[['x']]))        # + line (the spread model subtracts it)
    assert np.isnan(m.predict(df[['x']], line=np.full(len(df), np.nan))).all()
    with pytest.raises(ValueError):
        m.predict(df[['x']])


def test_residual_total_keeps_its_bias_when_the_level_drifts_beyond_the_training_range():
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.normal(size=n)
    line = np.linspace(200, 232, n)                       # scoring level drifts up over the history
    actual = line + 2 * x + rng.normal(0, 14, n)
    df = pd.DataFrame({'x': x, 'odds_total': line, 'target_total_pts': actual})
    tr, te = df.iloc[:2400], df.iloc[2400:]               # test games sit above every training total
    plain = lgb.LGBMRegressor(n_estimators=60, num_leaves=8, verbosity=-1, random_state=0).fit(tr[['x', 'odds_total']], tr['target_total_pts'])
    resid = mm.ResidualTotalModel(_reg(tr, mm.ResidualTotalModel.make_target(tr)))
    plain_bias = (te['target_total_pts'] - plain.predict(te[['x', 'odds_total']])).mean()
    resid_bias = (te['target_total_pts'] - resid.predict(te[['x']], line=te['odds_total'].to_numpy())).mean()
    assert plain_bias > 4 and abs(resid_bias) < 1.5


def test_predict_markets_uses_the_opening_total_for_a_residual_total_model():
    df = _games(400)
    sp = mm.ResidualSpreadModel(_reg(df, mm.ResidualSpreadModel.make_target(df)))
    tt = mm.ResidualTotalModel(_reg(df, mm.ResidualTotalModel.make_target(df)))
    ml = lgb.LGBMClassifier(n_estimators=20, verbosity=-1, random_state=0).fit(df[['x']], df['target_home_win'])
    cal = mm.MarginStackCalibrator().fit(ml.predict_proba(df[['x']])[:, 1], sp.predict(df[['x']], line=df['odds_home_spread']), 12.0, df['target_home_win'])
    bundle = mr.ModelBundle(suffix='_t', ml=ml, calibrator=cal, spread=sp, total=tt,
                            features={'ml': ['x'], 'spread': ['x'], 'total': ['x']}, calib_meta={'spread_std': 12.0, 'total_std': 18.0})
    out = mm.predict_markets(bundle, df)
    assert np.allclose(out['total_pred'], df['odds_total'] + tt.predict_residual(df[['x']]))
    assert mm.predict_markets(bundle, df.assign(odds_total=np.nan))['total_pred'].isna().all()


# ------------------------------------------------------------------ E18 fix (c) / E27: TotalLevelCalibrator

def _cal_block(n=3000, seed=5):
    """A calibration block whose model is a noisy, level-dependent signal: the model departs from the line mostly by noise, more so at high lines."""
    rng = np.random.default_rng(seed)
    line = np.round(rng.normal(224, 8, n))
    truth_gap = rng.normal(0, 3, n)
    actual = line + 0.3 * truth_gap + rng.normal(0, 18, n)
    pred = line + truth_gap + rng.normal(0, 3, n) - 0.4 * (line - 224)         # departs from the line, and leans down when the line is high
    return pd.DataFrame({'odds_total': line, 'total_pred': pred, 'target_total_pts': actual})


def _e18_fix_c(cal):
    """The reference: experiments/e18_calibration.py::make_fix('c') as registered."""
    from experiments.e18_calibration import make_fix
    return make_fix('c', cal)


def test_total_level_calibrator_matches_e18_fix_c():
    cal, test = _cal_block(3000, 5), _cal_block(800, 6)
    ref = _e18_fix_c(cal)
    c = mm.TotalLevelCalibrator().fit(cal['total_pred'], cal['odds_total'], cal['target_total_pts'])
    assert np.isclose(c.w0, ref.params['w0']) and np.isclose(c.w1, ref.params['w1']) and np.isclose(c.mean_level, ref.params['mean_level'])
    assert np.allclose(c.predict(test['total_pred'], test['odds_total']), ref(test), rtol=0, atol=1e-9)
    assert np.allclose(c.predict(cal['total_pred'], cal['odds_total']), ref(cal), rtol=0, atol=1e-9)


def test_total_level_calibrator_weight_is_clipped_and_keeps_games_without_a_line():
    c = mm.TotalLevelCalibrator()
    c.w0, c.w1, c.mean_level = 0.5, 0.2, 224.0                                    # slope so steep that the raw weight leaves [0, 1] at both ends
    line = np.array([200.0, 224.0, 230.0, np.nan])
    w = c.weight(line)
    assert np.allclose(w[:3], [0.0, 0.5, 1.0]) and np.isnan(w[3])
    pred = np.array([210.0, 230.0, 240.0, 233.0])
    out = c.predict(pred, line)
    assert np.allclose(out[:3], [200.0, 227.0, 240.0])                           # w=0 -> the line, w=1 -> the model
    assert out[3] == 233.0                                                       # no opening total: raw prediction kept
    assert np.isnan(c.predict(np.array([np.nan]), np.array([220.0]))).all()
    with pytest.raises(ValueError):
        mm.TotalLevelCalibrator().predict(pred, line)                            # not fitted


def test_total_level_calibrator_shrinks_a_noise_forecast_toward_the_line_and_leaves_a_good_one_alone():
    rng = np.random.default_rng(2)
    n = 6000
    line = np.round(rng.normal(224, 8, n))
    noise = line + rng.normal(0, 4, n)
    actual = line + rng.normal(0, 18, n)                                         # the forecast carries no information about the outcome
    c = mm.TotalLevelCalibrator().fit(noise, line, actual)
    assert c.weight(line).mean() < 0.15
    good = line + rng.normal(0, 4, n)
    actual2 = good + rng.normal(0, 4, n)                                         # the forecast IS the signal
    c2 = mm.TotalLevelCalibrator().fit(good, line, actual2)
    assert c2.weight(line).mean() > 0.85


def test_total_level_calibrator_ignores_rows_it_cannot_use_and_refuses_an_empty_fit():
    cal = _cal_block(500)
    c_full = mm.TotalLevelCalibrator().fit(cal['total_pred'], cal['odds_total'], cal['target_total_pts'])
    dirty = pd.concat([cal, pd.DataFrame({'odds_total': [np.nan, 220.0, 220.0], 'total_pred': [220.0, np.nan, 220.0], 'target_total_pts': [200.0, 200.0, np.nan]})])
    c_dirty = mm.TotalLevelCalibrator().fit(dirty['total_pred'], dirty['odds_total'], dirty['target_total_pts'])
    assert c_dirty.params == c_full.params and c_full.n_fit == 500
    with pytest.raises(ValueError):
        mm.TotalLevelCalibrator().fit([np.nan], [np.nan], [np.nan])


def _save_set_with_total_calibrator(d, suffix='_c'):
    df = _save_set(d, suffix, True)
    total = joblib.load(d / f'model_total{suffix}.pkl')
    raw = total.predict(df[['x', 'odds_total']])
    cal = mm.TotalLevelCalibrator().fit(raw, df['odds_total'], df['target_total_pts'])
    joblib.dump(cal, d / f'calibrator_total{suffix}.pkl')
    return df, cal, raw


def test_calibrator_total_round_trips_and_predict_markets_applies_it_only_when_present(tmp_path):
    df, cal, raw = _save_set_with_total_calibrator(tmp_path)
    b = mr.load_bundle('_c', tmp_path)
    assert isinstance(b.calibrator_total, mm.TotalLevelCalibrator) and b.calibrator_total.params == cal.params          # pickle round trip
    out = mm.predict_markets(b, df.iloc[:60])
    assert np.allclose(out['total_pred'], cal.predict(raw[:60], df['odds_total'].to_numpy()[:60]))
    assert not np.allclose(out['total_pred'], raw[:60])
    # ML and spread are untouched by the total calibrator
    b_plain = mr.load_bundle('_c', tmp_path)
    b_plain.calibrator_total = None
    plain = mm.predict_markets(b_plain, df.iloc[:60])
    for col in ('ml_prob_raw', 'ml_prob', 'spread_pred', 'spread_std', 'total_std'):
        assert np.array_equal(out[col].to_numpy(), plain[col].to_numpy())
    assert np.allclose(plain['total_pred'], raw[:60])
    # a game without an opening total keeps the raw prediction
    frame_nl = df.iloc[:10].assign(odds_total=np.nan)                              # (the total model itself reads odds_total, so its raw forecast moves too)
    no_line = mm.predict_markets(b, frame_nl)
    assert np.array_equal(no_line['total_pred'].to_numpy(), mm.predict_markets(b_plain, frame_nl)['total_pred'].to_numpy())


def test_a_set_that_declares_a_total_calibrator_but_lacks_the_file_is_refused(tmp_path):
    _save_set_with_total_calibrator(tmp_path)
    meta = json.loads((tmp_path / 'calibration_meta_c.json').read_text())
    (tmp_path / 'calibration_meta_c.json').write_text(json.dumps({**meta, 'total_calibrator': 'level_shrink_to_line'}))
    mr.load_bundle('_c', tmp_path)                                                # file present: fine
    (tmp_path / 'calibrator_total_c.pkl').unlink()
    with pytest.raises(FileNotFoundError, match='calibrator_total'):
        mr.load_bundle('_c', tmp_path)


def test_old_set_without_a_total_calibrator_predicts_exactly_as_before(tmp_path):
    for suffix, aware in (('_m', True), ('_old', False)):
        df = _save_set(tmp_path, suffix, aware)
        b = mr.load_bundle(suffix, tmp_path)
        assert b.calibrator_total is None
        out = mm.predict_markets(b, df.iloc[:40])
        assert np.array_equal(out['total_pred'].to_numpy(), b.total.predict(df.iloc[:40][['x', 'odds_total']]))       # the model's own numbers, bit for bit


def test_oof_errors_with_a_calibrator_are_the_errors_of_the_calibrated_forecast():
    df = _games(1000)
    feats, params = ['x', 'odds_total'], dict(n_estimators=20, num_leaves=4)
    idx0, err0 = mm.oof_abs_errors(df, feats, 'target_total_pts', params)
    idx1, err1 = mm.oof_abs_errors(df, feats, 'target_total_pts', params, calibrator=None)
    assert (idx0 == idx1).all() and np.array_equal(err0, err1)                                     # default path unchanged
    w0 = mm.TotalLevelCalibrator()
    w0.w0, w0.w1, w0.mean_level = 0.0, 0.0, 225.0                                                   # weight 0: the calibrated forecast is the line
    idx2, err2 = mm.oof_abs_errors(df, feats, 'target_total_pts', params, calibrator=w0)
    assert np.allclose(err2, np.abs(df['target_total_pts'].to_numpy()[idx2] - df['odds_total'].to_numpy()[idx2]))


def test_total_level_calibrator_reproduces_the_e18_fold_numbers_on_the_saved_walk_forward_run():
    """One fold of the real saved baseline walk-forward run: the fitted weights and the fold's total MAE equal what E18 recorded
    (reports/experiments/E18_calibration/results.json). Skipped where the saved predictions (gitignored) are not available."""
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    src = Path(os.environ.get('E27_SRC', 'C:/Users/jorda/nba-model-hybrid/reports/experiments')) / 'baseline'
    e18_path = root / 'reports' / 'experiments' / 'E18_calibration' / 'results.json'
    if not (src / 'predictions.csv').exists() or not (src / 'calibration_set_predictions.csv').exists() or not e18_path.exists():
        pytest.skip('saved baseline walk-forward predictions not available')
    e18 = json.loads(e18_path.read_text())
    test = pd.read_csv(src / 'predictions.csv', parse_dates=['gameDate'])
    calib = pd.read_csv(src / 'calibration_set_predictions.csv', parse_dates=['gameDate'])
    starts = [0] + [i for i in range(1, len(calib)) if calib['gameDate'].iloc[i] < calib['gameDate'].iloc[i - 1]] + [len(calib)]    # the file is 7 date-sorted blocks in fold order
    seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    if len(starts) != 8:
        pytest.skip('calibration file is not laid out as 7 date-sorted fold blocks')
    for k in (0, 4):                                                                        # the first and a middle fold
        s = seasons[k]
        blk = calib.iloc[starts[k]:starts[k + 1]]
        c = mm.TotalLevelCalibrator().fit(blk['total_pred'], blk['odds_total'], blk['target_total_pts'])
        ref = e18['exploratory']['fit_parameters']['c'][s]
        assert np.isclose(c.w0, ref['w0']) and np.isclose(c.w1, ref['w1']) and np.isclose(c.mean_level, ref['mean_level'])
        t = test[test['seasonYear'] == s]
        mae = np.nanmean(np.abs(t['target_total_pts'] - c.predict(t['total_pred'], t['odds_total'])))
        assert np.isclose(mae, e18['variants']['c']['mae_by_season'][s], atol=1e-9)
