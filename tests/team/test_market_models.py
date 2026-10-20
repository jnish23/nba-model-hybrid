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
