import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from props.models import calibrate, common, data, folds, minutes, outcome as oc, pipeline, rates, registry
from synthetic_models import synthetic_frame

SPEC = data.FeatureSpec('base')
FAST = {'num_threads': 4}


@pytest.fixture(scope='module')
def frame():
    return synthetic_frame()


@pytest.fixture(scope='module')
def fold(frame):
    return folds.make_fold(frame, '2021-22')


def test_minutes_model_recovers_the_two_regimes(frame, fold):
    mm = minutes.MinutesModel(SPEC, FAST).fit(frame.iloc[fold.fit_idx])
    test = frame.iloc[fold.test_idx]
    played = test[(test.lbl_played == 1) & test.m_min_h15.notna()]
    vals, wts = mm.distribution(played)
    assert vals.shape == (len(played), 2 * minutes.S) and np.allclose(wts.sum(axis=1), 1.0)
    assert (vals >= 0.5).all() and (vals <= 60).all()
    mean = mm.mean_of(vals, wts)
    assert np.mean(np.abs(played.lbl_minutes - mean)) < np.mean(np.abs(played.lbl_minutes - played.m_min_h15)) + 0.3
    c = mm.components(played)
    lim, defined = mm.limited_label(played)
    assert abs(c['p_limited'][defined].mean() - lim[defined].mean()) < 0.04               # the limited share is learned (about 10%)
    assert (vals < 0.55 * played.m_min_h15.to_numpy()[:, None]).mean() > 0.03             # the left tail exists in the distribution


def test_minutes_model_is_deterministic(frame, fold):
    a = minutes.MinutesModel(SPEC, FAST).fit(frame.iloc[fold.fit_idx])
    b = minutes.MinutesModel(SPEC, FAST).fit(frame.iloc[fold.fit_idx])
    rows = frame.iloc[fold.test_idx].head(500)
    assert np.allclose(a.components(rows)['mu_regular'], b.components(rows)['mu_regular'])


def test_rate_models_recover_known_rates_in_both_families(frame, fold):
    rm = rates.RateModels(SPEC, FAST).fit(frame.iloc[fold.fit_idx])
    test = frame.iloc[fold.test_idx]
    played = test[(test.lbl_played == 1) & (test.lbl_minutes > 0)]
    for fam in rates.FAMILIES:
        r = rm.predict(played, {c: fam for c in pipeline.COMPONENTS})
        assert (r[['p2', 'p3', 'pft']] > 0).all().all() and (r[['p2', 'p3', 'pft']] < 1).all().all()
        assert abs(r['p3'].mean() - 0.36) < 0.03 and abs(r['p2'].mean() - 0.52) < 0.03 and abs(r['pft'].mean() - 0.78) < 0.03
        truth = played['lbl_fga'].sum() - played['lbl_fg3a'].sum()
        assert abs((r['fg2a'] * played['lbl_minutes']).sum() / truth - 1) < 0.05            # the average predicted attempts track the truth
    scores = {f: rm.score(test, f) for f in rates.FAMILIES}
    assert set(scores['lgb']) == set(pipeline.COMPONENTS)
    k = rm.fit_dispersion(frame.iloc[fold.calib_idx], {c: 'lgb' for c in pipeline.COMPONENTS})
    assert k['reb'] > 50 and k['fg3a'] > 50                                                # the synthetic counts are Poisson given minutes


def test_family_selection_prefers_hier_on_ties():
    b = [{'family_scores': {'lgb': {c: 1.0 for c in pipeline.COMPONENTS}, 'hier': {c: 1.0004 for c in pipeline.COMPONENTS}}}]
    chosen, _ = pipeline.select_families(b)
    assert set(chosen.values()) == {'hier'}
    b[0]['family_scores']['hier']['fg3a'] = 1.05
    assert pipeline.select_families(b)[0]['fg3a'] == 'lgb'


def test_predictions_do_not_depend_on_the_test_seasons_labels(frame, fold):
    """Fit, calibrate and predict twice: once as is, once with the outcomes of the TEST season scrambled. The distributions must be identical."""
    fams = {c: 'hier' for c in pipeline.COMPONENTS}
    out = []
    test_rows = frame.index[fold.test_idx]
    for scramble in (False, True):
        f = frame.copy()
        if scramble:
            rng = np.random.default_rng(5)
            cols = [c for c in f.columns if c.startswith(('y_', 'lbl_')) and c not in ('lbl_played', 'lbl_minutes')]
            for c in cols:
                f.loc[test_rows, c] = rng.permutation(f.loc[test_rows, c].to_numpy())
        bundle = pipeline.train_fold(f, fold, SPEC, score_on_test=False)
        idx, pmfs, calibrator, played, scaler = pipeline.predict_fold(f, fold, bundle, fams, n_sims=300, seed=1)
        out.append((idx, pmfs))
    assert out[0][0]['minutes_mean'].to_numpy() == pytest.approx(out[1][0]['minutes_mean'].to_numpy())
    for s in oc.CAPS:
        assert np.array_equal(out[0][1][s], out[1][1][s]), s


def test_calibrator_repairs_a_miscalibrated_distribution():
    rng = np.random.default_rng(2)
    n = 6000
    mu = rng.uniform(2, 14, n)
    ys = np.arange(41)[None, :]
    pm = poisson.pmf(ys, mu[:, None]).astype(np.float32)
    pm[:, -1] += 1 - pm.sum(axis=1)
    obs = rng.poisson(1.25 * mu)                                                          # the model is 25% too low
    half = n // 2
    cal = calibrate.ThresholdCalibrator().fit('points', pm[:half], obs[:half])
    rep = calibrate.calibration_report('points', pm[half:], obs[half:], cal)
    assert rep['logloss_cal'] < rep['logloss_raw'] and rep['ece_cal'] < rep['ece_raw'] * 0.6
    line = np.floor(mu) + 0.5
    cp = cal.over_prob('points', pm[half:], line[half:])
    assert (cp > oc.prob_over(pm[half:], line[half:])).mean() > 0.7                        # raised, as it should be when the model is too low


def test_registry_versions_are_immutable_and_round_trip(tmp_path):
    registry.write_manifest('vX', {'a': 1}, root=tmp_path)
    with pytest.raises(FileExistsError):
        registry.write_manifest('vX', {'a': 2}, root=tmp_path)
    registry.update_manifest('vX', {'b': 2}, root=tmp_path)
    registry.save_bundle('vX', '2021-22', {'x': [1, 2, 3]}, root=tmp_path)
    assert registry.load_bundle('vX', '2021-22', root=tmp_path) == {'x': [1, 2, 3]}
    idx = pd.DataFrame({'game_id': ['0022100001'], 'player_id': [1]})
    registry.save_predictions('vX', '2021-22', idx, {'points': np.ones((1, 61), dtype=np.float32) / 61}, root=tmp_path)
    i2, p2 = registry.load_predictions('vX', '2021-22', root=tmp_path)
    assert i2['game_id'].iloc[0] == '0022100001' and p2['points'].shape == (1, 61)


def test_config_log_counts_every_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'CONFIG_LOG', tmp_path / 'log.jsonl')
    common.log_config('minutes', spec='base', rounds=10)
    common.log_config('rates_hier', component='reb', alpha=1.0)
    assert len((tmp_path / 'log.jsonl').read_text().strip().splitlines()) == 2


def test_refit_all_trains_on_calibration_rows_and_still_ignores_test_labels(frame, fold):
    fams = {c: 'hier' for c in pipeline.COMPONENTS}
    out = []
    test_rows = frame.index[fold.test_idx]
    for scramble in (False, True):
        f = frame.copy()
        if scramble:
            rng = np.random.default_rng(5)
            for c in [c for c in f.columns if c.startswith(('y_', 'lbl_')) and c not in ('lbl_played', 'lbl_minutes')]:
                f.loc[test_rows, c] = rng.permutation(f.loc[test_rows, c].to_numpy())
        bundle = pipeline.train_fold(f, fold, SPEC, score_on_test=False, refit_all=True)
        assert bundle['minutes'].info['regular']['rows'] < bundle['refit']['minutes'].info['regular']['rows']      # the refit models saw the calibration window too
        idx, pmfs, calibrator, played, scaler = pipeline.predict_fold(f, fold, bundle, fams, n_sims=300, seed=1)
        out.append(pmfs)
    for s in oc.CAPS:
        assert np.array_equal(out[0][s], out[1][s]), s


def test_forward_oof_never_uses_the_future():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 6000
    dates = np.repeat(pd.date_range('2022-01-01', periods=300).strftime('%Y-%m-%d'), 20)
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n)})
    y = 2.0 * X['a'].to_numpy() + rng.normal(0, 0.3, n)
    w = np.ones(n)
    params = {**common.DEFAULT_PARAMS, 'num_threads': 2}
    base = common.forward_oof('regression', X, y, w, dates, 30, n_blocks=5, params=params)
    assert np.isnan(base[dates < dates[int(n * 0.2)]]).all()                                  # block 0 only trains
    y2 = y.copy()
    y2[dates >= '2022-10-01'] = -y2[dates >= '2022-10-01']                                     # change the relationship late in the window
    alt = common.forward_oof('regression', X, y2, w, dates, 30, n_blocks=5, params=params)
    early = dates < '2022-08-01'                                                               # blocks that only see earlier data are unaffected
    both = early & ~np.isnan(base)
    assert both.sum() > 500 and np.allclose(base[both], alt[both])


def test_minutes_residual_bank_is_standardized_out_of_sample(frame, fold):
    mm = minutes.MinutesModel(SPEC, FAST).fit(frame.iloc[fold.fit_idx])
    for name in ('regular', 'limited'):
        bank = mm.reg[name]['bank']
        assert bank.shape == (len(minutes.BUCKET_EDGES) + 1, minutes.S) and np.isfinite(bank).all()
        assert np.all(np.diff(bank, axis=1) >= -1e-9)                                           # quantiles are ordered
        assert 0.5 < np.median(np.abs(bank[1])) < 3.0                                           # standardized: order one, not collapsed


def test_recency_weights_halve_per_half_life_and_leave_other_columns_alone(frame, fold):
    part = frame.iloc[fold.fit_idx].head(2000)
    ref = frame.iloc[fold.calib_idx]['game_date'].max()
    out = pipeline.recency_weights(part, ref, 365.0)
    age = (pd.Timestamp(ref) - pd.to_datetime(part['game_date'])).dt.days
    assert np.allclose(out['weight'], part['weight'] * 0.5 ** (age / 365.0))
    assert (out['weight'] <= part['weight'] + 1e-12).all() and out['weight'].iloc[age.argmax()] < out['weight'].iloc[age.argmin()]
    assert out.drop(columns='weight').equals(part.drop(columns='weight'))
    bundle = pipeline.train_fold(frame, fold, SPEC, score_on_test=False, recency_half_life_days=365.0)
    assert bundle['minutes'].info['regular']['rows'] > 0


def test_live_fold_has_no_test_part_and_calibrate_fold_matches_predict_fold(frame):
    import pytest as _pytest
    with _pytest.raises(ValueError):
        folds.make_live_fold(frame, '2021-22')                                                # the frame already has that season
    live_frame = frame[frame['season'] < '2021-22'].reset_index(drop=True)
    lf = folds.make_live_fold(live_frame, '2021-22')
    assert len(lf.test_idx) == 0 and live_frame['game_date'].to_numpy()[lf.fit_idx].max() < live_frame['game_date'].to_numpy()[lf.calib_idx].min()
    # the same arithmetic as predict_fold on a real fold
    fold = folds.make_fold(frame, '2021-22')
    fams = {c: 'hier' for c in pipeline.COMPONENTS}
    b1 = pipeline.train_fold(frame, fold, SPEC, score_on_test=False, refit_all=True)
    b2 = pipeline.train_fold(frame, fold, SPEC, score_on_test=False, refit_all=True)
    _, _, _, _, scaler_a = pipeline.predict_fold(frame, fold, b1, fams, n_sims=300, seed=1)
    scaler_b, _ = pipeline.calibrate_fold(frame, fold, b2, fams, n_sims=300, seed=1)
    assert scaler_a.c == scaler_b.c
    assert b1['refit']['rates'].nb_k == b2['refit']['rates'].nb_k
