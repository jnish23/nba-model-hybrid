"""
One walk-forward fold, end to end: train availability, minutes and both rate families on the fit window; choose a rate family per component across the
development folds; predict the played test rows' full distributions; calibrate on the calibration window.

The test season's labels are used only to SCORE (rate-family selection on development folds, the smoke report); no fitting step sees them.
"""
import numpy as np
import pandas as pd

from props.models import calibrate, common, outcome as oc
from props.models.availability import Availability
from props.models.data import FeatureSpec
from props.models.minutes import MinutesModel
from props.models.rates import COUNT_COMPONENTS, FAMILIES, MAKE_COMPONENTS, RateModels

COMPONENTS = list(COUNT_COMPONENTS) + list(MAKE_COMPONENTS)
OBS = {'points': 'y_points', 'rebounds': 'y_rebounds', 'assists': 'y_assists', 'threes': 'y_threes', 'pra': 'y_pra'}
PREFER_HIER_WITHIN = 0.001                 # if the two families are within 0.1% of each other, take the interpretable hierarchical model


def observed(frame, stat):
    y = frame
    if stat in OBS:
        return y[OBS[stat]].to_numpy(float)
    parts = {'pr': ('y_points', 'y_rebounds'), 'pa': ('y_points', 'y_assists'), 'ra': ('y_rebounds', 'y_assists')}[stat]
    return (y[parts[0]] + y[parts[1]]).to_numpy(float)


def train_fold(df, fold, spec: FeatureSpec, score_on_test=True):
    fit, cal, test = df.iloc[fold.fit_idx], df.iloc[fold.calib_idx], df.iloc[fold.test_idx]
    bundle = {'fold': fold.describe(), 'spec': spec, 'availability': Availability(spec).fit(fit), 'minutes': MinutesModel(spec).fit(fit),
              'rates': RateModels(spec).fit(fit), 'family_scores': {}}
    if score_on_test:                                     # development folds only: rate-family comparison with the actual minutes as exposure
        bundle['family_scores'] = {f: bundle['rates'].score(test, f) for f in FAMILIES}
    return bundle


def select_families(bundles):
    """Per component: the family with the lower mean loss across the development folds (hierarchical if within PREFER_HIER_WITHIN)."""
    chosen, table = {}, []
    for c in COMPONENTS:
        mean = {f: float(np.mean([b['family_scores'][f][c] for b in bundles])) for f in FAMILIES}
        pick = 'lgb' if mean['lgb'] < mean['hier'] * (1 - PREFER_HIER_WITHIN) else 'hier'
        chosen[c] = pick
        table.append({'component': c, **{f'{f}_loss': round(v, 5) for f, v in mean.items()}, 'chosen': pick})
    return chosen, pd.DataFrame(table)


def distributions(bundle, rows, families, n_sims, seed):
    mm, rm = bundle['minutes'], bundle['rates']
    vals, wts = mm.distribution(rows)
    r = rm.predict(rows, families)
    res = oc.outcome_distributions(vals, wts, {k: r[k].to_numpy() for k in r.columns}, rm.nb_k, n_sims=n_sims, seed=seed)
    res['minutes_mean'] = mm.mean_of(vals, wts)
    return res


def predict_fold(df, fold, bundle, families, n_sims=4000, seed=0):
    """Fits the calibration maps on the calibration window and returns (index frame, pmfs by stat, calibrator) for the played test rows."""
    rm = bundle['rates']
    cal = df.iloc[fold.calib_idx]
    cal = cal[(cal['lbl_played'] == 1) & (cal['lbl_minutes'] > 0)]
    rm.fit_dispersion(cal, families)
    cres = distributions(bundle, cal, families, n_sims, seed + 1)
    calibrator = calibrate.ThresholdCalibrator()
    for stat in oc.CAPS:
        y = observed(cal, stat)
        ok = ~np.isnan(y)
        calibrator.fit(stat, cres[stat][ok], y[ok])
    test = df.iloc[fold.test_idx]
    played = test[(test['lbl_played'] == 1) & (test['lbl_minutes'] > 0)]
    res = distributions(bundle, played, families, n_sims, seed)
    idx = played[['game_id', 'player_id', 'minutes_before', 'season', 'game_date']].reset_index(drop=True)
    idx['minutes_mean'] = res.pop('minutes_mean')
    idx['p_limited'] = bundle['minutes'].components(played)['p_limited']
    idx['p_play_model'] = bundle['availability'].predict(played)
    idx['p_play_table'] = played['inj_p_play'].to_numpy()
    res.pop('sim_means', None)
    return idx, res, calibrator, played
