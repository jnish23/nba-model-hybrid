"""Check 6: fold integrity of E25_with_odds, verified from its saved outputs (own code).
 - fold layout: calibration block size = n_train - int(0.8*n_train) for every fold (legacy 80/20 row split), no cap, no refit
 - calibration block lies strictly before the test season; test and calibration game ids are disjoint within each fold
 - sigma stored per fold == std of the calibration residuals (so it was fitted on the calibration block only)
 - ML calibrator: re-fit sklearn isotonic on (ml_prob_raw -> outcome) of the calibration block only, reproduce ml_prob on the test season exactly
 - spread/total predictions: which scheme (raw margin vs residual) by checking spread_pred against -odds_home_spread
 - features.json: no outcome-like or close_* columns; list sizes; every selected column exists in the training set"""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *   # noqa

t = load_train()
p = load_e25('predictions.csv')
c = load_e25('calibration_set_predictions.csv')
seasons = sorted(p.seasonYear.unique())
edges = np.r_[0, np.flatnonzero(np.diff(c.gameDate.values.astype('datetime64[D]').astype(int)) < 0) + 1, len(c)]
assert len(edges) - 1 == len(seasons), (len(edges), seasons)
rows = []
for k, s in enumerate(seasons):
    cb = c.iloc[edges[k]:edges[k + 1]]
    te = p[p.seasonYear == s]
    ntrain = int((t.seasonYear < s).sum())
    ncal_formula = ntrain - int(ntrain * 0.8)
    r = {'test_season': s, 'n_train_rows_(11-01 set)': ntrain, 'calib_rows_formula': ncal_formula, 'calib_rows_observed': len(cb),
         'calib_max_date': cb.gameDate.max().date(), 'test_min_date': te.gameDate.min().date(),
         'calib_before_test': bool(cb.gameDate.max() < te.gameDate.min()), 'id_overlap_test_vs_calib': len(set(cb.gameId) & set(te.gameId))}
    # sigma
    for tgt, pred, sig in (('target_point_diff', 'spread_pred', 'sigma_spread'), ('target_total_pts', 'total_pred', 'sigma_total')):
        res = (cb[tgt] - cb[pred])
        r[f'{sig}_stored'] = float(te[sig].iloc[0])
        r[f'{sig}_std_calib_resid'] = float(res.std(ddof=0))
        r[f'{sig}_std_test_resid'] = float((te[tgt] - te[pred]).std(ddof=0))
    # isotonic re-fit on the calibration block only
    iso = IsotonicRegression(out_of_bounds='clip').fit(cb.ml_prob_raw.to_numpy(), cb.target_home_win.to_numpy())
    r['isotonic_refit_max_abs_diff_on_test'] = float(np.abs(iso.predict(te.ml_prob_raw.to_numpy()) - te.ml_prob.to_numpy()).max())
    # model-trained-on-calibration check: calibration predictions come from the same model; their in-sample-ness shows up as a better score than test
    for nm, tgt, pr in (('spread', 'target_point_diff', 'spread_pred'), ('total', 'target_total_pts', 'total_pred')):
        r[f'{nm}_rmse_calib'] = float(np.sqrt(((cb[tgt] - cb[pr]) ** 2).mean()))
        r[f'{nm}_rmse_test'] = float(np.sqrt(((te[tgt] - te[pr]) ** 2).mean()))
    pc = cb.ml_prob_raw.clip(1e-6, 1 - 1e-6); pt = te.ml_prob_raw.clip(1e-6, 1 - 1e-6)
    ll = lambda pp, y: float(-(y * np.log(pp) + (1 - y) * np.log(1 - pp)).mean())
    r['ml_raw_logloss_calib'] = ll(pc, cb.target_home_win); r['ml_raw_logloss_test'] = ll(pt, te.target_home_win)
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(OUT / 'check6_fold_checks.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(df.T.to_string())

# test rows: one per game, no duplicates across folds, dates ordered
print('test game ids unique:', p.gameId.is_unique, '; test rows', len(p))
print('calibration rows are pre-test only for ALL folds:', bool(df.calib_before_test.all()))
# does the first test season row match training-set targets? (done in check 7) ; spread_pred scheme
x = p.dropna(subset=['odds_home_spread'])
print('corr(spread_pred, -odds_home_spread) =', round(float(np.corrcoef(x.spread_pred, -x.odds_home_spread)[0, 1]), 4),
      ' mean(spread_pred + odds_home_spread) =', round(float((x.spread_pred + x.odds_home_spread).mean()), 3), ' sd =', round(float((x.spread_pred + x.odds_home_spread).std()), 3))
x = p.dropna(subset=['odds_total'])
print('corr(total_pred, odds_total) =', round(float(np.corrcoef(x.total_pred, x.odds_total)[0, 1]), 4), ' sd(total_pred - odds_total) =', round(float((x.total_pred - x.odds_total).std()), 3))

# features.json
f = json.load(open(E25 / 'features.json'))
bad = []
allc = set(t.columns)
for s, v in f.items():
    for k, cols in v.items():
        for col in cols:
            if col not in allc:
                bad.append((s, k, col))
            if 'close' in col.lower() or col.startswith('target_') or col in ('points_home', 'points_away'):
                bad.append((s, k, col, 'outcome/close-like'))
print('selected columns missing from the 11-01 training set or outcome-like:', bad[:20], len(bad))
print('odds columns in the lists:', sorted({c_ for v in f.values() for cols in v.values() for c_ in cols if c_.startswith('odds')}))
print('features selected per fold (ml/spread/total):', {s: [len(v[k]) for k in ('ml', 'spread', 'total')] for s, v in f.items()})
