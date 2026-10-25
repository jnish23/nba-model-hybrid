"""E30 step 1 - freeze the four signals BEFORE any forward game is scored (see PREREGISTRATION.md, E30).
For each test (family x target) fit, on history only, the opening-line residual on the standardised family regressors with season dummies
(OLS), and store everything needed to evaluate the signal on a new game: columns, mean and SD of each regressor, coefficients, the centring
constant, the signal's SD, the residual SD and the 75th percentile of |signal|. Primary fit: 2019-20..2025-26 regular seasons; the fit on all 13
seasons is stored for the sensitivity run. Writes reports/experiments/E30_forward_tracking/frozen/params.json; refuses to overwrite.
python -m experiments.e30_freeze"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import odds_clean as oc                                  # noqa: E402
from experiments import e28_screen as sc                            # noqa: E402
from experiments import e30_signals as sg                           # noqa: E402
from team.data_paths import latest_training_set                     # noqa: E402

OUT = sg.ROOT / 'reports' / 'experiments' / 'E30_forward_tracking' / 'frozen'
CODE = ['experiments/features_tracking.py', 'experiments/e30_signals.py', 'core/travel_builder.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def history_frame(games):
    ts = pd.read_csv(latest_training_set(), usecols=['gameId', 'odds_total', 'odds_home_spread', 'target_total_pts', 'target_point_diff'], low_memory=False)
    ts['gameId'] = ts['gameId'].astype(str).str.zfill(10)
    ts = ts.drop_duplicates('gameId')
    f = games.merge(ts, on='gameId', how='inner')
    bad = ~f['odds_total'].between(*oc.TOTAL_RANGE) | (f['odds_home_spread'].abs() > oc.MAX_ABS_SPREAD) | f['odds_total'].isna() | f['odds_home_spread'].isna()
    f = f[~bad & (f['gameId'].str[:3] == '002')].copy()
    f['y_margin'] = f['target_point_diff'] + f['odds_home_spread']
    f['y_total'] = f['target_total_pts'] - f['odds_total']
    return f.reset_index(drop=True)


def fit_spec(f, family, target, seasons):
    X = sg.family_X(f, family, target)
    y = f[f'y_{target}'].to_numpy(float)
    S = f['seasonYear'].to_numpy()
    ok = X.notna().all(axis=1).to_numpy() & np.isfinite(y) & np.isin(S, seasons)
    mu, sd = X[ok].mean(), X[ok].std()
    use = [c for c in X.columns if sd[c] > 0]
    Z = ((X[use] - mu[use]) / sd[use]).to_numpy(float)
    lv = [s for s in seasons if (S[ok] == s).any()]
    D = np.column_stack([sc.season_dummies(S[ok], lv), Z[ok]])
    beta, draws = sc.fit_boot(D, y[ok], np.zeros(ok.sum()), 'ols', f['gameDate'].to_numpy()[ok])
    b = beta[len(lv):]
    sig = Z[ok] @ b
    centre = float(sig.mean())
    resid = y[ok] - D @ beta
    return {'family': family, 'target': target, 'fit_seasons': lv, 'n': int(ok.sum()), 'columns': use,
            'mu': {c: float(mu[c]) for c in use}, 'sd': {c: float(sd[c]) for c in use}, 'beta': {c: float(v) for c, v in zip(use, b)},
            'beta_ci': {c: [float(np.quantile(draws[:, len(lv) + i], .025)), float(np.quantile(draws[:, len(lv) + i], .975))] for i, c in enumerate(use)},
            'centre': centre, 'signal_sd': float((sig - centre).std()), 'residual_sd': float(resid.std()),
            'abs_signal_p75': float(np.quantile(np.abs(sig - centre), .75)), 'y_sd': float(y[ok].std())}


def main():
    if (OUT / 'params.json').exists():
        raise SystemExit(f'{OUT} is already frozen; refusing to overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    games = sg.feature_games()
    hist = history_frame(games)
    params = {'created': pd.Timestamp.now().isoformat(timespec='seconds'), 'history_through': str(hist['gameDate'].max().date()),
              'training_set': latest_training_set().name, 'training_set_sha256': sha(latest_training_set()),
              'code_sha256': {c: sha(sg.ROOT / c) for c in CODE}, 'tests': {}}
    for tid, (fam, tg) in sg.TESTS.items():
        params['tests'][tid] = {'primary': fit_spec(hist, fam, tg, sg.FIT_PRIMARY), 'all_seasons': fit_spec(hist, fam, tg, sg.FIT_ALL)}
        p = params['tests'][tid]['primary']
        print(tid, fam, tg, 'n', p['n'], 'signal sd', round(p['signal_sd'], 3), 'resid sd', round(p['residual_sd'], 2), 'p75|s|', round(p['abs_signal_p75'], 3), flush=True)
    (OUT / 'params.json').write_text(json.dumps(params, indent=1))


if __name__ == '__main__':
    main()
