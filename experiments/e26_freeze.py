"""E26 step 1 - freeze everything the forward test needs, BEFORE any forward game is scored (see PREREGISTRATION.md, E26).
    beta per market   edge slope of the deployed-form predictions (E13: E1 spread, E3 stack ML, baseline total) vs the opening line
    k (totals blend)  no-odds edge weight from E25_no_odds
    sigma             residual SDs used only for the descriptive betting readout
    no-odds total     a final LightGBM model on all seasons through 2025-26 (harness feature selection and grid tuning, refit on all rows)
Writes reports/experiments/E26_forward_shrink/frozen/. Refuses to overwrite. python -m experiments.e26_freeze"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import e25_decorrelation as d25                        # noqa: E402
from experiments.e01_residual import devig_open_prob                     # noqa: E402
from experiments.e25_regen import load_games_no_odds                    # noqa: E402
from experiments.lib import OUT, ROOT, TEST_SEASONS, fit_tuned          # noqa: E402
from team.data_paths import latest_training_set                         # noqa: E402
from team.train_baseline import FLAML_TRAIN_FRAC                        # noqa: E402
from team.tuning import GRID, make_est                                  # noqa: E402

FROZEN = OUT / 'E26_forward_shrink' / 'frozen'
DISC = d25.DISC
# the scorer rebuilds 2026-27 feature rows with this code; any change to these files after the freeze must be reported
FEATURE_CODE = ['team/game_features_v2.py', 'core/travel_builder.py', 'core/injury_report_features.py', 'team/train_baseline.py',
                'scripts/generate_training_set.py', 'core/odds_clean.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deployed_form_frame():
    """E13 predictions with the columns d25.frame() expects under the tag 'dep'."""
    p = pd.read_csv(OUT / 'E13_combined_winners' / 'predictions.csv', dtype={'gameId': str})
    p['gameDate'] = pd.to_datetime(p['gameDate'])
    p = p.rename(columns={'spread_pred': 'dep_spread_pred', 'total_pred': 'dep_total_pred', 'ml_prob': 'dep_ml_raw'})
    p['L_ml'] = d25.logit(np.clip(devig_open_prob(p), 1e-4, 1 - 1e-4))
    return p


def fit_beta(g, market, tag):
    """beta on all seasons and on discovery only, each with a day-bootstrap interval."""
    out = {}
    for lab, gg in (('all_seasons', g), ('discovery', g[g['seasonYear'].isin(DISC)])):
        gg = gg.reset_index(drop=True)
        e, r, L, _ = d25.frame(gg, market, tag)
        ok = np.isfinite(e) & np.isfinite(r) & np.isfinite(L)
        gg, e, r, L = gg[ok].reset_index(drop=True), e[ok], r[ok], L[ok]
        boots = [d25.beta(market, e[i], r[i], L[i]) for i in d25.day_draws(gg['gameDate'])]
        out[lab] = {'beta': d25.beta(market, e, r, L), 'ci': d25.ci(boots), 'n': int(len(gg))}
    return out


def main():
    FROZEN.mkdir(parents=True, exist_ok=True)
    if (FROZEN / 'params.json').exists():
        raise SystemExit(f'{FROZEN} is already frozen; refusing to overwrite')
    params = {'created': pd.Timestamp.now().isoformat(timespec='seconds'), 'history_seasons': TEST_SEASONS}

    dep = deployed_form_frame()
    params['beta'] = {mk: fit_beta(dep, mk, 'dep') for mk in ('spread', 'total', 'ml')}
    sig = {}
    for mk in ('spread', 'total'):
        e, r, L, _ = d25.frame(dep, mk, 'dep')
        b = params['beta'][mk]['all_seasons']['beta']
        res = r - b * e
        sig[mk] = float(np.nanstd(res))
    params['sigma_shrunk'] = sig

    nd = d25.load()
    g = d25.common_rows(nd, 'total')
    e, r, L, _ = d25.frame(g, 'total', 'no_odds')
    k_all = float((e * r).sum() / (e * e).sum())
    ed, rd, Ld, _ = d25.frame(g[g['seasonYear'].isin(DISC)], 'total', 'no_odds')
    params['k_total_blend'] = {'all_seasons': k_all, 'discovery': float((ed * rd).sum() / (ed * ed).sum()), 'n': int(len(g))}
    params['sigma_blend_total'] = float(np.std(r - k_all * e))

    csv = latest_training_set()
    m, df = load_games_no_odds(csv)
    train = df[df['seasonYear'] <= TEST_SEASONS[-1]].sort_values('gameDate', kind='stable').reset_index(drop=True)
    mt = train.iloc[:int(len(train) * 0.8)].reset_index(drop=True)
    n_fl = int(len(mt) * FLAML_TRAIN_FRAC)
    feats = m._select_features_from_subset(mt.iloc[:n_fl], 'target_total_pts', seed=42)
    _, info = fit_tuned('reg', mt[feats], mt['target_total_pts'], n_fl)
    model = make_est('reg', GRID[info['cfg']]).fit(train[feats], train['target_total_pts'])
    joblib.dump(model, FROZEN / 'no_odds_total.joblib')
    params['no_odds_total'] = {'features': feats, 'grid_cfg': int(info['cfg']), 'trained_rows': int(len(train)),
                               'trained_through': str(train['gameDate'].max().date()), 'training_set': str(Path(csv).name),
                               'training_set_sha256': sha(csv), 'model_sha256': sha(FROZEN / 'no_odds_total.joblib')}
    params['feature_code_sha256'] = {f: sha(ROOT / f) for f in FEATURE_CODE}
    from team.model_registry import DEPLOYED_SUFFIX
    params['deployed_suffix'] = DEPLOYED_SUFFIX
    (FROZEN / 'params.json').write_text(json.dumps(params, indent=1))
    print(json.dumps({k: v for k, v in params.items() if k not in ('no_odds_total', 'feature_code_sha256')}, indent=1))
    print('no-odds total:', {k: v for k, v in params['no_odds_total'].items() if k != 'features'}, len(feats), 'features')


if __name__ == '__main__':
    main()
