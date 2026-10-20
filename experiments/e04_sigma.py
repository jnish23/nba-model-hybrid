"""E4 - heteroskedastic sigma (see PREREGISTRATION.md). Baseline point predictions are unchanged; only the sigma used to
turn (pred - line) into a cover probability differs: global calibration-set std (baseline) vs a feature-dependent sigma
from a second model on out-of-fold |residuals| (4 forward-chaining blocks inside model-train)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import decide, metrics as M                                             # noqa: E402
from experiments.lib import GRID, OUT, SEED, TEST_SEASONS, label_calib_folds, load_games, load_run, make_est, split_fold   # noqa: E402

NAME = 'E04_sigma'
SIGMA_FEATURES = [
    'off_fieldGoalsAttempted_20g_home', 'off_fieldGoalsAttempted_20g_away', 'off_fieldGoalsAttempted_5g_home', 'off_fieldGoalsAttempted_5g_away',
    'off_percentageFieldGoalsAttempted3pt_20g_home', 'off_percentageFieldGoalsAttempted3pt_20g_away',
    'off_threePointersAttempted_20g_home', 'off_threePointersAttempted_20g_away',
    'inj_n_out_home', 'inj_n_out_away', 'inj_missing_home', 'inj_missing_away', 'inj_lost_home', 'inj_lost_away',
    'odds_home_spread', 'odds_total', 'days_rest_home', 'days_rest_away', 'season_game_number_home']
SIGMA_CFG = dict(learning_rate=0.05, n_estimators=150, num_leaves=4, min_child_samples=100, reg_lambda=20.0, colsample_bytree=0.8)


def oof_abs_resid(mt, feats, target, cfg_idx, kind_target, n_parts=5):
    """Forward-chaining out-of-fold residuals inside model-train: part k+1 predicted by a model fit on parts 1..k."""
    parts = np.array_split(np.arange(len(mt)), n_parts)
    idx, res = [], []
    for k in range(1, n_parts):
        tr, te = np.concatenate(parts[:k]), parts[k]
        est = make_est('reg', GRID[cfg_idx])
        est.fit(mt.iloc[tr][feats], mt.iloc[tr][target])
        res.append(np.abs(mt.iloc[te][target].to_numpy(float) - est.predict(mt.iloc[te][feats])))
        idx.append(te)
    return np.concatenate(idx), np.concatenate(res)


def cover_frame(df, market, sigma):
    pred = df[f'{market}_pred'].to_numpy(float)
    if market == 'spread':
        line, margin = df['odds_home_spread'].to_numpy(float), df['target_point_diff'].to_numpy(float)
        pe, res = pred + line, margin + line
    else:
        line, tot = df['odds_total'].to_numpy(float), df['target_total_pts'].to_numpy(float)
        pe, res = pred - line, tot - line
    p = np.clip(norm.cdf(pe / sigma), 1e-6, 1 - 1e-6)
    y = (res > 0).astype(float)
    ok = np.isfinite(p) & np.isfinite(res) & (res != 0)
    return pd.DataFrame({'p': p, 'y': y, 'ok': ok})


def main():
    base, cal = load_run('baseline')
    m, df = load_games()
    cal['fold'] = label_calib_folds(df, cal)
    feats = json.loads((OUT / 'baseline' / 'features.json').read_text())
    info = json.loads((OUT / 'baseline' / 'fold_info.json').read_text())
    sf = [c for c in SIGMA_FEATURES if c in df.columns]
    sig = {mk: pd.Series(np.nan, index=base.index) for mk in ('spread', 'total')}
    sig_rescaled = {mk: pd.Series(np.nan, index=base.index) for mk in ('spread', 'total')}
    for s in TEST_SEASONS:
        mt, calib, test, n_fl = split_fold(df, s)
        tm = (base['seasonYear'] == s).to_numpy()
        te_idx = base.index[tm]
        for mk in ('spread', 'total'):
            target = 'target_point_diff' if mk == 'spread' else 'target_total_pts'
            idx, res = oof_abs_resid(mt, feats[s][mk], target, info[s][mk]['cfg'], mk)
            g = lgb.LGBMRegressor(**SIGMA_CFG, random_state=SEED, deterministic=True, force_row_wise=True, verbosity=-1)
            g.fit(mt.iloc[idx][sf], res)
            e_abs_test = np.clip(g.predict(test[sf]), 1.0, None)
            sig[mk].loc[te_idx] = e_abs_test * np.sqrt(np.pi / 2)             # registered formula
            e_abs_cal = np.clip(g.predict(calib[sf]), 1.0, None)
            glob = float(base.loc[tm, f'sigma_{mk}'].iloc[0])
            sig_rescaled[mk].loc[te_idx] = e_abs_test / e_abs_cal.mean() * glob   # diagnostic only (not the registered rule)
    rows, lines, decisions = [], [f'# {NAME}\n'], {}
    for mk in ('spread', 'total'):
        g0 = cover_frame(base, mk, base[f'sigma_{mk}'].to_numpy(float))
        g1 = cover_frame(base, mk, sig[mk].to_numpy(float))
        g2 = cover_frame(base, mk, sig_rescaled[mk].to_numpy(float))
        ok = g0['ok'] & g1['ok'] & g2['ok']
        def loss(g):
            return pd.DataFrame({'cl': -(g['y'] * np.log(g['p']) + (1 - g['y']) * np.log(1 - g['p']))}).assign(
                seasonYear=base['seasonYear'].values, gameDate=base['gameDate'].values).set_index(base['gameId'].values)[ok.values]
        r = M.paired(loss(g0), loss(g1), 'cl')
        r2 = M.paired(loss(g0), loss(g2), 'cl')
        ece0, ece1, ece2 = (M.ece(g['y'][ok], g['p'][ok]) for g in (g0, g1, g2))
        dec = bool(decide.mc(r) and ece1 <= ece0 + 0.005)
        decisions[mk] = dec
        rows.append({'market': mk, 'global-sigma loglik': round(-float(loss(g0)['cl'].mean()), 5), 'hetero loglik': round(-float(loss(g1)['cl'].mean()), 5),
                     'diff (loss, exp-base)': round(r['diff'], 5), 'CI': f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]", 'seasons better': f"{r['seasons_improved']}/7",
                     'ECE global': round(ece0, 4), 'ECE hetero': round(ece1, 4), 'KEEP': dec,
                     'mean sigma global/hetero': f"{base[f'sigma_{mk}'].mean():.2f}/{sig[mk].mean():.2f}"})
        rows.append({'market': mk + ' (diagnostic: rescaled to global mean, NOT the registered rule)', 'hetero loglik': round(-float(loss(g2)['cl'].mean()), 5),
                     'diff (loss, exp-base)': round(r2['diff'], 5), 'CI': f"[{r2['ci_lo']:+.5f}, {r2['ci_hi']:+.5f}]",
                     'seasons better': f"{r2['seasons_improved']}/7", 'ECE hetero': round(ece2, 4)})
        lines.append(f"- {mk} per-season loss diff (registered): {r['per_season']}")
        # does sigma carry information at all? correlation of predicted vs realised |residual|
        act = np.abs((base['target_point_diff'] - base['spread_pred']) if mk == 'spread' else (base['target_total_pts'] - base['total_pred']))
        lines.append(f"- {mk}: corr(sigma_i, |residual|) = {np.corrcoef(sig[mk][ok.values], act[ok.values])[0, 1]:.3f}")
    lines += ['', pd.DataFrame(rows).to_markdown(index=False), '', '## Decision\n'] + [f'- {k}: **{"KEEP" if v else "REJECT"}**' for k, v in decisions.items()]
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    out = base[['gameId', 'seasonYear']].copy()
    out['sigma_spread_h'], out['sigma_total_h'] = sig['spread'].values, sig['total'].values
    out.to_csv(OUT / NAME / 'sigma.csv', index=False)
    (OUT / NAME / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'decision.json').write_text(json.dumps({'decision': decisions}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
