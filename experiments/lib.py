"""Walk-forward pipeline shared by every experiment.

BASELINE = the post-audit deployed configuration: opening odds as inputs, two-stage feature selection that stops at
the FLAML boundary, LightGBM hyperparameters tuned INSIDE each fold on a time-ordered validation block, isotonic
calibration and residual sigma fitted on the final 20% of the training window.

Folds: test season s in TEST_SEASONS, training = all earlier seasons (sorted by date); calibration = the last 20% of them,
capped at CALIB_MAX_ROWS (about one season, as in team/train_baseline.py's 365-day block), model_train = the rest; for tuning,
model_train is split again 80/20 in time (fit block / validation block). After the calibrators and sigma are fitted on the
calibration block, the tuned base models are refit on model_train + calibration (refit_on_all, the production default), so the
test season is predicted by a model that has seen everything up to its first game. Runs made before this (the artifacts under
reports/experiments/ from 2026-10) used the uncapped 80/20 split and no refit: split.json beside a run records which scheme made it
(run_split_params), and runs without one are read as the old scheme.
Everything is seeded (SEED) and LightGBM runs deterministically, so a rerun reproduces the numbers exactly.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from team.train_baseline import BaselineModel, FLAML_TRAIN_FRAC          # noqa: E402
from team.data_paths import latest_training_set                           # noqa: E402

SEED = 42
CALIB_MAX_ROWS = 1300            # about one NBA regular season of games
LEGACY_SPLIT = {'calib_cap': None, 'refit_on_all': False}      # how every run saved before the audit fix was made
TEST_SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
OUT = ROOT / 'reports' / 'experiments'
TARGETS = {'ml': 'target_home_win', 'spread': 'target_point_diff', 'total': 'target_total_pts'}

# GRID / make_est / raw_predict live in team/tuning.py (the final-model trainer uses the same search)
from team.tuning import GRID, make_est, raw_predict          # noqa: E402,F401


def fit_tuned(kind, X, y, n_fit, init=None, grid=GRID):
    """Pick the best grid config on rows [n_fit:] (fit on rows [:n_fit]); refit on all rows. Returns (est, info)."""
    Xa, ya, Xv, yv = X.iloc[:n_fit], y.iloc[:n_fit], X.iloc[n_fit:], y.iloc[n_fit:]
    ia = None if init is None else init[:n_fit]
    iv = None if init is None else init[n_fit:]
    best, best_score, scores = None, np.inf, []
    for i, cfg in enumerate(grid):
        est = make_est(kind, cfg)
        est.fit(Xa, ya, init_score=ia)
        r = raw_predict(est, kind, Xv, iv)
        if kind == 'clf':
            p = np.clip(expit(r), 1e-6, 1 - 1e-6)
            score = float(-(yv.to_numpy() * np.log(p) + (1 - yv.to_numpy()) * np.log(1 - p)).mean())
        else:
            score = float(np.mean((yv.to_numpy() - r) ** 2))
        scores.append(score)
        if score < best_score:
            best, best_score = i, score
    final = make_est(kind, grid[best])
    final.fit(X, y, init_score=init)
    return final, {'cfg': best, 'val_score': best_score, 'val_scores': scores}


def calib_rows(n_train, calib_cap=CALIB_MAX_ROWS):
    """Rows of the calibration block: the last 20% of the training window, at most `calib_cap` (None: no cap, the pre-audit rule)."""
    n_cal = n_train - int(n_train * 0.8)
    return n_cal if calib_cap is None else min(n_cal, calib_cap)


def run_split_params(name):
    """{'calib_cap', 'refit_on_all'} the run `name` was made with (LEGACY_SPLIT for runs saved before split.json existed)."""
    path = OUT / name / 'split.json'
    return {**LEGACY_SPLIT, **{k: v for k, v in json.loads(path.read_text()).items() if k in LEGACY_SPLIT}} if path.exists() else dict(LEGACY_SPLIT)


def split_fold(df, test_season, calib_cap=CALIB_MAX_ROWS):
    train = df[df['seasonYear'] < test_season].sort_values('gameDate', kind='stable').reset_index(drop=True)
    test = df[df['seasonYear'] == test_season].reset_index(drop=True)
    n_mt = len(train) - calib_rows(len(train), calib_cap)
    mt, calib = train.iloc[:n_mt].reset_index(drop=True), train.iloc[n_mt:].reset_index(drop=True)
    n_fl = int(len(mt) * FLAML_TRAIN_FRAC)
    return mt, calib, test, n_fl


def load_games(csv=None):
    """Training-set frame (sanitised exactly as BaselineModel does) plus closing lines from odds_clean."""
    m = BaselineModel(csv or latest_training_set())
    m.load_data(use_odds=True)
    df = m.df
    df['gameId'] = df['gameId'].astype(str).str.zfill(10)
    from core.odds_clean import load_clean_odds
    o = load_clean_odds(ROOT / 'data' / 'nba_stats.db')
    o['gameId'] = o['gameId'].astype(str).str.zfill(10)
    keep = ['gameId', 'close_home_spread', 'close_total', 'close_home_prob', 'open_home_prob', 'close_source']
    df = df.merge(o[keep], on='gameId', how='left')
    return m, df


DEFAULT_KEEP = ['gameId', 'gameDate', 'seasonYear', 'target_home_win', 'target_point_diff', 'target_total_pts',
                'odds_home_ml', 'odds_away_ml', 'odds_home_spread', 'odds_total', 'close_home_spread', 'close_total',
                'close_home_prob', 'open_home_prob']


def run_walk_forward(name, csv=None, features_from=None, hooks=None, seasons=TEST_SEASONS, save=True, df_override=None, selection_seed=42,
                     calib_cap=CALIB_MAX_ROWS, refit_on_all=True):
    """
    Run all folds. `hooks` (dict, optional) lets an experiment change ONE thing:
      'init_scores'(kind, frame, fold_ctx) -> array|None   market-residual init_score (E1)
      'extra_columns' - list of extra columns to keep in the outputs
    Returns (test_predictions, calib_predictions, info). Selected features are cached per fold in OUT/name/features.json
    and reused from `features_from` (another run's name) so that an experiment that does not touch features uses
    exactly the baseline feature sets.
    calib_cap / refit_on_all: see the module docstring (calib_cap=None, refit_on_all=False reproduces the pre-audit runs).
    """
    hooks = hooks or {}
    t0 = time.time()
    m, df = load_games(csv) if df_override is None else df_override
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    feats_path = out_dir / 'features.json'
    cached = {}
    if features_from:
        cached = json.loads((OUT / features_from / 'features.json').read_text())
    elif feats_path.exists():
        cached = json.loads(feats_path.read_text())
    chosen, tests, calibs, info = {}, [], [], {}
    keep = DEFAULT_KEEP + hooks.get('extra_columns', [])
    for s in seasons:
        mt, calib, test, n_fl = split_fold(df, s, calib_cap=calib_cap)
        ctx = {'season': s, 'n_fl': n_fl, 'mt': mt, 'calib': calib, 'test': test}
        fold_feats = cached.get(s)
        if fold_feats is None:
            sel_df = mt.iloc[:n_fl]
            fold_feats = {k: m._select_features_from_subset(sel_df, t, seed=selection_seed) for k, t in TARGETS.items()}
        chosen[s] = fold_feats
        res_test = test[[c for c in keep if c in test]].copy()
        res_cal = calib[[c for c in keep if c in calib]].copy()
        res_cal['fold'] = s
        fold_info = {}
        for kind_name, target in TARGETS.items():
            feats = fold_feats[kind_name]
            kind = 'clf' if kind_name == 'ml' else 'reg'
            init = None
            init_cal = init_test = None
            if 'init_scores' in hooks:
                init_all = hooks['init_scores'](kind_name, {'mt': mt, 'calib': calib, 'test': test}, ctx)
                if init_all is not None:
                    init, init_cal, init_test = init_all
            y_mt = hooks['target'](kind_name, mt, ctx) if 'target' in hooks else mt[target]
            est, finfo = fit_tuned(kind, mt[feats], y_mt, n_fl, init=init)
            fold_info[kind_name] = {**finfo, 'n_features': len(feats)}
            r_cal = raw_predict(est, kind, calib[feats], init_cal)
            r_test = raw_predict(est, kind, test[feats], init_test)
            if refit_on_all:
                # calibrators / sigma stay fitted on the model_train model's out-of-sample calibration predictions; the test season is
                # predicted by the same configuration refit on model_train + calibration (BaselineModel.train_and_save_all, refit_on_all)
                tv = pd.concat([mt, calib], ignore_index=True)
                y_tv = hooks['target'](kind_name, tv, ctx) if 'target' in hooks else tv[target]
                init_tv = None if init is None else np.concatenate([init, init_cal])
                final = make_est(kind, GRID[finfo['cfg']])
                final.fit(tv[feats], y_tv, init_score=init_tv)
                r_test = raw_predict(final, kind, test[feats], init_test)
                fold_info[kind_name]['refit_rows'] = len(tv)
            if kind_name == 'ml':
                p_cal, p_test = expit(r_cal), expit(r_test)
                iso = IsotonicRegression(out_of_bounds='clip').fit(p_cal, calib[target])
                res_cal['ml_prob_raw'], res_test['ml_prob_raw'] = p_cal, p_test
                res_cal['ml_prob'], res_test['ml_prob'] = iso.predict(p_cal), iso.predict(p_test)
            else:
                base_cal = hooks['pred_offset'](kind_name, calib, ctx) if 'pred_offset' in hooks else 0.0
                base_test = hooks['pred_offset'](kind_name, test, ctx) if 'pred_offset' in hooks else 0.0
                pc, pt = r_cal + base_cal, r_test + base_test
                res_cal[f'{kind_name}_pred'], res_test[f'{kind_name}_pred'] = pc, pt
                resid = calib[target].to_numpy(float) - pc
                ok = np.isfinite(resid)
                res_test[f'sigma_{kind_name}'] = float(np.std(resid[ok]))
                res_cal[f'sigma_{kind_name}'] = float(np.std(resid[ok]))
        tests.append(res_test)
        calibs.append(res_cal)
        info[s] = fold_info
        logging.info(f"[{name}] fold {s} done ({time.time()-t0:.0f}s)")
    test_df = pd.concat(tests, ignore_index=True)
    calib_df = pd.concat(calibs, ignore_index=True)
    if save:
        feats_path.write_text(json.dumps(chosen))
        test_df.to_csv(out_dir / 'predictions.csv', index=False)
        calib_df.to_csv(out_dir / 'calibration_set_predictions.csv', index=False)
        (out_dir / 'fold_info.json').write_text(json.dumps(info, indent=1))
        (out_dir / 'split.json').write_text(json.dumps({'calib_cap': calib_cap, 'refit_on_all': refit_on_all}))
    return test_df, calib_df, info


def label_calib_folds(df, calib_df, seasons=TEST_SEASONS, calib_cap='auto'):
    """Fold (test season) of each calibration-set row. Runs saved by run_walk_forward carry a `fold` column; older ones are in fold order,
    so the labels are rebuilt from the calibration-block sizes (calib_cap='auto' tries the capped block, then the pre-audit 20% block)."""
    if 'fold' in calib_df:
        return calib_df['fold'].to_numpy()
    for cap in ([CALIB_MAX_ROWS, None] if calib_cap == 'auto' else [calib_cap]):
        labels = []
        for s in seasons:
            labels += [s] * calib_rows(int((df['seasonYear'] < s).sum()), cap)
        if len(labels) == len(calib_df):
            return np.array(labels)
    raise AssertionError(f'calibration set has {len(calib_df)} rows, which fits no known fold layout')


def load_run(name):
    d = OUT / name
    t = pd.read_csv(d / 'predictions.csv', dtype={'gameId': str}, parse_dates=['gameDate'])
    c = pd.read_csv(d / 'calibration_set_predictions.csv', dtype={'gameId': str}, parse_dates=['gameDate'])
    return t, c
