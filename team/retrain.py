"""
Scheduled walk-forward retrain of the market-aware model set (never deploys).

Why: nothing retrained the deployed set (`_with_odds_v6`: base models end 2024-01-16, calibrated through 2026-06-13). This module
retrains on everything up to the newest game and answers "is the new set at least as good as what is deployed?" before anyone
promotes it.

Steps (run_retrain):
  1. GATE   train the same pipeline on games up to `gate_days` before the last game (into a scratch directory) and score the
            held-out recent window with that gate model AND with the deployed set. The deployed set trained earlier, so the window
            is out-of-sample for both; if the deployed set's calibration block reaches into the window the comparison is flagged.
  2. CANDIDATE  train on all games under a new dated suffix (refit_on_all, calibration = last half season, tuner='grid' by default:
            deterministic, minutes, the search the walk-forward backtests validated). An existing suffix is refused.
  3. REPORT write reports/retrain/retrain_<date>.json with the gate metrics, the candidate's hashes and the verdict.

Promotion is a deliberate human step: change team/model_registry.DEPLOYED_SUFFIX to the candidate's suffix in a reviewed commit.
"""
import datetime as dt
import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from team.market_models import predict_markets
from team.model_registry import DEPLOYED_SUFFIX, MODEL_DIR, load_bundle, read_manifest
from team.train_baseline import BaselineModel, CALIB_HOLDOUT_DAYS

GATE_DAYS = 120
GATE_MIN_GAMES = 150
# The candidate must not be worse than the deployed set by more than this on any gate metric (noise allowance on ~600 games)
TOLERANCE = {'ml_logloss': 0.004, 'spread_mae': 0.15, 'total_mae': 0.15}
# When the deployed set's calibrator was fitted on blocks that reach into the gate window, its calibrated log loss is in-sample on that
# window; the comparison then uses the calibration-free log loss of the raw classifier.
TOLERANCE_RAW = {'ml_logloss_raw': 0.004, 'spread_mae': 0.15, 'total_mae': 0.15}


def window_metrics(preds, frame):
    """Primary metrics of a prediction frame (ml_prob, spread_pred, total_pred) against the games' results."""
    y = frame['target_home_win'].to_numpy(float)
    p = np.clip(preds['ml_prob'].to_numpy(float), 1e-6, 1 - 1e-6)
    out = {'games': int(len(frame)),
           'ml_logloss': float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
           'ml_brier': float(brier_score_loss(y, p))}
    if 'ml_prob_raw' in preds:
        pr = np.clip(preds['ml_prob_raw'].to_numpy(float), 1e-6, 1 - 1e-6)
        out['ml_logloss_raw'] = float(-(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean())
    for tag, col, tgt in (('spread', 'spread_pred', 'target_point_diff'), ('total', 'total_pred', 'target_total_pts')):
        e = frame[tgt].to_numpy(float) - preds[col].to_numpy(float)
        ok = np.isfinite(e)
        out[f'{tag}_mae'] = float(np.abs(e[ok]).mean())
        out[f'{tag}_bias'] = float(e[ok].mean())
        out[f'{tag}_n'] = int(ok.sum())
    return out


def verdict(candidate, incumbent, tolerance=TOLERANCE):
    """{'ok': bool, 'worse': {metric: gap}} - the candidate passes unless it is worse than the incumbent by more than the tolerance."""
    worse = {m: float(candidate[m] - incumbent[m]) for m, tol in tolerance.items() if candidate[m] - incumbent[m] > tol}
    return {'ok': not worse, 'worse': worse}


def run_retrain(data_path, model_dir=MODEL_DIR, out_dir='reports/retrain', tuner='grid', time_budget=600,
                holdout_days=CALIB_HOLDOUT_DAYS, gate_days=GATE_DAYS, tag=None, incumbent_suffix=DEPLOYED_SUFFIX, today=None,
                incumbent_dir=None):
    today = today or dt.date.today()
    tag = tag or f'{today:%Y%m%d}'
    candidate_suffix = f'_with_odds_cand_{tag}'
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    incumbent_dir = Path(incumbent_dir) if incumbent_dir else model_dir
    full = BaselineModel(data_path)
    full.load_data(use_odds=True)
    games = full.df.sort_values('gameDate').reset_index(drop=True)
    last = games['gameDate'].max()
    report = {'run_at': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'training_set': Path(data_path).name, 'games': int(len(games)),
              'last_game': str(last.date()), 'tuner': tuner, 'candidate_suffix': candidate_suffix, 'incumbent_suffix': incumbent_suffix}

    # 1. gate: the same pipeline, one window earlier
    cut = last - pd.Timedelta(days=gate_days)
    window = games[games['gameDate'] > cut]
    if len(window) >= GATE_MIN_GAMES and (games['gameDate'] <= cut).sum() >= 2000:
        scratch = Path(tempfile.mkdtemp(prefix='retrain_gate_'))
        try:
            csv = scratch / 'gate_training_set.csv'
            pd.read_csv(data_path, low_memory=False).assign(_d=lambda d: pd.to_datetime(d['gameDate'])).query('_d <= @cut') \
                .drop(columns='_d').to_csv(csv, index=False)
            BaselineModel(csv, model_dir=str(scratch / 'saved')).train_and_save_all(
                use_odds=True, save_suffix='_gate', time_budget=time_budget, calib_holdout_days=holdout_days, tuner=tuner)
            gate = load_bundle('_gate', scratch / 'saved')
            report['gate'] = {'window': f'{window["gameDate"].min():%Y-%m-%d}..{last:%Y-%m-%d}', 'candidate': window_metrics(predict_markets(gate, window), window)}
            try:
                inc = load_bundle(incumbent_suffix, incumbent_dir)
                report['gate']['incumbent'] = window_metrics(predict_markets(inc, window), window)
                calib_end = pd.Timestamp(inc.calib_meta.get('calib_date_range', {}).get('end', 'NaT'))
                saw = bool(pd.notna(calib_end) and calib_end > cut)       # its calibrator/sigma were fitted on part of the window
                report['gate']['incumbent_calibrated_on_window'] = saw
                report['gate']['verdict'] = verdict(report['gate']['candidate'], report['gate']['incumbent'],
                                                    TOLERANCE_RAW if saw else TOLERANCE)
            except FileNotFoundError as e:
                report['gate']['incumbent'] = None
                report['gate']['note'] = f'no incumbent to compare with: {e}'
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    else:
        report['gate'] = {'skipped': f'fewer than {GATE_MIN_GAMES} games in the last {gate_days} days or <2000 before them'}

    # 2. candidate on everything (a new suffix: an existing one is refused)
    BaselineModel(data_path, model_dir=str(model_dir)).train_and_save_all(
        use_odds=True, save_suffix=candidate_suffix, time_budget=time_budget, calib_holdout_days=holdout_days, tuner=tuner)
    manifest = read_manifest(candidate_suffix, model_dir)
    report['candidate'] = {k: manifest[k] for k in ('model_set_hash', 'training_set_hash', 'trained_through', 'calib_holdout', 'created_at')}

    # 3. report
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'retrain_{tag}.json'
    path.write_text(json.dumps(report, indent=2, default=str))
    ok = report['gate'].get('verdict', {}).get('ok')
    logging.info(f"Candidate {candidate_suffix} trained through {report['candidate']['trained_through']}; gate "
                 f"{'passed' if ok else 'FAILED' if ok is False else 'not run'}; report {path}. NOT deployed: promotion = set "
                 f"DEPLOYED_SUFFIX in team/model_registry.py to '{candidate_suffix}' in a reviewed commit.")
    return report
