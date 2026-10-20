"""
Checks that a saved prediction log (predictions_log_v2 rows / predictions/predictions_<season>.csv) belongs to the model files on disk.

Two independent checks:
  provenance_report  every row carries a model_set_hash that matches a manifest in models/saved, and the files still hash to it
  rescore_gap        re-score the logged games with the model set and compare with what was logged (probability / points)

The 2025-26 out-of-sample log failed the second one: it held 32 distinct ml_prob values (an isotonic step function) while the
saved calibrator was a smooth margin stack that re-scores the same games to 1,321 distinct values (max gap 0.22 probability,
10.6 spread points, 12.4 total points), because the artifacts had been overwritten under the same suffix with no link to the log.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from team.model_registry import MODEL_DIR, model_set_hash


def known_hashes(model_dir=MODEL_DIR):
    """{model_set_hash: suffix} for every manifest in the directory."""
    out = {}
    for p in Path(model_dir).glob('model_manifest*.json'):
        m = json.loads(p.read_text())
        out[m['model_set_hash']] = m['suffix']
    return out


def provenance_report(log, model_dir=MODEL_DIR):
    """
    log: DataFrame with model_set_hash / training_set_hash columns (absent columns count as missing).
    Returns a dict: rows, rows_without_hash, hashes {hash: {'rows', 'suffix', 'files_unchanged'}}, unknown_hashes (no manifest).
    """
    hashes = log['model_set_hash'] if 'model_set_hash' in log else pd.Series([None] * len(log), index=log.index)
    missing = hashes.isna() | hashes.astype(str).str.strip().isin(['', 'None', 'nan'])
    known = known_hashes(model_dir)
    detail, unknown = {}, []
    for h, n in hashes[~missing].value_counts().items():
        suffix = known.get(h)
        if suffix is None:
            unknown.append(h)
        detail[h] = {'rows': int(n), 'suffix': suffix, 'files_unchanged': bool(suffix and model_set_hash(suffix, model_dir) == h)}
    return {'rows': int(len(log)), 'rows_without_hash': int(missing.sum()), 'hashes': detail, 'unknown_hashes': unknown}


def rescore_gap(log, bundle, frame, predict_markets):
    """
    Compare the logged ml_prob / spread_pred / total_pred with a fresh scoring of the same games.
    `frame` holds the model inputs and target_home_win (a training-set slice with a gameId column); `predict_markets` is
    team.market_models.predict_markets. Returns distinct-value counts, max abs gaps and log losses.
    """
    f = frame.assign(gameId=frame['gameId'].astype(str).str.zfill(10))
    fresh = predict_markets(bundle, f)
    fresh['gameId'] = f['gameId'].to_numpy()
    j = log.assign(gameId=log['game_id'].astype(str).str.zfill(10)).merge(
        fresh.merge(f[['gameId', 'target_home_win']], on='gameId'), on='gameId', suffixes=('_log', '_new'))
    y = j['target_home_win'].to_numpy(float)

    def ll(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return {
        'games': int(len(j)),
        'distinct_ml_logged': int(j['ml_prob_log'].round(6).nunique()), 'distinct_ml_rescored': int(j['ml_prob_new'].round(6).nunique()),
        'max_gap_ml': float((j['ml_prob_log'] - j['ml_prob_new']).abs().max()),
        'max_gap_spread': float((j['spread_pred_log'] - j['spread_pred_new']).abs().max()),
        'max_gap_total': float((j['total_pred_log'] - j['total_pred_new']).abs().max()),
        'logloss_logged': ll(j['ml_prob_log']), 'logloss_rescored': ll(j['ml_prob_new']),
    }
