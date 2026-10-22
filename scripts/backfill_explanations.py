"""
Explanations (team/explain.py) for games that were predicted without them: the 2025-26 out-of-sample season in predictions_log_v2 and any
other game whose model set can be named and whose features are in the training set.

Each season's rows are explained with the model set that was out-of-sample for it (model_version 'oos_backtest_<season>' ->
models/saved/*_oos_<season>), applied to that game's row of the current training set (loaded the way scripts/generate_season_predictions.py
loads it). The prediction log may hold numbers from an EARLIER run of the same set (the log is only rewritten when predictions are
regenerated), so the explained model number can differ from the logged one; the dashboard shows both when they differ. The breakdown
always adds up to the number it explains.

  python scripts/backfill_explanations.py                       # every season not yet explained, into data/nba_stats.db
  python scripts/backfill_explanations.py --db copy.db          # into another database (use a copy to try it out)
  python scripts/backfill_explanations.py --season 2025-26 --refresh

Games of the live model set (model_version v1.0.0_hybrid) are explained by the daily run (scripts/run_inference_v2.py) instead.
"""
import argparse
import logging
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from team.data_paths import latest_training_set
from team.explain import build_reference, explain_markets, load_explanations, load_oos_history, save_explanations
from team.model_registry import MODEL_DIR, load_bundle

OOS_VERSION = re.compile(r'^oos_backtest_(\d{4}-\d{2})$')


def games_to_explain(conn, explained_ids, season=None, refresh=False):
    """predictions_log_v2 rows (one per game) of out-of-sample seasons that still need explanations."""
    log = pd.read_sql("SELECT * FROM predictions_log_v2", conn)
    log['game_id'] = log['game_id'].astype(str).str.zfill(10)
    log['season'] = log['model_version'].map(lambda v: (OOS_VERSION.match(v or '') or [None, None])[1])
    log = log[log['season'].notna()]
    if season:
        log = log[log['season'] == season]
    if not refresh:
        log = log[~log['game_id'].isin(explained_ids)]
    return log.reset_index(drop=True)


def explain_log_rows(log, frame_all, season, model_dir, history, reference):
    """Explanation rows for the logged games of one season. Returns (rows, how many games' re-scored numbers differ from the log)."""
    bundle = load_bundle(f'_oos_{season}', model_dir)
    frame = frame_all[frame_all['gameId'].isin(log['game_id'])].reset_index(drop=True)
    lines = log.set_index('game_id').reindex(frame['gameId'])
    current = pd.DataFrame({'spread_line': lines['odds_home_spread'].to_numpy(float), 'total_line': lines['odds_total'].to_numpy(float),
                            'home_ml': lines['odds_home_ml'].to_numpy(float)})
    pdate = lines['prediction_date'].iloc[0] if len(lines) else None
    expl = explain_markets(bundle, frame, current, reference=reference, history=history, prediction_date=pdate)
    expl['prediction_date'] = expl['game_id'].map(lines['prediction_date'])
    return expl, frame


def differs_from_log(expl, log, spread_tol=0.5, total_tol=0.5, ml_tol=0.02):
    """Count games whose explained model number is further from the logged number than the tolerance."""
    pred = expl[(expl['kind'] == 'piece') & (expl['feature'] == 'prediction')].pivot(index='game_id', columns='market', values='value')
    L = log.set_index('game_id').reindex(pred.index)
    bad = pd.Series(False, index=pred.index)
    if 'spread' in pred:
        bad |= (pred['spread'] - L['spread_pred']).abs() > spread_tol
    if 'total' in pred:
        bad |= (pred['total'] - L['total_pred']).abs() > total_tol
    if 'moneyline' in pred:
        bad |= (pred['moneyline'] - L['ml_prob']).abs() > ml_tol
    return int(bad.sum()), len(pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default=str(ROOT / 'data' / 'nba_stats.db'))
    parser.add_argument('--season', default=None, help='only this season, e.g. 2025-26')
    parser.add_argument('--refresh', action='store_true', help='re-explain games that already have explanations')
    parser.add_argument('--model-dir', default=str(MODEL_DIR))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    from team.train_baseline import BaselineModel            # imported here: it pulls in the training stack
    with sqlite3.connect(args.db) as conn:
        explained = set(load_explanations(args.db)['game_id'])
        todo = games_to_explain(conn, explained, args.season, args.refresh)
    if todo.empty:
        logging.info("Nothing to explain.")
        return
    logging.info(f"{len(todo)} games in {sorted(todo['season'].unique())} to explain")

    loader = BaselineModel(latest_training_set())
    loader.load_data(use_odds=True)                          # same preprocessing as generate_season_predictions.py
    frame_all = loader.df.copy()
    frame_all['gameId'] = frame_all['gameId'].astype(str).str.zfill(10)
    history = load_oos_history()
    features = sorted({f for cols in load_bundle(f'_oos_{todo["season"].iloc[0]}', args.model_dir).features.values() for f in cols})
    recent = frame_all[frame_all['seasonYear'].isin(sorted(frame_all['seasonYear'].unique())[-3:])]
    reference = build_reference(recent, features)

    for season, part in todo.groupby('season'):
        expl, frame = explain_log_rows(part, frame_all, season, args.model_dir, history, reference)
        missing = len(part) - frame['gameId'].nunique()
        n = save_explanations(args.db, expl)
        off, total = differs_from_log(expl, part)
        logging.info(f"{season}: stored {n} rows for {expl['game_id'].nunique()} games ({missing} logged games have no row in the training set); "
                     f"{off} of {total} re-scored numbers differ from the logged ones by more than 0.5 pts / 2 prob points")


if __name__ == "__main__":
    main()
