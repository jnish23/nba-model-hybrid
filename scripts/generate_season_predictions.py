"""
Leak-free, out-of-sample predictions for every game of a completed season.

The live/backfilled prediction log for 2025-26 cannot be certified leak-free: most rows were
generated weeks after the games, with no record of which model (and therefore which training
data) produced them. This script regenerates the whole season cleanly:

  1. Train the deployed pipeline (opening odds as inputs, two-stage feature selection, FLAML
     tuning, calibration) on seasons BEFORE the target season only. Hyperparameter
     tuning, feature selection and the calibration holdout all come from that earlier data.
  2. Predict every target-season game from its point-in-time training-set row (features built
     from earlier games, the opening line, the game-day-noon injury report), with the same
     post-processing as live inference (calibrated ML probability, raw spread/total).
  3. Write the predictions in the predictions_log_v2 schema, marked model_version =
     'oos_backtest_<season>' with the generation time as prediction_date, to:
       - predictions/predictions_<season>.csv (replaced)
       - the predictions_log_v2 table rows for that season (replaced; the previous table is
         first copied to predictions_log_v2_backup_<YYYYMMDD>)
     The previous CSV working copy is saved to data/backups/ (gitignored).

Usage:
    python scripts/generate_season_predictions.py --season 2025-26 [--time-budget 600]
    python scripts/generate_season_predictions.py --season 2023-24 --eval-only   # backtest another season
    python scripts/generate_season_predictions.py --season 2025-26 --skip-train  # re-predict with saved models
"""
import argparse
import datetime as dt
import glob
import json
import logging
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_baseline import BaselineModel
from models.data_paths import latest_training_set
from models.prediction_logger import replace_rows_in_date_range
from models.model_registry import load_bundle
from models.market_models import predict_markets

DB_PATH = Path('data/nba_stats.db')
MODEL_DIR = Path('models/saved')


def team_names(conn):
    """teamId -> full team name (as used in the existing prediction log)."""
    n = pd.read_sql("SELECT home_team AS name, teamId_home AS teamId, gameDate FROM nba_odds "
                    "WHERE teamId_home IS NOT NULL ORDER BY gameDate", conn)
    latest = n.drop_duplicates('teamId', keep='last')  # most recent naming (e.g. 'LA Clippers' era)
    return dict(zip(latest['teamId'].astype(int), latest['name']))


def game_results(conn, season):
    tg = pd.read_sql("SELECT gameId, teamId, MATCHUP, points FROM team_game_full WHERE seasonYear = ?", conn, params=[season])
    home = tg[tg['MATCHUP'].str.contains(' vs. ', regex=False)].rename(columns={'teamId': 'teamId_home', 'points': 'home_score'})
    away = tg[~tg['MATCHUP'].str.contains(' vs. ', regex=False)][['gameId', 'teamId', 'points']].rename(
        columns={'teamId': 'teamId_away', 'points': 'away_score'})
    return home.drop(columns='MATCHUP').merge(away, on='gameId')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--season', default='2025-26')
    parser.add_argument('--time-budget', type=int, default=600, help='FLAML seconds per model')
    parser.add_argument('--skip-train', action='store_true',
                        help='Reuse the saved *_oos_<season> models and only re-predict (e.g. after the training set '
                             'gained new games from the target season; pre-season data must be unchanged)')
    parser.add_argument('--eval-only',action='store_true',
                        help='Write reports/oos_predictions_<season>.csv only; leave the predictions CSV and DB untouched')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    season, suffix = args.season, f'_oos_{args.season}'

    # 1. Train on earlier seasons only
    data_path = latest_training_set()
    full = pd.read_csv(data_path, low_memory=False)
    if args.skip_train:
        needed = [f'feature_config{suffix}.json', f'model_ml{suffix}.pkl', f'calibrator_ml{suffix}.pkl',
                  f'model_spread{suffix}.pkl', f'model_total{suffix}.pkl']
        missing = [n for n in needed if not (MODEL_DIR / n).exists()]
        if missing:
            raise FileNotFoundError(f"--skip-train needs previously trained models; missing {missing}")
        train_max = pd.to_datetime(full.loc[full['seasonYear'] < season, 'gameDate']).max()
        logging.info(f"Skipping training; reusing saved {suffix} models (pre-{season} data ends {train_max:%Y-%m-%d})")
    else:
        pre_path = data_path.with_name(f'training_set_pre_{season}.csv')
        full[full['seasonYear'] < season].to_csv(pre_path, index=False)
        logging.info(f"Training on {(full['seasonYear'] < season).sum()} games before {season} ({pre_path})")
        trainer = BaselineModel(pre_path)
        trainer.train_and_save_all(use_odds=True, save_suffix=suffix, time_budget=args.time_budget)
        train_max = trainer.df['gameDate'].max()

    # 2. Predict the target season with identical preprocessing
    loader = BaselineModel(data_path)
    loader.load_data(use_odds=True)
    test = loader.df[loader.df['seasonYear'] == season].copy()
    assert train_max < test['gameDate'].min(), f"Leak: training data ({train_max}) overlaps {season}"
    bundle = load_bundle(suffix, MODEL_DIR)     # validated set: market-aware (residual spread, margin-stack ML, sigma) or legacy
    preds = predict_markets(bundle, test)
    test['ml_prob'], test['spread_pred'], test['total_pred'] = preds['ml_prob'], preds['spread_pred'], preds['total_pred']

    # 3. Prediction-log rows
    with sqlite3.connect(DB_PATH) as conn:
        names = team_names(conn)
        results = game_results(conn, season)
    test['gameId'] = test['gameId'].astype(str).str.zfill(10)
    results['gameId'] = results['gameId'].astype(str).str.zfill(10)
    out = test.merge(results[['gameId', 'home_score', 'away_score']], on='gameId', how='left')
    generated = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log = pd.DataFrame({
        'game_id': out['gameId'],
        'game_date': out['gameDate'].dt.strftime('%Y-%m-%d'),
        'prediction_date': generated,
        'home_team': out['teamId_home'].astype(int).map(names),
        'away_team': out['teamId_away'].astype(int).map(names),
        'ml_prob': out['ml_prob'], 'spread_pred': out['spread_pred'], 'total_pred': out['total_pred'],
        'odds_home_ml': out['odds_home_ml'], 'odds_away_ml': out['odds_away_ml'],
        'odds_home_spread': out['odds_home_spread'], 'odds_total': out['odds_total'],
        'home_score': out['home_score'], 'away_score': out['away_score'],
        'status': np.where(out['home_score'].notna(), 'COMPLETED', 'PENDING'),
        'model_version': f'oos_backtest_{season}',
    })
    log.insert(5, 'matchup', log['away_team'] + ' @ ' + log['home_team'])
    log = log.sort_values(['game_date', 'game_id']).reset_index(drop=True)
    missing_names = log['home_team'].isna().sum() + log['away_team'].isna().sum()
    if missing_names:
        raise ValueError(f"{missing_names} team names could not be mapped")

    if args.eval_only:
        out_path = Path(f'reports/oos_predictions_{season}.csv')
        log.to_csv(out_path, index=False)
        logging.info(f"Eval-only: wrote {len(log)} predictions to {out_path}; training data ended {train_max:%Y-%m-%d}")
        return

    # Write CSV (back up the previous working copy first)
    csv_path = Path(f'predictions/predictions_{season}.csv')
    stamp = dt.date.today().strftime('%Y%m%d')
    if csv_path.exists():
        backup = Path('data/backups') / f'{csv_path.stem}_before_oos_{dt.datetime.now():%Y%m%d_%H%M%S}.csv'
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, backup)  # timestamped name: never overwrites an earlier backup
        logging.info(f"Backed up previous {csv_path} to {backup}")
    log.to_csv(csv_path, index=False)

    # Replace this season's rows in the DB table (after backing up the whole table)
    with sqlite3.connect(DB_PATH) as conn:
        backup_table = f'predictions_log_v2_backup_{stamp}'
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", [backup_table]).fetchone():
            conn.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM predictions_log_v2")
            logging.info(f"Backed up predictions_log_v2 to {backup_table}")
        lo, hi = replace_rows_in_date_range(conn, log)
    logging.info(f"Wrote {len(log)} out-of-sample predictions for {season} ({lo}..{hi}); "
                 f"training data ended {train_max:%Y-%m-%d}")


if __name__ == "__main__":
    main()
