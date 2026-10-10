from pathlib import Path
import sys
import os
import pandas as pd
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train_baseline import BaselineModel
from models.prediction_logger import PredictionLogger

logging.basicConfig(level=logging.INFO)

def backfill_logs():
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    data_dir = Path('data/processed')
    if not data_dir.exists():
        # Fallback
        data_dir = Path(r'c:\Users\jorda\nba-model-hybrid\data\processed')
    # Latest dated training set written by scripts/generate_training_set.py (ISO dates sort lexically)
    from models.data_paths import latest_training_set
    data_path = latest_training_set(data_dir)
        
    logging.info("Initializing Baseline Model for Backfill...")
    model = BaselineModel(data_path)
    model.load_data(use_odds=True)
    
    # Run backtest for 2025-26 ONLY
    # This ensures "entering" data is used for each prediction (no leakage)
    logging.info("Running Walk-Forward Backtest for 2025-26...")
    metrics, predictions_df = model.backtest_walk_forward(start_season='2025-26', use_odds=True)
    
    if predictions_df.empty:
        logging.error("No predictions generated!")
        return

    logging.info(f"Backfilling {len(predictions_df)} predictions to logs...")
    logger = PredictionLogger(db_path, log_path='predictions/predictions_2025-26_v2.csv')
    
    # Needs team names for the log entry text (optional, but good for 'matchup' field)
    # The training set has team IDs. We need a mapping.
    # We can query the DB.
    import sqlite3
    id_to_name = {}
    try:
        with sqlite3.connect(db_path) as conn:
            try:
                # Try team_info first
                team_map = pd.read_sql("SELECT teamId, teamName FROM team_info", conn)
            except:
                # Fallback to team_basic_stats if team_info doesn't exist
                team_map = pd.read_sql("SELECT teamId, teamName FROM team_basic_stats", conn)
            
            id_to_name = dict(zip(team_map['teamId'], team_map['teamName']))
    except Exception as e:
        logging.warning(f"Could not load team names: {e} - using IDs instead.")
    
    count = 0
    for _, row in predictions_df.iterrows():
        home_name = id_to_name.get(row['teamId_home'], f"ID_{row['teamId_home']}")
        away_name = id_to_name.get(row['teamId_away'], f"ID_{row['teamId_away']}")
        
        # Handle Timestamps safely
        g_date = row['gameDate']
        if pd.notnull(g_date):
            if hasattr(g_date, 'strftime'):
                g_date_str = g_date.strftime('%Y-%m-%d')
            else:
                g_date_str = str(g_date)
        else:
            g_date_str = 'UNKNOWN'

        # Construct log entry matching run_inference format
        entry = {
            'gameId': str(row['gameId']), # Ensure string
            'gameDate': g_date_str,
            # We don't have exact timestamp of prediction, assume 10am on game day? 
            # Or just use current time to indicate when it was logged. 
            # Ideally use gameDate + " 00:00:00" string.
            'prediction_timestamp': f"{g_date_str} 10:00:00", 
            'home_team': home_name,
            'away_team': away_name,
            
            'pred_home_win_prob': row['ml_prob'],
            'pred_spread': row['spread_pred'],
            'pred_total': row['total_pred'],
            
            # Odds
            'line_ml_home': row.get('odds_home_ml', 0),
            'line_ml_away': row.get('odds_away_ml', 0),
            'line_spread': row.get('odds_home_spread', 0),
            'line_total': row.get('odds_total', 0),
            
            'model_version': 'backfill_v2'
        }
        
        logger.log_prediction(entry, overwrite=False)
        count += 1
        
    logging.info(f"Successfully logged {count} historic predictions.")
    
    # Now update with actual scores
    logging.info("Updating actual scores/results in logs...")
    logger.update_actual_scores()
    logging.info("Backfill Complete.")

if __name__ == "__main__":
    backfill_logs()
