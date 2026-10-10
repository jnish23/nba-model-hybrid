
import pandas as pd
import logging
import sys
import os
from datetime import date

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from models.game_features_v2 import GameFeatureBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    output_dir = Path.cwd() / 'data' / 'processed'
    output_dir.mkdir(exist_ok=True)
    
    logging.info("Initializing GameFeatureBuilder...")
    builder = GameFeatureBuilder(db_path)
    
    # All seasons in the database from 2013-14 on (new seasons are picked up automatically)
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        seasons = sorted(pd.read_sql("SELECT DISTINCT seasonYear FROM team_game_full", conn)['seasonYear'])
    seasons = [s for s in seasons if s >= '2013-14']
    
    logging.info(f"Generating training set for seasons: {seasons}")
    archive_path = Path.cwd() / 'data' / 'nba_odds_archive.csv'
    df = builder.build_training_set(seasons, archive_path=archive_path)
    
    # Save to CSV
    today = date.today()
    output_path = output_dir / f'training_set_v1_reduced_{today}.csv'
    df.to_csv(output_path, index=False)
    
    logging.info(f"Training set saved to {output_path}")
    print(f"Final Odds Coverage: {df['odds_home_ml'].notna().mean():.2%}")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)[:20]} ...")

if __name__ == "__main__":
    main()
