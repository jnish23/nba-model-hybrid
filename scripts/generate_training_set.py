
import pandas as pd
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from models.game_features import GameFeatureBuilder

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
    
    # Define seasons to include
    seasons = [
        '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', 
        '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', 
        '2023-24', '2024-25', '2025-26'
    ]
    
    logging.info(f"Generating training set for seasons: {seasons}")
    archive_path = Path.cwd() / 'data' / 'nba_odds_archive.csv'
    df = builder.build_training_set(seasons, archive_path=archive_path)
    
    # Save to CSV
    output_path = output_dir / 'training_set_v4.csv'
    df.to_csv(output_path, index=False)
    
    logging.info(f"Training set saved to {output_path}")
    print(f"Final Odds Coverage: {df['odds_home_ml'].notna().mean():.2%}")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)[:20]} ...")

if __name__ == "__main__":
    main()
