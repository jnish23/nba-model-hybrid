
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
    import argparse
    parser = argparse.ArgumentParser(description="Build the point-in-time training set")
    parser.add_argument('--season-carryover', type=float, default=1.0,
                        help="EXPERIMENT: shrink team EWMAs toward the league mean at each season start "
                             "(1.0 = production behaviour). Output goes to a differently named file that "
                             "latest_training_set() never picks up.")
    args = parser.parse_args()
    db_path = Path.cwd() / 'data' / 'nba_stats.db'

    output_dir = Path.cwd() / 'data' / 'processed'
    output_dir.mkdir(exist_ok=True)
    
    logging.info("Initializing GameFeatureBuilder...")
    builder = GameFeatureBuilder(db_path, season_carryover=args.season_carryover)
    
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
    tag = '' if args.season_carryover >= 1.0 else f'_carryover{args.season_carryover:g}'
    output_path = output_dir / f'training_set_v1_reduced_{today}{tag}.csv'

    df.to_csv(output_path, index=False)
    
    ml_cov = df['odds_home_ml'].notna().mean() if 'odds_home_ml' in df.columns else 0
    spread_cov = df['odds_home_spread'].notna().mean() if 'odds_home_spread' in df.columns else 0
    logging.info(f"Final Odds Coverage -> Spread: {spread_cov:.2%}, Moneyline: {ml_cov:.2%}")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)[:20]} ...")

if __name__ == "__main__":
    main()
