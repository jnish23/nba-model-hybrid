
import sqlite3
from pathlib import Path
from update_database import NBADatabaseSync
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_odds_sync_only():
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    syncer = NBADatabaseSync(db_path)
    
    # Seasons to sync (adjust as needed, keeping the same list from update_database)
    seasons = [
        '2019-20', '2020-21', '2021-22', '2022-23', 
        '2023-24', '2024-25', '2025-26'
    ]
    
    print("--- Starting Odds-Only Sync ---")
    print("Logic checks existing data first:")
    print("1. If closing odds exist in 'nba_odds', they are SKIPPED.")
    print("2. If opening odds are missing from 'nba_odds_opening', they are FETCHED.")
    
    syncer.sync_odds_data(seasons)
    
    print("--- Odds Sync Complete ---")

if __name__ == "__main__":
    run_odds_sync_only()
