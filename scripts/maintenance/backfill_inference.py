"""
Backfill actual predictions for specific dates by running the full inference engine.
Usage: python backfill_inference.py
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from scripts.run_inference_v2 import InferenceEngine
from core.update_database import NBADatabaseSync
from team.model_registry import DEPLOYED_SUFFIX

if __name__ == "__main__":
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    
    # Dates to backfill
    dates_to_backfill = ['2026-01-26', '2026-01-27']
    
    # Sync database first
    logging.info("Syncing database...")
    syncer = NBADatabaseSync(db_path)
    syncer.sync_database(['2025-26'])
    
    # Run inference for each date
    for date_str in dates_to_backfill:
        logging.info(f"\n{'='*70}")
        logging.info(f"Running full inference for {date_str}")
        logging.info(f"{'='*70}")
        
        try:
            engine = InferenceEngine(db_path, suffix=DEPLOYED_SUFFIX, target_date=date_str)
            engine.run()
            logging.info(f"✓ Successfully backfilled predictions for {date_str}")
        except Exception as e:
            logging.error(f"✗ Failed to backfill {date_str}: {e}")
            import traceback
            traceback.print_exc()
    
    logging.info("\n✅ Backfill complete!")
    
    # Verify
    import sqlite3
    import pandas as pd
    
    conn = sqlite3.connect(db_path)
    verify_query = "SELECT game_date, COUNT(*) as prediction_count FROM predictions_log_v2 WHERE game_date IN ('2026-01-26', '2026-01-27') GROUP BY game_date ORDER BY game_date"
    result = pd.read_sql_query(verify_query, conn)
    conn.close()
    
    print("\nVerification - Predictions per date:")
    print(result.to_string(index=False))


