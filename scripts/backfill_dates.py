"""
Backfill predictions for specific dates by running inference.
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import unittest.mock as mock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def backfill_for_date(date_str):
    """
    Run inference for a specific date.
    """
    # Import here to avoid circular imports
    from scripts.run_inference import InferenceEngine
    from update_database import NBADatabaseSync
    
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    model_path = Path.cwd() / 'models' / 'saved' / 'model_ml_with_odds_v3.pkl'
    
    # Patch datetime.datetime.now() to return the target date at noon
    target_datetime = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=12, minute=0, second=0)
    
    with mock.patch('datetime.datetime') as mock_datetime:
        # Make datetime.now() return our target date
        mock_datetime.now.return_value = target_datetime
        # But allow other datetime functionality to work
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        try:
            engine = InferenceEngine(db_path, model_path)
            engine.run()
            print(f"✓ Successfully backfilled predictions for {date_str}")
        except Exception as e:
            print(f"✗ Failed to backfill {date_str}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Sync database first
    print("Syncing database...")
    from update_database import NBADatabaseSync
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    syncer = NBADatabaseSync(db_path)
    syncer.sync_database(['2025-26'])
    
    # Backfill dates
    dates = ['2026-01-26', '2026-01-27']
    print(f"\nBackfilling predictions for {dates}...")
    
    for date in dates:
        backfill_for_date(date)
    
    print("\nBackfill complete!")
