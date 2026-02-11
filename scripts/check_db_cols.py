import sqlite3
import pandas as pd
from pathlib import Path

db_path = Path.cwd() / 'data' / 'nba_stats.db'

def check_db_cols():
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM team_game_full LIMIT 1", conn)
        print("\n=== Column Names Sample ===")
        # Print columns containing 'Corner' or 'FG' to see formatting
        sample_cols = [c for c in df.columns if 'Corner' in c or 'Percentage' in c]
        for c in sample_cols[:20]:
            print(f"'{c}'")
            
        print(f"\nTotal columns: {len(df.columns)}")
        
        # Check specific problematic ones
        target = 'Corner 3 FG_PCT'
        print(f"\nChecking for '{target}': {'FOUND' if target in df.columns else 'MISSING'}")
        
        target_under = 'Corner_3_FG_PCT'
        print(f"Checking for '{target_under}': {'FOUND' if target_under in df.columns else 'MISSING'}")

if __name__ == "__main__":
    check_db_cols()
