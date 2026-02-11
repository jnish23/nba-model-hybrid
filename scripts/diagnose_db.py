import sqlite3
import pandas as pd
import logging
import random
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

DB_PATH = r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db'
OUTPUT_FILE = r'c:\Users\jorda\nba-model-hybrid\diagnostic_report.txt'

TABLES = [
    'player_basic_stats', 'team_basic_stats',
    'player_advanced_stats_v3', 'team_advanced_stats_v3',
    'player_scoring_stats_v3', 'team_scoring_stats_v3',
    'player_track_stats_v3', 'team_track_stats_v3',
    'player_miscellaneous_stats_v3', 'team_miscellaneous_stats_v3',
    'player_usage_stats_v3', 'team_usage_stats_v3',
    'team_shot_location_boxscores', 'player_shot_location_boxscores',
    'player_hustle_stats', 'team_hustle_stats',
    'defensive_stats_by_position', 'player_info', 'nba_odds'
]

CRITICAL_COLS = ['gameId', 'personId', 'teamId', 'seasonYear', 'gameDate', 'PLAYER_ID', 'TEAM_ID', 'GAME_ID', 'GAME_DATE']

def main():
    print(f"Starting Diagnostic for: {DB_PATH}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Redirect stdout to file context for cleaner code, or just pass 'f' to functions
        # Let's pass 'f' but print to console too? No, just file is safer for this task.
        
        def log(msg):
            print(msg)
            f.write(msg + '\n')
            
        log(f"Diagnostic Report ({datetime.now()})")
        log(f"Database: {DB_PATH}")

        conn = sqlite3.connect(DB_PATH)
        
        for table in TABLES:
            check_table_health(conn, table, log)
            
        check_referential_integrity(conn, log)
        sample_visual_verification(conn, log)
        
        conn.close()
        log("\nDiagnostic Complete.")
        print(f"Report saved to {OUTPUT_FILE}")

def check_table_health(conn, table, log_func):
    log_func(f"\n--- Checking Table: {table} ---")
    
    try:
        # Get count
        count = pd.read_sql(f"SELECT COUNT(*) as c FROM {table}", conn)['c'][0]
        log_func(f"Total Rows: {count}")
        
        if count == 0:
            log_func("WARNING: Table is empty.")
            return

        # Get columns
        cols_df = pd.read_sql(f"PRAGMA table_info({table})", conn)
        cols = cols_df['name'].tolist()
        
        # Check Date Range
        date_col = next((c for c in cols if c in ['gameDate', 'GAME_DATE', 'date']), None)
        if date_col:
            dates = pd.read_sql(f"SELECT MIN({date_col}) as min_d, MAX({date_col}) as max_d FROM {table}", conn)
            log_func(f"Date Range: {dates['min_d'][0]} to {dates['max_d'][0]}")
        else:
            log_func("Date column not found (skipped range check).")

        # Check NULLs in critical columns
        target_cols = [c for c in CRITICAL_COLS if c in cols]
        if target_cols:
            null_checks = [f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) as {c}_nulls" for c in target_cols]
            query = f"SELECT {', '.join(null_checks)} FROM {table}"
            null_res = pd.read_sql(query, conn)
            
            has_nulls = False
            for col in target_cols:
                n = null_res[f"{col}_nulls"][0]
                if n > 0:
                    log_func(f"CRITICAL WARNING: Column '{col}' has {n} NULL values!")
                    has_nulls = True
            if not has_nulls:
                log_func(f"Integrity Check: No NULLs found in {target_cols}")
        
        return count

    except Exception as e:
        log_func(f"ERROR checking table {table}: {e}")
        return 0

def check_referential_integrity(conn, log_func):
    log_func("\n--- Referential Integrity Check (Sample) ---")
    
    # Base set of games
    try:
        games_df = pd.read_sql("SELECT DISTINCT gameId FROM team_basic_stats", conn)
        all_games = set(games_df['gameId'].tolist())
        log_func(f"Base Game Count (team_basic_stats): {len(all_games)}")
        
        for table in ['player_basic_stats', 'team_advanced_stats_v3']:
            try:
                t_games = pd.read_sql(f"SELECT DISTINCT gameId FROM {table}", conn)['gameId'].tolist()
                missing = len(all_games - set(t_games))
                extra = len(set(t_games) - all_games)
                log_func(f"{table}: Missing {missing} games from base, Has {extra} extra games.")
            except:
                log_func(f"Could not check integrity for {table} (maybe no gameId?)")
                
    except Exception as e:
        log_func(f"Integrity check failed: {e}")

def sample_visual_verification(conn, log_func):
    log_func("\n--- Visual Verification (Random 5 Games) ---")
    try:
        games = pd.read_sql("SELECT DISTINCT gameId FROM player_basic_stats ORDER BY RANDOM() LIMIT 5", conn)
        
        for gid in games['gameId']:
            log_func(f"\nGame ID: {gid}")
            df = pd.read_sql(f"SELECT personId, playerName, points, assists, rebounds FROM player_basic_stats WHERE gameId = '{gid}' LIMIT 5", conn)
            # Safe string rep
            log_func(df.to_string(index=False))
            log_func("-" * 30)
            
    except Exception as e:
        log_func(f"Visual verification failed: {e}")



if __name__ == "__main__":
    main()
