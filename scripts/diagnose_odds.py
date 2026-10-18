import pandas as pd
import sqlite3
import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from team.game_features_v2 import GameFeatureBuilder

logging.basicConfig(level=logging.INFO)

def diagnose():
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    archive_path = Path.cwd() / 'data' / 'nba_odds_archive.csv'
    
    conn = sqlite3.connect(db_path)
    
    # 1. Load Training Set Frame (without odds)
    print("--- Loading Game Metadata ---")
    query = """
    SELECT seasonYear, gameDate, gameId, teamId as teamId_home, matchup as MATCHUP, wl as WL
    FROM team_game_full 
    WHERE seasonYear >= '2013-14'
    """
    games = pd.read_sql(query, conn)
    games['gameDate'] = pd.to_datetime(games['gameDate'])
    games['is_home'] = games['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
    home_games = games[games['is_home'] == 1].copy()
    
    print(f"Total Home Games to match: {len(home_games)}")
    print(f"Season counts:\n{home_games['seasonYear'].value_counts().sort_index()}")

    # 2. Check Archive Coverage (Integration Test)
    print("\n--- Testing build_training_set integration ---")
    builder = GameFeatureBuilder(db_path)
    
    # We'll just run build_training_set for a sample season to verify
    try:
        # Check an older season that needs Archive
        season_check = ['2015-16']
        df_train = builder.build_training_set(seasons=season_check, archive_path=archive_path)
        print(f"Build Shape: {df_train.shape}")
        
        cov = df_train['odds_home_ml'].notna().mean()
        print(f"Final Odds Coverage ({season_check[0]}): {cov:.2%}")
        
    except Exception as e:
        print(f"Error in build_training_set: {e}")
        import traceback
        traceback.print_exc()

    # 3. Check DB Coverage
    print("\n--- Inspecting DB Odds (Raw) ---")
    close_db, _ = builder._load_db_odds()
    if not close_db.empty:
        close_db['gameDate'] = pd.to_datetime(close_db['gameDate'])
        merged_db = pd.merge(home_games, close_db, on=['gameDate', 'teamId_home'], how='left')
        
        # Check simple Moneyline availability
        if 'odds_home_ml' in merged_db.columns:
             coverage_db = merged_db['odds_home_ml'].notna().mean()
        else:
            coverage_db = 0.0
            
        print(f"DB Coverage (Raw Match): {coverage_db:.2%}")
        merged_db['has_odds'] = merged_db.get('odds_home_ml', pd.Series(dtype=float)).notna()
        print("DB Missing by Season:")
        print(merged_db.groupby('seasonYear')['has_odds'].mean())
    else:
        print("DB Odds Empty/Failed to Load")
    
    # 4. Check Sample Mismatch
    # print("\n--- Sample Missing Game (Archive) ---")
    # if 'merged_archive' in locals():
    #     missing_archive = merged_archive[merged_archive['seasonYear'] == '2023-24']
    #     pass
    
    conn.close()

if __name__ == "__main__":
    diagnose()
