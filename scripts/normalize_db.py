import sqlite3
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

db_path = r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db'

# Define mappings based on the plan

# 1. Common Identifiers (Apply to ALL tables)
identifier_map = {
    'PLAYER_ID': 'personId',
    'TEAM_ID': 'teamId',
    'GAME_ID': 'gameId',
    'GAME_DATE': 'gameDate',
    'SEASON_YEAR': 'seasonYear',
    'TEAM_ABBREVIATION': 'teamTricode',
    'PLAYER_NAME': 'playerName',
    'TEAM_NAME': 'teamName',
    'SEASON': 'season', # Often appears as SEASON
    'derived_game_id': 'derivedGameId'
}

# 2. Basic Stats Expansion (Apply ONLY to player_basic_stats and team_basic_stats)
basic_stats_map = {
    'MIN': 'minutes',
    'FGM': 'fieldGoalsMade',
    'FGA': 'fieldGoalsAttempted',
    'FG_PCT': 'fieldGoalPercentage',
    'FG3M': 'threePointersMade',
    'FG3A': 'threePointersAttempted',
    'FG3_PCT': 'threePointerPercentage',
    'FTM': 'freeThrowsMade',
    'FTA': 'freeThrowsAttempted',
    'FT_PCT': 'freeThrowPercentage',
    'OREB': 'offensiveRebounds',
    'DREB': 'defensiveRebounds',
    'REB': 'rebounds',
    'AST': 'assists',
    'STL': 'steals',
    'BLK': 'blocks',
    'TOV': 'turnovers',
    'PF': 'personalFouls',
    'PTS': 'points',
    'PLUS_MINUS': 'plusMinus'
}

# List of tables to process
tables_to_process = [
    'player_basic_stats',
    'team_basic_stats',
    'player_advanced_stats_v3',
    'team_advanced_stats_v3',
    'player_scoring_stats_v3',
    'team_scoring_stats_v3',
    'player_track_stats_v3',
    'team_track_stats_v3',
    'player_miscellaneous_stats_v3',
    'team_miscellaneous_stats_v3',
    'player_usage_stats_v3',
    'team_usage_stats_v3',
    'team_shot_location_boxscores',
    'player_shot_location_boxscores',
    'player_hustle_stats',
    'team_hustle_stats',
    'defensive_stats_by_position',
    'player_info',
    'nba_odds'
]

basic_stats_tables = ['player_basic_stats', 'team_basic_stats']

def normalize_database():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for table in tables_to_process:
            logging.info(f"Processing table: {table}")
            
            # Get current columns
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                current_columns = [col[1] for col in cursor.fetchall()]
                
                if not current_columns:
                    logging.warning(f"Table {table} not found or has no columns. Skipping.")
                    continue
                
                # Determine allowed mapping for this table
                mapping = identifier_map.copy()
                if table in basic_stats_tables:
                    mapping.update(basic_stats_map)
                
                # Rename columns
                for col in current_columns:
                    if col in mapping:
                        new_name = mapping[col]
                        if new_name != col: # Only rename if different
                            try:
                                logging.info(f"  Renaming {col} -> {new_name}")
                                cursor.execute(f"ALTER TABLE {table} RENAME COLUMN {col} TO {new_name}")
                            except Exception as e:
                                logging.error(f"  Error renaming {col} in {table}: {e}")
            except Exception as e:
                logging.error(f"Error inspecting/processing table {table}: {e}")

        conn.commit()
        conn.close()
        logging.info("Database normalization completed.")

    except Exception as e:
        logging.error(f"Critical error connecting to database: {e}")

if __name__ == "__main__":
    normalize_database()
