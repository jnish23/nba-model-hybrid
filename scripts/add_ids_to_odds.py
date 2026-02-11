
import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_ids_to_tables():
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    archive_path = Path.cwd() / 'data' / 'nba_odds_archive.csv'
    
    # 1. Define ID Map (User Provided + Enhanced)


    tricode_map = {
        'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739, 'NOP': 1610612740, 'NO' : 1610612740,
        'CHI': 1610612741, 'DAL': 1610612742, 'DEN': 1610612743, 'GSW': 1610612744, 'GS' : 1610612744,
        'HOU': 1610612745, 'LAC': 1610612746, 'LAL': 1610612747, 'MIA': 1610612748,
        'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751, 'BK' : 1610612751, 'NYK': 1610612752, 'NY' : 1610612752, 
        'ORL': 1610612753, 'IND': 1610612754, 'PHI': 1610612755, 'PHX': 1610612756, 'PHO' : 1610612756,
        'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'SA': 1610612759, 'OKC': 1610612760,
        'TOR': 1610612761, 'UTA': 1610612762, 'MEM': 1610612763, 'WAS': 1610612764,
        'DET': 1610612765, 'CHA': 1610612766, 'CHAR' : 1610612766
    }
    
    # Backup Name Map (from GameFeatureBuilder)
    name_map = {
        'Atlanta Hawks': 1610612737, 'Boston Celtics': 1610612738, 
        'Brooklyn Nets': 1610612751, 'Charlotte Hornets': 1610612766, 'Charlotte Bobcats': 1610612766,
        'Chicago Bulls': 1610612741, 'Cleveland Cavaliers': 1610612739, 
        'Dallas Mavericks': 1610612742, 'Denver Nuggets': 1610612743, 
        'Detroit Pistons': 1610612765, 'Golden State Warriors': 1610612744, 
        'Houston Rockets': 1610612745, 'Indiana Pacers': 1610612754, 
        'LA Clippers': 1610612746, 'Los Angeles Clippers': 1610612746,
        'LA Lakers': 1610612747, 'Los Angeles Lakers': 1610612747,
        'Memphis Grizzlies': 1610612763, 'Miami Heat': 1610612748, 
        'Milwaukee Bucks': 1610612749, 'Minnesota Timberwolves': 1610612750, 
        'New Orleans Pelicans': 1610612740, 'New Orleans Hornets': 1610612740,
        'New York Knicks': 1610612752, 'Oklahoma City Thunder': 1610612760, 
        'Orlando Magic': 1610612753, 'Philadelphia 76ers': 1610612755, 
        'Phoenix Suns': 1610612756, 'Portland Trail Blazers': 1610612757, 
        'Sacramento Kings': 1610612758, 'San Antonio Spurs': 1610612759, 
        'Toronto Raptors': 1610612761, 'Utah Jazz': 1610612762, 
        'Washington Wizards': 1610612764,
        # Archive name mappings
        'Knicks': 1610612752, 'Mavericks': 1610612742, 'Lakers': 1610612747, 
        'Thunder': 1610612760, 'Warriors': 1610612744, 'Wizards': 1610612764, 
        'Hornets': 1610612766, 'Magic': 1610612753, 'Cavaliers': 1610612739, 
        'Pacers': 1610612754, 'Timberwolves': 1610612750, 'Spurs': 1610612759, 
        'Suns': 1610612756, 'Kings': 1610612758, 'Trailblazers': 1610612757, 
        'NewJersey': 1610612751, 'Heat': 1610612748, 'Bucks': 1610612749, 
        'Raptors': 1610612761, 'Hawks': 1610612737, 'Pistons': 1610612765, 
        'Pelicans': 1610612740, 'Grizzlies': 1610612763, 'Nuggets': 1610612743, 
        'Rockets': 1610612745, 'Celtics': 1610612738, 'Jazz': 1610612762, 
        'Clippers': 1610612746, 'Bulls': 1610612741, 'Seventysixers': 1610612755,
        'Nets': 1610612751, 'Golden State': 1610612744, 'Bobcats': 1610612766,
        '0': None # Garbage
    }

    # 2. Process Archive (CSV)
    logging.info("Processing Archive CSV...")
    if archive_path.exists():
        df_archive = pd.read_csv(archive_path)
        
        # Archive typically has 'home_team', 'away_team' as names (often city or short name)
        # Apply Name Map
        df_archive['teamId_home'] = df_archive['home_team'].map(name_map)
        df_archive['teamId_away'] = df_archive['away_team'].map(name_map)
        
        # Save back
        df_archive.to_csv(archive_path, index=False)
        logging.info(f"Updated archive with {df_archive['teamId_home'].notna().sum()} mapped rows")
    else:
        logging.warning("Archive file not found!")

    # 3. Process DB Tables (nba_odds, nba_odds_opening)
    logging.info("Processing DB Tables...")
    try:
        with sqlite3.connect(db_path) as conn:

            cursor = conn.cursor()
            cols = [info[1] for info in cursor.execute("PRAGMA table_info(nba_odds)").fetchall()]
            
            # Map function
            def update_table(table_name):
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                

                logging.info(f"Columns in {table_name}: {df.columns.tolist()[:10]}...")
                
                if 'home_team_abbr' in df.columns:
                    # Use Tricode Map
                    logging.info(f"Using Tricode Map for {table_name}")
                    df['teamId_home'] = df['home_team_abbr'].map(tricode_map)
                    df['teamId_away'] = df['away_team_abbr'].map(tricode_map)
                elif 'home_team' in df.columns:
                     # Use Name Map
                    logging.info(f"Using Name Map for {table_name}")
                    df['teamId_home'] = df['home_team'].map(name_map)
                    df['teamId_away'] = df['away_team'].map(name_map)
                
                # Verify coverage
                missing = df[df['teamId_home'].isna()]
                if not missing.empty:
                    logging.warning(f"Unmapped teams in {table_name}: {missing['home_team'].unique() if 'home_team' in df.columns else missing['home_team_abbr'].unique()}")
                
                # Save back to DB (replace)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                logging.info(f"Updated {table_name} with IDs")
            
            update_table('nba_odds')
            update_table('nba_odds_opening')
            
    except Exception as e:
        logging.error(f"Error updating DB: {e}")

if __name__ == "__main__":
    add_ids_to_tables()
