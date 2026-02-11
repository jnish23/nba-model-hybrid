
import pandas as pd
import sqlite3
from pathlib import Path

def export_debug():
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    archive_path = Path.cwd() / 'data' / 'nba_odds_archive.csv'
    
    # 1. Export Games (Target)
    conn = sqlite3.connect(db_path)
    query = """
    SELECT seasonYear, gameDate, gameId, teamId, matchup, wl 
    FROM team_game_full 
    WHERE seasonYear = '2015-16'
    """
    games = pd.read_sql(query, conn)
    games['gameDate'] = pd.to_datetime(games['gameDate'])
    
    # Add teamId_home/away logic for clarity
    games['is_home'] = games['matchup'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
    home_games = games[games['is_home'] == 1].copy()
    
    output_games = Path.cwd() / 'debug_games_2015.csv'
    home_games.to_csv(output_games, index=False)
    print(f"Exported {len(home_games)} games to {output_games}")
    
    # 2. Export Archive (Source)
    if archive_path.exists():
        df_archive = pd.read_csv(archive_path)
        # Filter for 2015-16 (approx dates)
        # 20151027 to 20160619
        df_archive = df_archive[
            (df_archive['date'].astype(str) >= '20151000') & 
            (df_archive['date'].astype(str) <= '20160700')
        ].copy()
        
        output_archive = Path.cwd() / 'debug_archive_2015.csv'
        df_archive.to_csv(output_archive, index=False)
        print(f"Exported {len(df_archive)} archive rows to {output_archive}")
        
    conn.close()

if __name__ == "__main__":
    export_debug()
