
import pandas as pd
import numpy as np
import sqlite3
import logging
import requests
from bs4 import BeautifulSoup
import difflib
import time

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LiveRotationBuilder:
    def __init__(self, db_path):
        self.db_path = db_path
        self.injury_url = "https://www.espn.com/nba/injuries"
        self.injury_map = {}
        

            
    def save_injury_report(self, injury_data):
        """
        Save scraped injury data to DB for future analysis.
        """
        logging.info(f"Saving {len(injury_data)} injury records to DB...")
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 1. Create Table if Not Exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS nba_injury_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scrape_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        player_name TEXT,
                        status TEXT,
                        source TEXT
                    )
                """)
                
                # 2. Insert Records
                # Add timestamp explicitly? No, DEFAULT CURRENT_TIMESTAMP handles it.
                records = [(name, status, 'ESPN') for name, status in injury_data.items()]
                conn.executemany("INSERT INTO nba_injury_logs (player_name, status, source) VALUES (?, ?, ?)", records)
                
            logging.info("Injury logs saved.")
        except Exception as e:
            logging.error(f"Failed to save injury logs: {e}")

    def fetch_injury_report(self):
        """
        Scrapes ESPN Injuries page.
        Returns: Dict {normalized_name: status}
        """
        logging.info(f"Scraping {self.injury_url}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(self.injury_url, headers=headers)
            if r.status_code != 200:
                logging.error(f"Failed to fetch injuries. Status: {r.status_code}")
                return {}
            
            soup = BeautifulSoup(r.content, 'html.parser')
            tables = soup.find_all('table')
            
            injury_data = {}
            count = 0
            
            for table in tables:
                rows = table.find_all('tr')
                # Header row is usually row 0, data starts at 1
                for row in rows:
                    cols = row.find_all('td')
                    if not cols: continue
                    
                    # Expected columns: Name, Pos, Date, Status, Comment
                    # But verify length
                    if len(cols) >= 4:
                        name = cols[0].get_text().strip()
                        status = cols[3].get_text().strip()
                        
                        injury_data[name] = status
                        count += 1
                        
            logging.info(f"Scraped {count} injury reports.")
            self.injury_map = injury_data
            
            # Save to DB
            self.save_injury_report(injury_data)
            
            return injury_data
            
        except Exception as e:
            logging.error(f"Error scraping injuries: {e}")
            return {}
            
    def get_roster_snapshot(self, season='2025-26'):
        """
        Load all players who have played this season + calculate their EWMA minutes.
        """
        logging.info(f"Loading roster snapshot for {season}...")
        with sqlite3.connect(self.db_path) as conn:
            # Get last game date
            last_date = pd.read_sql("SELECT MAX(gameDate) as d FROM player_basic_stats", conn).iloc[0,0]
            logging.info(f"Last DB Game Date: {last_date}")
            
            # Simple query: Get all stats for the season
            query = f"""
                SELECT personId, playerName, teamId, minutes, gameDate
                FROM player_basic_stats
                WHERE seasonYear = '{season}'
            """
            df = pd.read_sql(query, conn)
            
        if df.empty:
            logging.warning("No stats found for current season.")
            return pd.DataFrame()
            
        # Calculate EWMA Minutes per Player
        # Sort by Date
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        df = df.sort_values(['personId', 'gameDate'])
        
        # Calculate EWMA (Span=10 games for rotation stability)
        df['ewma_minutes'] = df.groupby('personId')['minutes'].transform(lambda x: x.ewm(span=10).mean())
        
        # Take the LAST available EWMA for each player
        snapshot = df.groupby('personId').last().reset_index()
        
        # Load Team Info for display
        # We need teamTricode
        # Assume existing team_info table or just merge later.
        
        # Filter out players who haven't played recently? 
        # No, if they are "Active" in DB, they are active.
        # But if they haven't played in 2 months, their EWMA might be stale?
        # Let's Assume Roster = Everyone who played this season.
        
        return snapshot[['personId', 'playerName', 'teamId', 'ewma_minutes', 'gameDate']]

    def match_names(self, roster_names, injury_names):
        """
        Map injury report names to DB names using fuzzy matching.
        Returns: Dict {db_name: injury_report_name}
        """
        mapping = {}
        for i_name in injury_names:
            # Exact Match
            if i_name in roster_names.values:
                mapping[i_name] = i_name
                continue
                
            # Fuzzy Match
            matches = difflib.get_close_matches(i_name, roster_names, n=1, cutoff=0.7)
            if matches:
                mapping[matches[0]] = i_name
                # logging.info(f"Matched '{i_name}' -> '{matches[0]}'")
            else:
                logging.warning(f"Could not match injury name: '{i_name}'")
                
        return mapping

    def build_projections(self, season='2025-26'):
        # 1. Get Data
        self.fetch_injury_report()
        roster_df = self.get_roster_snapshot(season)
        
        if roster_df.empty:
            return pd.DataFrame()
            
        # 2. Match Names
        # Invert mapping: We want to associate DB rows with Injury Status
        db_names = roster_df['playerName']
        injury_names = list(self.injury_map.keys())
        
        # Map DB Name -> Status
        # First, build map: DB_Name -> Injury_Name
        # Because we iterate through INJURY report to find DB matches.
        
        name_map = {} # db_name -> injury_name
        
        # Optimization: Create a set for O(1) exact lookup
        roster_set = set(db_names)
        
        for i_name in injury_names:
            # Try Exact
            if i_name in roster_set:
                name_map[i_name] = i_name
                continue
                
            # Try removing suffixes
            # e.g. "Kelly Oubre Jr." vs "Kelly Oubre"
            
            # Fuzzy
            matches = difflib.get_close_matches(i_name, db_names, n=1, cutoff=0.8) # High cutoff for safety
            if matches:
                name_map[matches[0]] = i_name
        
        # 3. Apply Projections
        def apply_status(row):
            name = row['playerName']
            base_min = row['ewma_minutes']
            
            if name in name_map:
                inj_name = name_map[name]
                status = self.injury_map.get(inj_name, '').lower()
                
                if 'out' in status:
                    return 0.0
                elif 'doubtful' in status:
                    return 0.0
                elif 'questionable' in status:
                    return base_min * 0.5
                elif 'day-to-day' in status:
                    return base_min * 0.75 # Close to playing
                else:
                    return base_min # Unknown status? Assume play.
            
            return base_min # No report = Healthy
            
        roster_df['proj_minutes'] = roster_df.apply(apply_status, axis=1)
        roster_df['status'] = roster_df['playerName'].map(lambda n: self.injury_map.get(name_map.get(n, ''), 'Active'))
        
        return roster_df

if __name__ == "__main__":
    builder = LiveRotationBuilder(r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db')
    logging.info("Building Live Projections...")
    df = builder.build_projections()
    
    print("\n--- Top Projected Out/Questionable Players ---")
    # Show items that are NOT Active
    mask = df['status'] != 'Active'
    out_df = df[mask].sort_values('ewma_minutes', ascending=False).head(20)[['playerName', 'status', 'ewma_minutes', 'proj_minutes']]
    try:
        print(out_df)
    except UnicodeEncodeError:
         print(out_df.to_string().encode('utf-8', errors='replace').decode('utf-8')) # Fallback
    
    print("\n--- Top Active Players ---")
    active_df = df[~mask].sort_values('ewma_minutes', ascending=False).head(10)[['playerName', 'status', 'ewma_minutes', 'proj_minutes']]
    try:
        print(active_df)
    except UnicodeEncodeError:
         print(active_df.to_string().encode('utf-8', errors='replace').decode('utf-8'))
