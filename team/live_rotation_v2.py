import pandas as pd
import numpy as np
import sqlite3
import logging
import requests
from bs4 import BeautifulSoup
import difflib
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LiveRotationBuilder:
    """
    Builds injury-adjusted roster features for inference.
    
    Approach: Calculate each player's BPM, then weight by injury-adjusted
    expected minutes to get team roster_bpm.
    """
    
    # Status multipliers - conservative estimates
    # Can be refined over time by tracking actual play rates
    STATUS_MULTIPLIERS = {
        'out': 0.0,
        'doubtful': 0.10,      # Rarely play
        'questionable': 0.50,  # Coin flip
        'day-to-day': 0.70,
        'probable': 0.90,
        'available': 1.0,
        'active': 1.0,
    }
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.injury_url = "https://www.espn.com/nba/injuries"
        self.injury_map = {}
        self.name_map = {}  # DB name -> injury report name
        
    def fetch_injury_report(self, max_retries=3):
        """Scrapes ESPN Injuries page with retry logic and caching fallback."""
        logging.info(f"Scraping {self.injury_url}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for attempt in range(max_retries):
            try:
                r = requests.get(self.injury_url, headers=headers, timeout=10)
                if r.status_code != 200:
                    logging.warning(f"Attempt {attempt+1}: Status {r.status_code}")
                    time.sleep(2)
                    continue
                
                soup = BeautifulSoup(r.content, 'html.parser')
                tables = soup.find_all('table')
                
                if not tables:
                    logging.warning(f"Attempt {attempt+1}: No tables found")
                    time.sleep(2)
                    continue
                
                injury_data = {}
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 4:
                            name = cols[0].get_text().strip()
                            status = cols[3].get_text().strip()
                            if name and status:
                                injury_data[name] = status
                
                if injury_data:
                    logging.info(f"Scraped {len(injury_data)} injury reports.")
                    self.injury_map = injury_data
                    self.save_injury_report(injury_data)
                    return injury_data
                    
            except requests.RequestException as e:
                logging.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        
        logging.warning("All attempts failed. Using cached data.")
        return self._load_cached_injuries()
    
    def _load_cached_injuries(self):
        """Load most recent injury data from DB as fallback."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT player_name, status 
                    FROM nba_injury_logs 
                    WHERE DATE(scrape_date) = (
                        SELECT DATE(MAX(scrape_date)) FROM nba_injury_logs
                    )
                """
                df = pd.read_sql(query, conn)
                if not df.empty:
                    logging.info(f"Loaded {len(df)} cached injury records.")
                    self.injury_map = dict(zip(df['player_name'], df['status']))
                    return self.injury_map
        except Exception as e:
            logging.error(f"Failed to load cached injuries: {e}")
        return {}
    
    def save_injury_report(self, injury_data):
        """Save scraped injury data to DB."""
        if not injury_data:
            return
            
        logging.info(f"Saving {len(injury_data)} injury records to DB...")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS nba_injury_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scrape_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        player_name TEXT,
                        status TEXT,
                        source TEXT
                    )
                """)
                records = [(name, status, 'ESPN') for name, status in injury_data.items()]
                conn.executemany(
                    "INSERT INTO nba_injury_logs (player_name, status, source) VALUES (?, ?, ?)",
                    records
                )
            logging.info("Injury logs saved.")
        except Exception as e:
            logging.error(f"Failed to save injury logs: {e}")

    def get_player_bpm_and_minutes(self, season='2025-26'):
        """
        Get each player's rolling BPM and expected minutes.
        Returns DataFrame with: personId, playerName, teamId, bpm, ewma_minutes
        """
        logging.info(f"Loading player BPM and minutes for {season}...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if we have pre-calculated BPM
            try:
                query = """
                    SELECT 
                        pbs.personId, pbs.playerName, pbs.teamId, 
                        pbs.minutes, pbs.gameDate,
                        pb.bpm
                    FROM player_basic_stats pbs
                    LEFT JOIN player_bpm pb 
                        ON pbs.personId = pb.personId AND pbs.gameId = pb.gameId
                    WHERE pbs.seasonYear = ?
                    ORDER BY pbs.personId, pbs.gameDate
                """
                df = pd.read_sql(query, conn, params=[season])
            except:
                # Fallback if player_bpm table doesn't exist
                logging.warning("player_bpm table not found, using basic stats only")
                query = """
                    SELECT personId, playerName, teamId, minutes, gameDate
                    FROM player_basic_stats
                    WHERE seasonYear = ?
                    ORDER BY personId, gameDate
                """
                df = pd.read_sql(query, conn, params=[season])
                df['bpm'] = 0  # Default BPM if not available
        
        if df.empty:
            logging.warning("No player stats found.")
            return pd.DataFrame()
        
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        df = df.sort_values(['personId', 'gameDate'])
        
        # Calculate EWMA for minutes (span=10 for stability)
        df['ewma_minutes'] = df.groupby('personId')['minutes'].transform(
            lambda x: x.ewm(span=10, adjust=False).mean()
        )
        
        # Calculate EWMA for BPM (if available)
        if 'bpm' in df.columns and df['bpm'].notna().any():
            df['ewma_bpm'] = df.groupby('personId')['bpm'].transform(
                lambda x: x.ewm(span=10, adjust=False).mean()
            )
        else:
            df['ewma_bpm'] = 0
        
        # Get latest values per player
        latest = df.groupby('personId').last().reset_index()
        
        return latest[['personId', 'playerName', 'teamId', 'ewma_bpm', 'ewma_minutes', 'gameDate']]
    
    def build_name_mapping(self, roster_df):
        """Build mapping between DB names and injury report names."""
        if not self.injury_map:
            return
            
        db_names = set(roster_df['playerName'])
        
        for injury_name in self.injury_map.keys():
            # Exact match
            if injury_name in db_names:
                self.name_map[injury_name] = injury_name
                continue
            
            # Fuzzy match (high cutoff for safety)
            matches = difflib.get_close_matches(injury_name, db_names, n=1, cutoff=0.8)
            if matches:
                self.name_map[matches[0]] = injury_name
    
    def get_status_multiplier(self, player_name):
        """Get the minutes multiplier based on injury status."""
        if player_name not in self.name_map:
            return 1.0  # Not on injury report = healthy
        
        injury_name = self.name_map[player_name]
        status_raw = self.injury_map.get(injury_name, '').lower()
        
        # Find matching status keyword
        for status_key, multiplier in self.STATUS_MULTIPLIERS.items():
            if status_key in status_raw:
                return multiplier
        
        # Unknown status - log and assume available
        logging.debug(f"Unknown status '{status_raw}' for {player_name}, assuming active")
        return 1.0
    
    def get_player_status(self, player_name):
        """Get injury status string for a player."""
        if player_name not in self.name_map:
            return 'Active'
        
        injury_name = self.name_map[player_name]
        return self.injury_map.get(injury_name, 'Active')
    
    def calculate_team_roster_bpm(self, team_roster_df):
        """
        Calculate injury-adjusted roster_bpm for a team.
        
        Formula: sum(player_bpm * adjusted_minutes) / sum(adjusted_minutes)
        """
        if team_roster_df.empty:
            return 0.0, 0.0
        
        total_weighted_bpm = 0.0
        total_adjusted_minutes = 0.0
        
        for _, player in team_roster_df.iterrows():
            multiplier = self.get_status_multiplier(player['playerName'])
            adjusted_minutes = player['ewma_minutes'] * multiplier
            
            total_weighted_bpm += player['ewma_bpm'] * adjusted_minutes
            total_adjusted_minutes += adjusted_minutes
        
        if total_adjusted_minutes == 0:
            return 0.0, 0.0
        
        roster_bpm = total_weighted_bpm / total_adjusted_minutes
        
        # Also return total adjusted minutes (useful for confidence)
        return roster_bpm, total_adjusted_minutes
    
    def build_roster_features(self, season='2025-26'):
        """
        Build injury-adjusted roster features for all teams.
        
        Returns DataFrame with: teamId, roster_bpm, roster_minutes, injury_count
        """
        # 1. Fetch injury data
        self.fetch_injury_report()
        
        # 2. Get player data
        player_df = self.get_player_bpm_and_minutes(season)
        if player_df.empty:
            return pd.DataFrame()
        
        # 3. Build name mapping
        self.build_name_mapping(player_df)
        logging.info(f"Matched {len(self.name_map)} players to injury report")
        
        # 4. Calculate per-team features
        team_features = []
        
        for team_id in player_df['teamId'].unique():
            team_roster = player_df[player_df['teamId'] == team_id]
            
            # Calculate injury-adjusted roster BPM
            roster_bpm, roster_minutes = self.calculate_team_roster_bpm(team_roster)
            
            # Count injuries on this team
            injury_count = 0
            minutes_lost = 0.0
            for _, player in team_roster.iterrows():
                status = self.get_player_status(player['playerName'])
                if status.lower() not in ['active', 'available', 'probable']:
                    injury_count += 1
                    multiplier = self.get_status_multiplier(player['playerName'])
                    minutes_lost += player['ewma_minutes'] * (1 - multiplier)
            
            team_features.append({
                'teamId': team_id,
                'roster_bpm': roster_bpm,
                'roster_minutes': roster_minutes,
                'injury_count': injury_count,
                'injury_minutes_lost': minutes_lost,
            })
        
        result = pd.DataFrame(team_features).set_index('teamId')
        logging.info(f"Built roster features for {len(result)} teams")
        
        return result
    
    def get_team_injury_report(self, team_id, player_df=None):
        """Get detailed injury report for a specific team (for display)."""
        if player_df is None:
            player_df = self.get_player_bpm_and_minutes()
        
        team_roster = player_df[player_df['teamId'] == team_id].copy()
        
        injuries = []
        for _, player in team_roster.iterrows():
            status = self.get_player_status(player['playerName'])
            if status != 'Active':
                multiplier = self.get_status_multiplier(player['playerName'])
                injuries.append({
                    'player': player['playerName'],
                    'status': status,
                    'ewma_minutes': round(player['ewma_minutes'], 1),
                    'adjusted_minutes': round(player['ewma_minutes'] * multiplier, 1),
                    'bpm': round(player['ewma_bpm'], 1),
                })
        
        return pd.DataFrame(injuries)


if __name__ == "__main__":
    builder = LiveRotationBuilder(r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db')
    
    # Build features
    roster_feats = builder.build_roster_features()
    
    print("\n" + "="*60)
    print("TEAM ROSTER FEATURES (Injury-Adjusted)")
    print("="*60)
    print(roster_feats.sort_values('roster_bpm', ascending=False).round(2))
    
    # Show injury details for teams with most injuries
    print("\n" + "="*60)
    print("TEAMS WITH MOST INJURY IMPACT")
    print("="*60)
    
    worst_hit = roster_feats.nlargest(3, 'injury_minutes_lost')
    player_df = builder.get_player_bpm_and_minutes()
    
    for team_id in worst_hit.index:
        injuries = builder.get_team_injury_report(team_id, player_df)
        if not injuries.empty:
            print(f"\nTeam {team_id}:")
            print(injuries.to_string(index=False))