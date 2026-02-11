import sqlite3
import pandas as pd
import logging
from typing import List
from sbrscrape import Scoreboard

# Add project root to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def get_nba_odds(date: str) -> pd.DataFrame:
    """Fetch NBA odds data for a specific date using sbrscrape"""
    logging.info(f"Fetching odds data for {date}")
    sb = Scoreboard(sport='NBA', date=date)
    odds_data = sb.games
    return pd.DataFrame(odds_data)


def process_odds_data(odds_df: pd.DataFrame, season: str, date: str) -> pd.DataFrame:
    """Process and flatten odds data"""
    if odds_df.empty:
        return odds_df

    try:
        # Flatten nested dictionaries for each sportsbook
        for column in self.odds_columns:
            if column in odds_df.columns:
                for sportsbook in self.sportsbooks:
                    odds_df[f'{column}_{sportsbook}'] = odds_df[column].apply(
                        lambda x: x.get(sportsbook) if isinstance(x, dict) else None
                    )
        
        # Drop original nested columns
        odds_df = odds_df.drop(columns=self.odds_columns, errors='ignore')
        
        # Add metadata
        odds_df['season'] = season
        odds_df['gameDate'] = date
        
        odds_df = odds_df.drop(columns=['date'])
        
        return odds_df
    except Exception as e:
        logging.error(f"Error processing odds data: {str(e)}")
        return pd.DataFrame()
    

def sync_odds_data(self, seasons: List[str]) -> None:
    """Sync odds data for specified seasons"""
    logging.info(f"Starting odds sync for seasons: {seasons}")
    
    try:
        with sqlite3.connect(self.db_path) as conn:
            for season in seasons:
                # Get existing dates
                existing_dates_query = """
                SELECT DISTINCT gameDate 
                FROM nba_odds 
                WHERE SEASON = ?
                """
                existing_dates = set(pd.read_sql_query(
                    existing_dates_query, 
                    conn, 
                    params=[season]
                )['gameDate'].tolist())
                
                
                
                # Get all game dates
                game_dates = self.fetch_game_date_and_season_type(season)
                
                for date, _ in game_dates:
                    if date not in existing_dates:
                        try:
                            odds_df = self.get_nba_odds(date)
                            if not odds_df.empty:
                                processed_df = self.process_odds_data(odds_df, season, date)
                                if not processed_df.empty:
                                    processed_df.to_sql('nba_odds', conn, if_exists='append', index=False)
                                    logging.info(f"Added odds data for {date}")
                        except Exception as e:
                            logging.error(f"Error processing odds for {date}: {str(e)}")
                            continue
                
            # Remove duplicates
            self.remove_duplicates('nba_odds', is_player_table=False)
            
        logging.info("Odds sync completed")
        
    except Exception as e:
        logging.error(f"Error in odds sync: {str(e)}")
        raise