import sqlite3
import numpy as np
import pandas as pd
import time
import random
import logging
import backoff
from pathlib import Path
from pandas.core.window import ewm
from requests.exceptions import RequestException, Timeout
from datetime import datetime
from typing import Set, Tuple, Optional, Dict, List
from dataclasses import dataclass
from nba_api.stats.endpoints import (
    leaguedashplayershotlocations,
    leaguegamefinder,
    boxscoreadvancedv3,
    boxscorescoringv3,
    boxscorehustlev2,
    boxscoreplayertrackv3,
    boxscoremiscv3,
    boxscoreusagev3,
    playergamelogs,
    leaguegamelog,
    leaguedashteamstats,
    leaguedashteamshotlocations
)
from sbrscrape import Scoreboard

# Add root to sys.path to allow importing models when run from scripts/
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prediction_logger import PredictionLogger

def custom_backoff_generator(**kwargs):
    """
    Custom backoff generator that yields fixed delays
    Yields: int: Delay in seconds
    """
    yield 120  # First retry after 2 minutes
    yield 240  # Second retry after 4 minutes
    yield 360  # Third retry after 6 minutes

@dataclass
class TablePair:
    """Class to hold related team and player table information"""
    team_table: str
    player_table: str
    id_column: str
    min_season: str  # Minimum season where this stat is available

class NBADatabaseSync:
    def __init__(self, db_path: str, log_path: str = 'nba_sync.log'):
        """
        Initialize the NBA database synchronization tool
        """
        self.db_path = db_path
        self.setup_logging(log_path)
        self.endpoint_map = {
            'team_advanced_stats_v3': boxscoreadvancedv3.BoxScoreAdvancedV3,
            'team_scoring_stats_v3': boxscorescoringv3.BoxScoreScoringV3,
            'team_hustle_stats': boxscorehustlev2.BoxScoreHustleV2,
            'team_track_stats_v3': boxscoreplayertrackv3.BoxScorePlayerTrackV3,
            'team_miscellaneous_stats_v3': boxscoremiscv3.BoxScoreMiscV3,
            'team_usage_stats_v3': boxscoreusagev3.BoxScoreUsageV3,
            'team_shot_locations': leaguedashteamshotlocations.LeagueDashTeamShotLocations,

        }
        
        # Define mappings for normalization
        self.identifier_map = {
            'PLAYER_ID': 'personId',
            'TEAM_ID': 'teamId',
            'GAME_ID': 'gameId',
            'GAME_DATE': 'gameDate',
            'SEASON_YEAR': 'seasonYear',
            'TEAM_ABBREVIATION': 'teamTricode',
            'PLAYER_NAME': 'playerName',
            'TEAM_NAME': 'teamName',
            'SEASON': 'season',
            'derived_game_id': 'derivedGameId'
        }

        self.basic_stats_map = {
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

        
        self.table_pairs = [
            TablePair('team_advanced_stats_v3', 'player_advanced_stats_v3', 'gameId', '2013-14'),
            TablePair('team_scoring_stats_v3', 'player_scoring_stats_v3', 'gameId', '2013-14'),
            TablePair('team_track_stats_v3', 'player_track_stats_v3', 'gameId', '2013-14'),
            # TablePair('team_miscellaneous_stats_v3', 'player_miscellaneous_stats_v3', 'gameId', '2013-14'),
            TablePair('team_usage_stats_v3', 'player_usage_stats_v3', 'gameId', '2013-14'),
            TablePair('team_hustle_stats', 'player_hustle_stats', 'gameId', '2015-16'),
            TablePair('defensive_stats_by_position', '', 'gameDate', '2016-17'),
            TablePair('team_shot_location_boxscores', '', 'gameDate', '2014-15'),
            TablePair('player_shot_location_boxscores', '', 'gameDate', '2014-15')


        ]
        
        self.POSITIONS = ['C', 'F', 'G']
        
        self.sportsbooks = ['betmgm', 'fanduel', 'caesars', 'bet365', 'draftkings', 'bet_rivers_ny']
        self.odds_columns = ['home_spread', 'home_spread_odds', 'away_spread', 
                           'away_spread_odds', 'home_ml', 'away_ml', 'total', 
                           'over_odds', 'under_odds']


    def setup_logging(self, log_path: str) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )


    def add_team_basic_boxscores(self, seasons: List[str]) -> None:
        """
        Pull and add basic team boxscores for specified seasons with improved DataFrame handling
        """
        logging.info(f"Adding team basic boxscores for seasons: {seasons}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    logging.info(f"Processing season {season}")
                    season_boxscores = []
                    
                    # Determine season types
                    season_types = ['Regular Season', 'PlayIn', 'Playoffs'] if season >= '2019-20' else ['Regular Season', 'Playoffs']
                    
                    for season_type in season_types:
                        try:
                            team_boxscores = leaguegamelog.LeagueGameLog(
                                season=season, 
                                season_type_all_star=season_type
                            ).get_data_frames()[0]
                            
                            # Only append if DataFrame is not empty
                            if not team_boxscores.empty:
                                # Remove VIDEO_AVAILABLE column if it exists
                                if 'VIDEO_AVAILABLE' in team_boxscores.columns:
                                    team_boxscores = team_boxscores.drop(columns=['VIDEO_AVAILABLE'])
                                
                                # Remove rows that have NaN values in 'WL' column indicating game not complete yet
                                team_boxscores = team_boxscores.dropna(subset=['WL'])
                                
                                season_boxscores.append(team_boxscores)
                                logging.info(f"Retrieved {len(team_boxscores)} games for {season_type} {season}")
                            else:
                                logging.warning(f"No data retrieved for {season_type} {season}")
                                
                        except Exception as e:
                            logging.error(f"Error retrieving {season_type} data for season {season}: {str(e)}")
                            continue
                        
                        time.sleep(4*random.random())

                    
                    if season_boxscores:
                        # Get column dtypes from first DataFrame to ensure consistency
                        dtype_dict = season_boxscores[0].dtypes.to_dict()
                        
                        # Ensure all DataFrames have the same columns and dtypes
                        processed_boxscores = []
                        for df in season_boxscores:
                            # Ensure all columns exist
                            for col, dtype in dtype_dict.items():
                                if col not in df.columns:
                                    df[col] = pd.Series(dtype=dtype)
                            
                            # Ensure correct dtypes
                            df = df.astype(dtype_dict)
                            processed_boxscores.append(df)
                        
                        # Concatenate with explicit dtype preservation
                        season_df = pd.concat(processed_boxscores, ignore_index=True)
                        season_df['SEASON_YEAR'] = season
                        
                        # Rename columns
                        season_df = season_df.rename(columns=self.identifier_map)
                        season_df = season_df.rename(columns=self.basic_stats_map)

                        # Add to database
                        table_name = 'team_basic_stats'
                        season_df.to_sql(table_name, conn, if_exists='append', index=False)
                        
                        # Remove duplicates
                        cur = conn.cursor()
                        cur.execute(f'''
                            DELETE FROM {table_name} 
                            WHERE rowid NOT IN (
                                SELECT max(rowid) 
                                FROM {table_name} 
                                GROUP BY teamId, gameId
                            )
                        ''')
                        conn.commit()
                        
                        logging.info(f"Successfully added {len(season_df)} games for season {season}")
                    else:
                        logging.warning(f"No valid data retrieved for season {season}")
                    
                    
        except Exception as e:
            logging.error(f"Error in add_team_basic_boxscores: {str(e)}")
            raise

    def add_player_basic_boxscores(self, seasons: List[str]) -> None:
        """
        Pull and add basic player boxscores for specified seasons with improved DataFrame handling
        """
        logging.info(f"Adding player basic boxscores for seasons: {seasons}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    logging.info(f"Processing season {season}")
                    season_boxscores = []
                    
                    season_types = ['Regular Season', 'PlayIn', 'Playoffs'] if season >= '2019-20' else ['Regular Season', 'Playoffs']
                    
                    for season_type in season_types:
                        try:
                            time.sleep(4*random.random())
                            player_boxscores = playergamelogs.PlayerGameLogs(
                                season_nullable=season,
                                season_type_nullable=season_type
                            ).get_data_frames()[0]
                            
                            if not player_boxscores.empty:
                                
                                # Remove rows that have NaN values in 'WL' column indicating game not complete yet (still being played)
                                player_boxscores = player_boxscores.dropna(subset=['WL'])


                                player_boxscores = player_boxscores.drop(columns=['TEAM_COUNT'])
                                season_boxscores.append(player_boxscores)
                                logging.info(f"Retrieved {len(player_boxscores)} player games for {season_type} {season}")
                            else:
                                logging.warning(f"No player data retrieved for {season_type} {season}")
                                
                        except Exception as e:
                            logging.error(f"Error retrieving {season_type} data for season {season}: {str(e)}")
                            continue
                        
                        time.sleep(4*random.random())


                    
                    if season_boxscores:
                        # Get column dtypes from first DataFrame
                        dtype_dict = season_boxscores[0].dtypes.to_dict()
                        
                        # Process each DataFrame
                        processed_boxscores = []
                        for df in season_boxscores:
                            # Ensure all columns exist
                            for col, dtype in dtype_dict.items():
                                if col not in df.columns:
                                    df[col] = pd.Series(dtype=dtype)
                            
                            # Ensure correct dtypes
                            df = df.astype(dtype_dict)
                            processed_boxscores.append(df)
                        
                        # Concatenate with explicit dtype preservation
                        season_df = pd.concat(processed_boxscores, ignore_index=True)
                        season_df['SEASON_YEAR'] = season
                        
                        # Rename columns
                        season_df = season_df.rename(columns=self.identifier_map)
                        season_df = season_df.rename(columns=self.basic_stats_map)

                        # Add to database
                        table_name = 'player_basic_stats'
                        season_df.to_sql(table_name, conn, if_exists='append', index=False)
                        
                        # Remove duplicates
                        cur = conn.cursor()
                        cur.execute(f'''
                            DELETE FROM {table_name} 
                            WHERE rowid NOT IN (
                                SELECT max(rowid) 
                                FROM {table_name} 
                                GROUP BY personId, gameId
                            )
                        ''')
                        conn.commit()
                        
                        logging.info(f"Successfully added {len(season_df)} player games for season {season}")
                    else:
                        logging.warning(f"No valid player data retrieved for season {season}")
                                        
        except Exception as e:
            logging.error(f"Error in add_player_basic_boxscores: {str(e)}")
            raise

    def sync_basic_stats(self, seasons: List[str]) -> None:
        """
        Sync both team and player basic stats for specified seasons
        
        Parameters:
        seasons (List[str]): List of seasons to sync
        """
        logging.info(f"Starting basic stats sync for seasons: {seasons}")
        
        try:
            # Sync team stats
            self.add_team_basic_boxscores(seasons)
            
            # Sync player stats
            self.add_player_basic_boxscores(seasons)
            
            logging.info("Basic stats sync completed successfully")
            
        except Exception as e:
            logging.error(f"Error in sync_basic_stats: {str(e)}")
            raise
    
    @backoff.on_exception(
        custom_backoff_generator,
        (RequestException, Timeout, Exception),
        max_tries=4,
        on_backoff=lambda details: logging.error(
            f"Retrying in {details['wait']}s after error: {details['exception']}"
        )
    )
    def get_defensive_stats(self, position: str, date: str, season: str, season_type: str) -> pd.DataFrame:
        """
        Get defensive stats for a specific position on a specific date
        
        Parameters:
        position (str): Position abbreviation ('C', 'F', or 'G')
        date (str): Date in YYYY-MM-DD format
        season (str): Season in YYYY-YY format
        
        Returns:
        pd.DataFrame: Defensive stats for the specified position
        """
        logging.info(f"Fetching defensive stats for position {position} on {date}")
        
        stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Opponent',
            per_mode_detailed='PerGame',
            season=season,
            season_type_all_star=season_type,
            date_from_nullable=date,
            date_to_nullable=date,
            player_position_abbreviation_nullable=position
        )
        
        df = stats.get_data_frames()[0]
        df['SEASON'] = season
        df['POSITION'] = position
        df['GAME_DATE'] = date
        
        # Rename columns (only identifiers here as it's not a basic stat table)
        df = df.rename(columns=self.identifier_map)

        return df

    def get_all_defensive_stats(self, date: str, season: str, season_type: str) -> Optional[pd.DataFrame]:
        """
        Get defensive stats for all positions on a specific date
        
        Parameters:
        date (str): Date in YYYY-MM-DD format
        season (str): Season in YYYY-YY format
        
        Returns:
        Optional[pd.DataFrame]: Combined defensive stats for all positions
        """
        all_data = []
        
        for position in self.POSITIONS:
            try:
                df = self.get_defensive_stats(position, date, season, season_type)
                all_data.append(df)
                logging.info(f"Successfully fetched data for position {position} on {date}")
            except Exception as e:
                logging.error(f"Failed to fetch data for position {position} on {date}: {str(e)}")
                continue
            
            time.sleep(1+random.gammavariate(1, 3))

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    def process_defensive_stats(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Process and pivot defensive stats
        
        Parameters:
        df (Optional[pd.DataFrame]): Raw defensive stats DataFrame
        
        Returns:
        Optional[pd.DataFrame]: Processed defensive stats
        """
        if df is None or df.empty:
            return None

        try:
            # Pivot the DataFrame
            pivot_cols = ['L', 'W_PCT', 'MIN', 'OPP_FGM',
                        'OPP_FGA', 'OPP_FG_PCT', 'OPP_FG3M', 'OPP_FG3A', 'OPP_FG3_PCT',
                        'OPP_FTM', 'OPP_FTA', 'OPP_FT_PCT', 'OPP_OREB', 'OPP_DREB', 'OPP_REB',
                        'OPP_AST', 'OPP_TOV', 'OPP_STL', 'OPP_BLK', 'OPP_BLKA', 'OPP_PF',
                        'OPP_PFD', 'OPP_PTS', 'PLUS_MINUS']
            
            df_pivoted = df.pivot(
                index=['season', 'teamId', 'teamName', 'gameDate'],
                columns='POSITION',
                values=pivot_cols
            )
            
            # Flatten column names
            df_pivoted.columns = [f'{stat}_{pos}' for stat, pos in df_pivoted.columns]
            df_pivoted = df_pivoted.reset_index() 
            
            return df_pivoted
            
        except Exception as e:
            logging.error(f"Error processing defensive stats: {str(e)}")
            return None

    def fetch_game_date_and_season_type(self, season: str) -> List[str]:
        """
        Fetch all game dates for a given season
        
        Parameters:
        season (str): Season in YYYY-YY format
        
        Returns:
        List[str]: List of game dates in YYYY-MM-DD format
        """
        try:
            query = """
            SELECT DISTINCT gameDate, SEASON_ID
            FROM team_basic_stats 
            WHERE seasonYear = ? 
            ORDER BY gameDate
            """
            
            with sqlite3.connect(self.db_path) as conn:
                dates = pd.read_sql_query(query, conn, params=[season])
                
                conditions = [
                    dates['SEASON_ID'].str.startswith('22'),
                    dates['SEASON_ID'].str.startswith('52'),
                    dates['SEASON_ID'].str.startswith('42')
                ]
                choices = ['Regular Season', 'PlayIn', 'Playoffs']
                
                
                dates['SEASON_TYPE'] = np.select(conditions, choices, default='Unknown')
                
                dates_and_season_type = dates[['gameDate', 'SEASON_TYPE']].values.tolist()
                
                return dates_and_season_type
                
        except Exception as e:
            logging.error(f"Error fetching game dates for season {season}: {str(e)}")
            return []

    def sync_defensive_stats(self, seasons: List[str]) -> None:
        """
        Sync defensive stats for specified seasons
        
        Parameters:
        seasons (List[str]): List of seasons to sync
        """
        logging.info(f"Starting defensive stats sync for seasons: {seasons}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    logging.info(f"Processing season: {season}")
                    
                    # Get existing dates
                    existing_dates_query = """
                    SELECT DISTINCT gameDate 
                    FROM defensive_stats_by_position 
                    WHERE season = ?
                    
                    """
                    existing_dates = set(pd.read_sql_query(
                        existing_dates_query, 
                        conn, 
                        params=[season]
                    )['gameDate'].tolist())
                    
                    # Get all game dates and season_types
                    game_dates_and_season_types = self.fetch_game_date_and_season_type(season)
                    
                    for date, season_type in game_dates_and_season_types:
                        if date not in existing_dates:
                            try:
                                raw_data = self.get_all_defensive_stats(date, season, season_type)
                                processed_data = self.process_defensive_stats(raw_data)
                                
                                if processed_data is not None:
                                    processed_data.to_sql(
                                        'defensive_stats_by_position',
                                        conn,
                                        if_exists='append',
                                        index=False
                                    )
                                    logging.info(f"Added defensive stats for {date}")
                                    
                            except Exception as e:
                                logging.error(f"Error processing date {date}: {str(e)}")
                                continue
                    
            logging.info("Defensive stats sync completed")
            
        except Exception as e:
            logging.error(f"Error in defensive stats sync: {str(e)}")
            raise
        
        
    @backoff.on_exception(
        custom_backoff_generator,
        (RequestException, Timeout, Exception),
        max_tries=4,
        on_backoff=lambda details: logging.error(
            f"Retrying in {details['wait']}s after error: {details['exception']}"
        )
    )
    def get_shot_location_stats(self, date: str, season: str, season_type: str) -> Optional[pd.DataFrame]:
        """
        Get shot location stats for a specific date
        """
        logging.info(f"Fetching shot location stats for date {date}")
        
        stats = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
            distance_range='By Zone',
            per_mode_detailed='PerGame',
            season=season,
            season_type_all_star=season_type,
            date_from_nullable=date,
            date_to_nullable=date,
        )
        
        df = stats.get_data_frames()[0]
        df['SEASON'] = season
        df['GAME_DATE'] = date
        
        # Process column names
        df.columns = df.columns.map(' '.join)
        df.columns = [col.strip() for col in df.columns]
        
        return df
        

    def sync_shot_location_stats(self, seasons: List[str]) -> None:
        """
        Sync shot location stats for specified seasons
        """
        logging.info(f"Starting shot location stats sync for seasons: {seasons}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    logging.info(f"Processing season: {season}")
                    
                    # Get existing dates
                    existing_dates_query = """
                    SELECT DISTINCT gameDate
                    FROM team_shot_location_boxscores 
                    WHERE season = ?
                    """
                    existing_dates_df = pd.read_sql_query(
                        existing_dates_query, 
                        conn, 
                        params=[season]
                        )
                    
                    existing_dates_list = existing_dates_df['gameDate'].astype(str).tolist()

                    existing_dates = set(existing_dates_list)

                    # Get all game dates and season types
                    game_dates_and_season_types = self.fetch_game_date_and_season_type(season)
                    
                    for date, season_type in game_dates_and_season_types:
                        if date not in existing_dates:
                            try:
                                df = self.get_shot_location_stats(date, season, season_type)
                                df = df.rename(columns = self.identifier_map)
                                if df is not None:
                                    df.to_sql(
                                        'team_shot_location_boxscores',
                                        conn,
                                        if_exists='append',
                                        index=False
                                    )
                                    logging.info(f"Added shot location stats for {date} ({season_type})")
                            except Exception as e:
                                logging.error(f"Error processing {date} ({season_type}): {str(e)}")
                                continue
                            
                            time.sleep(1+random.gammavariate(1, 3))

                # Remove duplicates
                self.remove_duplicates('team_shot_location_boxscores', is_player_table=False)
                
            logging.info("Shot location stats sync completed")
            
        except Exception as e:
            logging.error(f"Error in shot location stats sync: {str(e)}")
            raise

    @backoff.on_exception(
        custom_backoff_generator,
        (RequestException, Timeout, Exception),
        max_tries=4,
        on_backoff=lambda details: logging.error(
            f"Retrying in {details['wait']}s after error: {details['exception']}"
        )
    )
    def get_player_shot_location_stats(self, date: str, season: str, season_type: str) -> Optional[pd.DataFrame]:
            """
            Get player shot location stats for a specific date (for all players active that day)
            """
            logging.info(f"Fetching player shot location stats for date {date}")
            
            stats = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
                distance_range='By Zone',
                per_mode_detailed='PerGame',
                season=season,
                season_type_all_star=season_type,
                date_from_nullable=date,
                date_to_nullable=date,
            )
            
            df = stats.get_data_frames()[0]
            
            # Add metadata columns
            df['SEASON'] = season
            df['GAME_DATE'] = date
            
            # Process column names (flatten multi-level headers if present, strip whitespace)
            df.columns = df.columns.map(' '.join) if isinstance(df.columns[0], tuple) else df.columns
            df.columns = [col.strip() for col in df.columns]
            
            return df

    def sync_player_shot_location_stats(self, seasons: List[str]) -> None:
            """
            Sync player shot location stats for specified seasons
            """
            logging.info(f"Starting player shot location stats sync for seasons: {seasons}")
            
            try:
                with sqlite3.connect(self.db_path) as conn:
                    for season in seasons:
                        logging.info(f"Processing season: {season}")
                        
                        # 1. Get existing dates from the PLAYER table
                        try:
                            existing_dates_query = """SELECT gameDate FROM player_shot_location_boxscores WHERE season = ?"""
                            existing_dates_df = pd.read_sql_query(
                                existing_dates_query, 
                                conn, 
                                params=[season]
                                )
                        

                            existing_dates = set(existing_dates_df['gameDate'].astype(str).tolist())

                        except:
                            existing_dates = set()

                        # 2. Get all available game dates
                        game_dates_and_season_types = self.fetch_game_date_and_season_type(season)
                        
                        for date, season_type in game_dates_and_season_types:
                            if date not in existing_dates:
                                try:
                                    df = self.get_player_shot_location_stats(date, season, season_type)
                                    df = df.rename(columns = self.identifier_map)

                                    if df is not None:
                                        df.to_sql(
                                            'player_shot_location_boxscores',
                                             conn,
                                            if_exists='append',
                                            index=False
                                        )
                                        logging.info(f"Added player shot location stats for {date} ({season_type})")
                                except Exception as e:
                                    logging.error(f"Error processing {date} ({season_type}): {str(e)}")
                                    continue
                                        

                                # Sleep to be polite to the API
                                time.sleep(1+random.gammavariate(1, 3))

                    # 3. Remove duplicates from the PLAYER table
                    # Note: is_player_table=True is assumed to handle PLAYER_ID logic
                    self.remove_duplicates('player_shot_location_boxscores', is_player_table=True)
                    
                logging.info("Player shot location stats sync completed")
                
            except Exception as e:
                logging.error(f"Error in player shot location stats sync: {str(e)}")
                raise

    @backoff.on_exception(
        custom_backoff_generator,
        (RequestException, Timeout, Exception),
        max_tries=4,
        on_backoff=lambda details: logging.error(
            f"Retrying in {details['wait']}s after error: {details['exception']}"
        )
    )
    def fetch_game_ids(self, season: str) -> Set[str]:
        """
        Fetch all game IDs for a given season using the NBA API with custom backoff
        """
        logging.info(f"Fetching game IDs for season {season}")
        
        # Fetch games using NBA API
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00'  # NBA games only
        )
        
        games_df = gamefinder.get_data_frames()[0]
        games_df = games_df.dropna(subset=['WL'])
        
        # Filter for NBA games only
        nba_games = games_df.loc[
            games_df['SEASON_ID'].str.startswith(('22', '42', '52'))
        ]
        
        # Get unique game IDs
        game_ids = set(nba_games['GAME_ID'].unique())
        
        logging.info(f"Found {len(game_ids)} unique games for season {season}")
        
        if len(game_ids) == 0:
            logging.warning(f"No games found for season {season}")
            
        return game_ids
        
        
        
    def get_game_data(self, table: str, game_id: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Fetch game data from NBA API with exponential backoff
        
        Parameters:
        table (str): Table name
        game_id (str): Game ID
        
        Returns:
        Optional[Tuple[pd.DataFrame, pd.DataFrame]]: Player and team DataFrames if successful
        """
        endpoint = self.endpoint_map.get(table)
        if not endpoint:
            logging.warning(f"No endpoint found for table {table}")
            return None
            
        response = endpoint(game_id=game_id)
        time.sleep(1+random.gammavariate(1, 3))
        return response.get_data_frames()

    def process_game(self, game_id: str, table_pair: TablePair) -> bool:
        """
        Process a single game for a table pair
        
        Parameters:
        game_id (str): Game ID
        table_pair (TablePair): Table pair to process
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            game_data = self.get_game_data(table_pair.team_table, game_id)
            if game_data:
                player_df, team_df = game_data

                # Rename columns for both dataframes
                player_df = player_df.rename(columns=self.identifier_map)
                team_df = team_df.rename(columns=self.identifier_map)
                
                                                            
                with sqlite3.connect(self.db_path) as conn:
                    player_df.to_sql(table_pair.player_table, conn, if_exists='append', index=False)
                    team_df.to_sql(table_pair.team_table, conn, if_exists='append', index=False)
                
                logging.info(f"Successfully added data for game {game_id} to {table_pair.team_table}")
                return True
            
            logging.warning(f"No data available for game {game_id} in {table_pair.team_table}")
            return False
            
        except Exception as e:
            logging.error(f"Error processing game {game_id} for {table_pair.team_table}: {str(e)}")
            return False
        
        
    @backoff.on_exception(
        custom_backoff_generator,
        (RequestException, Timeout, Exception),
        max_tries=4,
        on_backoff=lambda details: logging.error(
            f"Retrying in {details['wait']}s after error: {details['exception']}"
        )
    )
    def get_nba_odds(self, date: str, current_line: bool = True) -> pd.DataFrame:
        """
        Fetch NBA odds data for a specific date using sbrscrape.
        
        Parameters:
        date (str): Date to fetch odds for
        current_line (bool): If True, fetch closing line (current). 
                             If False, fetch opening line.
        """
        logging.info(f"Fetching {'closing' if current_line else 'opening'} odds data for {date}")
        sb = Scoreboard(sport='NBA', date=date, current_line=current_line)
        odds_data = sb.games
        return pd.DataFrame(odds_data)


    def process_odds_data(self, odds_df: pd.DataFrame, season: str, date: str) -> pd.DataFrame:
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
        """Sync odds data (both opening and closing) for specified seasons"""


        logging.info(f"Starting odds sync for seasons: {seasons}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    if season < '2019-20':
                        continue
                    # --- 1. Closing Odds (nba_odds) ---
                    existing_closing_dates = set(pd.read_sql_query(
                        "SELECT DISTINCT gameDate FROM nba_odds WHERE SEASON = ?", 
                        conn, params=[season]
                    )['gameDate'].tolist())
                    
                    # --- 2. Opening Odds (nba_odds_opening) ---
                    # Create table if not exists (handled by to_sql usually, but ensuring read doesn't fail basic check)
                    try:
                         existing_opening_dates = set(pd.read_sql_query(
                            "SELECT DISTINCT gameDate FROM nba_odds_opening WHERE SEASON = ?",
                            conn, params=[season]
                        )['gameDate'].tolist())
                    except:
                        existing_opening_dates = set()

                    # Get all game dates
                    game_dates = self.fetch_game_date_and_season_type(season)
                    
                    for date, _ in game_dates:
                        # Sync Closing Odds
                        if date not in existing_closing_dates:
                            try:
                                odds_df = self.get_nba_odds(date, current_line=True)
                                if not odds_df.empty:
                                    processed_df = self.process_odds_data(odds_df, season, date)
                                    if not processed_df.empty:
                                        processed_df.to_sql('nba_odds', conn, if_exists='append', index=False)
                                        logging.info(f"Added closing odds data for {date}")
                            except Exception as e:
                                logging.error(f"Error processing closing odds for {date}: {str(e)}")
                        
                        # Sync Opening Odds
                        if date not in existing_opening_dates:
                            try:
                                odds_df = self.get_nba_odds(date, current_line=False)
                                if not odds_df.empty:
                                    processed_df = self.process_odds_data(odds_df, season, date)
                                    if not processed_df.empty:
                                        processed_df.to_sql('nba_odds_opening', conn, if_exists='append', index=False)
                                        logging.info(f"Added opening odds data for {date}")
                            except Exception as e:
                                logging.error(f"Error processing opening odds for {date}: {str(e)}")
                                continue
                                
                    
                # Remove duplicates
                self.remove_duplicates('nba_odds', is_player_table=False)
                self.remove_duplicates('nba_odds_opening', is_player_table=False)
                
            logging.info("Odds sync completed")
            
        except Exception as e:
            logging.error(f"Error in odds sync: {str(e)}")
            raise
        
        

    def remove_duplicates(self, table_name: str, is_player_table: bool = False) -> None:
        """
        Remove duplicate entries from specified table, keeping the latest entry
        
        Parameters:
        table_name (str): Name of the table to deduplicate
        is_player_table (bool): If True, use PLAYER_ID for deduplication, else use TEAM_ID
        """
        logging.info(f"Removing duplicates from {table_name}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:

                # Determine ID column based on table type
                
                if 'hustle' in table_name.lower() or 'v3' in table_name.lower():
                    id_column = 'personId' if is_player_table else 'teamId'
                    game_id_col = 'gameId'

                elif table_name.lower() in ('defensive_stats_by_position', 'team_shot_location_boxscores'):
                    id_column = 'teamId'
                    game_id_col = 'gameDate'

                elif table_name.lower() in ('player_shot_location_boxscores'):
                    id_column = 'personId'
                    game_id_col = 'gameDate'
                    
                elif table_name.lower() in ('nba_odds', 'nba_odds_opening'):
                    id_column = 'home_team'
                    game_id_col = 'gameDate'
                    
                    
                else:
                    id_column = 'personId' if is_player_table else 'teamId'
                    game_id_col = 'gameDate' 
                    
                
                # Construct deduplication query
                dedup_query = f"""
                DELETE FROM {table_name}
                WHERE rowid NOT IN (
                    SELECT MAX(rowid)
                    FROM {table_name}
                    GROUP BY {id_column}, {game_id_col}
                )
                """
                
                cursor = conn.cursor()
                
                # Get count before deletion
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count_before = cursor.fetchone()[0]
                
                # Execute deduplication
                cursor.execute(dedup_query)
                conn.commit()
                
                # Get count after deletion
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count_after = cursor.fetchone()[0]
                
                duplicates_removed = count_before - count_after
                logging.info(f"Removed {duplicates_removed} duplicates from {table_name}")
            
        except Exception as e:
            logging.error(f"Error removing duplicates from {table_name}: {str(e)}")
            raise


    def deduplicate_all_tables(self) -> None:
        """
        Remove duplicates from all tables in the database
        """
        logging.info("Starting database-wide deduplication")
        
        # Define tables and their types
        team_tables = [
            'team_basic_stats',
            'team_advanced_stats_v3',
            'team_scoring_stats_v3',
            'team_hustle_stats',
            'team_track_stats_v3',
            'team_miscellaneous_stats_v3',
            'team_usage_stats_v3',
            'defensive_stats_by_position',
            'team_shot_location_boxscores',
            'nba_odds_opening' # Added opening odds to team tables for dedupe
        ]
        
        player_tables = [
            'player_basic_stats',
            'player_advanced_stats_v3',
            'player_scoring_stats_v3',
            'player_hustle_stats',
            'player_track_stats_v3',
            'player_miscellaneous_stats_v3',
            'player_usage_stats_v3',
            'player_shot_location_boxscores'
        ]
        
        try:
            # Process team tables
            for table in team_tables:
                self.remove_duplicates(table, is_player_table=False)
                
            # Process player tables
            for table in player_tables:
                self.remove_duplicates(table, is_player_table=True)
                
            logging.info("Database-wide deduplication completed successfully")
            
        except Exception as e:
            logging.error(f"Error during database-wide deduplication: {str(e)}")
            raise

    def process_table_pair(self, season: str, table_pair: TablePair, all_game_ids: Set[str]) -> None:
        """
        Process a single table pair for a season
        
        Parameters:
        season (str): Season in format 'YYYY-YY'
        table_pair (TablePair): Table pair to process
        all_game_ids (Set[str]): Set of all game IDs for the season
        """
        # Skip if season is before the stat was tracked
        if season < table_pair.min_season:
            logging.info(f"Skipping {table_pair.team_table} for season {season} (stat not tracked)")
            return

        logging.info(f"Processing {table_pair.team_table} for season {season}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Skip defensive stats as they're handled separately
                if table_pair.team_table in ('defensive_stats_by_position', 'team_shot_location_boxscores', 'player_shot_location_boxscores'):
                    logging.info("Skipping defensive_stats_by_position and team_shot_location_boxscores as they are handled separately")
                    return
                    
                # Get existing game IDs
                existing_ids = set(pd.read_sql(
                    f"SELECT DISTINCT {table_pair.id_column} FROM {table_pair.player_table}",
                    conn
                )[table_pair.id_column])
                
                # Find missing games
                missing_ids = all_game_ids - existing_ids
                
                if not missing_ids:
                    logging.info(f"No missing games for {table_pair.team_table}")
                    return
                
                logging.info(f"Found {len(missing_ids)} missing games for {table_pair.team_table}")
                
                # Process missing games
                for game_id in missing_ids:
                    try:
                        self.process_game(game_id, table_pair)

                    except Exception as e:
                        logging.error(f"Error processing game {game_id}: {str(e)}")
                
                # Remove duplicates after processing
                # self.remove_duplicates(table_pair.team_table, is_player_table=False)
                # self.remove_duplicates(table_pair.player_table, is_player_table=True)
                
        except Exception as e:
            logging.error(f"Error processing {table_pair.team_table}: {str(e)}")

    def rebuild_team_game_full(self) -> None:
        """
        Rebuild the team_game_full table by joining all component tables.
        This ensures the unified table is up-to-date with latest synced data.
        """
        logging.info("Rebuilding team_game_full table...")
        
        query = """
        CREATE TABLE team_game_full AS
        SELECT 
            tb.seasonYear, tb.teamId, tb.teamTricode, tb.gameId, tb.gameDate, tb.matchup, tb.wl,
            
            -- Basic Stats
            tb.minutes, tb.points, tb.plusMinus, tb.fieldGoalsMade, tb.fieldGoalsAttempted, tb.fieldGoalPercentage,
            tb.threePointersMade, tb.threePointersAttempted, tb.threePointerPercentage,
            tb.freeThrowsMade, tb.freeThrowsAttempted,
            coalesce(tb.freeThrowPercentage, 0) as freeThrowPercentage,
            tb.offensiveRebounds, tb.defensiveRebounds, tb.rebounds, tb.assists, tb.steals, tb.blocks, tb.turnovers, tb.personalFouls,

            -- Advanced
            ta.offensiveRating, ta.defensiveRating, ta.netRating, ta.assistPercentage, ta.assistToTurnover, ta.assistRatio,
            ta.offensiveReboundPercentage, ta.defensiveReboundPercentage, ta.reboundPercentage, ta.effectiveFieldGoalPercentage,
            ta.trueShootingPercentage, ta.usagePercentage, ta.pace, ta.pie,

            -- Scoring
            ts.percentageFieldGoalsAttempted2pt, ts.percentageFieldGoalsAttempted3pt, ts.percentagePoints2pt, ts.percentagePointsMidrange2pt,
            ts.percentagePoints3pt, ts.percentagePointsFastBreak, ts.percentagePointsFreeThrow, ts.percentagePointsOffTurnovers,
            ts.percentagePointsPaint, ts.percentageAssisted2pt, ts.percentageUnassisted2pt, ts.percentageAssisted3pt,
            ts.percentageUnassisted3pt, ts.percentageAssistedFGM, ts.percentageUnassistedFGM,

            -- Track
            tt.distance, tt.touches, tt.secondaryAssists, tt.freeThrowAssists, tt.passes,
            tt.contestedFieldGoalsMade, tt.contestedFieldGoalsAttempted, tt.contestedFieldGoalPercentage,
            tt.uncontestedFieldGoalsMade, tt.uncontestedFieldGoalsAttempted, tt.uncontestedFieldGoalsPercentage,
            tt.defendedAtRimFieldGoalsMade, tt.defendedAtRimFieldGoalsAttempted, tt.defendedAtRimFieldGoalPercentage,
            tt.reboundChancesOffensive, tt.reboundChancesDefensive, tt.reboundChancesTotal,

            -- Hustle
            -- th.contestedShots, th.contestedShots2pt, th.contestedShots3pt, th.deflections, th.chargesDrawn,
            -- th.screenAssists, th.screenAssistPoints, th.looseBallsRecoveredOffensive, th.looseBallsRecoveredDefensive,
            -- th.looseBallsRecoveredTotal, th.offensiveBoxOuts, th.defensiveBoxOuts, th.boxOutPlayerTeamRebounds,
            -- th.boxOutPlayerRebounds, th.boxOuts,

            -- Shot Location
            tsl."Restricted Area FGM", tsl."Restricted Area FGA", tsl."Restricted Area FG_PCT",
            tsl."In The Paint (Non-RA) FGM", tsl."In The Paint (Non-RA) FGA", tsl."In The Paint (Non-RA) FG_PCT",
            tsl."Mid-Range FGM", tsl."Mid-Range FGA", tsl."Mid-Range FG_PCT",
            tsl."Above the Break 3 FGM", tsl."Above the Break 3 FGA", tsl."Above the Break 3 FG_PCT",
            tsl."Corner 3 FGM", tsl."Corner 3 FGA", tsl."Corner 3 FG_PCT",

            -- Defense
              coalesce(dp.OPP_FGM_C, 0) as OPP_FGM_C
            , coalesce(dp.OPP_FGM_F, 0) as OPP_FGM_F
            , coalesce(dp.OPP_FGM_G, 0) as OPP_FGM_G
            , coalesce(dp.OPP_FGA_C, 0) as OPP_FGA_C
            , coalesce(dp.OPP_FGA_F, 0) as OPP_FGA_F
            , coalesce(dp.OPP_FGA_G, 0) as OPP_FGA_G
            , coalesce(dp.OPP_FG_PCT_C, 0) as OPP_FG_PCT_C
            , coalesce(dp.OPP_FG_PCT_F, 0) as OPP_FG_PCT_F
            , coalesce(dp.OPP_FG_PCT_G, 0) as OPP_FG_PCT_G
            , coalesce(dp.OPP_FG3M_C, 0) as OPP_FG3M_C
            , coalesce(dp.OPP_FG3M_F, 0) as OPP_FG3M_F
            , coalesce(dp.OPP_FG3M_G, 0) as OPP_FG3M_G
            , coalesce(dp.OPP_FG3A_C, 0) as OPP_FG3A_C
            , coalesce(dp.OPP_FG3A_F, 0) as OPP_FG3A_F
            , coalesce(dp.OPP_FG3A_G, 0) as OPP_FG3A_G
            , coalesce(dp.OPP_FG3_PCT_C, 0) as OPP_FG3_PCT_C
            , coalesce(dp.OPP_FG3_PCT_F, 0) as OPP_FG3_PCT_F
            , coalesce(dp.OPP_FG3_PCT_G, 0) as OPP_FG3_PCT_G
            , coalesce(dp.OPP_FTM_C, 0) as OPP_FTM_C
            , coalesce(dp.OPP_FTM_F, 0) as OPP_FTM_F
            , coalesce(dp.OPP_FTM_G, 0) as OPP_FTM_G
            , coalesce(dp.OPP_FTA_C, 0) as OPP_FTA_C
            , coalesce(dp.OPP_FTA_F, 0) as OPP_FTA_F
            , coalesce(dp.OPP_FTA_G, 0) as OPP_FTA_G
            , coalesce(dp.OPP_FT_PCT_C, 0) as OPP_FT_PCT_C
            , coalesce(dp.OPP_FT_PCT_F, 0) as OPP_FT_PCT_F
            , coalesce(dp.OPP_FT_PCT_G, 0) as OPP_FT_PCT_G
            , coalesce(dp.OPP_OREB_C, 0) as OPP_OREB_C
            , coalesce(dp.OPP_OREB_F, 0) as OPP_OREB_F
            , coalesce(dp.OPP_OREB_G, 0) as OPP_OREB_G
            , coalesce(dp.OPP_DREB_C, 0) as OPP_DREB_C
            , coalesce(dp.OPP_DREB_F, 0) as OPP_DREB_F
            , coalesce(dp.OPP_DREB_G, 0) as OPP_DREB_G
            , coalesce(dp.OPP_REB_C, 0) as OPP_REB_C
            , coalesce(dp.OPP_REB_F, 0) as OPP_REB_F
            , coalesce(dp.OPP_REB_G, 0) as OPP_REB_G
            , coalesce(dp.OPP_AST_C, 0) as OPP_AST_C
            , coalesce(dp.OPP_AST_F, 0) as OPP_AST_F
            , coalesce(dp.OPP_AST_G, 0) as OPP_AST_G
            , coalesce(dp.OPP_TOV_C, 0) as OPP_TOV_C
            , coalesce(dp.OPP_TOV_F, 0) as OPP_TOV_F
            , coalesce(dp.OPP_TOV_G, 0) as OPP_TOV_G
            , coalesce(dp.OPP_STL_C, 0) as OPP_STL_C
            , coalesce(dp.OPP_STL_F, 0) as OPP_STL_F
            , coalesce(dp.OPP_STL_G, 0) as OPP_STL_G
            , coalesce(dp.OPP_BLK_C, 0) as OPP_BLK_C
            , coalesce(dp.OPP_BLK_F, 0) as OPP_BLK_F
            , coalesce(dp.OPP_BLK_G, 0) as OPP_BLK_G
            , coalesce(dp.OPP_BLKA_C, 0) as OPP_BLKA_C
            , coalesce(dp.OPP_BLKA_F, 0) as OPP_BLKA_F
            , coalesce(dp.OPP_BLKA_G, 0) as OPP_BLKA_G
            , coalesce(dp.OPP_PF_C, 0) as OPP_PF_C
            , coalesce(dp.OPP_PF_F, 0) as OPP_PF_F
            , coalesce(dp.OPP_PF_G, 0) as OPP_PF_G
            , coalesce(dp.OPP_PFD_C, 0) as OPP_PFD_C
            , coalesce(dp.OPP_PFD_F, 0) as OPP_PFD_F
            , coalesce(dp.OPP_PFD_G, 0) as OPP_PFD_G
            , coalesce(dp.OPP_PTS_C, 0) as OPP_PTS_C
            , coalesce(dp.OPP_PTS_F, 0) as OPP_PTS_F
            , coalesce(dp.OPP_PTS_G, 0) as OPP_PTS_G
            , coalesce(dp.PLUS_MINUS_C, 0) as PLUS_MINUS_C_DEF
            , coalesce(dp.PLUS_MINUS_F, 0) as PLUS_MINUS_F_DEF
            , coalesce(dp.PLUS_MINUS_G, 0) as PLUS_MINUS_G_DEF
            
        FROM team_basic_stats tb
        
        LEFT JOIN team_advanced_stats_v3 ta ON tb.teamId = ta.teamId AND tb.gameId = ta.gameId
        LEFT JOIN team_scoring_stats_v3 ts ON tb.teamId = ts.teamId AND tb.gameId = ts.gameId
        LEFT JOIN team_track_stats_v3 tt ON tb.teamId = tt.teamId AND tb.gameId = tt.gameId
        -- LEFT JOIN team_hustle_stats th ON tb.teamId = th.teamId AND tb.gameId = th.gameId
        
        LEFT JOIN team_shot_location_boxscores tsl ON tb.teamId = tsl.teamId AND tb.gameDate = tsl.gameDate
        LEFT JOIN defensive_stats_by_position dp ON tb.teamId = dp.teamId AND tb.gameDate = dp.gameDate
        where tb.seasonYear >= '2013-14'
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                
                logging.info("Dropping existing team_game_full table...")
                cur.execute("DROP TABLE IF EXISTS team_game_full")
                
                logging.info("Executing CREATE TABLE query...")
                cur.execute(query)
                
                rows = cur.execute("SELECT COUNT(*) FROM team_game_full").fetchone()[0]
                logging.info(f"Table created successfully with {rows} rows.")
                
                logging.info("Creating indexes on team_game_full...")
                cur.execute("CREATE INDEX idx_full_team_date ON team_game_full(teamId, gameDate)")
                cur.execute("CREATE INDEX idx_full_game_id ON team_game_full(gameId)")
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error rebuilding team_game_full: {e}")
            raise
            
            
        
    def sync_database(self, seasons: List[str]) -> None:
        """
        Main method to synchronize the database
        """
        start_time = datetime.now()
        logging.info(f"Starting database sync for seasons: {seasons}")
        
        try:
            # First sync basic stats
            self.sync_basic_stats(seasons)
            
            # Then sync defensive stats
            self.sync_defensive_stats(seasons)
            
            # Then sync shot location stats
            self.sync_shot_location_stats(seasons)

            self.sync_player_shot_location_stats(seasons)
            # Sync odds data
            
            self.sync_odds_data(seasons)

            # Then process detailed stats for each game
            for season in seasons:
                logging.info(f"Processing detailed stats for season {season}")
                
                # Fetch all game IDs for the season
                all_game_ids = self.fetch_game_ids(season)
                logging.info(f"Found {len(all_game_ids)} total games for season {season}")
                
                # Process each table pair
                for table_pair in self.table_pairs:
                    self.process_table_pair(season, table_pair, all_game_ids)
            
                # time.sleep(300)
            # Final deduplication of all tables
            self.deduplicate_all_tables()
            
            # Rebuild the full team game table (unified source for features)
            self.rebuild_team_game_full()
            
            # Update prediction results
            try:
                logging.info("Updating prediction logs with actual results...")
                logger = PredictionLogger(self.db_path)
                logger.update_actual_scores()
            except Exception as e:
                logging.warning(f"Failed to update prediction results: {e}")
            
        except Exception as e:
            logging.error(f"Error in database sync: {str(e)}")
            raise
            
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Database sync completed in {duration}")
        
# Usage example
if __name__ == "__main__":
    seasons_to_check = ['2025-26']
    db_path = Path.cwd() / 'data' / 'nba_stats.db'
    
    syncer = NBADatabaseSync(db_path)

    # Regular sync will now include deduplication

    syncer.sync_database(seasons_to_check)


