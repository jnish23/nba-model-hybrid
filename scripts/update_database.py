import sqlite3
import numpy as np
import pandas as pd
import time
import json
import random
import logging
import backoff
from pathlib import Path
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
from models.sbr import Scoreboard

# Add root to sys.path to allow importing models when run from scripts/
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prediction_logger import PredictionLogger

def custom_backoff_generator(**kwargs):
    """
    Fixed retry delays for stats.nba.com / SBR calls (seconds). Kept short so a bad API day
    cannot push the daily run (12:15 PM ET) toward tip-off: worst case ~2 min per call.
    """
    yield 10
    yield 30
    yield 60


# Only transient network / response errors are retried; programming errors fail immediately.
RETRYABLE = (RequestException, Timeout, ConnectionError, json.JSONDecodeError, KeyError, IndexError)
RETRY = dict(max_tries=4, max_time=180)

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
        """
        Log to `log_path` as well as the console. basicConfig is a no-op when the caller (e.g.
        run_inference_v2) already configured logging, so the file handler is attached directly.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        root = logging.getLogger()
        target = str(Path(log_path).resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == target for h in root.handlers):
            fh = logging.FileHandler(target)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            root.addHandler(fh)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)


    @backoff.on_exception(custom_backoff_generator, RETRYABLE, **RETRY,
                          on_backoff=lambda d: logging.error(f"Retrying in {d['wait']}s after error: {d['exception']}"))
    def _fetch_team_game_log(self, season: str, season_type: str) -> pd.DataFrame:
        return leaguegamelog.LeagueGameLog(season=season, season_type_all_star=season_type).get_data_frames()[0]

    @backoff.on_exception(custom_backoff_generator, RETRYABLE, **RETRY,
                          on_backoff=lambda d: logging.error(f"Retrying in {d['wait']}s after error: {d['exception']}"))
    def _fetch_player_game_logs(self, season: str, season_type: str) -> pd.DataFrame:
        return playergamelogs.PlayerGameLogs(season_nullable=season, season_type_nullable=season_type).get_data_frames()[0]

    @staticmethod
    def _dedupe_season(conn, table_name: str, key_cols: str, season: str) -> None:
        """Keep the latest row per key within one season (cheaper than a whole-table pass)."""
        conn.execute(f'''
            DELETE FROM {table_name}
            WHERE seasonYear = ? AND rowid NOT IN (
                SELECT max(rowid) FROM {table_name} WHERE seasonYear = ? GROUP BY {key_cols}
            )
        ''', [season, season])

    # ------------------------------------------------------------------ completeness helpers

    MAX_ATTEMPTS = 3          # give up on an item that keeps returning nothing (e.g. endpoint has no data)
    RECHECK_DAYS = 14         # dates this recent are re-fetched when a positional bucket is empty

    def _ensure_attempts_table(self, conn) -> None:
        conn.execute("""CREATE TABLE IF NOT EXISTS sync_attempts (
                            source TEXT, item TEXT, attempts INTEGER, last_attempt TEXT, last_error TEXT,
                            PRIMARY KEY (source, item))""")

    def _should_attempt(self, conn, source: str, item: str) -> bool:
        self._ensure_attempts_table(conn)
        row = conn.execute("SELECT attempts FROM sync_attempts WHERE source = ? AND item = ?", [source, item]).fetchone()
        return row is None or row[0] < self.MAX_ATTEMPTS

    def _record_attempt(self, conn, source: str, item: str, ok: bool, error: str = '') -> None:
        self._ensure_attempts_table(conn)
        if ok:
            conn.execute("DELETE FROM sync_attempts WHERE source = ? AND item = ?", [source, item])
        else:
            conn.execute("""INSERT INTO sync_attempts (source, item, attempts, last_attempt, last_error)
                            VALUES (?, ?, 1, datetime('now'), ?)
                            ON CONFLICT(source, item) DO UPDATE SET attempts = attempts + 1,
                                last_attempt = datetime('now'), last_error = excluded.last_error""",
                         [source, item, str(error)[:500]])
        conn.commit()

    def _append(self, conn, df: pd.DataFrame, table: str) -> None:
        """
        Append df to table, aligned to the table's existing columns. The NBA API adds columns
        over time (e.g. FP_HIGH_SCORE in PlayerGameLogs), and pandas' append fails on any column
        the table lacks - which aborted the whole sync. New columns are dropped (and logged);
        columns the API no longer returns are written as NULL.
        """
        if df is None or df.empty:
            return
        if self._table_exists(conn, table):
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            extra = [c for c in df.columns if c not in cols]
            if extra:
                logging.warning(f"{table}: ignoring {len(extra)} new API column(s) not in the table: {extra[:10]}")
            df = df.reindex(columns=cols)
        df.to_sql(table, conn, if_exists='append', index=False)

    def _table_exists(self, conn, table: str) -> bool:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", [table]).fetchone() is not None

    def _dates_to_fetch(self, conn, table: str, season: str, per: str = 'team', null_check_cols=None):
        """
        (date, season_type) pairs for `season` that are missing or incomplete in `table`:
          per='team' -> fewer distinct teamIds than team_basic_stats has for that date
          per='game' -> fewer rows than games that date (odds tables, one row per game)
        With null_check_cols, recent dates (RECHECK_DAYS) where any of those columns is NULL are
        re-fetched too. Items that already failed MAX_ATTEMPTS times are skipped.
        (Previously a date counted as done once ANY row existed, so a partial fetch - one
        positional call failing, or a sync while games were in progress - was never repaired.)
        """
        expected = pd.read_sql("""SELECT gameDate, COUNT(DISTINCT teamId) AS n_teams
                                  FROM team_basic_stats WHERE seasonYear = ? GROUP BY gameDate""",
                               conn, params=[season])
        if per == 'game':
            expected['n'] = expected['n_teams'] // 2
            have_sql = f"SELECT gameDate, COUNT(*) AS have FROM {table} WHERE season = ? GROUP BY gameDate"
        else:
            expected['n'] = expected['n_teams']
            have_sql = f"SELECT gameDate, COUNT(DISTINCT teamId) AS have FROM {table} WHERE season = ? GROUP BY gameDate"
        if self._table_exists(conn, table):
            have = pd.read_sql(have_sql, conn, params=[season])
        else:
            have = pd.DataFrame(columns=['gameDate', 'have'])
        m = expected.merge(have, on='gameDate', how='left')
        m['have'] = pd.to_numeric(m['have'], errors='coerce').fillna(0)
        todo = set(m.loc[m['have'] < m['n'], 'gameDate'])
        if null_check_cols and self._table_exists(conn, table):
            cutoff = (pd.Timestamp.today() - pd.Timedelta(days=self.RECHECK_DAYS)).strftime('%Y-%m-%d')
            cond = ' OR '.join(f'"{c}" IS NULL' for c in null_check_cols)
            todo |= set(pd.read_sql(f"SELECT DISTINCT gameDate FROM {table} WHERE season = ? AND gameDate >= ? AND ({cond})",
                                    conn, params=[season, cutoff])['gameDate'])
        types = dict(self.fetch_game_date_and_season_type(season))
        out = [(d, types.get(d, 'Regular Season')) for d in sorted(todo) if self._should_attempt(conn, table, d)]
        if out:
            logging.info(f"{table} {season}: {len(out)} dates missing or incomplete")
        return out

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
                            team_boxscores = self._fetch_team_game_log(season, season_type)
                            
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
                            # Fail loudly: silently continuing left the DB stale while inference ran anyway
                            raise RuntimeError(f"Could not retrieve team {season_type} game log for {season} "
                                               f"after retries: {e}") from e
                        
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
                        self._append(conn, season_df, table_name)
                        
                        # Remove duplicates within this season (keeps the latest row per team-game)
                        self._dedupe_season(conn, table_name, 'teamId, gameId', season)
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
                            player_boxscores = self._fetch_player_game_logs(season, season_type)
                            
                            if not player_boxscores.empty:
                                
                                # Remove rows that have NaN values in 'WL' column indicating game not complete yet (still being played)
                                player_boxscores = player_boxscores.dropna(subset=['WL'])


                                player_boxscores = player_boxscores.drop(columns=['TEAM_COUNT'], errors='ignore')
                                season_boxscores.append(player_boxscores)
                                logging.info(f"Retrieved {len(player_boxscores)} player games for {season_type} {season}")
                            else:
                                logging.warning(f"No player data retrieved for {season_type} {season}")
                                
                        except Exception as e:
                            raise RuntimeError(f"Could not retrieve player {season_type} game logs for {season} "
                                               f"after retries: {e}") from e
                        
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
                        self._append(conn, season_df, table_name)
                        
                        # Remove duplicates within this season (keeps the latest row per player-game)
                        self._dedupe_season(conn, table_name, 'personId, gameId', season)
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
        RETRYABLE,
        **RETRY,
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
                    
                    todo = self._dates_to_fetch(conn, 'defensive_stats_by_position', season,
                                                null_check_cols=[f'OPP_PTS_{p}' for p in self.POSITIONS])
                    for date, season_type in todo:
                        try:
                            raw_data = self.get_all_defensive_stats(date, season, season_type)
                            processed_data = self.process_defensive_stats(raw_data)
                            if processed_data is not None:
                                self._append(conn, processed_data, 'defensive_stats_by_position')
                                logging.info(f"Added defensive stats for {date}")
                            self._record_attempt(conn, 'defensive_stats_by_position', date, processed_data is not None, 'no data')
                        except Exception as e:
                            logging.error(f"Error processing date {date}: {str(e)}")
                            self._record_attempt(conn, 'defensive_stats_by_position', date, False, str(e))

                self.remove_duplicates('defensive_stats_by_position', is_player_table=False)
            logging.info("Defensive stats sync completed")
            
        except Exception as e:
            logging.error(f"Error in defensive stats sync: {str(e)}")
            raise
        
        
    @backoff.on_exception(
        custom_backoff_generator,
        RETRYABLE,
        **RETRY,
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
        
        # Process column names (flatten multi-level headers if present, strip whitespace)
        df.columns = df.columns.map(' '.join) if isinstance(df.columns[0], tuple) else df.columns
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
                    
                    for date, season_type in self._dates_to_fetch(conn, 'team_shot_location_boxscores', season):
                        try:
                            df = self.get_shot_location_stats(date, season, season_type)
                            ok = df is not None and not df.empty
                            if ok:
                                self._append(conn, df.rename(columns=self.identifier_map), 'team_shot_location_boxscores')
                                logging.info(f"Added shot location stats for {date} ({season_type})")
                            self._record_attempt(conn, 'team_shot_location_boxscores', date, ok, 'no data')
                        except Exception as e:
                            logging.error(f"Error processing {date} ({season_type}): {str(e)}")
                            self._record_attempt(conn, 'team_shot_location_boxscores', date, False, str(e))
                        time.sleep(1+random.gammavariate(1, 3))

                # Remove duplicates
                self.remove_duplicates('team_shot_location_boxscores', is_player_table=False)
                
            logging.info("Shot location stats sync completed")
            
        except Exception as e:
            logging.error(f"Error in shot location stats sync: {str(e)}")
            raise

    @backoff.on_exception(
        custom_backoff_generator,
        RETRYABLE,
        **RETRY,
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
                        
                        # Dates missing, or with fewer teams than played that day
                        for date, season_type in self._dates_to_fetch(conn, 'player_shot_location_boxscores', season):
                            try:
                                df = self.get_player_shot_location_stats(date, season, season_type)
                                ok = df is not None and not df.empty
                                if ok:
                                    self._append(conn, df.rename(columns=self.identifier_map), 'player_shot_location_boxscores')
                                    logging.info(f"Added player shot location stats for {date} ({season_type})")
                                self._record_attempt(conn, 'player_shot_location_boxscores', date, ok, 'no data')
                            except Exception as e:
                                logging.error(f"Error processing {date} ({season_type}): {str(e)}")
                                self._record_attempt(conn, 'player_shot_location_boxscores', date, False, str(e))
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
        RETRYABLE,
        **RETRY,
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
        Fetch one game's box score for a table pair and store player + team rows in ONE
        transaction (previously the player rows could be written without the team rows, and
        the game was then never retried because only the player table was checked).
        Returns True if rows were stored.
        """
        try:
            game_data = self.get_game_data(table_pair.team_table, game_id)
            if not game_data or len(game_data) < 2:
                logging.warning(f"No data available for game {game_id} in {table_pair.team_table}")
                return False
            player_df, team_df = game_data[0], game_data[1]
            if team_df.empty:
                logging.warning(f"Empty team data for game {game_id} in {table_pair.team_table}")
                return False

            player_df = player_df.rename(columns=self.identifier_map)
            team_df = team_df.rename(columns=self.identifier_map)

            with sqlite3.connect(self.db_path) as conn:
                try:
                    self._append(conn, player_df, table_pair.player_table)
                    self._append(conn, team_df, table_pair.team_table)
                except Exception:
                    # pandas commits each to_sql call, so undo by hand: remove this game from both
                    # tables (it was incomplete before) so it is cleanly re-fetched next run
                    for table in (table_pair.player_table, table_pair.team_table):
                        if self._table_exists(conn, table):
                            conn.execute(f"DELETE FROM {table} WHERE gameId = ?", [game_id])
                    conn.commit()
                    raise

            logging.info(f"Successfully added data for game {game_id} to {table_pair.team_table}")
            return True

        except Exception as e:
            logging.error(f"Error processing game {game_id} for {table_pair.team_table}: {str(e)}")
            return False

    @backoff.on_exception(
        custom_backoff_generator,
        RETRYABLE,
        **RETRY,
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
        """
        Sync opening (nba_odds_opening) and closing (nba_odds) odds for completed game dates.
        Dates with fewer odds rows than games are re-fetched (previously a date counted as done
        once any row existed). Closing lines are scraped after the games; for clean, timestamped
        pre-game lines see scripts/snapshot_odds.py (models/odds_clean.py prefers those).
        """
        logging.info(f"Starting odds sync for seasons: {seasons}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                for season in seasons:
                    if season < '2019-20':
                        continue
                    for table, current in (('nba_odds', True), ('nba_odds_opening', False)):
                        label = 'closing' if current else 'opening'
                        for date, _ in self._dates_to_fetch(conn, table, season, per='game'):
                            try:
                                odds_df = self.get_nba_odds(date, current_line=current)
                                processed_df = self.process_odds_data(odds_df, season, date) if not odds_df.empty else odds_df
                                ok = not processed_df.empty
                                if ok:
                                    self._append(conn, processed_df, table)
                                    logging.info(f"Added {label} odds data for {date}")
                                self._record_attempt(conn, table, date, ok, 'no data')
                            except Exception as e:
                                logging.error(f"Error processing {label} odds for {date}: {str(e)}")
                                self._record_attempt(conn, table, date, False, str(e))

                self.remove_duplicates('nba_odds', is_player_table=False)
                self.remove_duplicates('nba_odds_opening', is_player_table=False)
            logging.info("Odds sync completed")
        except Exception as e:
            logging.error(f"Error in odds sync: {str(e)}")
            raise

    @backoff.on_exception(custom_backoff_generator, RETRYABLE, **RETRY,
                          on_backoff=lambda d: logging.error(f"Retrying in {d['wait']}s after error: {d['exception']}"))
    def _fetch_player_info(self, person_id: int) -> pd.DataFrame:
        from nba_api.stats.endpoints import commonplayerinfo
        return commonplayerinfo.CommonPlayerInfo(player_id=person_id).get_data_frames()[0]

    def sync_player_info(self, seasons: List[str]) -> None:
        """
        Add player_info rows (position etc.) for players who appeared in `seasons` but are not in
        player_info yet - e.g. rookies. BPM uses the position; without a row a player gets the
        default position. One API call per new player.
        """
        with sqlite3.connect(self.db_path) as conn:
            known = set(pd.read_sql("SELECT PERSON_ID FROM player_info", conn)['PERSON_ID'])
            existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(player_info)")]
            placeholders = ','.join('?' * len(seasons))
            seen = pd.read_sql(f"SELECT DISTINCT personId FROM player_basic_stats WHERE seasonYear IN ({placeholders})",
                               conn, params=seasons)['personId']
            new_ids = sorted(set(seen) - known)
            logging.info(f"player_info: {len(new_ids)} new players to add")
            for pid in new_ids:
                if not self._should_attempt(conn, 'player_info', str(pid)):
                    continue
                try:
                    info = self._fetch_player_info(int(pid)).rename(columns=self.identifier_map)
                    ok = not info.empty
                    if ok:
                        info = info.reindex(columns=existing_cols)  # tolerate API column changes
                        self._append(conn, info, 'player_info')
                    self._record_attempt(conn, 'player_info', str(pid), ok, 'no data')
                except Exception as e:
                    logging.error(f"player_info for {pid} failed: {e}")
                    self._record_attempt(conn, 'player_info', str(pid), False, str(e))
                time.sleep(1 + random.random())

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

                elif table_name.lower() == 'player_shot_location_boxscores':
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
        Fetch games of `season` missing from the team OR player table of a per-game table pair.
        Games that keep returning no data are skipped after MAX_ATTEMPTS tries (sync_attempts).
        """
        if season < table_pair.min_season:
            logging.info(f"Skipping {table_pair.team_table} for season {season} (stat not tracked)")
            return
        if table_pair.team_table in ('defensive_stats_by_position', 'team_shot_location_boxscores',
                                     'player_shot_location_boxscores'):
            return  # date-based tables, synced by their own methods

        logging.info(f"Processing {table_pair.team_table} for season {season}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                def ids(table):
                    if not self._table_exists(conn, table):
                        return set()
                    return set(pd.read_sql(f"SELECT DISTINCT {table_pair.id_column} FROM {table}", conn)[table_pair.id_column])

                complete = ids(table_pair.team_table) & ids(table_pair.player_table)
                missing = sorted(g for g in all_game_ids - complete
                                 if self._should_attempt(conn, table_pair.team_table, g))
                if not missing:
                    logging.info(f"No missing games for {table_pair.team_table}")
                    return
                logging.info(f"Found {len(missing)} missing games for {table_pair.team_table}")

                for game_id in missing:
                    ok = self.process_game(game_id, table_pair)
                    self._record_attempt(conn, table_pair.team_table, game_id, ok, '' if ok else 'no data or error')

        except Exception as e:
            logging.error(f"Error processing {table_pair.team_table}: {str(e)}")

    # Exact opponent totals come from the opponent's own row in team_basic_stats.
    OPP_FROM_BOX = {
        'OPP_FGM': 'fieldGoalsMade', 'OPP_FGA': 'fieldGoalsAttempted', 'OPP_FG_PCT': 'fieldGoalPercentage',
        'OPP_FG3M': 'threePointersMade', 'OPP_FG3A': 'threePointersAttempted', 'OPP_FG3_PCT': 'threePointerPercentage',
        'OPP_FTM': 'freeThrowsMade', 'OPP_FTA': 'freeThrowsAttempted', 'OPP_FT_PCT': 'freeThrowPercentage',
        'OPP_OREB': 'offensiveRebounds', 'OPP_DREB': 'defensiveRebounds', 'OPP_REB': 'rebounds',
        'OPP_AST': 'assists', 'OPP_TOV': 'turnovers', 'OPP_STL': 'steals', 'OPP_BLK': 'blocks',
        'OPP_PF': 'personalFouls', 'OPP_PTS': 'points',
    }
    POSITIONAL_STATS = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
                        'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS']

    def rebuild_team_game_full(self) -> None:
        """
        Rebuild team_game_full by joining the component tables.

        - OPP_* (no position suffix): the opponent's actual totals, from the opponent's row in
          team_basic_stats. (Features used to sum the C/F/G columns below, which overstates
          opponent totals by ~23% because the position buckets overlap.)
        - OPP_<stat>_<C|F|G>, PLUS_MINUS_<C|F|G>_DEF: from LeagueDashTeamStats (Opponent) filtered
          by player position. The filter applies to the DEFENDING team's own players: these are the
          opponent's stats while this team's C/F/G-listed players were on the floor, NOT the
          stats of the opponent's centers/forwards/guards. Hybrid positions (G-F, F-C) fall in
          more than one bucket, so the three do not add up to the game total. Kept for future
          use; missing values stay NULL (they used to be coalesced to 0).
        - Built into a new table and swapped in within one transaction, so a failed rebuild
          never leaves the database without team_game_full.
        """
        logging.info("Rebuilding team_game_full table...")

        opp_cols = ',\n            '.join(f'opp.{src} AS {dst}' for dst, src in self.OPP_FROM_BOX.items())
        pos_cols = ',\n            '.join(
            [f'dp.OPP_{stat}_{p} AS OPP_{stat}_{p}' for stat in self.POSITIONAL_STATS for p in self.POSITIONS] +
            [f'dp.PLUS_MINUS_{p} AS PLUS_MINUS_{p}_DEF' for p in self.POSITIONS])

        query = f"""
        CREATE TABLE team_game_full_new AS
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

            -- Shot Location
            tsl."Restricted Area FGM", tsl."Restricted Area FGA", tsl."Restricted Area FG_PCT",
            tsl."In The Paint (Non-RA) FGM", tsl."In The Paint (Non-RA) FGA", tsl."In The Paint (Non-RA) FG_PCT",
            tsl."Mid-Range FGM", tsl."Mid-Range FGA", tsl."Mid-Range FG_PCT",
            tsl."Above the Break 3 FGM", tsl."Above the Break 3 FGA", tsl."Above the Break 3 FG_PCT",
            tsl."Corner 3 FGM", tsl."Corner 3 FGA", tsl."Corner 3 FG_PCT",

            -- Opponent totals (exact, from the opponent's box score)
            {opp_cols},

            -- Opponent stats while this team's C/F/G-listed players were on the floor (see docstring)
            {pos_cols}

        FROM team_basic_stats tb
        LEFT JOIN team_basic_stats opp ON opp.gameId = tb.gameId AND opp.teamId <> tb.teamId
        LEFT JOIN team_advanced_stats_v3 ta ON tb.teamId = ta.teamId AND tb.gameId = ta.gameId
        LEFT JOIN team_scoring_stats_v3 ts ON tb.teamId = ts.teamId AND tb.gameId = ts.gameId
        LEFT JOIN team_track_stats_v3 tt ON tb.teamId = tt.teamId AND tb.gameId = tt.gameId
        LEFT JOIN team_shot_location_boxscores tsl ON tb.teamId = tsl.teamId AND tb.gameDate = tsl.gameDate
        LEFT JOIN defensive_stats_by_position dp ON tb.teamId = dp.teamId AND tb.gameDate = dp.gameDate
        WHERE tb.seasonYear >= '2013-14'
        """

        conn = sqlite3.connect(self.db_path)
        conn.isolation_level = None  # manage the transaction explicitly (DDL included)
        try:
            cur = conn.cursor()
            cur.execute("DROP TABLE IF EXISTS team_game_full_new")
            cur.execute(query)
            n_new = cur.execute("SELECT COUNT(*) FROM team_game_full_new").fetchone()[0]
            n_base = cur.execute("SELECT COUNT(*) FROM team_basic_stats WHERE seasonYear >= '2013-14'").fetchone()[0]
            if n_new != n_base:
                raise RuntimeError(f"team_game_full_new has {n_new} rows but team_basic_stats has {n_base}: "
                                   "a component table has duplicate keys")
            cur.execute("BEGIN")
            cur.execute("DROP TABLE IF EXISTS team_game_full")
            cur.execute("ALTER TABLE team_game_full_new RENAME TO team_game_full")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_full_team_date ON team_game_full(teamId, gameDate)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_full_game_id ON team_game_full(gameId)")
            cur.execute("COMMIT")
            logging.info(f"team_game_full rebuilt with {n_new} rows.")
        except Exception as e:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.execute("DROP TABLE IF EXISTS team_game_full_new")
            logging.error(f"Error rebuilding team_game_full (existing table kept): {e}")
            raise
        finally:
            conn.close()

    def sync_database(self, seasons: List[str]) -> None:
        """
        Main method to synchronize the database
        """
        start_time = datetime.now()
        logging.info(f"Starting database sync for seasons: {seasons}")
        
        try:
            # First sync basic stats
            self.sync_basic_stats(seasons)

            # Positions etc. for players new to the database (e.g. rookies)
            try:
                self.sync_player_info(seasons)
            except Exception as e:
                logging.warning(f"player_info sync failed (continuing): {e}")
            
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
    from models.season_utils import season_for_date
    seasons_to_check = [season_for_date()]
    db_path = Path.cwd() / 'data' / 'nba_stats.db'

    syncer = NBADatabaseSync(db_path)
    syncer.sync_database(seasons_to_check)
