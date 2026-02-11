
import pandas as pd
import numpy as np
import sqlite3
import logging

class TravelFeatureBuilder:
    def __init__(self, db_path):
        self.db_path = db_path
        
    def load_data(self):
        with sqlite3.connect(self.db_path) as conn:
            # 1. Load Schedule (Team Basic Stats)
            # We need gameDate to sort.
            # We need MATCHUP to determine Home/Away and Opponent.
            self.df_games = pd.read_sql("SELECT teamId, teamTricode, gameId, gameDate, MATCHUP FROM team_basic_stats ORDER BY gameDate", conn)
            
            # 2. Load Arenas
            self.df_arenas = pd.read_sql("SELECT * FROM nba_arenas", conn)
            
    def _haversine_np(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees) - Vectorized
        """
        R = 3959.87433 # Miles
        
        dLat = np.radians(lat2 - lat1)
        dLon = np.radians(lon2 - lon1)
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
        
        a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def build_features(self):
        logging.info("Building Travel Features...")
        
        if not hasattr(self, 'df_games'):
            self.load_data()
            
        df = self.df_games.copy()
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        
        # 1. Identify Opponent and Location
        # Matchup format: "BOS @ NYC" (Away) or "BOS vs. NYC" (Home)
        
        def parse_opponent(matchup):
            # Split by space
            parts = matchup.split(' ')
            opp = parts[-1]
            return opp
            
        df['oppTricode'] = df['MATCHUP'].apply(parse_opponent)
        df['is_home'] = df['MATCHUP'].str.contains('vs.')
        
        # 2. Determine Current Game Venue (Tricode)
        # If Home: Venue = Me. If Away: Venue = Opp.
        df['venueTricode'] = np.where(df['is_home'], df['teamTricode'], df['oppTricode'])
        
        # 3. Join Arena Data to get Lat/Long/TZ/Alt for the VENUE
        # We join on 'venueTricode' -> 'teamTricode' in arenas
        # Filter columns to avoid collision with teamId
        arena_cols = ['teamTricode', 'lat', 'long', 'altitude', 'tz_offset']
        arena_map = self.df_arenas[arena_cols].set_index('teamTricode')
        
        # We need efficient join.
        df = df.merge(arena_map, left_on='venueTricode', right_index=True, how='left')
        
        # Now we have 'lat', 'long', 'altitude', 'tz_offset' for the CURRENT GAME LOCATION.
        
        # 4. Sort by Team and Date
        df = df.sort_values(['teamId', 'gameDate'])
        
        # 5. Shift to get Previous Location
        # Group by teamId
        grouped = df.groupby('teamId')
        
        df['prev_lat'] = grouped['lat'].shift(1)
        df['prev_long'] = grouped['long'].shift(1)
        df['prev_tz'] = grouped['tz_offset'].shift(1)
        
        # 6. Calculate Distance (Miles)
        # If prev is NaN (first game), distance = 0 (or NaN). Let's say 0.
        df['dist_miles'] = self._haversine_np(df['prev_lat'], df['prev_long'], df['lat'], df['long'])
        df['dist_miles'] = df['dist_miles'].fillna(0)
        
        # 7. Time Zone Delta
        # Current - Prev
        df['tz_delta'] = df['tz_offset'] - df['prev_tz']
        df['tz_delta'] = df['tz_delta'].fillna(0)
        
        # 8. Body Clock Lag
        # "Rolling average of last 3 days"
        # Since we have rows per game, and games aren't daily, strictly speaking we should use a time-series index.
        # But simpler: Rolling average of last 3 *games*? 
        # User said: "last 3 days' time zones". This implies filling missing dates.
        # That's complex. "estimate their 'internal' time vs. 'game' time."
        # Let's use an EWMA on the TZ Sequence with a small span (e.g., 3 games ~ 5-7 days).
        # Or span=3 (games).
        
        # We allow pandas ewm on the grouped object
        # internal_tz = ewma(tz_offset)
        # We want the internal clock *entering* the game, so we should shift the EWMA?
        # "If a team has been in ET for 24 hours... internal is still PT".
        # So Internal Clock is driven by Previous Locations.
        # Let's take EWMA of 'prev_tz'.
        
        df['body_clock_est'] = grouped['prev_tz'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
        # If NaN (start of season), assume matched.
        df['body_clock_est'] = df['body_clock_est'].fillna(df['tz_offset'])
        
        df['body_clock_lag'] = df['tz_offset'] - df['body_clock_est']
        
        # 9. Road Trip Length
        # Count consecutive 'is_home' == False
        # Vectorized approach:
        # Create groups where 'is_home' changes.
        # cumsum() of check.
        
        # Boolean series: is_away
        df['is_away'] = ~df['is_home']
        
        # Reset at Home:
        # We want a counter that increments if away, resets to 0 if home.
        # Logic:
        # g = (is_away != is_away.shift()).cumsum()
        # This groups consecutive runs.
        # Then inside groups, cumsum().
        # If is_home is True, set to 0.
        
        # Better: Iterate or use complex groupby.
        # Iteration is ok for 30 teams * 82 games * 20 years = 50k rows. Instant.
        
        def count_road_trip(series):
            # series is 'is_away' boolean
            counts = []
            c = 0
            for x in series:
                if x:
                    c += 1
                else:
                    c = 0
                counts.append(c)
            return pd.Series(counts, index=series.index)
            
        df['road_trip_count'] = grouped['is_away'].apply(count_road_trip).reset_index(level=0, drop=True)
        
        # 10. Altitude Advantage (Game Level)
        # User defined: "Home_Altitude - Away_Altitude".
        # This is constant for the game.
        # We need the HOME team's altitude (Venue Altitude)
        # And AWAY team's altitude (Venue of Opponent? No, altitude of Opponent's Home City?)
        # "Altitude Advantage" implies the playing conditions favor the acclimated team.
        # If Game is in Denver:
        #   Home (Denver): Alt 5280.
        #   Away (Boston): Alt 20.
        #   Advantage = 5280 - 20 = 5260.
        # If Game is in Boston:
        #   Home (Boston): Alt 20.
        #   Away (Denver): Alt 5280.
        #   Advantage = 20 - 5280 = -5260.
        # This implies "Home Field Altitude Advantage".
        # So Calculation: Venue_Altitude - Opponent_Home_Altitude.
        
        # We have Venue_Altitude (Already merged as 'altitude').
        # We need Opponent_Home_Altitude.
        # Make a lookup dict from arenas
        opp_alt_map = self.df_arenas.set_index('teamTricode')['altitude'].to_dict()
        
        df['opp_home_altitude'] = df['oppTricode'].map(opp_alt_map)
        df['altitude_advantage'] = df['altitude'] - df['opp_home_altitude']
        
        # If I am Away team, "Home_Altitude - Away_Altitude" refers to HOST - VISITOR.
        # If I am Away, I am the VISITOR. HOST is Opponent.
        # So calculation is: Venue_Alt (Opp's) - My_Home_Alt.
        # Variable `altitude_advantage` defined above:
        # Me=Away(BOS) at DEN. Venue=DEN(5280). MyHome=BOS(20).
        # Calc: 5280 - 20 = 5260.
        # This correctly represents the "Mountain Advantage" existing in the game.
        # But for the MODEL, is this feature "How much altitude advantage does THIS TEAM have?"
        # If I am Boston at Denver, I have a DISADVANTAGE (-5260).
        # My formula gave +5260.
        # I should invert it if I am Away?
        
        # "altitude advantage: Home_Altitude - Away_Altitude"
        # User definition is strictly Home - Away.
        # This is a Game Attribute. "This game has an alt diff of 5000".
        # It correlates with Home Win %.
        # If I feed this as a feature for BOTH teams:
        #   Home Team Feature: +5000.
        #   Away Team Feature: +5000.
        #   The model learns "When AltDiff is high, Home team wins more". Correct.
        # So I will stick to the user's explicit formula: Home_Alt - Away_Alt.
        
        # Wait, if I am Home: Altitude = MyAlt. OppHome = AwayAlt. Diff = My - Opp.
        # If I am Away: Altitude = OppAlt. OppHome = MyAlt. Diff = Opp - My.
        # So `altitude - opp_home_altitude` IS the formula `Home_Alt - Away_Alt`.
        # EXCEPT when it's a neutral game? Assuming no neutral for now.
        # Yes, `df['altitude']` is Venue Altitude.
        # If Home, Venue=Me.
        # If Away, Venue=Opp.
        # So `altitude` is always HOME altitude.
        # `opp_home_altitude` is ALWAYS the OTHER team's home altitude.
        # If I am Home, Opp=Away. Corret.
        # If I am Away, Opp=Home. Wait.
        # If I am Away, my `oppTricode` is the Home Team.
        # So `opp_home_altitude` is HOME altitude.
        # And `altitude` (Venue) is HOME altitude.
        # So Diff = 0?
        
        # LOGIC CHECK:
        # Case 1: BOS (Away) @ DEN (Home).
        #   Me=BOS. Opp=DEN.
        #   Venue = DEN (from is_home=False logic). `altitude` = 5280.
        #   OppTricode = DEN. `opp_home_altitude` = 5280.
        #   Diff = 0. WRONG.
        
        # Case 2: DEN (Home) vs BOS (Away).
        #   Me=DEN. Opp=BOS.
        #   Venue = DEN. `altitude` = 5280.
        #   OppTricode = BOS. `opp_home_altitude` = 20.
        #   Diff = 5260. CORRECT.
        
        # I need `Home_Team_Home_Altitude` and `Away_Team_Home_Altitude`.
        # If is_home:
        #   HomeAlt = My_Home_Alt. AwayAlt = Opp_Home_Alt.
        # If is_away:
        #   HomeAlt = Opp_Home_Alt. AwayAlt = My_Home_Alt.
        
        # Let's clean this.
        # My_Home_Alt lookup.
        my_alt_map = opp_alt_map # Same dict
        df['my_home_altitude'] = df['teamTricode'].map(my_alt_map)
        
        df['game_altitude_diff'] = np.where(
            df['is_home'],
            df['my_home_altitude'] - df['opp_home_altitude'],
            df['opp_home_altitude'] - df['my_home_altitude']
        )
        
        return df[['teamId', 'gameId', 'gameDate', 'dist_miles', 'tz_delta', 'body_clock_lag', 'road_trip_count', 'game_altitude_diff', 'venueTricode', 'lat', 'long']]

    def get_travel_features(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate travel features for a given schedule (live inference).
        
        Args:
            schedule_df: DataFrame with ['gameDate', 'away_team', 'home_team']
        """
        logging.info("Building live travel features...")
        
        # We need IDs or Tricodes for the teams in schedule.
        # Assuming schedule_df has team names. We need to map them.
        # This Builder typically works with IDs/Tricodes.
        # We need 'venueTricode' for distance calc.
        
        # 1. Load Arena Data
        if not hasattr(self, 'df_arenas'):
             with sqlite3.connect(self.db_path) as conn:
                 self.df_arenas = pd.read_sql("SELECT * FROM nba_arenas", conn)
                 
        # 2. Map Names to Tricodes/IDs
        unique_teams = pd.concat([schedule_df['home_team'], schedule_df['away_team']]).unique()
        
        name_map = {}
        with sqlite3.connect(self.db_path) as conn:
             # Team Name -> Tricode
             df_map = pd.read_sql("SELECT DISTINCT teamName, teamTricode FROM team_basic_stats", conn)
             name_map = dict(zip(df_map['teamName'], df_map['teamTricode']))
             
        # 3. Process Each Game in Schedule
        results = []
        
        # We need 'lat', 'long' of venues.
        arena_map = self.df_arenas.set_index('teamTricode')[['lat', 'long', 'altitude', 'tz_offset']]
        opp_alt_map = self.df_arenas.set_index('teamTricode')['altitude'].to_dict()

        for idx, row in schedule_df.iterrows():
            g_date = pd.to_datetime(row['gameDate'])
            
            # Teams
            home_name = row['home_team']
            away_name = row['away_team']
            
            home_tri = name_map.get(home_name)
            away_tri = name_map.get(away_name)
            
            if not home_tri or not away_tri:
                logging.warning(f"Could not resolve tricode for {home_name} or {away_name}")
                continue
                
            # Current Venue (Home Team's Arena)
            venue_tri = home_tri
            current_venue_info = arena_map.loc[venue_tri] if venue_tri in arena_map.index else None
            
            if current_venue_info is None:
                 logging.warning(f"No arena info for {venue_tri}")
                 continue
                 
            cur_lat = current_venue_info['lat']
            cur_lon = current_venue_info['long']
            cur_tz = current_venue_info['tz_offset']
            cur_alt = current_venue_info['altitude']
            
            # Helper to calculate team-specific travel metrics
            def get_metrics(team_tri, opponent_tri, is_home_team):
                # 1. Query Last Game
                with sqlite3.connect(self.db_path) as conn:
                    # We need last game date and LOCATION.
                    # team_game_full doesn't store Lat/Lon directly, but has MATCHUP -> is_home -> Venue.
                    # We need to construct the previous state.
                    query = f"""
                    SELECT gameDate, MATCHUP, teamTricode FROM team_game_full 
                    WHERE teamTricode = '{team_tri}' AND gameDate < '{row['gameDate']}'
                    ORDER BY gameDate DESC LIMIT 1
                    """
                    last_game = pd.read_sql(query, conn)
                    
                if last_game.empty:
                    # Default / Season Start
                    prev_lat, prev_lon, prev_tz = cur_lat, cur_lon, cur_tz # Assume already here
                    days_rest = 7
                    is_b2b = 0
                    games_last_7d = 0 # Approximate
                else:
                    lg = last_game.iloc[0]
                    lg_date = pd.to_datetime(lg['gameDate'])
                    days_rest = (g_date - lg_date).days
                    is_b2b = 1 if days_rest == 1 else 0
                    
                    # Previous Location
                    # Determine previous venue
                    # If last game was Home: Venue = Me. If Away: Venue = Opp.
                    lg_matchup = lg['MATCHUP']
                    lg_is_home = 'vs.' in lg_matchup
                    
                    if lg_is_home:
                        prev_venue = team_tri
                    else:
                        # Extract opp
                        prev_venue = lg_matchup.split(' ')[-1]
                        
                    # Lookup Prev Venue
                    if prev_venue in arena_map.index:
                        prev_info = arena_map.loc[prev_venue]
                        prev_lat = prev_info['lat']
                        prev_lon = prev_info['long']
                        prev_tz = prev_info['tz_offset']
                    else:
                        prev_lat, prev_lon, prev_tz = cur_lat, cur_lon, cur_tz # Fallback
                        
                    # Games in last 7 days?
                    # Keep it simple for live inference or query more history. 
                    # Let's say we assume the feature builder handles this if we just return rest/dist.
                    # But the model needs 'games_last_7d'. 
                    # Query count.
                    with sqlite3.connect(self.db_path) as conn:
                         q7 = f"""
                         SELECT COUNT(*) as c FROM team_game_full
                         WHERE teamTricode = '{team_tri}' 
                         AND gameDate < '{row['gameDate']}' 
                         AND gameDate >= date('{row['gameDate']}', '-7 days')
                         """
                         games_last_7d = pd.read_sql(q7, conn)['c'][0]
                
                # Calculations
                dist = self.calculate_distance(prev_lat, prev_lon, cur_lat, cur_lon)
                tz_delta = cur_tz - prev_tz
                body_clock_lag = cur_tz - prev_tz # Approximation of Lag = Delta for short term
                
                # Altitude Diff (Game Level Feature, but assigned to team row usually?)
                # Model expects 'game_altitude_diff'.
                # Formula: (MyHome - OppHome) if Home, else (OppHome - MyHome).
                # Wait, earlier definition:
                # df['game_altitude_diff'] = np.where(is_home, my_home - opp_home, opp_home - my_home)
                # So if I am Home: MyHomeAlt - OppHomeAlt.
                # If I am Away: OppHomeAlt - MyHomeAlt.
                # Effectively: Host_Home_Alt - Visitor_Home_Alt.
                # Wait.
                # If I am Home (Host): MyHome - OppHome (Visitor).
                # If I am Away (Visitor): OppHome (Host) - MyHome (Visitor).
                # So ALWAYS: Host - Visitor.
                # This feature is symmetric/Shared?
                # Actually, let's stick to the logic:
                # My Home Alt
                my_home_alt = opp_alt_map.get(team_tri, 0)
                opp_home_alt = opp_alt_map.get(opponent_tri, 0)
                
                if is_home_team:
                    alt_diff = my_home_alt - opp_home_alt
                else:
                    alt_diff = opp_home_alt - my_home_alt
                
                # Road Trip Count?
                # Requires history. Hard to query cheaply. Default to 0 or 1.
                road_trip = 0 # Placeholder
                
                return {
                    'team': team_tri, # Add col for join
                    'dist_miles': dist,
                    'tz_delta': tz_delta,
                    'body_clock_lag': body_clock_lag, # Approx
                    'days_rest': days_rest,
                    'is_b2b': is_b2b,
                    'games_last_7d': games_last_7d,
                    'game_altitude_diff': alt_diff,
                    'road_trip_count': road_trip,
                    'venueTricode': venue_tri
                }

            # Calc for Home
            h_metrics = get_metrics(home_tri, away_tri, True)
            h_metrics['team'] = home_name # Key for merge in inference
            
            # Calc for Away
            a_metrics = get_metrics(away_tri, home_tri, False)
            a_metrics['team'] = away_name
            
            results.append(h_metrics)
            results.append(a_metrics)
            
        return pd.DataFrame(results) 
        
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        return self._haversine_np(lat1, lon1, lat2, lon2)

if __name__ == "__main__":
    builder = TravelFeatureBuilder(r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db')
    builder.load_data()
    df = builder.build_features()
    
    # Verification: 10 CONSECUTIVE GAMES for one team
    print("--- Verification: Golden State Warriors (GSW) Schedule Sample ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    sample = df[df['teamId'] == 1610612744].sort_values('gameDate').iloc[10:20]
    
    # Columns to show
    cols = ['gameDate', 'venueTricode', 'dist_miles', 'tz_delta', 'body_clock_lag', 'road_trip_count', 'game_altitude_diff']
    print(sample[cols].to_string())
