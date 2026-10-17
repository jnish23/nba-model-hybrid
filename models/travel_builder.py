
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

    def build_features(self, upcoming=None):
        """Travel features for every historical team-game.

        upcoming: optional frame of NOT-yet-played games with the df_games columns
        (teamId, teamTricode, gameId, gameDate, MATCHUP). Their features are computed by exactly
        the same code as the historical rows - from history strictly before the earliest upcoming
        date - which is how live inference gets training-identical values (road_trip_count,
        body_clock_lag, ...).
        """
        logging.info("Building Travel Features...")
        
        if not hasattr(self, 'df_games'):
            self.load_data()
            
        df = self.df_games.copy()
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        if upcoming is not None and len(upcoming):
            upcoming = upcoming[df.columns].copy()
            upcoming['gameDate'] = pd.to_datetime(upcoming['gameDate'])
            # Games on/after the first upcoming date are not history (a live run may already find
            # that day's finished games in the DB; they must not leak into the features)
            df = pd.concat([df[df['gameDate'] < upcoming['gameDate'].min()], upcoming], ignore_index=True)
        
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
        Travel features for a live schedule, computed by build_features() itself so they equal
        the training values (distance, time-zone delta, body-clock lag, road-trip length, altitude).

        Args:
            schedule_df: DataFrame with ['gameDate', 'away_team', 'home_team'] (team names);
                an optional 'gameId' column is carried through.
        Returns one row per team per game with a 'team' (name) column and the feature columns.
        """
        logging.info("Building live travel features...")
        if not hasattr(self, 'df_games'):
            self.load_data()

        with sqlite3.connect(self.db_path) as conn:
            teams = pd.read_sql("SELECT DISTINCT teamId, teamName, teamTricode FROM team_basic_stats", conn)
        by_name = teams.drop_duplicates('teamName').set_index('teamName')

        rows, keep = [], []
        for i, g in enumerate(schedule_df.itertuples(index=False)):
            if g.home_team not in by_name.index or g.away_team not in by_name.index:
                logging.warning(f"Could not resolve team ids for {g.home_team} / {g.away_team}")
                continue
            home, away = by_name.loc[g.home_team], by_name.loc[g.away_team]
            gid = f"live_{i}"
            rows.append({'teamId': home['teamId'], 'teamTricode': home['teamTricode'], 'gameId': gid,
                         'gameDate': g.gameDate, 'MATCHUP': f"{home['teamTricode']} vs. {away['teamTricode']}"})
            rows.append({'teamId': away['teamId'], 'teamTricode': away['teamTricode'], 'gameId': gid,
                         'gameDate': g.gameDate, 'MATCHUP': f"{away['teamTricode']} @ {home['teamTricode']}"})
            keep += [(gid, g.home_team), (gid, g.away_team)]
        if not rows:
            return pd.DataFrame()

        upcoming = pd.DataFrame(rows)
        feats = self.build_features(upcoming=upcoming)
        feats = feats[feats['gameId'].isin(upcoming['gameId'])]
        names = dict(zip(zip(upcoming['gameId'], upcoming['teamId']), [k[1] for k in keep]))
        feats = feats.assign(team=[names[(gid, tid)] for gid, tid in zip(feats['gameId'], feats['teamId'])])
        return feats.drop(columns=['teamId', 'gameId', 'gameDate', 'venueTricode', 'lat', 'long']).reset_index(drop=True)

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
