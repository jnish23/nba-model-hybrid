import pandas as pd

class RotationModel:
    def __init__(self):
        pass

    def generate_roster_df(self, pbs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the 'Entering Minutes Projection' for every player-game.
        """
        df = pbs_df.copy()
        
        # Sort history
        if 'gameDate' in df.columns:
            df = df.sort_values(['personId', 'gameDate'])
        else:
            df = df.sort_values(['personId', 'seasonYear', 'gameId'])
            
        # Rolling Average of Minutes
        df['rolling_avg_minutes'] = (
            df.groupby('personId')['minutes']
            .transform(lambda x: x.ewm(span=10, min_periods=1).mean())
        )
        
        # Shift to get Entering Value
        df['mins_ewm'] = df.groupby('personId')['rolling_avg_minutes'].shift(1)
        df['projected_minutes'] = df['mins_ewm'].fillna(10.0)
        
        roster_df = df[['seasonYear', 'teamId', 'teamName', 'personId', 'playerName', 'mins_ewm', 'projected_minutes', 'gameId']].copy()
        
        if 'gameDate' in df.columns:
            roster_df['gameDate'] = df['gameDate']
            
        return roster_df

    def predict_minutes(self, roster_df, injury_list=None):
        if injury_list is None:
            injury_list = []
            
        df = roster_df.copy()
        
        # Injury Logic
        df.loc[df['personId'].isin(injury_list), 'projected_minutes'] = 0
        
        # Identify Active players (naive check)
        # In a real scenario, we'd distribute "missing minutes" logic here
        # But for backtest, we just take the projection derived from history.
        
        # Redistribution Logic 
        # (This is expensive to do row-by-row in pandas for 50k games)
        # For the backtest speed, we often skip redistribution or do it vectorized.
        # But let's include the simple scalar check from the script.
        
        # Actually, doing this per-game in a loop is what the previous script did implicitly?
        # No, the previous script `predict_minutes` function took a SINGLE roster_df and injury list.
        # But we are processing ALL games.
        # The `generate_roster_df` ALREADY has the 'entering' value.
        # So unless we are simulating SPECIFIC injuries on specific days, we just use 'projected_minutes'.
        
        df['projected_minutes'] = df['projected_minutes'].clip(upper=48)
        
        return df[['seasonYear', 'gameDate', 'gameId', 'teamId', 'personId', 'projected_minutes']]

    def calculate_roster_delta(self, minutes_df: pd.DataFrame, player_skill_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines Minutes Projection with Player Skill.
        """
        print("--- Calculating Roster Delta ---")
        
        m_df = minutes_df.copy()
        s_df = player_skill_df.copy()
        
        m_df = minutes_df.copy()
        s_df = player_skill_df.copy()
        
        m_df['gameId'] = m_df['gameId'].astype(int)
        m_df['personId'] = m_df['personId'].astype(int)
        
        s_df['gameId'] = s_df['gameId'].astype(int)
        s_df['personId'] = s_df['personId'].astype(int)
        
        merged = pd.merge(
            m_df, 
            s_df, 
            left_on=['gameId', 'personId'], 
            right_on=['gameId', 'personId'],
            how='left' 
        )
        
        # Fill missing skill with 0 (League Average)
        merged['skill_per_minute'] = merged['skill_per_minute'].fillna(0)
        
        merged['player_impact'] = merged['projected_minutes'] * merged['skill_per_minute']
        
        team_daily_score = merged.groupby(['gameId', 'teamId'])['player_impact'].sum().reset_index()
        team_daily_score.rename(columns={'player_impact': 'team_player_score'}, inplace=True)
        
        # Baseline
        team_daily_score['avg_roster_rating'] = (
            team_daily_score.groupby('teamId')['team_player_score']
            .transform(lambda x: x.ewm(span=20, min_periods=5).mean().shift(1))
        )
        
        team_daily_score['roster_delta'] = team_daily_score['team_player_score'] - team_daily_score['avg_roster_rating']
        team_daily_score['roster_delta'] = team_daily_score['roster_delta'].fillna(0)
        
        return team_daily_score
