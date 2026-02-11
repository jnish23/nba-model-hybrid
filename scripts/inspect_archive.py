
import pandas as pd
from pathlib import Path

def inspect():
    path = Path.cwd() / 'data' / 'nba_odds_archive.csv'
    df = pd.read_csv(path)
    # Check 2015
    # Date YYYYMMDD
    # 2015-16 season starts Oct 2015
    df = df[df['date'].astype(str).str.startswith('201511')]
    print(f"Rows in Nov 2015: {len(df)}")
    print(df[['date', 'home_team', 'away_team', 'teamId_home']].head(10))
    
    print("\nCheck if teamId_home is NaN:")
    print(df['teamId_home'].isna().sum())

if __name__ == "__main__":
    inspect()
