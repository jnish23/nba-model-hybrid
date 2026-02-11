
import pandas as pd
import sqlite3
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_PATH = 'data/nba_stats.db'
CSV_PATH = 'data/nba_arena_data.csv'

def ingest_arenas():
    logging.info("Loading Arena Data...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Clean Altitude (remove commas)
    if df['Altitude (ft)'].dtype == 'O':
        df['altitude'] = df['Altitude (ft)'].astype(str).str.replace(',', '').astype(float)
    else:
        df['altitude'] = df['Altitude (ft)']
        
    # 2. Map TimeZones to Offset integers (relative to ET or UTC)
    # NBA is in US.
    # ET = -5, CT = -6, MT = -7, PT = -8
    # We can just use ints: ET=0, CT=-1, MT=-2, PT=-3
    # Or strict UTC offsets. Let's use UTC offsets for clarity, assuming Standard Time (DST handling is complex, but usually relative diff stays same).
    # ET: -5, CT: -6, MT: -7, PT: -8
    
    tz_map = {
        'ET': -5,
        'CT': -6,
        'MT': -7,
        'PT': -8
    }
    
    df['tz_offset'] = df['timeZone'].map(tz_map)
    
    # Check for unmapped
    if df['tz_offset'].isna().any():
        logging.warning(f"Unmapped TimeZones found: {df[df['tz_offset'].isna()]['timeZone'].unique()}")
        # Fill with matching?
        
    # Select columns
    final_df = df[['TeamId', 'teamTricode', 'Arena', 'Lat', 'Long', 'altitude', 'tz_offset']]
    
    # Rename for consistency
    final_df = final_df.rename(columns={
        'TeamId': 'teamId', 
        'Lat': 'lat', 
        'Long': 'long',
        'Arena': 'arena_name'
    })
    
    logging.info(f"Ingesting {len(final_df)} arenas into {DB_PATH}...")
    
    with sqlite3.connect(DB_PATH) as conn:
        final_df.to_sql('nba_arenas', conn, if_exists='replace', index=False)
        
    logging.info("Done.")
    print(final_df.head())

if __name__ == "__main__":
    ingest_arenas()
