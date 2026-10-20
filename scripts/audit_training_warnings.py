"""
Explain the data-quality warnings printed by generate_training_set.py, writing one CSV per warning
to reports/audit/:

  unmatched_team_names.csv   injury-report team cells that matched no team ("Unmatched team names")
  fuzzy_player_matches.csv   injury-report players linked by fuzzy name match ("N via fuzzy match")
  invalid_archive_spreads.csv  archive games whose opening spread is > 25 points ("invalid odds_home_spread")
  odds_missing_team_ids.csv  nba_odds_opening rows lacking teamId_home, and whether the name lookup filled them

Usage: python scripts/audit_training_warnings.py
"""
import os
import sqlite3
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import pdfplumber  # noqa: F401
except ImportError:  # only needed to parse new PDFs; the parsed cache is used here
    sys.modules['pdfplumber'] = types.ModuleType('pdfplumber')

from core import injury_report_features as irf

DB = Path('data/nba_stats.db')
OUT = Path('reports/audit')
OUT.mkdir(parents=True, exist_ok=True)


def main():
    reports = irf.parse_all_reports(report_dir=Path('nonexistent'))
    reports = reports[reports['source_file'].map(irf.report_slot).isin(irf.GRID_SLOTS)]
    linked = irf.link_reports(reports, DB)

    # 1. Unmatched team names
    bad = linked[linked['teamId'].isna() & linked['team'].notna()]
    t = (bad.groupby('team', dropna=False)
            .agg(rows=('team', 'size'), n_reports=('source_file', 'nunique'),
                 first_date=('game_date', 'min'), last_date=('game_date', 'max'),
                 example_matchup=('matchup', 'first'), example_player=('player', 'first'),
                 example_file=('source_file', 'first')).reset_index())
    t.to_csv(OUT / 'unmatched_team_names.csv', index=False)
    print(f"\n[1] Unmatched team names:\n{t.to_string(index=False)}")

    # 2. Fuzzy player matches
    with sqlite3.connect(DB) as conn:
        pl = pd.read_sql("SELECT DISTINCT personId, playerName FROM player_basic_stats", conn)
        tm = pd.read_sql("SELECT DISTINCT teamId, teamTricode FROM team_game_full", conn)
    f = pd.DataFrame(irf.FUZZY_MATCHES)
    if not f.empty:
        f = (f.merge(pl, on='personId', how='left').merge(tm, on='teamId', how='left')
              .rename(columns={'playerName': 'matched_db_player'}))
        f['game_date'] = pd.to_datetime(f['game_date']).dt.strftime('%Y-%m-%d')
        f = f[['seasonYear', 'game_date', 'teamTricode', 'report_name', 'matched_db_player', 'report_key',
               'matched_key', 'personId', 'n_rows']].sort_values(['seasonYear', 'game_date'])
        s = f.groupby(['report_name', 'matched_db_player', 'teamTricode', 'seasonYear'], as_index=False)['n_rows'].sum()
        s.sort_values('n_rows', ascending=False).to_csv(OUT / 'fuzzy_player_matches_summary.csv', index=False)
    f.to_csv(OUT / 'fuzzy_player_matches.csv', index=False)
    print(f"\n[2] Fuzzy player matches: {len(f)} (report, team, game) groups, "
          f"{int(f['n_rows'].sum()) if len(f) else 0} rows -> fuzzy_player_matches(.csv|_summary.csv)")

    # 3. Archive spreads beyond +-25
    a = pd.read_csv('data/nba_odds_archive.csv')
    a['gameDate'] = pd.to_datetime(a['date'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
    cols = ['season', 'gameDate', 'home_team', 'away_team', 'home_open_spread', 'away_open_spread',
            'home_close_spread', 'away_close_spread', 'open_over_under', 'close_over_under', 'home_final', 'away_final']
    inv = a[(a['home_open_spread'].abs() > 25) | (a['away_open_spread'].abs() > 25)][cols]
    inv.to_csv(OUT / 'invalid_archive_spreads.csv', index=False)
    print(f"\n[3] Invalid archive opening spreads:\n{inv.to_string(index=False)}")

    # 4. nba_odds_opening rows without teamId_home
    with sqlite3.connect(DB) as conn:
        o = pd.read_sql("SELECT * FROM nba_odds_opening", conn)
        tmap = pd.read_sql("SELECT DISTINCT teamId, teamName FROM team_basic_stats WHERE teamId IS NOT NULL", conn)
    name_to_id = dict(zip(tmap['teamName'].str.strip(), tmap['teamId']))
    m = o[o['teamId_home'].isna()].copy()
    m['filled_teamId'] = m['home_team'].str.strip().map(name_to_id)
    m['filled'] = m['filled_teamId'].notna()
    m['gameDate'] = m['gameDate'].astype(str).str[:10]
    m[['gameDate', 'season', 'home_team', 'away_team', 'teamId_away', 'filled', 'filled_teamId']].to_csv(
        OUT / 'odds_missing_team_ids.csv', index=False)
    print(f"\n[4] nba_odds_opening rows missing teamId_home: {len(m)} "
          f"(filled by name: {int(m['filled'].sum())}, still missing and DROPPED: {int((~m['filled']).sum())})")
    print("   by season:\n" + m.groupby(['season', 'filled']).size().unstack(fill_value=0).to_string())
    if (~m['filled']).any():
        print("   unfilled team names:\n" + m.loc[~m['filled'], 'home_team'].value_counts().to_string())
    print(f"\nCSVs written to {OUT}/")


if __name__ == '__main__':
    main()
