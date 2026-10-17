import sqlite3

import numpy as np
import pandas as pd
import pytest

from models.game_features_v2 import GameFeatureBuilder


@pytest.fixture()
def odds_db(tmp_path):
    path = tmp_path / 'odds.db'
    opening = pd.DataFrame({
        'gameDate': ['2025-11-01', '2025-11-01', '2025-11-02'],
        'home_team': ['Team A', 'Team C', 'Team A'], 'away_team': ['Team B', 'Team D', 'Team C'],
        'teamId_home': [1.0, np.nan, 1.0], 'teamId_away': [2.0, np.nan, 3.0],     # one row needs the name lookup
        'home_spread_draftkings': [-3.5, np.nan, np.nan], 'home_spread_fanduel': [-4.0, -1.5, -2.0],
        'home_ml_draftkings': [-160.0, np.nan, np.nan], 'home_ml_fanduel': [-165.0, -110.0, -130.0],
        'away_ml_draftkings': [140.0, np.nan, np.nan], 'away_ml_fanduel': [145.0, -110.0, 110.0],
        'total_draftkings': [221.5, np.nan, np.nan], 'total_fanduel': [222.0, 215.0, 218.5],
    })
    teams = pd.DataFrame({'teamId': [1, 2, 3, 4], 'teamName': ['Team A', 'Team B', 'Team C', 'Team D']})
    with sqlite3.connect(path) as conn:
        opening.to_sql('nba_odds_opening', conn, index=False)
        teams.to_sql('team_basic_stats', conn, index=False)
    return str(path)


def test_db_odds_use_book_priority_per_market_and_fill_team_ids(odds_db):
    out = GameFeatureBuilder(odds_db)._load_db_odds().set_index(['gameDate', 'teamId_home'])
    first = out.loc[('2025-11-01', 1)]
    assert first['odds_home_spread'] == -3.5 and first['odds_total'] == 221.5      # DraftKings first
    assert first['odds_away_ml'] == 140.0
    filled = out.loc[('2025-11-01', 3)]                                            # id recovered from 'Team C'
    assert filled['odds_home_spread'] == -1.5                                      # DraftKings absent -> FanDuel
    assert out.loc[('2025-11-02', 1)]['odds_home_ml'] == -130.0


def test_archive_odds_drop_impossible_spreads(tmp_path):
    csv = tmp_path / 'archive.csv'
    pd.DataFrame({'date': [20240110, 20240111], 'teamId_home': [1, 2],
                  'home_open_spread': [-3.5, 41.0], 'away_open_spread': [3.5, -41.0],
                  'open_over_under': [221.0, 219.0]}).to_csv(csv, index=False)
    out = GameFeatureBuilder('unused')._load_archive_odds(csv)
    assert out.loc[0, 'odds_home_spread'] == -3.5 and out.loc[0, 'odds_total'] == 221.0
    assert np.isnan(out.loc[1, 'odds_home_spread']) and np.isnan(out.loc[1, 'odds_away_spread'])
    assert out['gameDate'].dt.strftime('%Y-%m-%d').tolist() == ['2024-01-10', '2024-01-11']


def test_missing_odds_tables_do_not_crash(tmp_path):
    path = tmp_path / 'empty.db'
    sqlite3.connect(path).close()
    assert GameFeatureBuilder(str(path))._load_db_odds().empty
    assert GameFeatureBuilder('unused')._load_archive_odds(tmp_path / 'nope.csv').empty
