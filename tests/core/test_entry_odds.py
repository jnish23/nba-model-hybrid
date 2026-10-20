"""The entry price is the first pre-game line at/after the feature cutoff, never the opening line."""
import sqlite3

import pandas as pd
import pytest

from core import odds_clean, snapshot_odds

START = '2026-10-20T23:30:00Z'      # 7:30 PM ET


def snap(ts, spread, total, hml=-160, aml=140, book='fanduel', in_game=0, start=START):
    return (ts, '2026-10-20', start, 'Scheduled', in_game, 'Detroit Pistons', 'San Antonio Spurs', 'DET', 'SAS', 'current', book,
            spread, -110, -spread, -110, hml, aml, total, -110, -110)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / 'odds.db'
    conn = sqlite3.connect(path)
    conn.execute(snapshot_odds.SCHEMA)
    rows = [
        snap('2026-10-20 09:00:00', -3.0, 222.5),                 # before noon: the market the model could not yet price against
        snap('2026-10-20 11:59:00', -3.5, 223.0),                 # still before the cutoff
        snap('2026-10-20 12:30:00', -4.5, 224.5),                 # first at/after noon -> the entry price
        snap('2026-10-20 15:00:00', -5.5, 225.5),                 # later movement
        snap('2026-10-20 23:45:00', -9.5, 231.5, in_game=1),      # in-game: never used
    ]
    conn.executemany('INSERT INTO odds_snapshots VALUES (' + ','.join('?' * 20) + ')', rows)
    conn.execute('CREATE TABLE nba_odds_opening (gameDate TEXT, home_team TEXT, away_team TEXT, teamId_home INT, teamId_away INT)')
    conn.execute('CREATE TABLE nba_odds (gameDate TEXT, home_team TEXT, away_team TEXT, teamId_home INT, teamId_away INT)')
    for t in ('nba_odds_opening', 'nba_odds'):
        conn.execute(f"INSERT INTO {t} VALUES ('2026-10-20', 'Detroit Pistons', 'San Antonio Spurs', 1, 2)")
    conn.execute('CREATE TABLE team_game_full (gameId TEXT, gameDate TEXT, teamId INT, MATCHUP TEXT, points INT, seasonYear TEXT)')
    conn.executemany('INSERT INTO team_game_full VALUES (?,?,?,?,?,?)', [
        ('0022600001', '2026-10-20', 1, 'DET vs. SAS', 110, '2026-27'), ('0022600001', '2026-10-20', 2, 'SAS @ DET', 100, '2026-27')])
    conn.commit()
    conn.close()
    return path


def test_entry_price_is_first_snapshot_at_or_after_noon(db):
    e = odds_clean.load_entry_odds(db)
    assert len(e) == 1
    r = e.iloc[0]
    assert r['gameId'] == '0022600001'
    assert r['entry_home_spread'] == -4.5 and r['entry_total'] == 224.5     # not the 9 AM, 11:59 AM, 3 PM or in-game line
    assert r['entry_fetched_at'] == '2026-10-20 12:30:00'


def test_entry_cutoff_is_capped_at_tip_minus_30m(tmp_path):
    early = pd.DataFrame({'game_date': ['2026-10-20'], 'game_start': ['2026-10-20T16:00:00Z']})   # 12:00 PM ET tip
    assert odds_clean.entry_cutoff(early).iloc[0] == pd.Timestamp('2026-10-20 11:30:00')
    late = pd.DataFrame({'game_date': ['2026-10-20'], 'game_start': [START]})
    assert late.pipe(odds_clean.entry_cutoff).iloc[0] == pd.Timestamp('2026-10-20 12:00:00')
    unknown = pd.DataFrame({'game_date': ['2026-10-20'], 'game_start': [None]})
    assert odds_clean.entry_cutoff(unknown).iloc[0] == pd.Timestamp('2026-10-20 12:00:00')


def test_game_seen_only_before_the_cutoff_has_no_entry_price(db):
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM odds_snapshots WHERE fetched_at_et >= '2026-10-20 12:00:00'")
    conn.commit()
    conn.close()
    assert odds_clean.load_entry_odds(db).empty            # the opener is NOT a fallback


def test_no_snapshot_table_gives_an_empty_frame(tmp_path):
    path = tmp_path / 'empty.db'
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE nba_odds_opening (gameDate TEXT, home_team TEXT, away_team TEXT, teamId_home INT, teamId_away INT)')
    conn.execute('CREATE TABLE nba_odds (gameDate TEXT, home_team TEXT, away_team TEXT, teamId_home INT, teamId_away INT)')
    conn.execute('CREATE TABLE team_game_full (gameId TEXT, gameDate TEXT, teamId INT, MATCHUP TEXT, points INT, seasonYear TEXT)')
    conn.commit()
    conn.close()
    assert odds_clean.load_entry_odds(path).empty
