import sqlite3

import pandas as pd
import pytest

import core.update_database as ud


def test_ident_accepts_plain_names_and_refuses_everything_else():
    for ok in ('team_basic_stats', 'teamId_home', 'seasonYear', 'nba_odds_opening'):
        assert ud._ident(ok) == ok
    for bad in ('x; DROP TABLE y', 'a b', 'a"b', "a'b", '1abc', '', 'a-b'):
        with pytest.raises(ValueError):
            ud._ident(bad)


def test_statements_that_interpolate_identifiers_still_work():
    with sqlite3.connect(':memory:') as conn:
        conn.execute("CREATE TABLE nba_odds (home_team TEXT, away_team TEXT, teamId_home INTEGER, teamId_away INTEGER)")
        conn.executemany("INSERT INTO nba_odds (home_team, away_team) VALUES (?, ?)",
                         [('Boston Celtics', 'Miami Heat'), ('Nowhere FC', 'Boston Celtics')])
        filled = ud.fill_odds_team_ids(conn, tables=('nba_odds',))
        assert filled == 3                                            # unknown name left alone
        got = pd.read_sql('SELECT * FROM nba_odds', conn)
        assert got.loc[0, 'teamId_home'] == 1610612738 and got.loc[0, 'teamId_away'] == 1610612748
        assert pd.isna(got.loc[1, 'teamId_home']) and got.loc[1, 'teamId_away'] == 1610612738


def test_season_filter_values_are_bound_not_interpolated():
    from core.availability_features import _team_schedule
    with sqlite3.connect(':memory:') as conn:
        pd.DataFrame({'gameId': ['1', '2'], 'teamId': [1, 1], 'gameDate': ['2025-11-01', '2024-11-01'],
                      'seasonYear': ["2025-26", "2024-25"]}).to_sql('team_game_full', conn, index=False)
        assert len(_team_schedule(conn, ['2025-26'])) == 1
        assert len(_team_schedule(conn, ["x' OR '1'='1"])) == 0       # a quote in a value is just a value
