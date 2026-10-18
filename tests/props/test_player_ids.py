import sqlite3

import pandas as pd
import pytest

from props import odds_store
from props.player_ids import PlayerMatcher, normalize, resolve_game_id, resolve_store, strip_suffix

BOS, NYK, LAL = 1610612738, 1610612752, 1610612747


@pytest.fixture()
def nba_db(tmp_path):
    path = tmp_path / 'nba.db'
    games = pd.DataFrame([
        # personId, name, team, date
        (1, 'Jayson Tatum', BOS, '2026-10-01T00:00:00'), (1, 'Jayson Tatum', BOS, '2026-10-15T00:00:00'),
        (2, 'Jaren Jackson Jr.', NYK, '2026-10-10T00:00:00'),
        (3, 'P.J. Washington', NYK, '2026-10-12T00:00:00'),
        (4, 'Mike Smith', BOS, '2026-10-12T00:00:00'),       # two players with the same name on different teams
        (5, 'Mike Smith', LAL, '2026-10-12T00:00:00'),
        (6, 'Moved Guy', LAL, '2026-10-01T00:00:00'), (6, 'Moved Guy', NYK, '2026-10-14T00:00:00'),
    ], columns=['personId', 'playerName', 'teamId', 'gameDate'])
    info = pd.DataFrame({'PERSON_ID': [1, 2, 3, 4, 5, 6, 7], 'DISPLAY_FIRST_LAST': [
        'Jayson Tatum', 'Jaren Jackson Jr.', 'P.J. Washington', 'Mike Smith', 'Mike Smith', 'Moved Guy', 'Rookie Newman']})
    teams = pd.DataFrame({'teamId': [BOS, NYK, LAL, LAL], 'teamName': ['Boston Celtics', 'New York Knicks',
                          'Los Angeles Lakers', 'Los Angeles Clippers'], 'teamTricode': ['BOS', 'NYK', 'LAL', 'LAC']})
    tbs = pd.DataFrame({'teamId': [BOS, NYK], 'gameId': ['0022600001', '0022600001'],
                        'gameDate': ['2026-10-20T00:00:00'] * 2, 'MATCHUP': ['BOS vs. NYK', 'NYK @ BOS']})
    with sqlite3.connect(path) as c:
        games.assign(MATCHUP='x').to_sql('player_basic_stats', c, index=False)
        info.to_sql('player_info', c, index=False)
        teams.merge(tbs, on='teamId', how='left').to_sql('team_basic_stats', c, index=False)
    return str(path)


def test_normalize_and_suffix():
    assert normalize('P.J. Washington') == 'pj washington'
    assert normalize('Shai Gilgeous-Alexander') == 'shai gilgeous alexander'
    assert normalize('Nikola Jokić') == 'nikola jokic'
    assert normalize("De'Aaron Fox") == 'deaaron fox'
    assert strip_suffix(normalize('Jaren Jackson Jr.')) == 'jaren jackson'


def test_match_rules(nba_db, tmp_path):
    m = PlayerMatcher(nba_db, overrides_path=tmp_path / 'none.csv')
    teams = (BOS, NYK)
    assert m.match('Jayson Tatum', '2026-10-20', teams) == (1, 'team_exact')
    assert m.match('Jaren Jackson', '2026-10-20', teams) == (2, 'team_variant')       # vendor drops the Jr.
    assert m.match('PJ Washington', '2026-10-20', teams) == (3, 'team_exact')          # punctuation differs
    assert m.match('Mike Smith', '2026-10-20', teams) == (4, 'team_exact')             # team context breaks the tie
    assert m.match('Mike Smith', '2026-10-20', (NYK, LAL)) == (5, 'team_exact')
    assert m.match('Rookie Newman', '2026-10-20', teams) == (7, 'unique_name')         # no NBA game yet
    assert m.match('Nobody Here', '2026-10-20', teams)[0] is None


def test_team_is_the_latest_team_before_the_date(nba_db, tmp_path):
    m = PlayerMatcher(nba_db, overrides_path=tmp_path / 'none.csv')
    assert m.team_on(6, '2026-10-05') == LAL and m.team_on(6, '2026-10-20') == NYK
    assert m.team_on(6, '2026-09-01') is None                       # no earlier game
    assert m.match('Moved Guy', '2026-10-20', (BOS, NYK))[0] == 6


def test_fuzzy_match_needs_a_single_close_candidate(nba_db, tmp_path):
    m = PlayerMatcher(nba_db, overrides_path=tmp_path / 'none.csv')
    assert m.match('Jayson Tatumm', '2026-10-20', (BOS, NYK)) == (1, 'fuzzy_team')


def test_override_wins(nba_db, tmp_path):
    ov = tmp_path / 'o.csv'
    ov.write_text('name_raw,player_id,note\nNicky Tatum,1,alias\n')
    assert PlayerMatcher(nba_db, overrides_path=ov).match('Nicky Tatum', '2026-10-20', (BOS, NYK)) == (1, 'override')


def test_team_name_aliases(nba_db, tmp_path):
    m = PlayerMatcher(nba_db, overrides_path=tmp_path / 'none.csv')
    assert m.team_id('Boston Celtics') == BOS and m.team_id('BOS') == BOS


def test_resolve_game_id(nba_db):
    assert resolve_game_id(nba_db, '2026-10-20', BOS) == '0022600001'
    assert resolve_game_id(nba_db, '2026-10-21', BOS) is None


def test_resolve_store_fills_ids_and_logs_unmatched_without_dropping_rows(nba_db, tmp_path):
    conn = odds_store.connect(tmp_path / 'p.db')
    odds_store.upsert_event(conn, 'ev1', '2026-10-21T00:00:00Z', 'Boston Celtics', 'New York Knicks')
    rows = pd.DataFrame({'snapshot_ts_utc': ['2026-10-20T20:00:00Z'] * 3, 'event_id_vendor': ['ev1'] * 3,
                         'book': ['dk'] * 3, 'player_name_raw': ['Jayson Tatum', 'Jaren Jackson', 'Mystery Man'],
                         'market': ['points'] * 3, 'line': [27.5, 15.5, 8.5], 'over_price': [-110] * 3,
                         'under_price': [-110] * 3, 'is_alt_line': [0] * 3, 'source': ['t'] * 3})
    odds_store.insert_props(conn, rows)
    out = resolve_store(conn, PlayerMatcher(nba_db, overrides_path=tmp_path / 'none.csv'), nba_db)
    got = dict(conn.execute('select player_name_raw, player_id from prop_snapshots').fetchall())
    assert got == {'Jayson Tatum': 1, 'Jaren Jackson': 2, 'Mystery Man': None}     # unmatched row is kept
    assert conn.execute('select game_id from events').fetchone()[0] == '0022600001'
    assert conn.execute('select game_id from prop_snapshots limit 1').fetchone()[0] == '0022600001'
    assert out['matched_rows'] == 2 and out['unmatched_rows'] == 1 and out['match_rate'] == pytest.approx(2 / 3)
    assert conn.execute('select player_name_raw, reason from unmatched_names').fetchall() == [('Mystery Man', 'no_candidate')]
    resolve_store(conn, PlayerMatcher(nba_db, overrides_path=tmp_path / 'none.csv'), nba_db)   # idempotent
    assert conn.execute('select count(*) from unmatched_names').fetchone()[0] == 1
