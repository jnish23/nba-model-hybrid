"""core/live_feed.py from saved ESPN responses (tests/fixtures/live_feed, captured 2026-10-25 with odds and play-by-play stripped). No test touches the network."""
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from core import live_feed as F

FIX = Path(__file__).resolve().parent.parent / 'fixtures' / 'live_feed'
LIVE_EVENT, FINAL_EVENT = '401909867', '401909861'          # MEM @ SAC in the 3rd quarter; LAL @ UTA final
SCHEDULE = pd.DataFrame({'game_id': ['0022600118', '0022600112'], 'game_date': ['2026-10-25', '2026-10-25'],
                         'home_team': ['Sacramento Kings', 'Utah Jazz'], 'away_team': ['Memphis Grizzlies', 'Los Angeles Lakers']})
MEM_SAC_BETS = pd.DataFrame({'player_id': [203897, 1627734, 1629631, 1629660, 1642377, 1642907, 1643409, 1643411],
                             'player_name': ['Zach LaVine', 'Domantas Sabonis', "De'Andre Hunter", 'Ty Jerome', 'Jaylen Wells', 'Cedric Coward',
                                             'Cameron Boozer', 'Darius Acuff Jr.']})


def fixture(name):
    return json.loads((FIX / name).read_text(encoding='utf-8'))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError('test tried to reach the network')
    monkeypatch.setattr(requests, 'get', refuse)


@pytest.fixture()
def espn(monkeypatch):
    """Serve the saved responses through _get, recording each request."""
    calls = []
    files = {('scoreboard', '20261025'): 'espn_scoreboard_2026-10-25.json', ('summary', LIVE_EVENT): f'espn_summary_in_{LIVE_EVENT}.json',
             ('summary', FINAL_EVENT): f'espn_summary_post_{FINAL_EVENT}.json'}

    def fake_get(path, params=None):
        key = (path, (params or {}).get('dates') or (params or {}).get('event'))
        calls.append(key)
        return fixture(files[key]) if key in files else None
    monkeypatch.setattr(F, '_get', fake_get)
    return calls


def test_every_espn_team_maps_to_an_nba_tricode():
    teams = fixture('espn_teams.json')
    assert len(teams) == 30
    codes = {t['abbreviation']: F.tricode(t['abbreviation']) for t in teams}
    assert None not in codes.values() and len(set(codes.values())) == 30
    assert {k: v for k, v in codes.items() if k != v} == {'GS': 'GSW', 'NO': 'NOP', 'NY': 'NYK', 'SA': 'SAS', 'UTAH': 'UTA', 'WSH': 'WAS'}
    assert all(F.tricode(t['displayName']) == codes[t['abbreviation']] for t in teams)    # full names as bets.csv / predictions_log carry them


def test_tricode_unknown_is_none():
    assert F.tricode('XYZ') is None and F.tricode(None) is None and F.tricode('Seattle SuperSonics') is None


@pytest.mark.parametrize('name, key', [('Darius Acuff Jr.', 'darius acuff'), ("De'Andre Hunter", 'deandre hunter'), ('Nikola Jokić', 'nikola jokic'),
                                       ('Luka Dončić', 'luka doncic'), ('Olivier-Maxence Prosper', 'olivier maxence prosper'),
                                       ('Gary Trent Jr', 'gary trent'), ('Robert Williams III', 'robert williams'), ('  GG  Jackson ', 'gg jackson')])
def test_norm_name(name, key):
    assert F.norm_name(name) == key


def test_scoreboard_parses_live_and_final_games_and_attaches_nba_ids(espn):
    b = F.scoreboard('2026-10-25', SCHEDULE)
    assert list(b.columns) == F.SCOREBOARD_COLS and len(b) == 7
    live = b[b['espn_event_id'] == LIVE_EVENT].iloc[0]
    assert (live['nba_game_id'], live['status'], live['period'], live['clock']) == ('0022600118', 'live', 3, '39.5')
    assert (live['home_tricode'], live['away_tricode'], live['home_score'], live['away_score']) == ('SAC', 'MEM', 84, 92)
    final = b[b['espn_event_id'] == FINAL_EVENT].iloc[0]
    assert (final['nba_game_id'], final['status'], final['clock'], final['home_tricode'], final['home_score'], final['away_score']) == ('0022600112', 'final', None, 'UTA', 123, 132)
    assert b['nba_game_id'].isna().sum() == 5                      # games not in the schedule keep an empty id
    assert set(b.loc[b['espn_event_id'] == '401909863', ['home_tricode', 'away_tricode']].iloc[0]) == {'NYK', 'ORL'}


def test_scheduled_game_has_no_score_or_clock():
    data = fixture('espn_scoreboard_2026-10-25.json')
    ev = data['events'][0]
    ev['status']['type'].update({'state': 'pre', 'completed': False, 'name': 'STATUS_SCHEDULED'})
    row = F.parse_scoreboard({'events': [ev]}).iloc[0]
    assert row['status'] == 'scheduled' and row['home_score'] is None and row['clock'] is None and row['period'] is None


def test_box_live_game_matches_every_bet_player(espn):
    board = F.scoreboard('2026-10-25', SCHEDULE)
    lines = F.box('0022600118', board, MEM_SAC_BETS, overrides={})
    assert list(lines.columns) == F.BOX_COLS and len(lines) == 8
    acuff = lines.set_index('player_id').loc[1643411]
    assert (acuff['player_name'], acuff['team_tricode'], acuff['minutes'], acuff['points']) == ('Darius Acuff Jr.', 'SAC', 26.0, 18)
    wells = lines.set_index('player_id').loc[1642377]
    assert (wells['minutes'], wells['points'], wells['on_court']) == (22.0, 12, True)
    assert lines.set_index('player_id').loc[1629631, 'player_name'] == "De'Andre Hunter"
    assert espn == [('scoreboard', '20261025'), ('summary', LIVE_EVENT)]


def test_box_final_game_values_and_no_on_court(espn):
    board = F.scoreboard('2026-10-25', SCHEDULE)
    roster = pd.DataFrame({'player_id': [1641718, 1629029], 'player_name': ['Keyonte George', 'Luka Dončić']})   # accented spelling on our side
    lines = F.box('0022600112', board, roster, overrides={}).set_index('player_id')
    george = lines.loc[1641718]
    assert (george['minutes'], george['points'], george['rebounds'], george['assists'], george['threes']) == (33.0, 24, 9, 10, 2)
    assert lines.loc[1629029, 'points'] == 52 and lines['on_court'].isna().all()


def test_unmatched_and_ambiguous_players_are_dropped_not_guessed(espn):
    espn_rows = F.parse_summary(fixture(f'espn_summary_in_{LIVE_EVENT}.json'))
    roster = pd.DataFrame({'player_id': [1, 2, 3, 1642377], 'player_name': ['Nobody Here', 'Ty Jerome', 'Ty Jerome Jr.', 'Jaylen Wells']})
    lines, unmatched = F.match_players(espn_rows, roster, overrides={})
    assert list(lines['player_id']) == [1642377]                  # two roster players share Ty Jerome's key: neither is taken
    assert unmatched == ['Nobody Here', 'Ty Jerome', 'Ty Jerome Jr.']


def test_override_maps_an_espn_id_the_names_cannot(espn):
    espn_rows = F.parse_summary(fixture(f'espn_summary_in_{LIVE_EVENT}.json'))
    coward = espn_rows.loc[espn_rows['espn_name'] == 'Cedric Coward', 'espn_id'].iloc[0]
    roster = pd.DataFrame({'player_id': [1642907], 'player_name': ['C. J. Coward']})
    assert F.match_players(espn_rows, roster, overrides={})[0].empty
    lines, unmatched = F.match_players(espn_rows, roster, overrides={coward: 1642907})
    assert list(lines['player_id']) == [1642907] and unmatched == []


def test_overrides_csv_reads_and_tolerates_missing(tmp_path):
    p = tmp_path / 'o.csv'
    p.write_text('espn_athlete_id,player_id,note\n# comment\n4903027,1642907,test\n', encoding='utf-8')
    assert F.load_overrides(p) == {'4903027': 1642907}
    assert F.load_overrides(tmp_path / 'missing.csv') == {}
    assert F.load_overrides(F.OVERRIDES) == {}                    # the shipped file is a header only


def test_parse_summary_skips_dnps():
    rows = F.parse_summary(fixture(f'espn_summary_in_{LIVE_EVENT}.json'))
    assert len(rows) == 20 and 'Kris Murray' not in set(rows['espn_name'])


@pytest.mark.parametrize('payload', [None, {}, {'events': None}, {'events': [{'id': 'x'}]}, {'boxscore': {'players': [{'team': {}, 'statistics': [{}]}]}}])
def test_empty_or_malformed_payloads_give_empty_frames(payload):
    assert F.parse_scoreboard(payload).empty and list(F.parse_scoreboard(payload).columns) == F.SCOREBOARD_COLS
    assert F.parse_summary(payload).empty


def test_failed_fetch_returns_empty_frames_and_never_raises(monkeypatch, caplog):
    def boom(*a, **k):
        raise requests.ConnectionError('feed down')
    monkeypatch.setattr(requests, 'get', boom)
    board = F.scoreboard('2026-10-25', SCHEDULE)
    assert board.empty and list(board.columns) == F.SCOREBOARD_COLS
    assert any('failed' in r.message for r in caplog.records)
    fake_board = pd.DataFrame([{**dict.fromkeys(F.SCOREBOARD_COLS), 'nba_game_id': '0022600118', 'espn_event_id': LIVE_EVENT, 'status': 'live'}])
    lines = F.box('0022600118', fake_board, MEM_SAC_BETS)
    assert lines.empty and list(lines.columns) == F.BOX_COLS


def test_http_error_status_is_a_failure(monkeypatch):
    class Resp:
        status_code = 503

        def raise_for_status(self):
            raise requests.HTTPError('503')
    monkeypatch.setattr(requests, 'get', lambda *a, **k: Resp())
    assert F._get('scoreboard') is None


def test_box_for_unknown_game_or_empty_roster_makes_no_request(espn):
    board = F.scoreboard('2026-10-25', SCHEDULE)
    espn.clear()
    assert F.box('0029999999', board, MEM_SAC_BETS).empty
    assert F.box('0022600118', board, MEM_SAC_BETS.iloc[:0]).empty
    assert espn == []
