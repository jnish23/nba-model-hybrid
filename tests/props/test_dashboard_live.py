"""The props board's live panel logic (dashboard_props/logic.py), on the saved ESPN responses of MEM @ SAC in the 3rd quarter (2026-10-25) and the real open bets of that game."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import live_feed as F
from dashboard_props import logic as L
from props.live.scores import STAT_VALUE

FIX = Path(__file__).resolve().parent.parent / 'fixtures' / 'live_feed'
GAME = '0022600118'
NOW = pd.Timestamp('2026-10-26T02:50:00Z')
BETS = pd.DataFrame([            # player_id, player_name, stat, side, line, expected_minutes: the recorded bets of this game
    (203897, 'Zach LaVine', 'ra', 'under', 6.5, 28.3), (1627734, 'Domantas Sabonis', 'assists', 'under', 6.5, 28.3),
    (1629631, "De'Andre Hunter", 'rebounds', 'under', 4.5, 20.4), (1629660, 'Ty Jerome', 'assists', 'under', 6.5, 23.9),
    (1642377, 'Jaylen Wells', 'assists', 'under', 1.5, 21.1), (1642907, 'Cedric Coward', 'ra', 'under', 8.5, 20.1),
    (1643409, 'Cameron Boozer', 'rebounds', 'under', 9.5, 33.1), (1643411, 'Darius Acuff Jr.', 'assists', 'over', 4.5, 33.307624)],
    columns=['player_id', 'player_name', 'stat', 'side', 'line', 'expected_minutes']).assign(
    game_id=GAME, game_date='2026-10-25', tip_ref_utc='2026-10-26T01:00:00Z', home_team='Sacramento Kings', away_team='Memphis Grizzlies', y=np.nan)


def fixture(name):
    return json.loads((FIX / name).read_text(encoding='utf-8'))


@pytest.fixture()
def board():
    sched = BETS[['game_id', 'game_date', 'home_team', 'away_team']].drop_duplicates()
    return F.attach_nba_ids(F.parse_scoreboard(fixture('espn_scoreboard_2026-10-25.json')), sched)


@pytest.fixture()
def lines():
    out, unmatched = F.match_players(F.parse_summary(fixture('espn_summary_in_401909867.json')), BETS, overrides={})
    assert unmatched == []
    return out.assign(nba_game_id=GAME)


def test_live_stat_matches_the_settlement_definitions():
    nba = {'points': 21, 'reboundsTotal': 7, 'assists': 5, 'threePointersMade': 3}
    espn = {'points': 21, 'rebounds': 7, 'assists': 5, 'threes': 3}
    assert set(L.LIVE_STAT) == set(STAT_VALUE)
    assert all(L.LIVE_STAT[s](espn) == STAT_VALUE[s](nba) for s in STAT_VALUE)


def test_live_view_of_every_open_bet(board, lines):
    v = L.live_view(BETS, board, lines).set_index('Player')
    assert list(L.live_view(BETS, board, lines).columns) == L.LIVE_COLS and len(v) == 8
    assert set(v['Game']) == {'MEM 92 @ SAC 84'} and set(v['Clock']) == {'Q3 39.5'}
    expect = {'Zach LaVine': (1, '5 more allowed'), 'Domantas Sabonis': (2, '4 more allowed'), "De'Andre Hunter": (5, 'busted'), 'Ty Jerome': (5, '1 more allowed'),
              'Jaylen Wells': (2, 'busted'), 'Cedric Coward': (7, '1 more allowed'), 'Cameron Boozer': (7, '2 more allowed'), 'Darius Acuff Jr.': (6, 'hit')}
    assert {p: (int(v.loc[p, 'Now']), v.loc[p, 'Status']) for p in v.index} == expect
    assert v.loc['Darius Acuff Jr.', 'Min'] == 26.0 and v.loc['Darius Acuff Jr.', 'On pace'] == pytest.approx(6 / 26 * 33.307624)
    assert v.loc['Darius Acuff Jr.', 'Bet'] == 'OVER Assists 4.5' and v.loc['Darius Acuff Jr.', 'kind'] == 'hit'


def test_bet_whose_player_has_no_box_row_yet(board, lines):
    v = L.live_view(BETS, board, lines[lines['player_id'] != 1643411]).set_index('Player')
    assert np.isnan(v.loc['Darius Acuff Jr.', 'Now']) and v.loc['Darius Acuff Jr.', 'Status'] == 'not in the box score yet' and v.loc['Darius Acuff Jr.', 'kind'] == 'none'
    assert L.live_view(BETS, board, None)['kind'].eq('none').all()


def test_final_or_unscheduled_games_are_not_shown(board, lines):
    assert L.live_view(BETS, board.assign(status='final'), lines).empty
    assert L.live_view(BETS.assign(game_id='0022600999'), board, lines).empty
    assert L.live_view(BETS, board.iloc[:0], lines).empty and L.live_view(BETS.iloc[:0], board, lines).empty


@pytest.mark.parametrize('side, line, cur, kind, text', [
    ('over', 4.5, 4, 'alive', 'needs 1 more'), ('over', 4.5, 5, 'hit', 'hit'), ('over', 5.0, 5, 'hit', 'hit'), ('over', 9.5, 0, 'alive', 'needs 10 more'),
    ('under', 6.5, 6, 'alive', '0 more allowed'), ('under', 6.5, 7, 'busted', 'busted'), ('under', 7.0, 7, 'alive', '0 more allowed'), ('under', 7.0, 8, 'busted', 'busted')])
def test_live_status(side, line, cur, kind, text):
    assert L.live_status(side, line, cur) == (kind, text)


def test_on_pace():
    assert L.on_pace(6, 24, 36) == pytest.approx(9.0)
    assert L.on_pace(6, 24, np.nan) == pytest.approx(6 / 24 * L.DEFAULT_MINUTES)
    assert L.on_pace(6, 40, 30) == pytest.approx(6.0)                      # already past the expected minutes: no shrinking
    assert np.isnan(L.on_pace(2, 3, 30)) and np.isnan(L.on_pace(np.nan, 20, 30))


def test_period_label():
    assert [L.period_label(p, c) for p, c in [(1, '12:00'), (4, '0.4'), (5, '3:10'), (7, '1:00'), (None, None)]] == ['Q1 12:00', 'Q4 0.4', 'OT 3:10', '3OT 1:00', '']


def test_live_candidates_only_open_bets_of_tipped_recent_games():
    b = pd.concat([BETS.iloc[:2],
                   BETS.iloc[2:3].assign(y=3.0),                                          # settled
                   BETS.iloc[3:4].assign(tip_ref_utc='2026-10-26T03:30:00Z'),             # not tipped yet
                   BETS.iloc[4:5].assign(tip_ref_utc='2026-10-25T20:00:00Z')])            # tipped more than 5 h ago
    assert list(L.live_candidates(b, NOW)['player_name']) == ['Zach LaVine', 'Domantas Sabonis']
