import numpy as np
import pandas as pd

from props.live import scores

NOW = pd.Timestamp('2026-10-26T04:00:00Z')


def _player(pid, minutes, pts=0, reb=0, ast=0, tpm=0):
    return {'personId': pid, 'statistics': {'minutes': minutes, 'points': pts, 'reboundsTotal': reb, 'assists': ast, 'threePointersMade': tpm}}


def _box(team_minutes=('240:00', '240:00')):
    return {'homeTeam': {'players': [_player(1, '30:00', 20, 8, 5, 2), _player(2, '', 0)], 'statistics': {'minutes': team_minutes[0]}},
            'awayTeam': {'players': [_player(3, '25:10', 10, 3, 9, 0)], 'statistics': {'minutes': team_minutes[1]}}}


def _bets():
    rows = [('b1', '0022600001', 1, 'rebounds', 'over', 7.5), ('b2', '0022600001', 1, 'ra', 'under', 12.5), ('b3', '0022600001', 2, 'assists', 'over', 1.5),
            ('b4', '0022600001', 3, 'assists', 'over', 9.0), ('b5', '0022600001', 9, 'threes', 'under', 1.5), ('b6', '0022600002', 1, 'threes', 'over', 1.5)]
    d = pd.DataFrame(rows, columns=['bet_id', 'game_id', 'player_id', 'stat', 'side', 'line'])
    d['tip_ref_utc'] = ['2026-10-26T00:00:00Z'] * 5 + ['2026-10-26T03:00:00Z']
    return d


def test_minutes_and_grade():
    assert scores.minutes_float('29:30') == 29.5 and scores.minutes_float('') == 0.0 and scores.minutes_float(None) == 0.0
    assert scores.grade('over', 7.5, 8, 30) == 'win' and scores.grade('under', 7.5, 8, 30) == 'loss' and scores.grade('over', 9.0, 9, 25) == 'push'
    assert scores.grade('over', 1.5, 0, 0) == 'void' and scores.grade('over', 1.5, np.nan, 0.0) == 'void'


def test_a_game_is_final_only_when_both_teams_reach_240_minutes():
    assert scores.player_lines(_box())[0] is True
    assert scores.player_lines(_box(('240:00', '212:30')))[0] is False
    assert scores.player_lines(None)[0] is False and scores.player_lines({'homeTeam': {}, 'awayTeam': {}})[0] is False


def test_provisional_results_grade_finished_games_only():
    calls = []

    def fetch(gid):
        calls.append(gid)
        return _box() if gid == '0022600001' else None

    r = scores.provisional_results(_bets(), NOW, fetch).set_index('bet_id')
    assert r.loc['b1', 'provisional'] == 'win' and r.loc['b1', 'actual'] == 8                                  # rebounds 8 > 7.5
    assert r.loc['b2', 'provisional'] == 'loss' and r.loc['b2', 'actual'] == 13                                  # rebounds + assists = 13 > 12.5: the under loses
    assert r.loc['b3', 'provisional'] == 'void' and r.loc['b4', 'provisional'] == 'push' and r.loc['b5', 'provisional'] == 'void'      # no minutes / not on the box score
    assert 'b6' not in r.index and calls == ['0022600001']                                                       # the 03:00 game cannot be over yet: never even requested
    assert (r['game_final'] == 1).all()


def test_nothing_pending_gives_an_empty_frame():
    assert scores.provisional_results(_bets().iloc[0:0], NOW, lambda g: _box()).columns.tolist() == scores.COLS
