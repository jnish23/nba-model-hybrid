"""Today's Games live scores against the picks (dashboard/live_scores.py), on the saved ESPN scoreboard of 2026-10-25: MEM @ SAC in the 3rd (92-84), LAL @ UTA final (132-123)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from core import live_feed as F
from dashboard import live_scores as LS

FIX = Path(__file__).resolve().parent.parent / 'fixtures' / 'live_feed'
GAMES = pd.DataFrame({
    'game_id': ['0022600118', '0022600112', '0022600199'], 'game_date': ['2026-10-25'] * 3,
    'home_team': ['Sacramento Kings', 'Utah Jazz', 'Boston Celtics'], 'away_team': ['Memphis Grizzlies', 'Los Angeles Lakers', 'Miami Heat'],
    'ml_prob': [0.35, 0.50, 0.60], 'odds_home_ml': [-130.0, -110.0, -150.0], 'odds_away_ml': [110.0, -110.0, 130.0],
    'spread_pred': [1.0, -10.0, 4.0], 'odds_home_spread': [-2.5, 5.5, -3.5], 'total_pred': [225.0, 245.0, 220.0], 'odds_total': [230.0, 240.5, 218.5]})


@pytest.fixture()
def board():
    raw = F.parse_scoreboard(json.loads((FIX / 'espn_scoreboard_2026-10-25.json').read_text(encoding='utf-8')))
    return F.attach_nba_ids(raw, GAMES)


def test_pick_sides_follow_the_table():
    g = LS.pick_sides(GAMES)
    assert list(g['ml_side']) == ['away', None, None]
    assert list(g['spread_side']) == ['away', 'away', 'home'] and list(g['total_side']) == ['under', 'over', 'over']
    assert LS.pick_sides(GAMES.assign(odds_total=float('nan')))['total_side'].isna().all()


def test_live_view_of_a_live_and_a_final_game(board):
    full = LS.live_view(LS.pick_sides(GAMES), board)
    assert list(full.columns) == LS.LIVE_COLS + ['ml_result', 'spread_result', 'total_result']
    assert full[['ml_result', 'spread_result', 'total_result']].values.tolist() == [['', '', ''], ['', 'won', 'won']]   # live under not decided; final LAL has no ML pick
    v = full[LS.LIVE_COLS].set_index('Game')
    assert list(v.index) == ['MEM @ SAC', 'LAL @ UTA']
    assert v.loc['MEM @ SAC'].to_dict() == {'Score': 'MEM 92 - 84 SAC', 'Clock': 'Q3 39.5', 'ML': 'MEM (pick) leads by 8',
                                            'Spread': 'MEM +2.5: covering by 10.5', 'Total': 'Under 230: 176 now, 54 below'}
    assert v.loc['LAL @ UTA'].to_dict() == {'Score': 'LAL 132 - 123 UTA', 'Clock': 'Final', 'ML': 'LAL won by 9',
                                            'Spread': 'LAL -5.5: covered by 3.5', 'Total': 'Over 240.5: won (255)'}


def test_the_logs_own_score_and_status_columns_do_not_interfere(board):
    logged = GAMES.assign(home_score=[None, 123.0, None], away_score=[None, 132.0, None], status=['SCHEDULED', 'COMPLETED', 'SCHEDULED'], period=None)
    assert LS.live_view(LS.pick_sides(logged), board).equals(LS.live_view(LS.pick_sides(GAMES), board))


def test_scheduled_unknown_or_missing_board_gives_nothing(board):
    g = LS.pick_sides(GAMES)
    assert LS.live_view(g, board.assign(status='scheduled')).empty
    assert LS.live_view(g, board.iloc[:0]).empty and LS.live_view(g.iloc[:0], board).empty and LS.live_view(g, None).empty


@pytest.mark.parametrize('side, hs, as_, text', [(None, 100, 100, 'tied'), (None, 90, 100, 'MIA leads by 10'), ('home', 90, 100, 'BOS (pick) trails by 10'),
                                                 ('away', 90, 100, 'MIA (pick) leads by 10'), ('home', 95, 95, 'BOS (pick) tied')])
def test_ml_state(side, hs, as_, text):
    assert LS.ml_state(side, 'BOS', 'MIA', hs, as_) == text


@pytest.mark.parametrize('side, line, hs, as_, text', [
    ('home', -3.5, 100, 95, 'BOS -3.5: covering by 1.5'), ('home', -3.5, 100, 98, 'BOS -3.5: not covering by 1.5'), ('home', -3.0, 100, 97, 'BOS -3: push now'),
    ('away', -3.5, 100, 98, 'MIA +3.5: covering by 1.5'), ('away', -3.5, 100, 90, 'MIA +3.5: not covering by 6.5'), (None, -3.5, 100, 90, '')])
def test_spread_state(side, line, hs, as_, text):
    assert LS.spread_state(side, line, 'BOS', 'MIA', hs, as_) == text


@pytest.mark.parametrize('side, line, hs, as_, text', [
    ('over', 224.5, 90, 86, 'Over 224.5: 176 now, needs 49 more'), ('over', 224.0, 90, 86, 'Over 224: 176 now, needs 49 more'),
    ('over', 224.5, 120, 105, 'Over 224.5: over already (225)'), ('under', 224.5, 90, 86, 'Under 224.5: 176 now, 48.5 below'),
    ('under', 224.5, 120, 105, 'Under 224.5: busted (225)'), ('under', 224.0, 112, 112, 'Under 224: 224 now, 0 below'), (None, 224.5, 1, 1, '')])
def test_total_state(side, line, hs, as_, text):
    assert LS.total_state(side, line, hs, as_) == text


def test_period_label():
    assert [LS.period_label(p, c, s) for p, c, s in [(2, '0.0', 'live'), (5, '2:00', 'live'), (6, None, 'final'), (4, None, 'final'), (5, None, 'final')]] == \
        ['Q2 0.0', 'OT 2:00', 'Final (2OT)', 'Final', 'Final (OT)']


@pytest.mark.parametrize('kind,side,line,home_line,hs,as_,final,res', [
    ('ml', 'home', None, None, 100, 95, True, 'won'), ('ml', 'away', None, None, 100, 95, True, 'lost'), ('ml', 'home', None, None, 100, 95, False, ''),
    ('ml', None, None, None, 100, 95, True, ''),
    ('spread', 'home', None, -3.5, 100, 98, True, 'lost'), ('spread', 'away', None, -3.5, 100, 98, True, 'won'), ('spread', 'home', None, -2.0, 100, 98, True, ''),
    ('spread', 'home', None, float('nan'), 100, 98, True, ''), ('spread', 'home', None, 5.5, 90, 100, False, ''),
    ('total', 'over', 224.5, None, 110, 100, True, 'lost'), ('total', 'under', 224.5, None, 110, 100, True, 'won'), ('total', 'under', 210.0, None, 110, 100, True, ''),
    ('total', 'over', 200.5, None, 110, 100, False, 'won'), ('total', 'under', 200.5, None, 110, 100, False, 'lost'), ('total', 'over', 224.5, None, 110, 100, False, ''),
    ('total', 'under', 224.5, None, 110, 100, False, ''), ('total', 'over', float('nan'), None, 110, 100, True, '')])
def test_pick_result(kind, side, line, home_line, hs, as_, final, res):
    assert LS.pick_result(kind, side, line, home_line, hs, as_, final) == res


def test_final_wording():
    assert LS.ml_state('away', 'BOS', 'MIA', 100, 95, final=True) == 'MIA (pick) lost by 5' and LS.ml_state(None, 'BOS', 'MIA', 100, 95, final=True) == 'BOS won by 5'
    assert LS.spread_state('home', -3.5, 'BOS', 'MIA', 100, 98, final=True) == 'BOS -3.5: did not cover by 1.5'
    assert LS.spread_state('home', -2.0, 'BOS', 'MIA', 100, 98, final=True) == 'BOS -2: push'
    assert [LS.total_state(s, 224.5, 110, 100, final=True) for s in ('over', 'under')] == ['Over 224.5: lost (210)', 'Under 224.5: won (210)']
    assert LS.total_state('under', 210.0, 110, 100, final=True) == 'Under 210: push (210)'
