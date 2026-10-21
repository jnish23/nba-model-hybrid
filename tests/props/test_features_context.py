import sqlite3

import numpy as np
import pandas as pd
import pytest

from conftest import DB_PATH
from props.features import context


def test_pregame_ewm_never_includes_the_row_itself():
    df = pd.DataFrame({'team': [1, 1, 1, 1, 2, 2], 'date': list('abcdab'), 'season': ['s'] * 6, 'x': [10., 20, 30, 40, 5, 7]})
    out = context.pregame_ewm(df, 'team', ['date'], ['x'], span=3).sort_index()['x']
    alpha = 2 / (3 + 1)
    assert np.isnan(out[0]) and np.isnan(out[4])                                   # first game of a team: no past
    assert out[1] == pytest.approx(10.0)
    assert out[2] == pytest.approx(((1 - alpha) * 10 + 20) / ((1 - alpha) + 1))    # weights 1 (latest) and (1-alpha)
    # changing a row and everything after it leaves its own value unchanged
    df2 = df.copy()
    df2.loc[3, 'x'] = 1e6
    out2 = context.pregame_ewm(df2, 'team', ['date'], ['x'], span=3).sort_index()['x']
    assert out2[3] == pytest.approx(out[3]) and out2[:4].tolist()[:3] == pytest.approx(out[:3].tolist(), nan_ok=True)


def test_add_context_separates_own_team_from_opponent():
    ctx = pd.DataFrame({'game_id': ['g1', 'g1'], 'team_id': [1, 2], 'is_playoff': [0, 0], 'season_game_number': [5, 6],
                        'off_rtg': [110., 105.], 'days_rest': [3., 1.]})
    rows = pd.DataFrame({'game_id': ['g1', 'g1'], 'player_id': [10, 20], 'team_id': [1, 2], 'opp_team_id': [2, 1]})
    out = context.add_context(rows, ctx).set_index('player_id')
    assert out.loc[10, ['tm_off_rtg', 'opp_off_rtg']].tolist() == [110., 105.]
    assert out.loc[20, ['tm_off_rtg', 'opp_off_rtg']].tolist() == [105., 110.]
    assert out.loc[10, 'ctx_rest_advantage'] == 2 and out.loc[20, 'ctx_rest_advantage'] == -2
    assert out.loc[10, 'ctx_team_game_number'] == 5


@pytest.mark.db
def test_zone_profile_allowed_is_the_opponents_past_shots():
    """Recompute one team's allowed restricted-area share from the raw zone table, using only games before the target game."""
    ctx = context.team_context(str(DB_PATH))
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
        g = pd.read_sql("SELECT teamId AS team_id, gameId AS game_id, substr(gameDate,1,10) AS d, seasonYear AS season FROM team_game_full", conn)
        z = pd.read_sql('SELECT teamId AS team_id, substr(gameDate,1,10) AS d, "Restricted Area FGA" AS ra, '
                        '"In The Paint (Non-RA) FGA" AS pt, "Mid-Range FGA" AS mr, "Corner 3 FGA" AS c3, "Above the Break 3 FGA" AS ab, '
                        '"Backcourt FGA" AS bc FROM team_shot_location_boxscores', conn)
    g['game_id'] = g['game_id'].astype(str).str.zfill(10)
    z = z.merge(g, on=['team_id', 'd'])
    z['share'] = z['ra'] / z[['ra', 'pt', 'mr', 'c3', 'ab', 'bc']].sum(axis=1)
    team = 1610612747
    mine = g[(g.team_id == team) & (g.season == '2024-25')].sort_values('d')
    target = mine.iloc[40]
    past = mine.iloc[:40]
    opp_share = []
    for gid in past.game_id:
        opp_share.append(z[(z.game_id == gid) & (z.team_id != team)]['share'].iloc[0])
    alpha = 2 / (context.SPAN + 1)
    w = (1 - alpha) ** np.arange(len(opp_share))[::-1]
    # the team-model EWMA runs across seasons; use a long enough window that the earlier season's tail is negligible check:
    expected_recent = float(np.dot(w, opp_share) / w.sum())
    got = ctx[(ctx.game_id == target.game_id) & (ctx.team_id == team)]['zone_ra_share_allowed'].iloc[0]
    assert got == pytest.approx(expected_recent, abs=0.01)           # equal up to the previous-season tail, which carries weight < 1%
