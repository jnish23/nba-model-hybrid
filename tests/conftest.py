import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DB_PATH = ROOT / 'data' / 'nba_stats.db'


def pytest_configure(config):
    config.addinivalue_line('markers', 'db: needs data/nba_stats.db (skipped in CI)')


def pytest_collection_modifyitems(config, items):
    import pytest
    if DB_PATH.exists():
        return
    skip = pytest.mark.skip(reason='data/nba_stats.db not available')
    for item in items:
        if 'db' in item.keywords:
            item.add_marker(skip)


import sqlite3

import numpy as np
import pandas as pd
import pytest


def make_game_log(n_per_season=24, seed=0):
    """Two teams playing each other (alternating home/away) over two seasons, with an offseason gap,
    a back-to-back, a playoff-style game id and a NaN stat. Returns a team_game_full-shaped frame
    (upper-case WL / MATCHUP, as in the real table)."""
    rng = np.random.default_rng(seed)
    off = ['offensiveRating', 'defensiveRating', 'netRating', 'fieldGoalPercentage', 'threePointerPercentage',
           'freeThrowPercentage', 'points', 'plusMinus', 'offensiveRebounds', 'defensiveRebounds', 'assists',
           'turnovers', 'assistToTurnover', 'percentagePointsPaint', 'percentagePointsFastBreak',
           'percentagePointsOffTurnovers', 'percentageFieldGoalsAttempted3pt', 'fieldGoalsAttempted',
           'threePointersAttempted', 'freeThrowsAttempted', 'steals', 'blocks']
    opp = [f'OPP_{s}' for s in ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                                'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PTS']]
    rows = []
    gid = 0
    for season, start in (('2024-25', '2024-10-22'), ('2025-26', '2025-10-22')):
        d = pd.Timestamp(start)
        for g in range(n_per_season):
            d = d + pd.Timedelta(days=1 if g in (5, 6) else int(rng.integers(2, 4)))   # a couple of B2Bs
            home, away = (1, 2) if g % 2 == 0 else (2, 1)
            game_id = f"{'004' if (season == '2025-26' and g >= n_per_season - 2) else '002'}{gid:07d}"
            gid += 1
            pts = {home: int(rng.integers(95, 125)), away: int(rng.integers(95, 125))}
            for team, opp_team, is_home in ((home, away, True), (away, home, False)):
                tri = f'T{team}'
                otri = f'T{opp_team}'
                row = {'seasonYear': season, 'teamId': team, 'teamTricode': tri, 'gameId': game_id,
                       'gameDate': d.strftime('%Y-%m-%d'),
                       'MATCHUP': f'{tri} vs. {otri}' if is_home else f'{tri} @ {otri}',
                       'WL': 'W' if pts[team] > pts[opp_team] else 'L'}
                row.update({c: float(rng.normal(50, 10)) for c in off})
                row.update({c: float(rng.normal(30, 8)) for c in opp})
                row['points'] = float(pts[team])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[3, 'steals'] = np.nan
    return df


@pytest.fixture()
def team_db(tmp_path):
    path = tmp_path / 'teams.db'
    df = make_game_log()
    with sqlite3.connect(path) as conn:
        df.to_sql('team_game_full', conn, index=False)
        df[['teamId', 'teamTricode']].drop_duplicates().assign(teamName=lambda d: 'Team ' + d['teamTricode']) \
            .to_sql('team_basic_stats', conn, index=False)
    return str(path)
