"""E30 - shared pieces of the forward test of the four E28 opening-line candidates (frozen by e30_freeze.py, scored by e30_score.py).

The game-level feature rows are rebuilt straight from nba_stats.db with experiments/features_tracking.py, for any date range, so the frozen
signals can be evaluated on games the training set does not contain. Every feature for game t uses only games before t (tested), so rows
rebuilt after the fact are point-in-time in content; whether the previous night's tracking data had ARRIVED by noon is a separate question,
handled by the log-based `live_feasible` flag in the scorer.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.travel_builder import TravelFeatureBuilder          # noqa: E402
from experiments import features_tracking as ft               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'nba_stats.db'
# test id -> (family, target); margin = home margin, total = game total
TESTS = {'T1': ('F1', 'margin'), 'T2': ('F3', 'margin'), 'T3': ('F4', 'margin'), 'T4': ('F5', 'total')}
FIT_PRIMARY = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
FIT_ALL = ['2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19'] + FIT_PRIMARY


def game_table(db_path=DB):
    """One row per game in team_game_full: ids, date, season, is_b2b (days since the team's previous game == 1) and dist_miles per side."""
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as c:
        t = pd.read_sql('SELECT seasonYear, teamId, gameId, gameDate, MATCHUP FROM team_game_full', c)
    t['gameId'] = t['gameId'].astype(str).str.zfill(10)
    t['gameDate'] = pd.to_datetime(t['gameDate'])
    t = t.sort_values(['teamId', 'gameDate', 'gameId'], kind='stable').reset_index(drop=True)
    days = (t['gameDate'] - t.groupby('teamId')['gameDate'].shift(1)).dt.days.fillna(7)
    t['is_b2b'] = (days == 1).astype(float)
    tr = TravelFeatureBuilder(str(db_path)).build_features()
    tr['gameId'] = tr['gameId'].astype(str).str.zfill(10)
    t = t.merge(tr[['teamId', 'gameId', 'dist_miles']].drop_duplicates(['teamId', 'gameId']), on=['teamId', 'gameId'], how='left')
    t['is_home'] = t['MATCHUP'].astype(str).str.contains('vs.', regex=False)
    h = t[t['is_home']].rename(columns={'teamId': 'teamId_home', 'is_b2b': 'is_b2b_home', 'dist_miles': 'dist_miles_home'})
    a = t[~t['is_home']].rename(columns={'teamId': 'teamId_away', 'is_b2b': 'is_b2b_away', 'dist_miles': 'dist_miles_away'})
    g = h[['gameId', 'gameDate', 'seasonYear', 'teamId_home', 'is_b2b_home', 'dist_miles_home']].merge(
        a[['gameId', 'teamId_away', 'is_b2b_away', 'dist_miles_away']], on='gameId', how='inner')
    return g.sort_values(['gameDate', 'gameId']).reset_index(drop=True)


def feature_games(db_path=DB):
    """Game table plus every trk_* column, built with the same code as the E28 variant."""
    games = game_table(db_path)
    feats = ft.build_team_features(ft.load_inputs(db_path))
    return ft.add_to_games(games, feats)


def family_X(f, family, target):
    """Regressors of a family, exactly as experiments/e28_screen.py builds them (diff columns for the margin, home + away sums for the total)."""
    cols = {}
    for n in ft.FAMILIES[family]:
        if n == 'one_side_tired':
            v = f['trk_one_side_tired']
            cols[n] = v.abs() if target == 'total' else v
        elif target == 'total':
            cols[n] = f[f'trk_{n}_home'] + f[f'trk_{n}_away']
        else:
            cols[n] = f[f'diff_trk_{n}']
    return pd.DataFrame(cols)


def signal_points(f, spec):
    """Frozen signal in points (centred): ((X - mu) / sd) @ beta - centre. NaN where any regressor is missing."""
    X = family_X(f, spec['family'], spec['target'])[spec['columns']]
    Z = (X - pd.Series(spec['mu'])[spec['columns']]) / pd.Series(spec['sd'])[spec['columns']]
    s = Z.to_numpy(float) @ np.array([spec['beta'][c] for c in spec['columns']]) - spec['centre']
    return pd.Series(np.where(Z.notna().all(axis=1), s, np.nan), index=f.index)
