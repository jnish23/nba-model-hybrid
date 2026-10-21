"""
The player-game universe and the per-game box-score history.

roster rows: every player listed for a game in the v3 box-score tables, played or not (DNP/inactive players carry a reason in `comment`).
They are the rows of the dataset (from FIRST_DATASET_SEASON on) and, for earlier seasons, the history that rate features start from.

LABELS vs FEATURES: everything in this module that describes the game itself (minutes, box-score stats, the starter flag, the DNP reason) is a label
for that game. It may be used as a feature only for LATER games of the same player, which is what features/history.py does with a strict shift.
Label columns carry the prefixes `y_` (targets) or `lbl_` (other facts about the game), so a leakage test can find them by name.
"""
import sqlite3

import numpy as np
import pandas as pd

from core.season_utils import season_for_date
from props.features import guards

FIRST_HISTORY_SEASON = '2013-14'
FIRST_DATASET_SEASON = '2019-20'
GAME_TYPES = ('002', '004', '005')               # regular season, playoffs, play-in

BASIC_COLS = {'minutes': 'minutes', 'points': 'points', 'rebounds': 'rebounds', 'assists': 'assists',
              'threePointersMade': 'fg3m', 'threePointersAttempted': 'fg3a', 'fieldGoalsAttempted': 'fga',
              'fieldGoalsMade': 'fgm', 'freeThrowsMade': 'ftm', 'freeThrowsAttempted': 'fta', 'offensiveRebounds': 'oreb', 'defensiveRebounds': 'dreb',
              'turnovers': 'tov', 'personalFouls': 'pf', 'steals': 'stl', 'blocks': 'blk'}
TRACK_COLS = {'reboundChancesTotal': 'reb_chances', 'reboundChancesOffensive': 'reb_chances_off',
              'reboundChancesDefensive': 'reb_chances_def', 'touches': 'touches', 'passes': 'passes',
              'secondaryAssists': 'secondary_ast', 'freeThrowAssists': 'ft_ast',
              'contestedFieldGoalsAttempted': 'contested_fga'}
USAGE_COLS = {'usagePercentage': 'usage_pct'}
TARGETS = {'points': 'y_points', 'rebounds': 'y_rebounds', 'assists': 'y_assists', 'fg3m': 'y_threes'}


def _q(conn, sql, params=()):
    return pd.read_sql(sql, conn, params=params)


def load_games(conn):
    """One row per (game, team): date, home flag, opponent. From team_basic_stats ('X vs. Y' is home, 'X @ Y' away)."""
    t = _q(conn, "SELECT gameId AS game_id, teamId AS team_id, substr(gameDate, 1, 10) AS game_date, MATCHUP AS matchup "
                 "FROM team_basic_stats")
    t['game_id'] = t['game_id'].astype(str).str.zfill(10)
    t['is_home'] = t['matchup'].str.contains(' vs. ', regex=False).astype(int)
    opp = t[['game_id', 'team_id']].rename(columns={'team_id': 'opp_team_id'})
    assert t.groupby('game_id')['team_id'].nunique().max() <= 2, 'a game with more than two teams'
    t = t.merge(opp, on='game_id')                                                  # many-to-many by construction: each game's two teams meet twice
    t = guards.dedupe(t[t['team_id'] != t['opp_team_id']].drop(columns='matchup'), ['game_id', 'team_id'], 'team games')
    t['game_type'] = t['game_id'].str[:3]
    t['season'] = [season_for_date(d) for d in t['game_date']]
    m = _q(conn, "SELECT gameId AS game_id, teamId AS team_id, points - OPP_PTS AS lbl_team_margin FROM team_game_full")
    m['game_id'] = m['game_id'].astype(str).str.zfill(10)
    return t.merge(guards.dedupe(m, ['game_id', 'team_id'], 'team margins'), on=['game_id', 'team_id'], how='left', validate='one_to_one')


def load_history(db_path, first_season=FIRST_HISTORY_SEASON, game_types=GAME_TYPES):
    """
    Roster rows with box-score stats, from `first_season` on, sorted by (player, date, game). Columns:
    game_id, player_id, team_id, opp_team_id, game_date, season, game_type, is_home,
    lbl_played, lbl_started, lbl_dnp_reason, minutes (NaN if not played), the stat columns, the y_ targets.
    """
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
        games = load_games(conn)
        roster = _q(conn, "SELECT gameId AS game_id, personId AS player_id, teamId AS team_id, position AS pos_flag, "
                          "comment AS lbl_dnp_reason FROM player_advanced_stats_v3")
        basic = _q(conn, "SELECT gameId AS game_id, personId AS player_id, " + ', '.join(
            f'"{k}" AS {v}' for k, v in BASIC_COLS.items()) + " FROM player_basic_stats")
        track = _q(conn, "SELECT gameId AS game_id, personId AS player_id, " + ', '.join(
            f'"{k}" AS {v}' for k, v in TRACK_COLS.items()) + " FROM player_track_stats_v3")
        usage = _q(conn, "SELECT gameId AS game_id, personId AS player_id, " + ', '.join(
            f'"{k}" AS {v}' for k, v in USAGE_COLS.items()) + " FROM player_usage_stats_v3")
    for d in (roster, basic, track, usage):
        d['game_id'] = d['game_id'].astype(str).str.zfill(10)
    keys = ['game_id', 'player_id']
    df = guards.dedupe(roster, keys, 'roster').merge(games, on=['game_id', 'team_id'], how='inner', validate='many_to_one')
    df = df.merge(guards.dedupe(basic, keys, 'basic stats'), on=keys, how='left', validate='one_to_one')
    df = df.merge(guards.dedupe(track, keys, 'tracking stats'), on=keys, how='left', validate='one_to_one')
    df = df.merge(guards.dedupe(usage, keys, 'usage stats'), on=keys, how='left', validate='one_to_one')
    df = df[df['game_type'].isin(game_types) & (df['season'] >= first_season)].copy()
    df['lbl_played'] = (df['minutes'].fillna(0) > 0).astype(int)
    df['lbl_started'] = df['pos_flag'].fillna('').astype(str).str.len().gt(0).astype(int)
    df.loc[df['lbl_played'] == 0, df.columns.intersection(
        list(BASIC_COLS.values()) + list(TRACK_COLS.values()) + list(USAGE_COLS.values()))] = np.nan
    df = df.drop(columns='pos_flag')
    for src, tgt in TARGETS.items():
        df[tgt] = df[src]
    df['y_pra'] = df[['points', 'rebounds', 'assists']].sum(axis=1, min_count=3)
    return df.sort_values(['player_id', 'game_date', 'game_id']).reset_index(drop=True)
