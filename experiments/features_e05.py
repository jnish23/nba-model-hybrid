"""E5 - pace-normalised, opponent-adjusted team features.

For every team-game: per-100-possession box-score stats (offense: own, defense: what the opponent did), minus what the
opponent usually does against everyone else, using ONLY the opponent's games before this one:

    adj_off_s(A, t) = off100_s(A, t) - EWMA20_{prior games of B}[ def100_s ]      (B = opponent of A at t)
    adj_def_s(A, t) = def100_s(A, t) - EWMA20_{prior games of B}[ off100_s ]

then EWMAs (spans 5 and 20) of those adjusted series over A's prior games. All inputs are shifted so that nothing from
game t or later is used (tests/test_experiment_lookahead.py corrupts the future and checks).
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

STATS = {  # own column -> opponent-total column
    'points': 'OPP_PTS', 'fieldGoalsAttempted': 'OPP_FGA', 'threePointersAttempted': 'OPP_FG3A',
    'freeThrowsAttempted': 'OPP_FTA', 'turnovers': 'OPP_TOV', 'offensiveRebounds': 'OPP_OREB',
    'defensiveRebounds': 'OPP_DREB', 'assists': 'OPP_AST', 'steals': 'OPP_STL', 'blocks': 'OPP_BLK'}
SPANS = (5, 20)
ALLOW_SPAN = 20


def _pre_game_ewma(df, cols, span):
    """EWMA of `cols` over each team's PRIOR games (strictly before the row)."""
    ew = df.groupby('teamId')[cols].ewm(span=span).mean().reset_index(level=0, drop=True).reindex(df.index)
    return ew.groupby(df['teamId']).shift(1)


def build_features(log):
    """log: team_game_full-like frame (teamId, gameId, gameDate, minutes, pace + STATS columns). Returns one row per
    (gameId, teamId) with e5_* feature columns."""
    df = log.sort_values(['teamId', 'gameDate', 'gameId'], kind='stable').reset_index(drop=True).copy()
    poss = df['pace'] * df['minutes'] / 240.0
    own, opp = list(STATS), list(STATS.values())
    off100 = pd.DataFrame({f'o_{c}': df[c] / poss * 100 for c in own})
    def100 = pd.DataFrame({f'd_{c}': df[STATS[c]] / poss * 100 for c in own})
    work = pd.concat([df[['teamId', 'gameId', 'gameDate']], off100, def100], axis=1)
    work['poss'] = poss
    # what each team usually produces / allows, from its own PRIOR games
    prior = _pre_game_ewma(work, [c for c in work.columns if c.startswith(('o_', 'd_'))], ALLOW_SPAN)
    prior.columns = [f'prior_{c}' for c in prior.columns]
    work = pd.concat([work, prior], axis=1)
    # opponent's entering norms, matched on the same game
    opp_cols = [c for c in work.columns if c.startswith('prior_')]
    o = work[['gameId', 'teamId'] + opp_cols].rename(columns={'teamId': 'oppId', **{c: f'opp_{c}' for c in opp_cols}})
    m = work.merge(o, on='gameId')
    m = m[m['teamId'] != m['oppId']].drop_duplicates(['gameId', 'teamId'])
    work = work[['teamId', 'gameId', 'gameDate'] + [c for c in work.columns if c.startswith(('o_', 'd_')) or c == 'poss']].merge(
        m[['gameId', 'teamId'] + [f'opp_{c}' for c in opp_cols]], on=['gameId', 'teamId'], how='left')
    work = work.sort_values(['teamId', 'gameDate', 'gameId'], kind='stable').reset_index(drop=True)
    adj = {}
    for c in own:
        adj[f'adjoff_{c}'] = work[f'o_{c}'] - work[f'opp_prior_d_{c}']      # opponent's usual allowance
        adj[f'adjdef_{c}'] = work[f'd_{c}'] - work[f'opp_prior_o_{c}']      # opponent's usual output
    adj['pace'] = work['poss']
    A = pd.concat([work[['teamId', 'gameId', 'gameDate']], pd.DataFrame(adj)], axis=1)
    feats = [A[['teamId', 'gameId']]]
    for span in SPANS:
        f = _pre_game_ewma(A, list(adj), span)
        f.columns = [f'e5_{c}_{span}g' for c in f.columns]
        feats.append(f)
    return pd.concat(feats, axis=1)


def add_to_games(games, feats):
    """games: one row per game with gameId, teamId_home, teamId_away. Adds e5_*_home/_away and diff_e5_*."""
    fcols = [c for c in feats.columns if c.startswith('e5_')]
    out = games.copy()
    out['_gid'] = out['gameId'].astype(str).str.zfill(10)
    f = feats.copy()
    f['_gid'] = f['gameId'].astype(str).str.zfill(10)
    new = {}
    for side in ('home', 'away'):
        sf = f[['_gid', 'teamId'] + fcols].rename(columns={'teamId': f'teamId_{side}', **{c: f'{c}_{side}' for c in fcols}})
        out = out.merge(sf, on=['_gid', f'teamId_{side}'], how='left')
    for c in fcols:
        new[f'diff_{c}'] = out[f'{c}_home'] - out[f'{c}_away']
    out = pd.concat([out.drop(columns='_gid'), pd.DataFrame(new, index=out.index)], axis=1)
    return out


def build_variant(csv_in, csv_out, db_path):
    with sqlite3.connect(db_path) as conn:
        log = pd.read_sql("SELECT * FROM team_game_full", conn)
    log['gameDate'] = pd.to_datetime(log['gameDate'])
    feats = build_features(log)
    games = pd.read_csv(csv_in, low_memory=False)
    games['teamId_home'] = games['teamId_home'].astype('int64')
    games['teamId_away'] = games['teamId_away'].astype('int64')
    out = add_to_games(games, feats)
    out.to_csv(csv_out, index=False)
    return out
