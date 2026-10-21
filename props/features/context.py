"""
Team-level context for a player-game (T-independent): the player's own team, the opponent, and the schedule. Team level only: no player-vs-team
history appears anywhere.

Reuses the team model's pre-game builders, which already have train/serve parity tests:
  team.team_features_v2.TeamFeatureBuilder.build_entire_history()   EWMAs of a team's PAST games (5- and 20-game spans) at each game
  core.travel_builder.TravelFeatureBuilder.build_features()         rest, distance, time zones, altitude
and adds the opponent-allowed shot-zone profile (the opponent's shots in each of this team's past games), which decides how many rebounds
are available. Columns are prefixed tm_ (the player's team) and opp_ (the opponent).
"""
import sqlite3

import numpy as np
import pandas as pd

from team.team_features_v2 import TeamFeatureBuilder, ewma_pre_game
from core.travel_builder import TravelFeatureBuilder
from props.features import guards

ZONES = {'ra': 'Restricted Area', 'paint': 'In The Paint (Non-RA)', 'mid': 'Mid-Range', 'c3': 'Corner 3', 'atb3': 'Above the Break 3'}
SPAN = 20

# name -> source column in the team history frame
TEAM_COLS = {
    'off_rtg': 'off_offensiveRating_20g', 'def_rtg': 'off_defensiveRating_20g', 'net_rtg': 'off_netRating_20g',
    'pts': 'off_points_20g', 'pts_5g': 'off_points_5g', 'oreb': 'off_offensiveRebounds_20g', 'tov': 'off_turnovers_20g',
    'ast': 'off_assists_20g', 'fg3a': 'off_threePointersAttempted_20g', 'fta': 'off_freeThrowsAttempted_20g',
    'pts_allowed': 'def_OPP_PTS_20g', 'pts_allowed_5g': 'def_OPP_PTS_5g', 'fga_allowed': 'def_OPP_FGA_20g',
    'fg_pct_allowed': 'def_OPP_FG_PCT_20g', 'fg3a_allowed': 'def_OPP_FG3A_20g', 'fg3_pct_allowed': 'def_OPP_FG3_PCT_20g',
    'fta_allowed': 'def_OPP_FTA_20g', 'reb_allowed': 'def_OPP_REB_20g', 'reb_allowed_5g': 'def_OPP_REB_5g',
    'oreb_allowed': 'def_OPP_OREB_20g', 'ast_allowed': 'def_OPP_AST_20g', 'ast_allowed_5g': 'def_OPP_AST_5g',
    'tov_forced': 'def_OPP_TOV_20g', 'win_pct_last10': 'win_pct_last10', 'days_rest': 'days_rest', 'is_b2b': 'is_b2b',
    'games_last_7d': 'games_last_7d'}
TRAVEL_COLS = ['dist_miles', 'tz_delta', 'body_clock_lag', 'road_trip_count', 'game_altitude_diff']


def pregame_ewm(df, group, order, cols, span=SPAN, season_col='season'):
    """Exponentially weighted mean of each group's PAST rows (never the row itself), via the team model's ewma_pre_game."""
    df = df.sort_values([group] + order)
    out = {}
    for c in cols:
        res = np.full(len(df), np.nan)
        pos = 0
        for _, g in df.groupby(group, sort=False):
            res[pos:pos + len(g)] = ewma_pre_game(g[c].to_numpy(dtype=float), g[season_col].to_numpy(), span)
            pos += len(g)
        out[c] = res
    return pd.DataFrame(out, index=df.index)


def _zone_allowed(db_path):
    """For each (game, team): EWM of the zone profile the team's DEFENSE has allowed in earlier games."""
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
        cols = ', '.join(f'"{z} FGA" AS {k}_a, "{z} FGM" AS {k}_m' for k, z in ZONES.items())
        z = pd.read_sql(f"SELECT teamId AS team_id, substr(gameDate, 1, 10) AS game_date, season, {cols} FROM team_shot_location_boxscores", conn)
        g = pd.read_sql("SELECT teamId AS team_id, substr(gameDate, 1, 10) AS game_date, gameId AS game_id FROM team_game_full", conn)
    g['game_id'] = g['game_id'].astype(str).str.zfill(10)
    z = guards.dedupe(z.merge(g, on=['team_id', 'game_date']), ['game_id', 'team_id'], 'zone profiles')
    tot = z[[f'{k}_a' for k in ZONES]].sum(axis=1)
    for k in ZONES:
        z[f'{k}_share'] = z[f'{k}_a'] / tot.where(tot > 0)
        z[f'{k}_pct'] = z[f'{k}_m'] / z[f'{k}_a'].where(z[f'{k}_a'] > 0)
    prof = z[['game_id', 'team_id'] + [f'{k}_{s}' for k in ZONES for s in ('share', 'pct')]]
    # what team X's defense allowed in game g = the OTHER team's offensive profile in g
    assert prof.groupby('game_id')['team_id'].nunique().max() <= 2, 'a game with more than two team profiles'
    other = prof.merge(prof, on='game_id', suffixes=('_def', '_off'))        # many-to-many by construction: the two teams of a game meet twice
    other = other[other['team_id_def'] != other['team_id_off']]
    allowed = other[['game_id', 'team_id_def'] + [f'{k}_{s}_off' for k in ZONES for s in ('share', 'pct')]]
    allowed.columns = ['game_id', 'team_id'] + [f'{k}_{s}_allowed' for k in ZONES for s in ('share', 'pct')]
    meta = z[['game_id', 'team_id', 'game_date', 'season']]
    allowed = allowed.merge(meta, on=['game_id', 'team_id'], validate='one_to_one')
    cols = [c for c in allowed.columns if c.endswith('_allowed')]
    ew = pregame_ewm(allowed, 'team_id', ['game_date', 'game_id'], cols)
    res = allowed.loc[ew.index, ['game_id', 'team_id']].join(ew)
    return res.rename(columns={c: f'zone_{c}' for c in cols})


def team_context(db_path):
    """One row per (game_id, team_id) with every context feature, unprefixed."""
    tf = TeamFeatureBuilder(str(db_path)).build_entire_history()
    tf = tf.rename(columns={'gameId': 'game_id', 'teamId': 'team_id'})
    tf['game_id'] = tf['game_id'].astype(str).str.zfill(10)
    ctx = tf[['game_id', 'team_id', 'is_playoff', 'season_game_number']].copy()
    for name, src in TEAM_COLS.items():
        ctx[name] = tf[src].to_numpy(dtype=float)
    # possessions proxy (pace) and rebound-supply proxies from the same EWMAs
    ctx['pace_proxy'] = (tf['off_fieldGoalsAttempted_20g'] + 0.44 * tf['off_freeThrowsAttempted_20g'] + tf['off_turnovers_20g']
                         - tf['off_offensiveRebounds_20g']).to_numpy()
    ctx['misses'] = (tf['off_fieldGoalsAttempted_20g'] * (1 - tf['off_fieldGoalPercentage_20g'])).to_numpy()
    ctx['misses_allowed'] = (tf['def_OPP_FGA_20g'] * (1 - tf['def_OPP_FG_PCT_20g'])).to_numpy()
    tr = TravelFeatureBuilder(str(db_path)).build_features().rename(columns={'gameId': 'game_id', 'teamId': 'team_id'})
    tr['game_id'] = tr['game_id'].astype(str).str.zfill(10)
    ctx = guards.dedupe(ctx, ['game_id', 'team_id'], 'team context')
    ctx = ctx.merge(guards.dedupe(tr[['game_id', 'team_id'] + TRAVEL_COLS], ['game_id', 'team_id'], 'travel features'), on=['game_id', 'team_id'], how='left', validate='one_to_one')
    ctx = ctx.merge(guards.dedupe(_zone_allowed(db_path), ['game_id', 'team_id'], 'zone-allowed features'), on=['game_id', 'team_id'], how='left', validate='one_to_one')
    return ctx


def add_context(rows, ctx):
    """rows: needs game_id, team_id, opp_team_id. Adds tm_* (own team), opp_* (opponent) and ctx_* (game-level) columns."""
    value_cols = [c for c in ctx.columns if c not in ('game_id', 'team_id', 'is_playoff', 'season_game_number')]
    own = ctx[['game_id', 'team_id'] + value_cols].rename(columns={c: f'tm_{c}' for c in value_cols})
    opp = ctx[['game_id', 'team_id'] + value_cols].rename(columns={'team_id': 'opp_team_id', **{c: f'opp_{c}' for c in value_cols}})
    out = rows.merge(own, on=['game_id', 'team_id'], how='left', validate='many_to_one').merge(opp, on=['game_id', 'opp_team_id'], how='left', validate='many_to_one')
    game = ctx[['game_id', 'team_id', 'is_playoff', 'season_game_number']].rename(
        columns={'is_playoff': 'ctx_is_playoff', 'season_game_number': 'ctx_team_game_number'})
    out = out.merge(game, on=['game_id', 'team_id'], how='left', validate='many_to_one')
    out['ctx_rest_advantage'] = out['tm_days_rest'] - out['opp_days_rest']
    return out
