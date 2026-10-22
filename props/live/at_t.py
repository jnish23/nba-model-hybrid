"""
The T-dependent block for the live roster: injury status at T (reports stamped at least the publish lag before T), play probabilities and teammate/opponent availability, through
the same functions as the batch build (props.features.injury_at_t.at_t_block).
"""
import sqlite3

import pandas as pd

from props.features import injury_at_t
from props.features.rows import FIRST_DATASET_SEASON


def play_tables(features_db, grid, live_season=None):
    """
    {minutes_before: Series(season, status, bucket -> P(played))} from the stored dataset, by the training function. For a season that already exists in the dataset the table equals what
    training used; `live_season` adds a season with no rows yet, whose table is 'all earlier seasons' (exactly how training treats any season).
    """
    with sqlite3.connect(f'file:{features_db}?mode=ro', uri=True) as conn:
        at_t = pd.read_sql('SELECT game_id, player_id, minutes_before, inj_status FROM player_game_at_T', conn)
        base = pd.read_sql('SELECT game_id, player_id, season, m_bucket FROM player_game_base', conn)
        lab = pd.read_sql('SELECT game_id, player_id, lbl_played FROM player_game_labels', conn)
    tables = {}
    for x in grid:
        keyed = at_t[at_t['minutes_before'] == x].merge(base, on=['game_id', 'player_id'], validate='one_to_one').merge(lab, on=['game_id', 'player_id'], validate='one_to_one')
        tables[x] = injury_at_t.play_rate_table(keyed, x, FIRST_DATASET_SEASON, extra_season=live_season)
    return tables


def live_at_t(base_live, reports, tips_et, minutes_before, table):
    """T-dependent columns for every row of the live base at one decision time (same column set as the stored at-T table, minus the market environment)."""
    chosen, games = injury_at_t.latest_report_rows(reports, tips_et, minutes_before)
    status = injury_at_t.status_for_rows(base_live[['game_id', 'player_id', 'team_id']], chosen, games)
    t = injury_at_t.at_t_block(base_live, status, table)
    t.insert(2, 'minutes_before', minutes_before)
    return t


def asof_reports(reports, tips_et, minutes_before, publish_lag_min=injury_at_t.PUBLISH_LAG_MIN):
    """Only the report rows stamped at least the publish lag before T (the stamp is the scheduled slot). Explicit, so the as-if test does not depend on the builder ignoring later reports."""
    r = reports.dropna(subset=['gameId']).copy()
    r['gameId'] = r['gameId'].astype(str).str.zfill(10)
    cutoff = r['gameId'].map(tips_et) - pd.Timedelta(minutes=minutes_before + publish_lag_min)
    return r[cutoff.notna() & (pd.to_datetime(r['report_ts']) <= cutoff)]


def report_players(reports_asof, tips_et, minutes_before):
    """(game_id, player_id, team_id) of every player named on the chosen (latest usable) report of each game at T; the team is the one the report lists him under."""
    chosen, _ = injury_at_t.latest_report_rows(reports_asof, tips_et, minutes_before)
    c = chosen.dropna(subset=['personId'])
    return pd.DataFrame({'game_id': c['gameId'].astype(str), 'player_id': c['personId'].astype('int64'), 'team_id': pd.to_numeric(c['teamId'], errors='coerce')}).drop_duplicates(
        ['game_id', 'player_id']).reset_index(drop=True)
