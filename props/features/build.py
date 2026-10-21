"""
Build the Phase 2 dataset: three tables joined on (player_id, game_id).

  player_game_labels   outcomes and facts about the game itself (y_*, lbl_*, minutes). NEVER features.
  player_game_base     T-independent features: minutes/role/rest/rates (history.py) + team and opponent context (context.py)
  player_game_at_T     T-dependent features, one row per (player, game, minutes_before): injury status and probability, teammate and opponent
                       availability (injury_at_t.py), market game environment (market_env.py)

Decision times are parameters (default 240, 60, 5 minutes before the reference tip). Reports are linked once and cached.
"""
import logging
from pathlib import Path

import pandas as pd

from core import injury_report_features as irf
from core.odds_clean import load_clean_odds
from props import schedule as sched_mod
from props.features import context, history, injury_at_t, market_env, rows as rows_mod

GRID = (240, 60, 5)
MARKET_ERA_FIRST_SEASON = '2023-24'              # first season with our own market snapshots
OUT_DB = Path('data/props_features.db')
LINKED_CACHE = Path('data/processed/props_injury_linked.pkl')
MAIN_REPORT_DIR = irf.REPORT_DIR
HOURLY_DIR = Path('data/raw/injury_reports_hourly')
MAIN_CACHE = Path('data/processed/props_injury_main_parsed.csv')
HOURLY_CACHE = Path('data/processed/props_injury_hourly_parsed.csv')
SEED_CACHE = Path('data/processed/injury_reports_parsed.csv')        # the team pipeline's cache: copied once, never written to here


def load_linked_reports(db_path, refresh=False):
    """Parse both report folders (incremental caches of their own), link to teams, games and players. Cached as parquet."""
    if LINKED_CACHE.exists() and not refresh:
        return pd.read_pickle(LINKED_CACHE)
    if not MAIN_CACHE.exists() and SEED_CACHE.exists():
        MAIN_CACHE.write_bytes(SEED_CACHE.read_bytes())               # start from the team pipeline's parse, then add only what is new
    frames = [irf.parse_all_reports(MAIN_REPORT_DIR, MAIN_CACHE)]
    if HOURLY_DIR.exists():
        frames.append(irf.parse_all_reports(HOURLY_DIR, HOURLY_CACHE))
    reports = pd.concat(frames, ignore_index=True).drop_duplicates(['source_file', 'team', 'player', 'status', 'matchup'])
    linked = irf.link_reports(reports, str(db_path))
    LINKED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    linked.to_pickle(LINKED_CACHE)
    return linked


def reference_tips_et(seasons, offset_min=sched_mod.TIP_OFFSET_MIN):
    """game_id -> naive US/Eastern reference tip (NBA scheduled tip + offset)."""
    sch = pd.concat([sched_mod.load_schedule(s) for s in seasons], ignore_index=True).drop_duplicates('game_id')
    tip = pd.to_datetime(sch['nba_tip_utc'], utc=True) + pd.Timedelta(minutes=offset_min)
    et = tip.dt.tz_convert('America/New_York').dt.tz_localize(None)
    return pd.Series(et.to_numpy(), index=sch['game_id'].astype(str).str.zfill(10))


def build_base(db_path, first_season=rows_mod.FIRST_DATASET_SEASON, max_date=None):
    hist = rows_mod.load_history(db_path)
    if max_date is not None:
        hist = hist[hist['game_date'] <= max_date]
    pos = history.position_groups(db_path)
    base = history.add_history_features(hist, pos, first_season)
    base = context.add_context(base, context.team_context(db_path))
    label_cols = ['game_id', 'player_id', 'lbl_played', 'lbl_started', 'lbl_dnp_reason', 'lbl_team_margin', 'minutes'] + \
                 [c for c in hist.columns if c.startswith('y_')] + list(rows_mod.BASIC_COLS.values()) + list(rows_mod.TRACK_COLS.values())
    labels = hist.loc[hist['season'] >= first_season, list(dict.fromkeys(c for c in label_cols if c in hist.columns))].reset_index(drop=True)
    # every outcome column is marked as a label by its name, so no feature table can contain one by accident
    labels = labels.rename(columns={c: f'lbl_{c}' for c in labels.columns
                                    if c not in ('game_id', 'player_id') and not c.startswith(('y_', 'lbl_'))})
    return base, labels


def build_at_t(base, labels, db_path, props_db, grid=GRID, reports=None):
    """T-dependent features for every base row and every decision time."""
    seasons = sorted(base['season'].unique())
    tips = reference_tips_et(seasons)
    reports = load_linked_reports(db_path) if reports is None else reports
    lab = labels[['game_id', 'player_id', 'lbl_played']]
    snap = market_env.snapshot_env(props_db)
    # The SBR proxy only stands in for seasons without snapshots; in the market era a game with no snapshot gets NaN, never a substitute,
    # so every market-era env value comes from the same source.
    proxy = market_env.sbr_proxy(load_clean_odds(Path(db_path)))
    pre_market = set(base.loc[base['season'] < MARKET_ERA_FIRST_SEASON, 'game_id'])
    env = market_env.combine(snap, proxy[proxy['game_id'].isin(pre_market)])
    feat_cols = ['game_id', 'player_id', 'm_min_h15', 'r_usage_pct_h15', 'r_reb_chances_h15', 'r_assists_h15', 'r_secondary_ast_h15', 'r_points_h15']
    out = []
    for x in grid:
        chosen, games = injury_at_t.latest_report_rows(reports, tips, x)
        status = injury_at_t.status_for_rows(base[['game_id', 'player_id', 'team_id']], chosen, games)
        keyed = status.merge(base[['game_id', 'player_id', 'season', 'm_bucket']], on=['game_id', 'player_id'], validate='one_to_one').merge(lab, on=['game_id', 'player_id'], validate='one_to_one')
        table = injury_at_t.play_rate_table(keyed, x, rows_mod.FIRST_DATASET_SEASON)
        d = injury_at_t.attach_p_play(status, base, table)
        d = d.merge(base[['game_id', 'player_id', 'opp_team_id']], on=['game_id', 'player_id'], validate='one_to_one')
        av = injury_at_t.teammate_availability(d, base[feat_cols])
        env_x = market_env.team_view(base[['game_id', 'player_id', 'is_home']], env, x)
        t = d[['game_id', 'player_id', 'inj_status', 'inj_status_code', 'inj_p_play', 'inj_reason_class', 'inj_report_age_min',
               'inj_report_missing']].merge(av.drop(columns=['team_id', 'opp_team_id']), on=['game_id', 'player_id'], validate='one_to_one').merge(
            env_x, on=['game_id', 'player_id'], validate='one_to_one')
        t.insert(2, 'minutes_before', x)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def write_tables(out_db, **tables):
    """Dataset tables go to their own SQLite file (no parquet engine is installed, and none is needed)."""
    import sqlite3
    out_db = Path(out_db)
    out_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(out_db) as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists='replace', index=False, chunksize=20000)
            idx = ['game_id', 'player_id'] + (['minutes_before'] if 'minutes_before' in df.columns else [])
            conn.execute(f'CREATE INDEX IF NOT EXISTS ix_{name} ON {name} ({", ".join(idx)})')


def read_tables(out_db=OUT_DB):
    import sqlite3
    with sqlite3.connect(f'file:{out_db}?mode=ro', uri=True) as conn:
        return {n: pd.read_sql(f'SELECT * FROM {n}', conn) for n in ('player_game_base', 'player_game_labels', 'player_game_at_T')}


def build_all(db_path='data/nba_stats.db', props_db='data/props_odds.db', out_db=OUT_DB, grid=GRID, max_date=None, refresh_reports=False):
    base, labels = build_base(db_path, max_date=max_date)
    reports = load_linked_reports(db_path, refresh=refresh_reports)
    at_t = build_at_t(base, labels, db_path, props_db, grid, reports=reports)
    write_tables(out_db, player_game_base=base, player_game_labels=labels, player_game_at_T=at_t)
    logging.info(f'wrote {len(base)} base rows, {len(at_t)} at-T rows, {len(labels)} label rows to {out_db}')
    return base, labels, at_t
