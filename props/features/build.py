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
from props.features import context, history, injury_at_t, market_env, pregame_roster, rows as rows_mod

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


def build_at_t(base, labels, db_path, props_db, grid=GRID, reports=None, membership='box', hist=None, pos=None, only_game_ids=None):
    """
    T-dependent features for every base row and every decision time.
    membership: the teammate set of the av_* / opp_av_* columns (props.features.pregame_roster): 'box' (default, today's behaviour: the box-score rows),
    'v2nq' or 'pregame_all' (pre-game sets; need the listed history `hist` and positions `pos`, loaded from db_path when not given).
    only_game_ids: development only; output rows of these games (the play-rate tables still come from every base row).
    """
    if membership not in pregame_roster.MEMBERSHIPS:
        raise ValueError(f'unknown membership {membership!r}')
    rows = base if only_game_ids is None else base[base['game_id'].isin(set(only_game_ids))].reset_index(drop=True)
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
    pre = None
    if membership != 'box':
        hist = rows_mod.load_history(db_path) if hist is None else hist
        pos = history.position_groups(db_path) if pos is None else pos
        pre, summary = pregame_roster.prepare_members(rows, hist, pos, reports, tips, grid, membership)
        logging.info(f'membership {membership}: {summary}')
    out = []
    for x in grid:
        chosen, games = injury_at_t.latest_report_rows(reports, tips, x)
        status = injury_at_t.status_for_rows(base[['game_id', 'player_id', 'team_id']], chosen, games)
        keyed = status.merge(base[['game_id', 'player_id', 'season', 'm_bucket']], on=['game_id', 'player_id'], validate='one_to_one').merge(lab, on=['game_id', 'player_id'], validate='one_to_one')
        table = injury_at_t.play_rate_table(keyed, x, rows_mod.FIRST_DATASET_SEASON)
        status_rows = status if only_game_ids is None else status[status['game_id'].isin(set(rows['game_id']))].reset_index(drop=True)
        env_x = market_env.team_view(rows[['game_id', 'player_id', 'is_home']], env, x)
        if pre is None:
            block = injury_at_t.at_t_block(rows, status_rows, table)
        else:
            members, member_feat = pre[x]
            member_status = injury_at_t.status_for_rows(members[['game_id', 'player_id', 'team_id']], chosen, games)
            block = pregame_roster.at_t_members_block(rows, status_rows, table, members, member_status, member_feat)
        t = block.merge(env_x, on=['game_id', 'player_id'], validate='one_to_one')
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


def read_tables(out_db=OUT_DB, game_ids=None):
    """The three dataset tables; `game_ids` limits them to those games (the live parity test needs about 40 dates, not the whole dataset)."""
    import sqlite3
    where = ''
    if game_ids is not None:
        ids = [str(g) for g in game_ids]
        assert all(g.isdigit() for g in ids), 'game ids are digit strings'
        where = ' WHERE game_id IN (%s)' % ','.join(f"'{g}'" for g in ids)
    with sqlite3.connect(f'file:{out_db}?mode=ro', uri=True) as conn:
        return {n: pd.read_sql(f'SELECT * FROM {n}{where}', conn) for n in ('player_game_base', 'player_game_labels', 'player_game_at_T')}


def build_all(db_path='data/nba_stats.db', props_db='data/props_odds.db', out_db=OUT_DB, grid=GRID, max_date=None, refresh_reports=False, membership='box',
              reports=None):
    """membership: the teammate set of av_* (see build_at_t; default 'box'). reports: the linked report rows, when already loaded (then refresh_reports is not used)."""
    base, labels = build_base(db_path, max_date=max_date)
    reports = load_linked_reports(db_path, refresh=refresh_reports) if reports is None else reports
    hist = pos = None
    if membership != 'box':                                   # the same listed history build_base used (same max_date cut)
        hist = rows_mod.load_history(db_path)
        hist = hist if max_date is None else hist[hist['game_date'] <= max_date]
        pos = history.position_groups(db_path)
    at_t = build_at_t(base, labels, db_path, props_db, grid, reports=reports, membership=membership, hist=hist, pos=pos)
    write_tables(out_db, player_game_base=base, player_game_labels=labels, player_game_at_T=at_t)
    logging.info(f'wrote {len(base)} base rows, {len(at_t)} at-T rows, {len(labels)} label rows to {out_db}')
    return base, labels, at_t
