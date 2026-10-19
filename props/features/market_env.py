"""
Market game environment at T: game spread, total and win probability, and what they imply for one player's team (decision D4).

Source, in order: our snapshots (props_odds.db `team_lines`, the consensus across books at the snapshot taken at T) for 2023-24 on; for earlier seasons a
flagged proxy from the SBR open/close lines (core.odds_clean): T-240 gets the opening line, T-5 the closing line, T-60 gets nothing. `env_source`
says which: 0 none, 1 snapshot, 2 SBR open, 3 SBR close. The player's PROP lines are never used here.

Blowout probability: the team's final margin is modelled as Normal(expected margin from the spread, MARGIN_SD); the expected margin of the home team
is -home_spread. MARGIN_SD = 12 points is a fixed, documented constant (NBA margins around the closing spread have a standard deviation of about 12).
"""
import sqlite3

import numpy as np
import pandas as pd
from scipy.special import ndtr

from core import devig

MARGIN_SD = 12.0
BLOWOUT_MARGIN = 15.0
MIN_BOOKS = 3
SOURCE_CODE = {'none': 0, 'snapshot': 1, 'sbr_open': 2, 'sbr_close': 3}
GRID = (240, 60, 5)


def snapshot_env(props_db, grid=GRID):
    """Consensus game lines at each grid point from our snapshots (a snapshot later than T, e.g. after a revised tip time, is never used). One row per (game_id, minutes_before): home_spread, total, home_win_prob, n_books, dispersion."""
    modes = [f'hist_T-{x}' for x in grid]
    with sqlite3.connect(f'file:{props_db}?mode=ro', uri=True, timeout=120) as conn:
        q = f"""SELECT r.game_id, l.mode, t.book, t.home_spread, t.total, t.home_ml, t.away_ml
                FROM team_lines t
                JOIN snapshot_log l ON l.snapshot_ts_utc = t.snapshot_ts_utc AND l.event_id_vendor = t.event_id_vendor
                JOIN event_ref r ON r.event_id_vendor = t.event_id_vendor
                LEFT JOIN snapshot_ref s ON s.snapshot_ts_utc = t.snapshot_ts_utc AND s.event_id_vendor = t.event_id_vendor
                WHERE l.mode IN ({','.join('?' * len(modes))}) AND r.game_id IS NOT NULL AND COALESCE(s.in_game, 0) = 0
                  AND s.minutes_to_ref >= CAST(substr(l.mode, 8) AS REAL) - 1.0"""
        d = pd.read_sql(q, conn, params=modes)
    d['minutes_before'] = d['mode'].str.replace('hist_T-', '', regex=False).astype(int)
    fair_home, _, _ = devig.devig_prices(d['home_ml'], d['away_ml'], 'multiplicative')
    d['home_win_prob'] = fair_home
    g = d.groupby(['game_id', 'minutes_before'])
    out = pd.DataFrame({
        'home_spread': g['home_spread'].median(), 'total': g['total'].median(), 'home_win_prob': g['home_win_prob'].median(),
        'n_books': g['home_spread'].count(), 'spread_dispersion': g['home_spread'].agg(lambda s: s.max() - s.min())})
    out.loc[out['n_books'] < MIN_BOOKS, ['home_spread', 'total', 'home_win_prob']] = np.nan
    out['env_source'] = SOURCE_CODE['snapshot']
    return out.reset_index()


def sbr_proxy(clean_odds, grid=GRID):
    """Flagged proxy for seasons without snapshots: SBR opening line as T-240, closing line as T-5, nothing for T-60."""
    o = clean_odds.rename(columns={'gameId': 'game_id'}).copy()
    o['game_id'] = o['game_id'].astype(str).str.zfill(10)
    frames = []
    for x, pre, code in ((240, 'open_', SOURCE_CODE['sbr_open']), (5, 'close_', SOURCE_CODE['sbr_close'])):
        if x not in grid:
            continue
        f = pd.DataFrame({'game_id': o['game_id'], 'minutes_before': x, 'home_spread': o[f'{pre}home_spread'],
                          'total': o[f'{pre}total'], 'home_win_prob': o[f'{pre}home_prob'],
                          'n_books': o[f'{pre}n_books_spread'], 'spread_dispersion': np.nan, 'env_source': code})
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def combine(snapshots, proxy):
    """Snapshot rows win; the proxy fills only (game, T) pairs with no snapshot row."""
    key = ['game_id', 'minutes_before']
    keep = proxy.merge(snapshots[key].assign(_s=1), on=key, how='left')
    keep = keep[keep['_s'].isna()].drop(columns='_s')
    return pd.concat([snapshots, keep], ignore_index=True)


def team_view(rows, env, minutes_before):
    """
    rows: player rows with game_id and is_home. Returns env_* columns for decision time `minutes_before`.
    env_team_spread < 0 means the player's team is the favourite.
    """
    e = env[env['minutes_before'] == minutes_before].drop(columns='minutes_before')
    d = rows[['game_id', 'player_id', 'is_home']].merge(e, on='game_id', how='left')
    hs = d['home_spread'].to_numpy(dtype=float)
    home = d['is_home'].to_numpy() == 1
    team_spread = np.where(home, hs, -hs)
    total = d['total'].to_numpy(dtype=float)
    mu = -team_spread                                           # expected margin of the player's team
    out = d[['game_id', 'player_id']].copy()
    out['env_team_spread'] = team_spread
    out['env_total'] = total
    out['env_team_total'] = total / 2.0 - team_spread / 2.0
    out['env_opp_total'] = total / 2.0 + team_spread / 2.0
    hwp = d['home_win_prob'].to_numpy(dtype=float)
    out['env_team_win_prob'] = np.where(home, hwp, 1.0 - hwp)
    out['env_team_big_win_prob'] = 1.0 - ndtr((BLOWOUT_MARGIN - mu) / MARGIN_SD)
    out['env_team_big_loss_prob'] = ndtr((-BLOWOUT_MARGIN - mu) / MARGIN_SD)
    out['env_blowout_prob'] = out['env_team_big_win_prob'] + out['env_team_big_loss_prob']
    out['env_n_books'] = d['n_books'].to_numpy()
    out['env_spread_dispersion'] = d['spread_dispersion'].to_numpy()
    out['env_source'] = d['env_source'].fillna(SOURCE_CODE['none']).astype(int).to_numpy()
    return out
