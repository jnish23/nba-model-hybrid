"""
Player tiers for coverage and quality reporting. Defined from games strictly BEFORE the game date (no look-ahead), using the
trailing TRAIL games from nba_stats.db:

    star      trailing mean points >= 20
    starter   trailing mean minutes >= 24 (and not a star)
    rotation  everything else (including players with fewer than MIN_GAMES prior games)

The thresholds are fixed here, before any results were looked at; change them only with a note in the report that uses them.
"""
import sqlite3

import numpy as np
import pandas as pd

TRAIL = 20
MIN_GAMES = 5
STAR_PPG = 20.0
STARTER_MPG = 24.0


def trailing_form(nba_db_path):
    """player_id, gameDate, mean minutes / points over the previous TRAIL games (excluding the date itself)."""
    with sqlite3.connect(f'file:{nba_db_path}?mode=ro', uri=True) as conn:
        g = pd.read_sql("SELECT personId AS player_id, substr(gameDate, 1, 10) AS gameDate, minutes, points "
                        "FROM player_basic_stats", conn)
    g = g.dropna(subset=['minutes']).sort_values(['player_id', 'gameDate'])
    grp = g.groupby('player_id')
    g['mpg'] = grp['minutes'].transform(lambda s: s.shift(1).rolling(TRAIL, min_periods=MIN_GAMES).mean())
    g['ppg'] = grp['points'].transform(lambda s: s.shift(1).rolling(TRAIL, min_periods=MIN_GAMES).mean())
    return g[['player_id', 'gameDate', 'mpg', 'ppg']]


def tier_from_form(mpg, ppg):
    mpg, ppg = np.asarray(mpg, dtype=float), np.asarray(ppg, dtype=float)
    return np.where(ppg >= STAR_PPG, 'star', np.where(mpg >= STARTER_MPG, 'starter', 'rotation'))


def assign_tiers(df, nba_db_path=None, date_col='et_game_date', form=None):
    """Adds mpg, ppg, tier to df (needs player_id and a game-date column). Latest form strictly before the date.
    Pass `form=trailing_form(...)` to reuse it across calls."""
    form = trailing_form(nba_db_path) if form is None else form
    left = df.reset_index().rename(columns={'index': '_row'})
    left['_d'] = pd.to_datetime(left[date_col])
    right = form.assign(_d=pd.to_datetime(form['gameDate'])).sort_values('_d')
    left = left.dropna(subset=['player_id']).assign(player_id=lambda d: d['player_id'].astype('int64')).sort_values('_d')
    m = pd.merge_asof(left, right[['player_id', '_d', 'mpg', 'ppg']], on='_d', by='player_id',
                      direction='backward', allow_exact_matches=False)
    out = df.copy()
    m = m.set_index('_row')
    out['mpg'], out['ppg'] = m['mpg'], m['ppg']
    out['tier'] = tier_from_form(out['mpg'], out['ppg'])
    out.loc[out['mpg'].isna(), 'tier'] = 'rotation'
    return out
