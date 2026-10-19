"""
Minutes, role and per-minute rate features from a player's OWN earlier games (the T-independent part of the dataset).

Every number is computed from games strictly before the row's game (props/features/decay.py), so the same function serves training and live
use. Hyperparameters live in HYPER and are documented in the feature dictionary; the shrinkage pseudo-counts are to be tuned on 2019-22 only.

Rates are per minute, as a ratio of decayed sums, shrunk toward a prior built from the league's EARLIER seasons for players of the same
position group and trailing-minutes bucket:   rate = (S_stat + K * prior_rate) / (S_minutes + K)
"""
import sqlite3

import numpy as np
import pandas as pd

from props.features import decay
from props.features.rows import FIRST_DATASET_SEASON

HALF_LIVES = (5, 15, 40)                       # in games played
HYPER = {
    'k_minutes': 200.0,                        # pseudo-minutes for per-minute rates (~6 games of a starter)
    'k_attempts': 25.0,                        # pseudo-attempts for shooting percentages
    'k_blowout_games': 3.0,                    # pseudo-games for minutes split by blowout / close games
    'blowout_margin': 15,                      # |final margin| at or above this counts as a blowout
    'minutes_bucket_edges': [0, 12, 20, 28, 60],
}
RATE_STATS = ['points', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm', 'rebounds', 'oreb', 'dreb', 'assists', 'tov', 'pf', 'stl', 'blk',
              'reb_chances', 'reb_chances_off', 'reb_chances_def', 'touches', 'passes', 'secondary_ast', 'ft_ast', 'contested_fga']
PCT_STATS = {'fg_pct': ('fgm', 'fga'), 'fg3_pct': ('fg3m', 'fg3a'), 'ft_pct': ('ftm', 'fta')}
POS_MAP = {'Guard': 'G', 'Forward': 'F', 'Center': 'C', 'Guard-Forward': 'GF', 'Forward-Guard': 'GF',
           'Forward-Center': 'FC', 'Center-Forward': 'FC'}


def position_groups(db_path):
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
        p = pd.read_sql("SELECT PERSON_ID AS player_id, POSITION FROM player_info", conn)
    return dict(zip(p['player_id'].astype(int), p['POSITION'].map(POS_MAP).fillna('F')))


def _bucket(mean_minutes):
    edges = HYPER['minutes_bucket_edges']
    b = pd.cut(pd.Series(mean_minutes), edges, labels=False, right=False)
    return b.fillna(-1).astype(int).to_numpy()          # -1: no history yet


def add_history_features(hist, pos_by_player, first_dataset_season=FIRST_DATASET_SEASON):
    """
    hist: output of rows.load_history (sorted by player, date, game; all seasons from the history start).
    Returns the rows of seasons >= first_dataset_season with keys and feature columns (prefixes: m_ minutes, role_, rest_, r_ rates, c_ conversion).
    """
    h = hist.reset_index(drop=True)
    g = h['player_id'].to_numpy()
    played = h['lbl_played'].to_numpy().astype(bool)
    all_rows = np.ones(len(h), dtype=bool)
    minutes = h['minutes'].to_numpy(dtype=float)
    out = h[['game_id', 'player_id', 'team_id', 'opp_team_id', 'game_date', 'season', 'game_type', 'is_home']].copy()

    # ---- minutes: decayed mean minutes per game played, and its spread
    for hl in HALF_LIVES:
        a = decay.alpha_for(hl)
        s1 = decay.decayed_before(minutes, g, played, a)
        s0 = decay.decayed_count_before(g, played, a)
        s2 = decay.decayed_before(minutes ** 2, g, played, a)
        mean = np.where(s0 > 0, s1 / np.where(s0 > 0, s0, 1), np.nan)
        out[f'm_min_h{hl}'] = mean
        out[f'm_games_eff_h{hl}'] = s0
        if hl == 15:
            var = np.where(s0 > 0, s2 / np.where(s0 > 0, s0, 1) - mean ** 2, np.nan)
            out['m_min_sd_h15'] = np.sqrt(np.clip(var, 0, None))
    # ---- availability and role over ALL listed rows (DNP rows count): who actually takes the floor, and who starts
    for hl in (5, 15):
        a = decay.alpha_for(hl)
        n_all = decay.decayed_count_before(g, all_rows, a)
        out[f'role_p_played_h{hl}'] = np.where(n_all > 0, decay.decayed_before(h['lbl_played'].to_numpy(float), g, all_rows, a)
                                               / np.where(n_all > 0, n_all, 1), np.nan)
        out[f'role_start_share_h{hl}'] = np.where(n_all > 0, decay.decayed_before(h['lbl_started'].to_numpy(float), g, all_rows, a)
                                                  / np.where(n_all > 0, n_all, 1), np.nan)
    # ---- blowout sensitivity: minutes in blowouts vs close games (a player's own history), shrunk toward his overall mean
    a = decay.alpha_for(40)
    blow = (h['lbl_team_margin'].abs() >= HYPER['blowout_margin']).to_numpy() & played
    close = played & ~blow
    kb = HYPER['k_blowout_games']
    overall = out['m_min_h40'].to_numpy()
    for name, sel in (('blowout', blow), ('close', close)):
        s1 = decay.decayed_before(np.where(sel, minutes, 0.0), g, played, a)
        s0 = decay.decayed_before(sel.astype(float), g, played, a)
        out[f'm_min_{name}_h40'] = (s1 + kb * overall) / (s0 + kb)
    out['m_blowout_ratio_h40'] = out['m_min_blowout_h40'] / out['m_min_close_h40']

    # ---- rest, returns and season workload
    d = pd.to_datetime(h['game_date'])
    dp = d.where(played)
    last_played = dp.groupby(h['player_id']).ffill().groupby(h['player_id']).shift(1)       # date of the previous PLAYED game
    gap = (d - last_played).dt.days
    out['rest_days_since_played'] = gap
    same_season = h['season'].eq(h.groupby('player_id')['season'].shift(1))
    returning = (gap >= 10) & same_season
    ret_flag = returning.astype(int)
    run = ret_flag.groupby(h['player_id']).cumsum()                                          # index of the current absence episode
    played_i = pd.Series(played.astype(int), index=h.index)
    since = played_i.groupby([h['player_id'], run]).cumsum() - played_i                         # played games since the return game (excl. this row)
    out['rest_returning'] = ret_flag
    out['rest_games_since_return'] = np.where(run > 0, since, np.nan)
    out['rest_games_played_season'] = (played_i.groupby([h['player_id'], h['season']]).cumsum() - played_i).to_numpy()

    # ---- per-minute rates and conversion percentages, with league priors from EARLIER seasons
    pos = h['player_id'].map(pos_by_player).fillna('F').to_numpy()
    bucket = _bucket(out['m_min_h15'])
    keys_df = pd.DataFrame({'season': h['season'], 'pos': pos, 'bucket': bucket})
    stats = {s: h[s].to_numpy(dtype=float) for s in RATE_STATS if s in h.columns}
    usage = h['usage_pct'].to_numpy(dtype=float) * minutes if 'usage_pct' in h.columns else None
    if usage is not None:
        stats['usage_x_min'] = usage
    priors = _league_priors(h, keys_df, stats, minutes, played)
    k = HYPER['k_minutes']
    for hl in HALF_LIVES:
        a = decay.alpha_for(hl)
        s_min = decay.decayed_before(minutes, g, played, a)
        for s, vals in stats.items():
            s_stat = decay.decayed_before(vals, g, played, a)
            pr = priors.lookup(s, keys_df)
            name = 'r_usage_pct' if s == 'usage_x_min' else f'r_{s}'
            out[f'{name}_h{hl}'] = (s_stat + k * pr) / (s_min + k)
    # shooting conversion: decayed made / decayed attempts, shrunk toward the league percentage for the cell
    ka = HYPER['k_attempts']
    for name, (made, att) in PCT_STATS.items():
        for hl in (15, 40):
            a = decay.alpha_for(hl)
            s_m = decay.decayed_before(stats[made], g, played, a)
            s_a = decay.decayed_before(stats[att], g, played, a)
            pm, pa = priors.lookup(made, keys_df), priors.lookup(att, keys_df)
            league_pct = np.where(pa > 0, pm / np.where(pa > 0, pa, 1), np.nan)
            out[f'c_{name}_h{hl}'] = (s_m + ka * league_pct) / (s_a + ka)
    out['pos_group'] = pos
    out['m_bucket'] = bucket
    return out[out['season'] >= first_dataset_season].reset_index(drop=True)


class _Priors:
    """League per-minute rates by (season, position group, minutes bucket) from EARLIER seasons only; coarser levels fill thin cells."""

    def __init__(self, tables):
        self.tables = tables                    # stat -> {'cell': Series, 'bucket': Series, 'all': Series}

    def lookup(self, stat, keys_df):
        t = self.tables[stat]
        r = t['cell'].reindex(pd.MultiIndex.from_frame(keys_df[['season', 'pos', 'bucket']])).to_numpy()
        miss = np.isnan(r)
        if miss.any():                          # same season and bucket, any position
            r = np.where(miss, t['bucket'].reindex(pd.MultiIndex.from_frame(keys_df[['season', 'bucket']])).to_numpy(), r)
            miss = np.isnan(r)
            if miss.any():                      # same season, any bucket
                r = np.where(miss, t['all'].reindex(keys_df['season']).to_numpy(), r)
        return r


def _expanding_prior(sums):
    """sums: DataFrame (num, den) indexed (season, *cell). Rate over all STRICTLY EARLIER seasons of the same cell."""
    sums = sums.sort_index()
    cell_levels = list(range(1, sums.index.nlevels))
    cum = sums.groupby(level=cell_levels)[['num', 'den']].cumsum() if cell_levels else sums[['num', 'den']].cumsum()
    before = cum - sums[['num', 'den']]                       # excludes the row's own season
    return before['num'] / before['den'].where(before['den'] > 0)


def _league_priors(h, keys_df, stats, minutes, played):
    tables = {}
    den = np.where(played, np.nan_to_num(minutes), 0.0)
    for s, vals in stats.items():
        d = keys_df.assign(num=np.where(played, np.nan_to_num(vals), 0.0), den=den)
        tables[s] = {
            'cell': _expanding_prior(d.groupby(['season', 'pos', 'bucket'])[['num', 'den']].sum()),
            'bucket': _expanding_prior(d.groupby(['season', 'bucket'])[['num', 'den']].sum()),
            'all': _expanding_prior(d.groupby('season')[['num', 'den']].sum())}
    return _Priors(tables)
