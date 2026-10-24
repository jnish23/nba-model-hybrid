"""E28 / E29 - running-load, speed, physical-previous-game and schedule-density features built from tracking data.

Definitions are registered in reports/experiments/PREREGISTRATION.md ("E28") before any relationship with an outcome was
computed. Every rolling statistic uses only a team's (or player's) PRIOR games; the only forward-looking input is the
published schedule (`trip_last_game` reads where the team's NEXT game is played). tests/team/test_tracking_features.py
corrupts everything from game t onward and checks the features at t.

Naming follows the existing convention: trk_<name>_home / trk_<name>_away / diff_trk_<name> (home minus away).
Tracking rows that are all zero (distance 0 and touches 0, 350 team-games in 2025-26) are read as MISSING, never as low load.
Lagged load features use "the most recent game with tracking data" (at most 3 games back), see audit.md.
"""
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SPAN = 20            # EWMA span of the "norm"
SD_WIN = 20          # rolling window of the "sd"
MIN_NORM = 5         # prior valid games before a norm exists
MIN_SD = 8           # prior valid games before an sd exists
SD_FLOOR_FRAC = 0.01  # sd floor, as a fraction of |norm|
MAX_STALE = 3        # a "last game with data" feature may be at most this many games old
MIN_PLAYER_MIN = 5.0   # minutes for a player-game to enter a norm
TEAM_DPM_RANGE = (0.05, 0.095)   # plausible team miles per minute (data: 0.0366 at the 0.05th percentile; the 25 rows under 12 miles are broken)
PLAYER_SPEED_RANGE = (2.5, 6.5)  # mph
PLAYER_IMPLIED_SPEED_RANGE = (1.5, 8.0)   # distance / hours played, mph
ROTATION_MIN = 10.0
MIN_ROT_PLAYERS = 4
HIGH_ALTITUDE_FT = 4000.0
CONTINUITY_DAYS = 7  # a trip / "next game" / "previous game" link must not cross a gap longer than this
MIN_TIRED_HISTORY = 200
Z_CLIP = 6.0         # z-scores are clipped to +-Z_CLIP (a handful of broken records otherwise give |z| > 12)

TEAM_FEATURES = [
    # F1 excess running load
    'dist_per_min_last', 'dist_per_poss_last', 'xload_last', 'xload_4n', 'rot_xload_last',
    # F2 speed drop
    'rot_speed_z_last', 'rot_speed_trend5', 'star_speed_z_last',
    # F3 physical previous game
    'prev_ot', 'prev_close', 'prev_n40', 'prev_hustle_z', 'prev_opp_pace',
    # F4 schedule density
    'g3in4', 'g4in6', 'games_last7', 'trip_last_game', 'homecoming', 'prev_altitude',
]
FAMILIES = {
    'F1': ['dist_per_min_last', 'dist_per_poss_last', 'xload_last', 'xload_4n', 'rot_xload_last'],
    'F2': ['rot_speed_z_last', 'rot_speed_trend5', 'star_speed_z_last'],
    'F3': ['prev_ot', 'prev_close', 'prev_n40', 'prev_hustle_z', 'prev_opp_pace'],
    'F4': ['g3in4', 'g4in6', 'games_last7', 'trip_last_game', 'homecoming', 'prev_altitude'],
    'F5': ['xload_4n_x_b2b', 'xload_last_x_dist', 'one_side_tired'],
}
# per-team F5 pieces (computed after the home/away merge)
F5_TEAM = ['xload_4n_x_b2b', 'xload_last_x_dist']
HUSTLE_COLS = ['contestedShots', 'deflections', 'looseBallsRecoveredTotal', 'boxOuts']


def family_columns(family):
    """Column names (home, away, diff; the game-level one_side_tired alone) that belong to a family."""
    out = []
    for n in FAMILIES[family]:
        if n == 'one_side_tired':
            out.append('trk_one_side_tired')
        else:
            out += [f'trk_{n}_home', f'trk_{n}_away', f'diff_trk_{n}']
    return out


def parse_minutes(s):
    """'MM:SS' -> float minutes (vectorised); blank / None -> NaN."""
    s = pd.Series(s, copy=False).astype('string')
    parts = s.str.extract(r'^\s*(\d+):(\d{1,2})')
    out = pd.to_numeric(parts[0], errors='coerce') + pd.to_numeric(parts[1], errors='coerce') / 60.0
    alt = pd.to_numeric(s.where(parts[0].isna()), errors='coerce')
    return out.fillna(alt).astype(float).to_numpy()


def prior_norm_sd(df, key, col):
    """Norm (EWMA span SPAN) and sd (rolling SD_WIN) of `col` over each key's PRIOR rows with a non-NaN value.
    df must be sorted by (key, date); returns two Series on df.index, NaN until MIN_NORM / MIN_SD prior values exist."""
    v = df[[key, col]].dropna(subset=[col])
    g = v.groupby(key, sort=False)[col]
    norm = g.transform(lambda s: s.ewm(span=SPAN).mean().shift(1))
    sd = g.transform(lambda s: s.rolling(SD_WIN, min_periods=MIN_SD).std().shift(1))
    prior_n = g.cumcount()
    norm = norm.where(prior_n >= MIN_NORM)
    sd = np.maximum(sd, SD_FLOOR_FRAC * norm.abs())
    return norm.reindex(df.index), sd.reindex(df.index)


def zscore(df, key, col):
    norm, sd = prior_norm_sd(df, key, col)
    return ((df[col] - norm) / sd).where(sd > 0).clip(-Z_CLIP, Z_CLIP)


def last_with_data(df, key, col):
    """Value of `col` at the team's most recent PRIOR game that has one, at most MAX_STALE games back."""
    ff = df.groupby(key, sort=False)[col].transform(lambda s: s.ffill(limit=MAX_STALE))
    return ff.groupby(df[key], sort=False).shift(1)


# ------------------------------------------------------------------ input loading
def load_inputs(db_path):
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as c:
        d = {
            'tgf': pd.read_sql('SELECT seasonYear, teamId, teamTricode, gameId, gameDate, MATCHUP, minutes, plusMinus, pace FROM team_game_full', c),
            'tt': pd.read_sql('SELECT gameId, teamId, distance, touches FROM team_track_stats_v3', c),
            'pt': pd.read_sql('SELECT gameId, teamId, personId, minutes, speed, distance FROM player_track_stats_v3', c),
            'th': pd.read_sql('SELECT gameId, teamId, ' + ', '.join(HUSTLE_COLS) + ' FROM team_hustle_stats', c),
            'arenas': pd.read_sql('SELECT teamId, altitude FROM nba_arenas', c),
        }
    return d


def _norm_ids(d):
    d = {k: v.copy() for k, v in d.items()}
    for k in ('tgf', 'tt', 'pt', 'th'):
        d[k]['gameId'] = d[k]['gameId'].astype(str).str.zfill(10)
    d['tgf']['gameDate'] = pd.to_datetime(d['tgf']['gameDate'])
    return d


# ------------------------------------------------------------------ team-game base table
def team_base(d):
    tg = d['tgf'][['teamId', 'gameId', 'gameDate', 'MATCHUP', 'minutes', 'plusMinus', 'pace']].copy()
    tg = tg.sort_values(['teamId', 'gameDate', 'gameId'], kind='stable').reset_index(drop=True)
    tg['is_home'] = tg['MATCHUP'].astype(str).str.contains('vs.', regex=False)
    opp = tg[['gameId', 'teamId']].rename(columns={'teamId': 'oppId'})
    tg = tg.merge(opp, on='gameId', how='left')
    tg = tg[tg['teamId'] != tg['oppId']].drop_duplicates(['gameId', 'teamId']).sort_values(['teamId', 'gameDate', 'gameId'], kind='stable').reset_index(drop=True)
    alt = d['arenas'].set_index('teamId')['altitude']
    tg['venue_alt'] = np.where(tg['is_home'], tg['teamId'].map(alt), tg['oppId'].map(alt))
    tm = tg['minutes'].astype(float)
    tg['tm'] = tm
    tg['poss'] = tg['pace'] * tm / 240.0
    tt = d['tt'][['gameId', 'teamId', 'distance', 'touches']]
    tg = tg.merge(tt, on=['gameId', 'teamId'], how='left')
    valid = (tg['distance'] > 0) & (tg['touches'] > 0)
    dpm = tg['distance'] / tg['tm']
    valid &= dpm.between(*TEAM_DPM_RANGE)
    tg['dist'] = tg['distance'].where(valid)
    tg['dist_per_min'] = tg['dist'] / tg['tm']
    tg['dist_per_poss'] = tg['dist'] / tg['poss']
    th = d['th'][['gameId', 'teamId'] + HUSTLE_COLS]
    tg = tg.merge(th, on=['gameId', 'teamId'], how='left')
    tg['hustle'] = tg[HUSTLE_COLS].sum(axis=1, min_count=len(HUSTLE_COLS))
    return tg.drop(columns=HUSTLE_COLS + ['distance', 'touches'])


def player_aggregates(d, tg):
    """Per team-game: minutes-weighted z(distance per minute) and z(speed) over the rotation, the same for the top-2 by
    minutes, and the number of players at 40+ minutes. Player norms use that player's own prior valid games only."""
    pt = d['pt'][['gameId', 'teamId', 'personId', 'minutes', 'speed', 'distance']].copy()
    pt['pm'] = parse_minutes(pt['minutes'])
    pt = pt.merge(tg[['gameId', 'teamId', 'gameDate']], on=['gameId', 'teamId'], how='inner')
    pt = pt.sort_values(['personId', 'gameDate', 'gameId'], kind='stable').reset_index(drop=True)
    implied = pt['distance'] / pt['pm'] * 60.0
    ok = (pt['pm'] >= MIN_PLAYER_MIN) & pt['speed'].between(*PLAYER_SPEED_RANGE) & implied.between(*PLAYER_IMPLIED_SPEED_RANGE)
    pt['dpm'] = (pt['distance'] / pt['pm']).where(ok)
    pt['spd'] = pt['speed'].where(ok)
    pt['z_dpm'] = zscore(pt, 'personId', 'dpm')
    pt['z_spd'] = zscore(pt, 'personId', 'spd')
    out = []
    for (gid, tid), g in pt.groupby(['gameId', 'teamId'], sort=False):
        row = {'gameId': gid, 'teamId': tid, 'n40': float((g['pm'] >= 40).sum()) if g['pm'].notna().any() else np.nan}
        rot = g[(g['pm'] >= ROTATION_MIN)]
        for name, col in (('rot_xload', 'z_dpm'), ('rot_speed_z', 'z_spd')):
            r = rot.dropna(subset=[col])
            row[name] = float(np.average(r[col], weights=r['pm'])) if len(r) >= MIN_ROT_PLAYERS else np.nan
        v = g.dropna(subset=['z_spd']).nlargest(2, 'pm')
        row['star_speed_z'] = float(np.average(v['z_spd'], weights=v['pm'])) if len(v) == 2 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def _window_sum(dates, vals, lo_days, hi_days):
    """For each i: sum / count / count-valid of vals[j] over j with date_j in [date_i - lo_days, date_i - hi_days]."""
    n = len(dates)
    dn = dates.astype('datetime64[D]').astype(np.int64)
    s, cnt, cv = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        lo = np.searchsorted(dn, dn[i] - lo_days, side='left')
        hi = np.searchsorted(dn, dn[i] - hi_days, side='right')
        w = vals[lo:hi]
        cnt[i] = len(w)
        m = ~np.isnan(w)
        cv[i] = m.sum()
        s[i] = w[m].sum()
    return s, cnt, cv


def _slope_last5(x):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - 5):i]
        idx = np.arange(len(w), dtype=float)
        m = ~np.isnan(w)
        if m.sum() >= 3:
            out[i] = np.polyfit(idx[m], w[m], 1)[0]
    return out


def build_team_features(d):
    """One row per (gameId, teamId) with trk_* team features. `d`: dict of tgf / tt / pt / th / arenas frames."""
    d = _norm_ids(d)
    tg = team_base(d)
    k = 'teamId'
    tg['xload'] = zscore(tg, k, 'dist_per_poss')
    tg['hustle_z'] = zscore(tg, k, 'hustle')
    pa = player_aggregates(d, tg)
    tg = tg.merge(pa, on=['gameId', 'teamId'], how='left')
    tg = tg.sort_values(['teamId', 'gameDate', 'gameId'], kind='stable').reset_index(drop=True)
    g = tg.groupby(k, sort=False)
    f = pd.DataFrame({'gameId': tg['gameId'], 'teamId': tg['teamId']})
    # --- F1
    f['trk_dist_per_min_last'] = last_with_data(tg, k, 'dist_per_min')
    f['trk_dist_per_poss_last'] = last_with_data(tg, k, 'dist_per_poss')
    f['trk_xload_last'] = last_with_data(tg, k, 'xload')
    f['trk_rot_xload_last'] = last_with_data(tg, k, 'rot_xload')
    # --- F2
    f['trk_rot_speed_z_last'] = last_with_data(tg, k, 'rot_speed_z')
    f['trk_star_speed_z_last'] = last_with_data(tg, k, 'star_speed_z')
    # --- F3 (previous game itself: the box score is always there)
    f['trk_prev_ot'] = g['tm'].shift(1).gt(240).astype(float).where(g['tm'].shift(1).notna())
    f['trk_prev_close'] = g['plusMinus'].shift(1).abs().le(5).astype(float).where(g['plusMinus'].shift(1).notna())
    f['trk_prev_n40'] = g['n40'].shift(1)
    f['trk_prev_hustle_z'] = g['hustle_z'].shift(1)
    # Both teams in a game share its pace, so the previous game's raw pace is mostly the team's own season pace (and the total line: r = +0.16
    # with odds_total in E28). The feature is the previous game's pace SHOCK: that pace minus the team's own pace habit going into that game
    # (EWMA of its games before it), i.e. how much faster/slower than usual the previous opponent made the game.
    own_pace_habit = g['pace'].transform(lambda s: s.shift(2).ewm(span=10, min_periods=3).mean())
    f['trk_prev_opp_pace'] = g['pace'].shift(1) - own_pace_habit
    # --- per-team loops: xload_4n, speed trend, schedule density
    cols = {c: np.full(len(tg), np.nan) for c in ('xload_4n', 'rot_speed_trend5', 'g3in4', 'g4in6', 'games_last7', 'trip_last_game', 'homecoming', 'prev_altitude')}
    for tid, idx in tg.groupby(k, sort=False).indices.items():
        sub = tg.iloc[idx]
        dates = sub['gameDate'].to_numpy()
        dn = dates.astype('datetime64[D]').astype(np.int64)
        x = sub['xload'].to_numpy(float)
        s, cnt, cv = _window_sum(dates, x, 4, 1)
        had_valid_before = np.concatenate([[0], np.cumsum(~np.isnan(x))[:-1]]) > 0
        x4 = np.where(had_valid_before & ((cnt == 0) | (cv > 0)), s, np.nan)
        cols['xload_4n'][idx] = x4
        cols['rot_speed_trend5'][idx] = _slope_last5(sub['rot_speed_z'].to_numpy(float))
        _, c3, _ = _window_sum(dates, np.zeros(len(dn)), 3, 0)      # games in [t-3, t], incl. t
        _, c5, _ = _window_sum(dates, np.zeros(len(dn)), 5, 0)
        _, c7, _ = _window_sum(dates, np.zeros(len(dn)), 7, 1)
        cols['g3in4'][idx] = (c3 >= 3).astype(float)
        cols['g4in6'][idx] = (c5 >= 4).astype(float)
        cols['games_last7'][idx] = c7
        home = sub['is_home'].to_numpy(bool)
        alt = sub['venue_alt'].to_numpy(float)
        n = len(sub)
        trip, comeback, prev_alt = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        run = 0                                       # consecutive away games ending at the previous game
        for i in range(n):
            if i == 0:
                trip_run_prev = 0
            else:
                gap = dn[i] - dn[i - 1]
                trip_run_prev = run if gap <= CONTINUITY_DAYS else 0
                comeback[i] = float(home[i] and trip_run_prev >= 3)
                prev_alt[i] = float(gap <= CONTINUITY_DAYS and alt[i - 1] >= HIGH_ALTITUDE_FT and alt[i] < HIGH_ALTITUDE_FT)
            run = (trip_run_prev + 1) if not home[i] else 0
            if i + 1 < n:
                trip[i] = float((not home[i]) and home[i + 1] and (dn[i + 1] - dn[i]) <= CONTINUITY_DAYS)
            else:
                trip[i] = 0.0                          # no later game in the data
        cols['trip_last_game'][idx] = trip
        cols['homecoming'][idx] = comeback
        cols['prev_altitude'][idx] = prev_alt
    for c, v in cols.items():
        f[f'trk_{c}'] = v
    keep = ['gameId', 'teamId'] + [f'trk_{n}' for n in TEAM_FEATURES]
    # carried alongside for F5 (not features themselves)
    f['trk_prev_ot_raw'] = f['trk_prev_ot']
    return f[keep + ['trk_prev_ot_raw']]


# ------------------------------------------------------------------ game level
def one_side_tired(dates, home_score, away_score):
    """+1 when the home side's tired score is above the 80th percentile of all PRIOR team-games' scores and the away side's
    is below their median, -1 for the mirror image, 0 otherwise. Percentiles use only games dated strictly before the game
    (expanding, so nothing depends on a fold); NaN while fewer than MIN_TIRED_HISTORY scores exist or either score is NaN."""
    dates = pd.to_datetime(pd.Series(np.asarray(dates)))
    h, a = np.asarray(home_score, float), np.asarray(away_score, float)
    out = np.full(len(h), np.nan)
    order = np.argsort(dates.to_numpy(), kind='stable')
    pool = []
    for day in pd.unique(dates.to_numpy()[order]):
        rows = np.where(dates.to_numpy() == day)[0]
        if len(pool) >= MIN_TIRED_HISTORY:
            arr = np.asarray(pool)
            p80, p50 = np.percentile(arr, 80), np.percentile(arr, 50)
            for i in rows:
                if np.isnan(h[i]) or np.isnan(a[i]):
                    continue
                out[i] = float(h[i] > p80 and a[i] < p50) - float(a[i] > p80 and h[i] < p50)
        pool.extend([v for v in np.concatenate([h[rows], a[rows]]) if not np.isnan(v)])
    return out


def add_to_games(games, feats):
    """games: one row per game (gameId, teamId_home, teamId_away, gameDate, is_b2b_*, dist_miles_*). Adds the trk_* columns."""
    out = games.copy()
    gid = out['gameId'].astype(str).str.zfill(10)
    fcols = [c for c in feats.columns if c.startswith('trk_')]
    f = feats.copy()
    f['_gid'] = f['gameId'].astype(str).str.zfill(10)
    tmp = pd.DataFrame({'_gid': gid})
    for side in ('home', 'away'):
        sf = f[['_gid', 'teamId'] + fcols].rename(columns={'teamId': f'teamId_{side}', **{c: f'{c}_{side}' for c in fcols}})
        part = pd.DataFrame({'_gid': gid.to_numpy(), f'teamId_{side}': out[f'teamId_{side}'].astype('int64').to_numpy()})
        part = part.merge(sf, on=['_gid', f'teamId_{side}'], how='left')
        for c in fcols:
            tmp[f'{c}_{side}'] = part[f'{c}_{side}'].to_numpy()
    new = {}
    names = [c for c in fcols if c != 'trk_prev_ot_raw']
    for c in names:
        new[f'{c}_home'], new[f'{c}_away'] = tmp[f'{c}_home'].to_numpy(), tmp[f'{c}_away'].to_numpy()
        new[f'diff_{c}'] = tmp[f'{c}_home'].to_numpy() - tmp[f'{c}_away'].to_numpy()
    # F5: interactions and asymmetry (use the existing schedule columns)
    for side in ('home', 'away'):
        b2b = out[f'is_b2b_{side}'].to_numpy(float)
        dist = out[f'dist_miles_{side}'].to_numpy(float) / 1000.0
        new[f'trk_xload_4n_x_b2b_{side}'] = tmp[f'trk_xload_4n_{side}'].to_numpy() * b2b
        new[f'trk_xload_last_x_dist_{side}'] = tmp[f'trk_xload_last_{side}'].to_numpy() * dist
        new[f'_tired_{side}'] = tmp[f'trk_xload_4n_{side}'].to_numpy() + tmp[f'trk_prev_ot_raw_{side}'].to_numpy() + b2b
    for n in F5_TEAM:
        new[f'diff_trk_{n}'] = new[f'trk_{n}_home'] - new[f'trk_{n}_away']
    new['trk_one_side_tired'] = one_side_tired(out['gameDate'], new['_tired_home'], new['_tired_away'])
    new.pop('_tired_home'), new.pop('_tired_away')
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def build_variant(csv_in, csv_out, db_path):
    """Append the tracking / fatigue columns to a copy of the training set (existing columns and row order untouched)."""
    feats = build_team_features(load_inputs(db_path))
    games = pd.read_csv(csv_in, low_memory=False)
    games['teamId_home'] = games['teamId_home'].astype('int64')
    games['teamId_away'] = games['teamId_away'].astype('int64')
    out = add_to_games(games, feats)
    out.to_csv(csv_out, index=False)
    return out


def mechanism_targets(db_path):
    """Outcomes of game t used as mechanism checks in the screen (NEVER features): team 3P%, FT%, rim FG%, pace, points per
    possession. One row per (gameId, teamId)."""
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as c:
        t = pd.read_sql('SELECT teamId, gameId, minutes, pace, points, threePointersMade, threePointersAttempted, freeThrowsMade, freeThrowsAttempted, '
                        '"Restricted Area FGM" AS raf_m, "Restricted Area FGA" AS raf_a FROM team_game_full', c)
    t['gameId'] = t['gameId'].astype(str).str.zfill(10)
    poss = t['pace'] * t['minutes'] / 240.0
    return pd.DataFrame({'gameId': t['gameId'], 'teamId': t['teamId'],
                         'mech_3p_pct': t['threePointersMade'] / t['threePointersAttempted'].replace(0, np.nan),
                         'mech_ft_pct': t['freeThrowsMade'] / t['freeThrowsAttempted'].replace(0, np.nan),
                         'mech_rim_pct': t['raf_m'] / t['raf_a'].replace(0, np.nan),
                         'mech_pace': t['pace'], 'mech_ppp': t['points'] / poss})
