"""Checks 1b/3: independent re-implementation of roster_bpm / roster_bpm_ewma from player_basic_stats (own code: explicit per-player
recursions, own position map, own 365-day trailing league mean). Compared with the 2026-11-01 training set for ALL games of the 392-game
stratified sample (seed 42) and, as a stronger test, for every game of the 2025-26 season. Also a 'poison' test: re-run the recomputation
after overwriting every player-game dated on/after a cut date with garbage and confirm values before the cut do not change
(proves no same-day / future rows enter)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *   # noqa

STATS = ['minutes', 'fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersMade', 'threePointersAttempted', 'freeThrowsMade',
         'freeThrowsAttempted', 'offensiveRebounds', 'defensiveRebounds', 'assists', 'steals', 'blocks', 'turnovers',
         'personalFouls', 'points']


def load_players():
    with conn() as c:
        ps = pd.read_sql(f"select gameId,teamId,personId,gameDate,seasonYear,{','.join(STATS)} from player_basic_stats", c)
        pi = pd.read_sql('select PERSON_ID personId, POSITION from player_info', c)
    ps['gameDate'] = pd.to_datetime(ps['gameDate'])
    ps['gameId'] = ps['gameId'].astype(str).str.zfill(10)
    ps = ps[ps['seasonYear'] >= '2013-14']
    return ps.merge(pi, on='personId', how='left')


def pos_scalar(p):
    if not isinstance(p, str):
        return 3.0
    u = p.upper()
    if 'GUARD' in u and 'FORWARD' in u: return 2.5
    if 'FORWARD' in u and 'CENTER' in u: return 4.5
    if 'GUARD' in u: return 1.5
    if 'FORWARD' in u: return 3.5
    if 'CENTER' in u: return 5.0
    return {'PG': 1.0, 'SG': 2.0, 'SF': 3.0, 'PF': 4.0, 'C': 5.0, 'G': 1.5, 'F': 3.5}.get(u, 3.0)


def lerp(a, b, s):
    return a + (b - a) / 4.0 * (s - 1.0)


def player_game_bpm(ps):
    """Per player-game pre-game BPM from that player's own PRIOR games only."""
    ps = ps.sort_values(['personId', 'gameDate', 'gameId']).reset_index(drop=True)
    ps[STATS] = ps[STATS].fillna(0)
    X = ps[STATS].to_numpy(float)
    pid = ps['personId'].to_numpy()
    n = len(ps)
    alpha = 2.0 / 41.0
    E = np.full_like(X, np.nan)
    cum = np.zeros(n)
    start = 0
    bounds = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        cur = None
        csum = 0.0
        for i in range(a, b):
            if cur is not None:
                E[i] = cur
            cum[i] = csum
            x = X[i]
            cur = x.copy() if cur is None else (1 - alpha) * cur + alpha * x
            csum += x[0]
    e = {s: E[:, k] for k, s in enumerate(STATS)}
    poss = e['minutes'] * (100.0 / 48.0)
    poss = np.where(poss == 0, 1.0, poss)
    p = {s: e[s] / poss * 100.0 for s in STATS}
    rel = np.clip(cum / 200.0, 0, 1)
    ps_pos = ps['POSITION'].map({v: pos_scalar(v) for v in ps['POSITION'].unique()}).astype(float).to_numpy()
    role = np.clip(5.0 - 0.4 * p['assists'], 1.0, 5.0)
    raw = (p['points'] * 0.860 + p['threePointersMade'] * 0.389 + p['assists'] * lerp(0.580, 1.034, ps_pos)
           + p['turnovers'] * -0.964 + p['offensiveRebounds'] * lerp(0.613, 0.181, ps_pos)
           + p['defensiveRebounds'] * lerp(0.116, 0.181, ps_pos) + p['steals'] * lerp(1.369, 1.008, ps_pos)
           + p['blocks'] * lerp(1.327, 0.703, ps_pos) + p['personalFouls'] * -0.367
           + p['fieldGoalsAttempted'] * lerp(-0.560, -0.780, role)
           + (p['freeThrowsAttempted'] * 0.44) * lerp(-0.246, -0.343, role))
    raw = raw * rel
    ps['raw'] = raw
    ps['e_min'] = e['minutes']
    # league mean of raw over player-games in [d-365d, d), computed by day sums with searchsorted
    ok = ~np.isnan(raw)
    day = ps.loc[ok].groupby('gameDate')['raw'].agg(['sum', 'count']).sort_index()
    days = day.index.to_numpy()
    cs, cc = np.r_[0, np.cumsum(day['sum'].to_numpy())], np.r_[0, np.cumsum(day['count'].to_numpy())]
    d_all = ps['gameDate'].to_numpy()
    hi = np.searchsorted(days, d_all, side='left')                      # days strictly before d
    lo = np.searchsorted(days, d_all - np.timedelta64(365, 'D'), side='left')   # days >= d-365
    cnt = cc[hi] - cc[lo]
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(cnt > 0, (cs[hi] - cs[lo]) / cnt, 0.0)
    ps['bpm'] = ps['raw'] - mean
    return ps


def team_game_roster(ps):
    ps = ps.copy()
    ps['wv'] = ps['bpm'] * ps['e_min']
    g = ps.groupby(['gameId', 'teamId', 'gameDate']).agg(wv=('wv', 'sum'), em=('e_min', 'sum')).reset_index()
    g['cur'] = g['wv'] / g['em'].replace(0, 1)
    g = g.sort_values(['teamId', 'gameDate', 'gameId']).reset_index(drop=True)
    out = []
    for tid, h in g.groupby('teamId'):
        cur = h['cur'].to_numpy()
        prev = np.r_[np.nan, cur[:-1]]
        ew = np.full(len(cur), np.nan)
        a = 2.0 / 11.0
        state = None
        for i in range(len(cur)):
            if state is not None:
                ew[i] = state
            if not np.isnan(cur[i]):
                state = cur[i] if state is None else (1 - a) * state + a * cur[i]
        h = h.assign(roster_bpm=np.where(np.isnan(prev), 0.0, prev), roster_bpm_ewma=np.where(np.isnan(ew), 0.0, ew))
        out.append(h[['gameId', 'teamId', 'gameDate', 'roster_bpm', 'roster_bpm_ewma', 'cur']])
    return pd.concat(out)


def compare(tr, side_frame, label):
    m = tr.merge(side_frame, on='gameId', how='left')
    rows = []
    for side in ('home', 'away'):
        for f in ('roster_bpm', 'roster_bpm_ewma'):
            r = tr[['gameId', f'teamId_{side}', f'{f}_{side}']].merge(
                side_frame.rename(columns={'teamId': f'teamId_{side}', f: f + '_mine'})[['gameId', f'teamId_{side}', f + '_mine']],
                on=['gameId', f'teamId_{side}'], how='left')
            d = (r[f'{f}_{side}'] - r[f + '_mine']).abs()
            rows.append({'set': label, 'feature': f'{f}_{side}', 'n': len(r), 'missing_mine': int(r[f + '_mine'].isna().sum()),
                         'max_absdiff': float(d.max()), 'p99': float(d.quantile(.99)), 'median': float(d.median()),
                         'n_gt_1e-6': int((d > 1e-6).sum())})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    t = load_train()
    ps = load_players()
    pg = player_game_bpm(ps)
    roster = team_game_roster(pg)
    print('team-games from player table:', len(roster))
    rng = np.random.default_rng(SEED)
    samp = pd.concat([g.iloc[rng.choice(len(g), min(28, len(g)), replace=False)] for _, g in t.groupby('seasonYear')])
    res = pd.concat([compare(samp, roster, 'stratified_392'), compare(t[t.seasonYear == '2025-26'], roster, '2025-26_all'),
                     compare(t, roster, 'all_games')])
    res.to_csv(OUT / 'check3_bpm_recompute.csv', index=False)
    print(res.to_string())

    # --- poison test: corrupt everything on/after the cut date; values for games strictly before the cut must not change
    cut = pd.Timestamp('2025-01-15')
    ps2 = ps.copy()
    late = ps2['gameDate'] >= cut
    rng2 = np.random.default_rng(1)
    for s in STATS:
        ps2.loc[late, s] = rng2.uniform(0, 50, late.sum())
    r2 = team_game_roster(player_game_bpm(ps2))
    a = roster.merge(r2, on=['gameId', 'teamId'], suffixes=('', '_p'))
    before = a[a['gameDate'] < cut]
    on_cut = a[a['gameDate'] == cut]
    print('poison test: games before cut', len(before), 'max |roster_bpm diff|', (before['roster_bpm'] - before['roster_bpm_p']).abs().max(),
          'max |ewma diff|', (before['roster_bpm_ewma'] - before['roster_bpm_ewma_p']).abs().max())
    print('games ON the cut date (their own box score is poisoned, prior games are not): max |roster_bpm diff|',
          (on_cut['roster_bpm'] - on_cut['roster_bpm_p']).abs().max(), ' max |ewma diff|',
          (on_cut['roster_bpm_ewma'] - on_cut['roster_bpm_ewma_p']).abs().max())
    after = a[a['gameDate'] > cut]
    print('games after the cut (should differ): median |diff|', (after['roster_bpm'] - after['roster_bpm_p']).abs().median())

    # --- first game of a season / first-ever game, traded players
    first = roster.sort_values(['teamId', 'gameDate']).copy()
    first['season'] = first['gameId'].map(ps.drop_duplicates('gameId').set_index('gameId')['seasonYear'])
    first['k'] = first.groupby(['teamId', 'season']).cumcount()
    # simpler: how many first-of-season rows have roster_bpm == 0 (no history) in the dataset
    z = t.merge(first[first.k == 0][['gameId', 'teamId']], left_on=['gameId', 'teamId_home'], right_on=['gameId', 'teamId'])
    print('first-of-season home rows:', len(z), ' with roster_bpm_home==0:', int((z['roster_bpm_home'] == 0).sum()),
          ' (only the first-ever game of 2013-14 should be 0)')
