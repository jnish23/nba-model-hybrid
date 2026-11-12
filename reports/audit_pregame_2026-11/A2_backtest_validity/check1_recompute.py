"""Check 1a: recompute team-form features for a stratified random sample of games (seed 42) straight from team_game_full using
ONLY rows dated strictly before the game, and compare with the 2026-11-01 training set. Own code (explicit recursions, no pandas ewm)."""
import re
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *   # noqa

t = load_train()
cols = list(t.columns)
with conn() as c:
    tg = pd.read_sql('select * from team_game_full', c)
tg['gameDate'] = pd.to_datetime(tg['gameDate'])
tg = tg.sort_values(['teamId', 'gameDate', 'gameId']).reset_index(drop=True)
tg['gameId'] = tg['gameId'].astype(str).str.zfill(10)
assert not tg.duplicated(['teamId', 'gameDate']).any()

# EWMA feature names present in the training set
ew = {}
for col in cols:
    m = re.match(r'^(off|def)_(.+)_(5|20)g_home$', col)
    if m:
        ew[col[:-5]] = (m.group(1), m.group(2), int(m.group(3)))
print('EWMA features (per side):', len(ew))
missing_src = [k for k, (p, s, sp) in ew.items() if s not in tg.columns]
print('source columns missing from team_game_full:', missing_src)


def ewma_before(x, span):
    """EWMA with pandas adjust=True semantics (weights (1-a)^k), over x (oldest first), ignoring NaN; recursion, not pandas."""
    a = 2.0 / (span + 1.0)
    w = 1.0 - a
    num = den = 0.0
    for v in x:
        num *= w
        den *= w
        if not np.isnan(v):
            num += v
            den += 1.0
    return num / den if den > 0 else np.nan


rng = np.random.default_rng(SEED)
per_season = 28
samp = []
for s, g in t.groupby('seasonYear'):
    n = min(per_season, len(g))
    samp.append(g.iloc[rng.choice(len(g), n, replace=False)])
samp = pd.concat(samp)
print('sample games:', len(samp), 'seasons:', samp.seasonYear.nunique())

by_team = {tid: g for tid, g in tg.groupby('teamId')}
rows = []      # one row per (game, side, feature)
for _, r in samp.iterrows():
    for side in ('home', 'away'):
        tid = int(r[f'teamId_{side}'])
        g = by_team[tid]
        hist = g[g['gameDate'] < r['gameDate']]           # STRICTLY before the game date
        assert (g[g['gameId'] == r['gameId']].shape[0]) == 1
        rec = {}
        for name, (pre, stat, span) in ew.items():
            rec[name] = ewma_before(hist[stat].to_numpy(float), span)
        # trends
        for name in list(ew):
            if name.endswith('_5g'):
                base = name[:-3]
                if f'{base}_20g' in rec:
                    rec[f'{base}_trend'] = rec[name] - rec[f'{base}_20g']
        # rest and schedule load
        d = r['gameDate']
        if len(hist):
            rec['days_rest'] = float((d - hist['gameDate'].max()).days)
        else:
            rec['days_rest'] = 7.0
        rec['is_b2b'] = float(rec['days_rest'] == 1)
        rec['games_last_7d'] = float(((hist['gameDate'] >= d - pd.Timedelta(days=7))).sum())
        # season game number (all games, incl. playoffs, of that seasonYear before this one, +1)
        rec['season_game_number'] = float((hist['seasonYear'] == r['seasonYear']).sum() + 1)
        # momentum: wins in the previous games
        win = (hist['WL'] == 'W').astype(int).to_numpy()
        last10 = win[-10:]
        rec['win_pct_last10'] = float(last10.mean()) if len(win) >= 3 else np.nan
        if len(win):
            run = 1
            while run < len(win) and win[-1 - run] == win[-1]:
                run += 1
            rec['win_streak'] = float(run if win[-1] == 1 else -run)
        else:
            rec['win_streak'] = 0.0
        ishome = hist['MATCHUP'].str.contains('vs.', regex=False).astype(int).to_numpy()
        hw = ishome * win
        l20h, l20w = ishome[-20:], hw[-20:]
        rec['home_win_pct_20g'] = float(l20w.sum() / l20h.sum()) if (len(win) >= 5 and l20h.sum() > 0) else np.nan
        rec['is_playoff'] = float(str(r['gameId'])[:3] == '004')
        for k, v in rec.items():
            col = f'{k}_{side}'
            if col in t.columns:
                rows.append({'gameId': r['gameId'], 'season': r['seasonYear'], 'side': side, 'feature': k,
                             'recomputed': v, 'dataset': float(r[col])})
res = pd.DataFrame(rows)
res['absdiff'] = (res['recomputed'] - res['dataset']).abs()
res['nan_mismatch'] = res['recomputed'].isna() != res['dataset'].isna()
res.loc[res['recomputed'].isna() & res['dataset'].isna(), 'absdiff'] = 0.0
res.to_csv(OUT / 'check1_recompute_detail.csv', index=False)

summ = res.groupby('feature').agg(n=('absdiff', 'size'), nan_mismatch=('nan_mismatch', 'sum'), max_absdiff=('absdiff', 'max'),
                                  p99=('absdiff', lambda x: np.nanpercentile(x, 99)), median=('absdiff', 'median'),
                                  n_gt_1e6=('absdiff', lambda x: int((x > 1e-6).sum())),
                                  scale=('dataset', lambda x: float(np.nanstd(x)))).reset_index()
summ.to_csv(OUT / 'check1_recompute_summary.csv', index=False)
print('features compared:', summ.feature.nunique(), 'rows:', len(res))
print('features with any |diff|>1e-6 or NaN mismatch:')
bad = summ[(summ.n_gt_1e6 > 0) | (summ.nan_mismatch > 0)]
print(bad.sort_values('max_absdiff', ascending=False).head(40).to_string())
print('overall max absdiff', res.absdiff.max(), ' share of values within 1e-6:', float((res.absdiff <= 1e-6).mean()))
core = summ[summ.feature.isin(['days_rest', 'is_b2b', 'games_last_7d', 'win_pct_last10', 'win_streak', 'season_game_number',
                               'home_win_pct_20g', 'is_playoff'])]
print(core.to_string())
ex = res[res.feature.str.startswith('off_offensiveRating') | res.feature.str.startswith('off_points')
         | res.feature.str.startswith('def_OPP_PTS')].groupby('feature').absdiff.agg(['max', 'median'])
print(ex)
