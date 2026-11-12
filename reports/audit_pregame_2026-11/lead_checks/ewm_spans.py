"""Lead's check: does the EWM window (spans 5 and 20) matter? Pre-game EWMA of each team's point margin and total at several spans,
home-minus-away difference (sum for totals), scored walk-forward (fit on all earlier seasons, test on each season 2019-20..2025-26):
 (1) alone, against the result;  (2) on top of the opening line, against the market residual."""
import sqlite3
import numpy as np
import pandas as pd

pd.set_option('display.width', 220)
c = sqlite3.connect('file:data/nba_stats.db?mode=ro', uri=True)
t = pd.read_sql("select gameId, teamId, gameDate, seasonYear, MATCHUP as matchup, points, plusMinus from team_game_full order by teamId, gameDate", c)
t['gameDate'] = pd.to_datetime(t.gameDate)
t['opp_pts'] = t.points - t.plusMinus
t['tot'] = t.points + t.opp_pts
SPANS = [3, 5, 10, 20, 40, 82, 164]
for s in SPANS:
    for col in ('plusMinus', 'tot'):
        t[f'{col}_{s}'] = t.groupby('teamId')[col].transform(lambda x: x.ewm(span=s).mean().shift(1))
t['is_home'] = t.matchup.str.contains('vs.', regex=False)
h, a = t[t.is_home], t[~t.is_home]
g = h.merge(a, on='gameId', suffixes=('_h', '_a'))
g = g.rename(columns={'seasonYear_h': 'season', 'gameDate_h': 'date'})
g['margin'] = g.points_h - g.points_a
g['total'] = g.points_h + g.points_a
for s in SPANS:
    g[f'm{s}'] = g[f'plusMinus_{s}_h'] - g[f'plusMinus_{s}_a']
    g[f't{s}'] = (g[f'tot_{s}_h'] + g[f'tot_{s}_a']) / 2
tr = pd.read_csv('data/processed/training_set_v1_reduced_2026-11-01.csv', usecols=['gameId', 'odds_home_spread', 'odds_total'], dtype={'gameId': str})
g['gameId'] = g.gameId.astype(str).str.zfill(10); tr['gameId'] = tr.gameId.str.zfill(10)
g = g.merge(tr, on='gameId', how='left').sort_values('date').reset_index(drop=True)
g = g.dropna(subset=[f'm{SPANS[-1]}', f't{SPANS[-1]}'])
TEST = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']


def wf(cols, target, need=None):
    """Walk-forward OLS: pooled out-of-sample MAE and RMSE, and seasons in which it beats the reference set (5, 20)."""
    err = []
    for s in TEST:
        d = g if need is None else g.dropna(subset=need)
        a_, b_ = d[d.season < s], d[d.season == s]
        X = np.column_stack([a_[cols].to_numpy(float), np.ones(len(a_))]) if cols else np.ones((len(a_), 1))
        co = np.linalg.lstsq(X, a_[target].to_numpy(float), rcond=None)[0]
        Xb = np.column_stack([b_[cols].to_numpy(float), np.ones(len(b_))]) if cols else np.ones((len(b_), 1))
        err.append(pd.DataFrame({'season': s, 'date': b_.date.values, 'e': b_[target].to_numpy(float) - Xb @ co}))
    return pd.concat(err, ignore_index=True)


def table(prefix, target, need=None, title=''):
    ref = wf([f'{prefix}5', f'{prefix}20'], target, need)
    rows = []
    sets = [('none (constant)', [])] + [(f'span {s}', [f'{prefix}{s}']) for s in SPANS] + \
           [('5 + 20 (production)', [f'{prefix}5', f'{prefix}20']), ('10 + 40', [f'{prefix}10', f'{prefix}40']), ('20 + 82', [f'{prefix}20', f'{prefix}82']),
            ('5 + 20 + 82', [f'{prefix}5', f'{prefix}20', f'{prefix}82']), ('all seven', [f'{prefix}{s}' for s in SPANS])]
    rng = np.random.default_rng(42)
    days = ref.date.dt.normalize()
    codes, uniq = pd.factorize(days)
    for name, cols in sets:
        e = wf(cols, target, need)
        d = e.e.abs().to_numpy() - ref.e.abs().to_numpy()
        sums = np.bincount(codes, weights=d); cnt = np.bincount(codes)
        boot = [sums[i].sum() / cnt[i].sum() for i in (rng.integers(0, len(uniq), len(uniq)) for _ in range(1000))]
        better = sum((e[e.season == s].e.abs().mean() < ref[ref.season == s].e.abs().mean()) for s in TEST)
        rows.append({'features': name, 'MAE': e.e.abs().mean(), 'RMSE': np.sqrt((e.e ** 2).mean()), 'MAE minus (5+20)': d.mean(),
                     'CI lo': np.percentile(boot, 2.5), 'CI hi': np.percentile(boot, 97.5), 'seasons better than 5+20': better})
    print(f'\n{title}  (n={len(ref)})')
    print(pd.DataFrame(rows).round(3).to_string(index=False))


g['resid_spread'] = g.margin + g.odds_home_spread
g['resid_total'] = g.total - g.odds_total
table('m', 'margin', title='MARGIN from the EWMA of point margin alone (no market)')
table('t', 'total', title='TOTAL from the EWMA of game totals alone (no market)')
table('m', 'resid_spread', need=['odds_home_spread'], title='MARGIN minus opening line (what is left once the line is known)')
table('t', 'resid_total', need=['odds_total'], title='TOTAL minus opening line (what is left once the line is known)')
k = g.dropna(subset=['odds_home_spread', 'odds_total'])
k = k[k.season.isin(TEST)]
print('\nReference: the opening line alone on the same test games: spread MAE %.3f, total MAE %.3f' % (k.resid_spread.abs().mean(), k.resid_total.abs().mean()))
