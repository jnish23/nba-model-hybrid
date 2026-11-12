"""Checks 5b, 5c, 5d.
5b  load_clean_odds (project function, under test) vs my own consensus recomputed from the raw per-book columns on a random sample
    (seed 42): sign conventions, de-vigged probabilities, and the basis difference between the training set's odds_* (one book, DraftKings
    first) and the consensus open used to evaluate.
5c  games where 'close' could be an in-game / post-game value: |close-open| > 8 (spread) / > 15 (total); closing-vs-opening accuracy ratio by
    month (the signature of in-game closing lines used by the project's own guard).
5d  seasons 2013-14..2018-19: odds availability in the training set and how a missing ML price enters the model."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
sys.path.insert(0, str(ROOT := __import__('pathlib').Path('C:/Users/jorda/nba-pregame-audit')))
from common import *   # noqa
from core.odds_clean import load_clean_odds   # project function under test

BOOKS = ['betmgm', 'fanduel', 'caesars', 'bet365', 'draftkings', 'bet_rivers_ny']
o = load_clean_odds(DB)
o['gameId'] = o['gameId'].astype(str).str.zfill(10)
o['gameDate'] = pd.to_datetime(o['gameDate'])
print('clean odds rows', len(o), o.gameDate.min().date(), o.gameDate.max().date())
o['margin'] = o['home_score'] - o['away_score']
o['tot'] = o['home_score'] + o['away_score']
o['home_win'] = (o['margin'] > 0).astype(int)

# ---------- sign conventions on the whole set
for pre in ('open_', 'close_'):
    d = o.dropna(subset=[pre + 'home_spread'])
    fav = d[d[pre + 'home_spread'] < 0]
    dog = d[d[pre + 'home_spread'] > 0]
    print(f'{pre}home_spread: n={len(d)} corr with margin={d[pre+"home_spread"].corr(d["margin"]):.3f}; home win rate when home favoured '
          f'{fav.home_win.mean():.3f} (n={len(fav)}), when home dog {dog.home_win.mean():.3f} (n={len(dog)}); mean(margin+spread) {np.mean(d.margin + d[pre+"home_spread"]):.3f}')
    dp = o.dropna(subset=[pre + 'home_prob'])
    print(f'{pre}home_prob: corr with home_win={dp[pre+"home_prob"].corr(dp.home_win):.3f}; mean prob {dp[pre+"home_prob"].mean():.3f} vs actual {dp.home_win.mean():.3f}; '
          f'corr with -home_spread={dp[pre+"home_prob"].corr(-dp[pre+"home_spread"]):.3f}')
    dt = o.dropna(subset=[pre + 'total'])
    print(f'{pre}total: mean(total - line) {np.mean(dt.tot - dt[pre+"total"]):.3f}')

# ---------- my own consensus from raw columns, on a random sample
with conn() as c:
    raw = {'open_': pd.read_sql('select * from nba_odds_opening', c), 'close_': pd.read_sql('select * from nba_odds', c)}
    names = pd.concat([pd.read_sql('select distinct home_team name, teamId_home id from nba_odds where teamId_home is not null', c),
                       pd.read_sql('select distinct away_team name, teamId_away id from nba_odds where teamId_away is not null', c)]).drop_duplicates()
n2id = dict(zip(names.name.replace({'Charlotte Bobcats': 'Charlotte Hornets', 'LA Clippers': 'Los Angeles Clippers'}), names.id.astype(int)))
rng = np.random.default_rng(SEED)
samp = o[(o.gameDate < '2026-10-01')].sample(500, random_state=SEED)


def my_consensus(row, d, kind):
    hs = np.array([pd.to_numeric(row.get(f'home_spread_{b}'), errors='coerce') for b in BOOKS], float)
    as_ = np.array([pd.to_numeric(row.get(f'away_spread_{b}'), errors='coerce') for b in BOOKS], float)
    ok = (np.abs(hs) <= 30) & (np.abs(hs + as_) < 0.01)
    hs = np.where(ok, hs, np.nan)
    if np.isfinite(hs).any():
        med = np.nanmedian(hs)
        hs = np.where(np.abs(hs - med) <= 3, hs, np.nan)
    sp = np.nanmedian(hs) if np.isfinite(hs).any() else np.nan
    tt = np.array([pd.to_numeric(row.get(f'total_{b}'), errors='coerce') for b in BOOKS], float)
    tt = np.where((tt >= 170) & (tt <= 290), tt, np.nan)
    if np.isfinite(tt).any():
        med = np.nanmedian(tt)
        tt = np.where(np.abs(tt - med) <= 6, tt, np.nan)
    tot = np.nanmedian(tt) if np.isfinite(tt).any() else np.nan
    hm = np.array([pd.to_numeric(row.get(f'home_ml_{b}'), errors='coerce') for b in BOOKS], float)
    am = np.array([pd.to_numeric(row.get(f'away_ml_{b}'), errors='coerce') for b in BOOKS], float)
    good = (np.abs(hm) >= 100) & (np.abs(hm) <= 20000) & (np.abs(am) >= 100) & (np.abs(am) <= 20000)
    imp = lambda x: np.where(x < 0, -x / (-x + 100), 100 / (x + 100))
    ph, pa = imp(hm), imp(am)
    over = ph + pa
    fair = np.where(good & (over >= 1) & (over <= 1.15), ph / over, np.nan)
    if np.isfinite(fair).any():
        med = np.nanmedian(fair)
        fair = np.where(np.abs(fair - med) <= 0.03, fair, np.nan)
    pr = np.nanmedian(fair) if np.isfinite(fair).any() else np.nan
    return sp, tot, pr


import warnings
warnings.filterwarnings('ignore')
rows = []
for kind, pre in (('open', 'open_'), ('close', 'close_')):
    d = raw[pre].copy()
    d['gameDate'] = pd.to_datetime(d['gameDate']).dt.strftime('%Y-%m-%d')
    for col in ('home_team', 'away_team'):
        d[col] = d[col].replace({'Charlotte Bobcats': 'Charlotte Hornets', 'LA Clippers': 'Los Angeles Clippers'})
    d['hid'] = d['home_team'].map(n2id)
    d['aid'] = d['away_team'].map(n2id)
    d = d.drop_duplicates(['gameDate', 'hid', 'aid'], keep='last').set_index(['gameDate', 'hid', 'aid'])
    for _, r in samp.iterrows():
        k = (r.gameDate.strftime('%Y-%m-%d'), int(r.teamId_home), int(r.teamId_away))
        if k not in d.index:
            continue
        sp, tot, pr = my_consensus(d.loc[k], d, kind)
        rows.append({'gameId': r.gameId, 'which': kind, 'spread_proj': r[pre + 'home_spread'], 'spread_mine': sp, 'total_proj': r[pre + 'total'],
                     'total_mine': tot, 'prob_proj': r[pre + 'home_prob'], 'prob_mine': pr, 'src': r.close_source})
cmp_ = pd.DataFrame(rows)
cmp_.to_csv(OUT / 'check5b_sample_comparison.csv', index=False)
for kind in ('open', 'close'):
    x = cmp_[cmp_.which == kind]
    # for 2026-27 snapshot-based close the raw nba_odds table is not the source, excluded by sampling date < 2026-10
    for f in ('spread', 'total', 'prob'):
        a, b = x[f + '_proj'], x[f + '_mine']
        both = a.notna() & b.notna()
        print(f'{kind:5s} {f:6s} n={int(both.sum())} max|diff|={float((a[both]-b[both]).abs().max()):.2e}  nan-mismatch={int((a.isna()!=b.isna()).sum())}')
# explicit raw-sign spot check: a game where one book's home spread is negative -> that book lists the home team as favourite -> home ML negative
spot = raw['close_'].copy()
spot['hs'] = pd.to_numeric(spot['home_spread_draftkings'], errors='coerce')
spot['hm'] = pd.to_numeric(spot['home_ml_draftkings'], errors='coerce')
x = spot.dropna(subset=['hs', 'hm'])
x = x[(x.hs.abs() > 0.5) & (x.hm.abs() < 5000)]
print('raw DraftKings: home spread<0 and home ML<0 agree in', round(float(((x.hs < 0) == (x.hm < 0)).mean()), 4), 'of', len(x), 'games')

# ---------- 5c: implausible open->close moves
o2 = o.dropna(subset=['open_home_spread', 'close_home_spread']).copy()
o2['mv_s'] = o2.close_home_spread - o2.open_home_spread
big_s = o2[o2.mv_s.abs() > 8]
o3 = o.dropna(subset=['open_total', 'close_total']).copy()
o3['mv_t'] = o3.close_total - o3.open_total
big_t = o3[o3.mv_t.abs() > 15]
print('games with both open and close spread:', len(o2), ' |close-open|>8:', len(big_s), '; >5:', int((o2.mv_s.abs() > 5).sum()),
      '; mean |move|', round(o2.mv_s.abs().mean(), 3), ' p99', round(o2.mv_s.abs().quantile(.99), 2))
print('games with both open and close total:', len(o3), ' |close-open|>15:', len(big_t), '; >10:', int((o3.mv_t.abs() > 10).sum()),
      '; mean |move|', round(o3.mv_t.abs().mean(), 3), ' p99', round(o3.mv_t.abs().quantile(.99), 2))
cols = ['gameDate', 'gameId', 'open_home_spread', 'close_home_spread', 'open_total', 'close_total', 'margin', 'tot', 'close_source']
print(big_s[cols].to_string()); print(big_t[cols].to_string())
pd.concat([big_s[cols], big_t[cols]]).drop_duplicates().to_csv(OUT / 'check5c_big_moves.csv', index=False)
# in-game signature: close closer to the result than open by an implausible margin, by month
o2['err_open'] = (o2.margin + o2.open_home_spread).abs()
o2['err_close'] = (o2.margin + o2.close_home_spread).abs()
o2['ym'] = o2.gameDate.dt.to_period('M')
mm = o2.groupby('ym').agg(n=('err_open', 'size'), err_open=('err_open', 'mean'), err_close=('err_close', 'mean'))
mm['ratio'] = mm.err_close / mm.err_open
print('months with n>=30 and spread close/open abs-error ratio < 0.93 or > 1.05:')
print(mm[(mm.n >= 30) & ((mm.ratio < 0.93) | (mm.ratio > 1.05))].round(3).to_string())
sea = o2.assign(season=np.where(o2.gameDate.dt.month >= 8, o2.gameDate.dt.year, o2.gameDate.dt.year - 1)).groupby('season').agg(
    n=('err_open', 'size'), err_open=('err_open', 'mean'), err_close=('err_close', 'mean'))
sea['ratio'] = sea.err_close / sea.err_open
print('by season:'); print(sea.round(3).to_string())
print('how many games per source: ', o.close_source.value_counts().to_dict())
print('games with close missing among games with open (by season):')
o['season'] = np.where(o.gameDate.dt.month >= 8, o.gameDate.dt.year, o.gameDate.dt.year - 1)
print(o.groupby('season').agg(n=('gameId', 'size'), open_sp=('open_home_spread', lambda x: x.notna().mean()), close_sp=('close_home_spread', lambda x: x.notna().mean())).round(3).to_string())

# ---------- basis: training-set (single book) opening vs consensus opening
t = load_train()
tt = t[['gameId', 'odds_home_spread', 'odds_total', 'odds_home_ml', 'odds_away_ml']].merge(
    o[['gameId', 'open_home_spread', 'open_total', 'open_home_prob', 'close_home_spread', 'close_total', 'close_home_prob']], on='gameId')
tt = tt.dropna(subset=['odds_home_spread', 'open_home_spread'])
d = tt.odds_home_spread - tt.open_home_spread
print('training-set odds_home_spread (one book) minus consensus open: n=%d, share equal %.3f, mean|d| %.3f, p95 %.2f, max %.1f' %
      (len(d), (d == 0).mean(), d.abs().mean(), d.abs().quantile(.95), d.abs().max()))
d2 = tt.dropna(subset=['odds_total', 'open_total'])
d2 = d2.odds_total - d2.open_total
print('odds_total minus consensus open_total: share equal %.3f, mean|d| %.3f, p95 %.2f' % ((d2 == 0).mean(), d2.abs().mean(), d2.abs().quantile(.95)))
tt['impl_open'] = np.where(tt.odds_home_ml < 0, -tt.odds_home_ml / (-tt.odds_home_ml + 100), 100 / (tt.odds_home_ml + 100))
tt['impl_a'] = np.where(tt.odds_away_ml < 0, -tt.odds_away_ml / (-tt.odds_away_ml + 100), 100 / (tt.odds_away_ml + 100))
tt['devig_book'] = tt.impl_open / (tt.impl_open + tt.impl_a)
dd = (tt.devig_book - tt.open_home_prob).dropna()
print('de-vigged single-book open prob minus consensus open prob: mean|d| %.4f p95 %.4f' % (dd.abs().mean(), dd.abs().quantile(.95)))
# the move statistic: close - open using single-book open vs consensus open
mv_book = tt.close_home_spread - tt.odds_home_spread
mv_cons = tt.close_home_spread - tt.open_home_spread
m = mv_book.notna() & mv_cons.notna()
print('SD of (close - open) spread: with single-book open %.3f vs with consensus open %.3f (n=%d)' % (mv_book[m].std(), mv_cons[m].std(), m.sum()))
mt = tt.dropna(subset=['odds_total', 'open_total', 'close_total'])
print('SD of (close - open) total: single-book open %.3f vs consensus open %.3f' % ((mt.close_total - mt.odds_total).std(), (mt.close_total - mt.open_total).std()))
