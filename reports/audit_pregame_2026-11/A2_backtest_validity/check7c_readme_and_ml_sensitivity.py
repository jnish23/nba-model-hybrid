"""Check 7c: (1) reproduce the E25_decorrelation README 'with odds' edge slopes (confirmation seasons 2022-23..2025-26, opening line, and the
close) with day-bootstrap CIs; (2) ML sensitivity to the degenerate isotonic probabilities (exact 0/1) in the legacy E25 run."""
import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from common import *
from mymetrics import *
import warnings; warnings.filterwarnings('ignore')
p = load_e25()
conf = p[p.seasonYear >= '2022-23']
def slope_boot(d, y, x, n=1000, seed=SEED):
    days, inv = np.unique(d.gameDate.to_numpy(), return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(days))]
    rng = np.random.default_rng(seed); Y, X = y.to_numpy(float), x.to_numpy(float)
    pt = np.polyfit(X, Y, 1)[0]
    b = np.empty(n)
    for k in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(days), len(days))]); b[k] = np.polyfit(X[idx], Y[idx], 1)[0]
    return pt, np.percentile(b, 2.5), np.percentile(b, 97.5)
print('README (E25_decorrelation, with odds, confirmation 2022-23..2025-26): total 0.19 [0.05, 0.34], spread 0.27 [0.01, 0.55], ML (log-odds) 0.39 [0.09, 0.72]; vs close ~0 (-0.05..+0.05)')
for nm, d, act, pred, op, cl, sg in (('spread', conf.dropna(subset=['odds_home_spread', 'close_home_spread']), 'target_point_diff', 'spread_pred', 'odds_home_spread', 'close_home_spread', 1),
                                     ('total', conf.dropna(subset=['odds_total', 'close_total']), 'target_total_pts', 'total_pred', 'odds_total', 'close_total', 0)):
    if sg:   # spread: margin units; line in margin = -spread
        yo, xo = d[act] + d[op], d[pred] + d[op]; yc, xc = d[act] + d[cl], d[pred] + d[cl]
    else:
        yo, xo = d[act] - d[op], d[pred] - d[op]; yc, xc = d[act] - d[cl], d[pred] - d[cl]
    a = slope_boot(d, yo, xo); b = slope_boot(d, yc, xc)
    print(f'{nm:7s} n={len(d)}  slope vs OPEN {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}]   slope vs CLOSE (result-close on model-close) {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]')
d = conf.dropna(subset=['open_home_prob', 'close_home_prob'])
lg = lambda q: np.log(q.clip(1e-4, 1 - 1e-4) / (1 - q.clip(1e-4, 1 - 1e-4)))
# README's ML beta is in log-odds: regress (logit outcome-proxy)? use the probability-space slope and a log-odds-gap slope via logistic offset (Newton)
def logit_offset_beta(df, offset, x):
    X = x.to_numpy(float); y = df.target_home_win.to_numpy(float); o = offset.to_numpy(float); b = 0.0
    for _ in range(50):
        pr = 1 / (1 + np.exp(-(o + b * X))); g = ((y - pr) * X).sum(); h = (pr * (1 - pr) * X * X).sum(); b += g / h
    return b
def boot_logit(df, offset, x, n=500):
    days, inv = np.unique(df.gameDate.to_numpy(), return_inverse=True); groups = [np.flatnonzero(inv == i) for i in range(len(days))]
    rng = np.random.default_rng(SEED); bs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(days), len(days))]); bs.append(logit_offset_beta(df.iloc[idx], offset.iloc[idx], x.iloc[idx]))
    return logit_offset_beta(df, offset, x), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
off = lg(d.open_home_prob); x = lg(d.ml_prob) - off
print('ML log-odds slope vs OPEN (logistic with offset logit(open)): %+.3f [%+.3f, %+.3f] (n=%d; ml_prob clipped to [1e-4,1-1e-4] because the isotonic output has exact 0/1)' % (*boot_logit(d, off, x), len(d)))
offc = lg(d.close_home_prob); xc = lg(d.ml_prob) - offc
print('ML log-odds slope vs CLOSE: %+.3f [%+.3f, %+.3f]' % boot_logit(d, offc, xc))

# ML sensitivity
print('\nML sensitivity (pooled, bets at open prices):')
def roi(df, col):
    q = df.copy(); q['ml_prob'] = q[col]
    b, _ = make_bets(q, markets=('ml',)); st = day_boot_stat(b)
    return st
for nm, d_, col in (('as saved (isotonic, may be exactly 0/1)', p, 'ml_prob'),
                    ('excluding the 55 games with ml_prob<=0.001 or >=0.999', p[(p.ml_prob > 0.001) & (p.ml_prob < 0.999)], 'ml_prob'),
                    ('raw classifier prob (no calibration)', p, 'ml_prob_raw'),
                    ('ml_prob clipped to [0.02, 0.98]', p.assign(ml_prob_c=p.ml_prob.clip(.02, .98)), 'ml_prob_c')):
    st = roi(d_, col)
    print(f'  {nm:58s} bets {st["bets"]:5d} hit {st["hit"]:.1f}% ROI {st["roi"]:+.2f}% [{st["roi_ci"][0]:+.1f}, {st["roi_ci"][1]:+.1f}]')

# README's ML beta uses the RAW classifier probability (ml_prob_raw) with an intercept; reproduce that definition
def beta_int(df, offset, x):
    X = np.column_stack([np.ones(len(x)), x.to_numpy(float)]); y = df.target_home_win.to_numpy(float); o = offset.to_numpy(float); b = np.zeros(2)
    for _ in range(50):
        pr = 1 / (1 + np.exp(-(o + X @ b))); w = pr * (1 - pr); H = X.T @ (X * w[:, None]) + 1e-9 * np.eye(2); st = np.linalg.solve(H, X.T @ (y - pr)); b += st
        if np.abs(st).max() < 1e-10: break
    return b[1]
def boot_int(df, offset, x, n=500):
    days, inv = np.unique(df.gameDate.to_numpy(), return_inverse=True); groups = [np.flatnonzero(inv == i) for i in range(len(days))]
    rng = np.random.default_rng(SEED); bs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(days), len(days))]); bs.append(beta_int(df.iloc[idx], offset.iloc[idx], x.iloc[idx]))
    return beta_int(df, offset, x), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
xr = lg(d.ml_prob_raw) - off
print('\nML log-odds beta using ml_prob_raw + intercept (the README definition), vs open: %+.3f [%+.3f, %+.3f] (README 0.39 [0.09, 0.72])' % boot_int(d, off, xr))
xr2 = lg(d.ml_prob_raw) - offc
print('   ... vs close: %+.3f [%+.3f, %+.3f]' % boot_int(d, offc, xr2))

# offset = de-vigged SINGLE-BOOK open price (what the model read; README's devig_open_prob), games with both prices
d2 = conf.dropna(subset=['odds_home_ml', 'odds_away_ml']).copy()
ph, pa = implied(d2.odds_home_ml), implied(d2.odds_away_ml)
off2 = lg(pd.Series(ph / (ph + pa), index=d2.index))
print('ML beta (raw prob, intercept) with single-book de-vigged open as offset: %+.3f [%+.3f, %+.3f] n=%d' % (*boot_int(d2, off2, lg(d2.ml_prob_raw) - off2), len(d2)))
