"""Lead's check (exploratory, decides nothing): does capping unusual games in training, or a robust loss, make the model better on ordinary games?
Deployed feature lists and hyperparameters; training rows 2019-20 onward (rows with an opening line); test seasons 2021-22..2025-26, each trained on all
earlier seasons. Variants differ ONLY in the training target / loss. Errors are always measured against the real, uncapped result."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

pd.set_option('display.width', 220)
SUF = '_with_odds_cand_20261019'
feats = json.load(open(f'models/saved/feature_config{SUF}.json'))
par = {t: json.load(open(f'models/saved/best_config_{t}{SUF}.json')) for t in ('spread', 'total')}
need = sorted(set(feats['spread']) | set(feats['total']) | {'gameId', 'gameDate', 'seasonYear', 'target_point_diff', 'target_total_pts', 'odds_home_spread', 'odds_total'})
d = pd.read_csv('data/processed/training_set_v1_reduced_2026-11-01.csv', usecols=lambda c: c in need)
d = d[(d.seasonYear >= '2019-20') & (d.seasonYear <= '2025-26')].dropna(subset=['odds_home_spread', 'odds_total']).sort_values('gameDate').reset_index(drop=True)
d['y_spread'] = d.target_point_diff + d.odds_home_spread          # residual target, as deployed
d['y_total'] = d.target_total_pts                                 # level target, as deployed
d['line_total'] = d.odds_total
TEST = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']


def fit_predict(kind, tr, te, variant):
    X, Xt = tr[feats[kind]].astype(float), te[feats[kind]].astype(float)
    y = tr[f'y_{kind}'].to_numpy(float)
    base = tr.line_total.to_numpy(float) if kind == 'total' else np.zeros(len(tr))       # centre of 'unusual' = the line
    resid = y - base
    sd = resid.std()
    extra, w = {}, None
    if variant == 'L2 (deployed)':
        pass
    elif variant.startswith('cap'):
        kcap = float(variant.split()[1].rstrip('sd'))
        y = base + np.clip(resid, -kcap * sd, kcap * sd)
    elif variant.startswith('drop'):
        kcap = float(variant.split()[1].rstrip('sd'))
        keep = np.abs(resid) <= kcap * sd
        X, y = X[keep], y[keep]
    elif variant == 'Huber (1 sd)':
        extra = dict(objective='huber', alpha=float(sd))
    elif variant == 'L1 (median)':
        extra = dict(objective='regression_l1')
    m = lgb.LGBMRegressor(**par[kind], **extra, verbosity=-1, random_state=42, deterministic=True, force_row_wise=True)
    m.fit(X, y)
    return m.predict(Xt)


VARIANTS = ['L2 (deployed)', 'cap 2.5sd', 'cap 2sd', 'cap 1.5sd', 'drop 2sd', 'Huber (1 sd)', 'L1 (median)']
for kind in ('spread', 'total'):
    out = {v: [] for v in VARIANTS}
    rows = []
    for s in TEST:
        tr, te = d[d.seasonYear < s], d[d.seasonYear == s]
        for v in VARIANTS:
            p = fit_predict(kind, tr, te, v)
            actual = te[f'y_{kind}'].to_numpy(float)
            line = te.line_total.to_numpy(float) if kind == 'total' else np.zeros(len(te))
            out[v].append(pd.DataFrame({'season': s, 'date': te.gameDate.values, 'err': actual - p, 'line_err': actual - line, 'gap': p - line}))
    ref = pd.concat(out['L2 (deployed)'], ignore_index=True)
    sd_all = ref.line_err.std()
    ordinary = ref.line_err.abs() <= 1.5 * sd_all                   # 'ordinary' game = result within 1.5 sd of the opening line (about 87% of games)
    codes, uniq = pd.factorize(pd.to_datetime(ref.date))
    rng = np.random.default_rng(42)
    draws = [rng.integers(0, len(uniq), len(uniq)) for _ in range(1000)]
    for v in VARIANTS:
        e = pd.concat(out[v], ignore_index=True)
        dd = e.err.abs().to_numpy() - ref.err.abs().to_numpy()
        sums, cnt = np.bincount(codes, weights=dd), np.bincount(codes)
        b = [sums[i].sum() / cnt[i].sum() for i in draws]
        do = np.where(ordinary, dd, 0.0); so, co = np.bincount(codes, weights=do), np.bincount(codes, weights=ordinary.astype(float))
        bo = [so[i].sum() / co[i].sum() for i in draws]
        side_ok = (np.sign(e.gap) == np.sign(e.line_err))[(e.gap != 0) & (e.line_err != 0)]
        rows.append({'training target / loss': v, 'MAE all': e.err.abs().mean(), 'vs deployed': dd.mean(), 'CI': f'[{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]',
                     'MAE ordinary games': e.err.abs()[ordinary].mean(), 'ordinary vs deployed': dd[ordinary].mean(),
                     'CI ord': f'[{np.percentile(bo, 2.5):+.3f}, {np.percentile(bo, 97.5):+.3f}]',
                     'MAE unusual games': e.err.abs()[~ordinary].mean(), 'mean error': e.err.mean(), 'median error': e.err.median(),
                     'right side of the open': side_ok.mean(), 'sd of (model - line)': e.gap.std()})
    print(f'\n===== {kind.upper()}  ({len(ref)} test games; opening line alone: MAE {ref.line_err.abs().mean():.3f}, ordinary {ref.line_err.abs()[ordinary].mean():.3f}; '
          f'ordinary = {ordinary.mean():.0%} of games)')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
k = d[d.seasonYear.isin(TEST)]
r = k.target_total_pts - k.odds_total
print('\nTotals result minus opening line: mean %+.2f, median %+.2f, share over %.3f (pushes excluded), skew %.2f' % (r.mean(), r.median(), (r > 0).sum() / (r != 0).sum(), r.skew()))
r = k.target_point_diff + k.odds_home_spread
print('Margin result minus opening line: mean %+.2f, median %+.2f, skew %.2f, kurtosis %.2f' % (r.mean(), r.median(), r.skew(), r.kurt()))
