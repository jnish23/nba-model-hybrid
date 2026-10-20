"""
Descriptive look at the minutes and count distributions the Phase 3 models must reproduce. Uses seasons up to 2023-24 only (never the 2024-25
validation or the 2025-26 final holdout). Fits nothing but variance/dispersion summaries.
    python scripts/props_phase3_eda.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.features import build  # noqa: E402

LAST_SEASON = '2023-24'
STATS = {'points': ('y_points', 'r_points_h15'), 'rebounds': ('y_rebounds', 'r_rebounds_h15'), 'assists': ('y_assists', 'r_assists_h15'),
         'threes': ('y_threes', 'r_fg3m_h15')}


def nb_logpmf(y, mu, k):
    mu = np.maximum(mu, 1e-6)
    return gammaln(y + k) - gammaln(k) - gammaln(y + 1) + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu))


def pois_logpmf(y, mu):
    mu = np.maximum(mu, 1e-6)
    return y * np.log(mu) - mu - gammaln(y + 1)


def main():
    t = build.read_tables()
    d = t['player_game_base'].merge(t['player_game_labels'], on=['game_id', 'player_id'])
    d = d[d['season'] <= LAST_SEASON]
    pd.set_option('display.width', 200)
    print('rows by season (all listed / played):')
    print(d.groupby('season').agg(listed=('player_id', 'size'), played=('lbl_played', 'sum')).T.to_string())
    p = d[(d.lbl_played == 1) & (d.m_min_h15 >= 10)].copy()
    p['min_ratio'] = p['lbl_minutes'] / p['m_min_h15']
    print('\n1. MINUTES given played (expected minutes >= 10): realized / expected, quantiles')
    q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    p['exp_bucket'] = pd.cut(p['m_min_h15'], [10, 20, 28, 60])
    print(p.groupby('exp_bucket', observed=True)['min_ratio'].quantile(q).unstack().round(2).to_string())
    for thr in (0.3, 0.5, 0.7):
        print(f'   share of played games with minutes < {thr:.0%} of expected: {(p.min_ratio < thr).mean():.3f}')
    print(f'   sd of (minutes - expected): {np.std(p.lbl_minutes - p.m_min_h15):.2f}; mean abs {np.mean(np.abs(p.lbl_minutes - p.m_min_h15)):.2f}')
    print('\n2. COUNTS given played: overdispersion with expected = rate x EXPECTED minutes vs rate x ACTUAL minutes (oracle)')
    train = p[p.season <= '2022-23']
    test = p[p.season == LAST_SEASON]
    rows = []
    for name, (y, r) in STATS.items():
        for label, minutes in (('expected minutes', 'm_min_h15'), ('actual minutes (oracle)', 'lbl_minutes')):
            mu_tr = (train[r] * train[minutes]).to_numpy()
            y_tr = train[y].to_numpy()
            # method of moments for k in Var = mu + mu^2 / k, pooled
            resid2 = (y_tr - mu_tr) ** 2 - mu_tr
            k = np.sum(mu_tr ** 2) / max(np.sum(resid2), 1e-9)
            k = float(np.clip(k, 0.5, 1000))
            mu_te = (test[r] * test[minutes]).to_numpy()
            y_te = test[y].to_numpy()
            rows.append({'stat': name, 'minutes': label, 'mean': y_te.mean(), 'var/mean': y_te.var() / y_te.mean(),
                         'resid var/mean': np.var(y_te - mu_te) / y_te.mean(), 'k (NB)': round(k, 1),
                         'loglik Poisson': pois_logpmf(y_te, mu_te).mean(), 'loglik NB': nb_logpmf(y_te, mu_te, k).mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('\n   zeros among played (expected minutes >= 10), test season:', {n: round(float((test[y] == 0).mean()), 3) for n, (y, _) in STATS.items()})
    print('\n3. RESIDUAL CORRELATION of points, rebounds, assists (z = (y - mu) / sqrt(mu), mu = rate x expected minutes), train seasons')
    z = pd.DataFrame({n: (train[y] - train[r] * train['m_min_h15']) / np.sqrt(np.maximum(train[r] * train['m_min_h15'], 0.5)) for n, (y, r) in STATS.items()})
    print(z.corr().round(3).to_string())
    zo = pd.DataFrame({n: (train[y] - train[r] * train['lbl_minutes']) / np.sqrt(np.maximum(train[r] * train['lbl_minutes'], 0.5)) for n, (y, r) in STATS.items()})
    print('   with ACTUAL minutes (shared minutes removed):'); print(zo.corr().round(3).to_string())
    print('\n4. Does a blowout shorten minutes? (seasons with a market proxy at T-5)')
    a = t['player_game_at_T']
    a = a[a.minutes_before == 5][['game_id', 'player_id', 'env_blowout_prob']]
    b = p.merge(a, on=['game_id', 'player_id'])
    b['bb'] = pd.qcut(b['env_blowout_prob'], 4, duplicates='drop')
    print(b[b.m_min_h15 >= 25].groupby('bb', observed=True).agg(ratio=('min_ratio', 'mean'), n=('min_ratio', 'size')).round(3).to_string())


if __name__ == '__main__':
    main()
