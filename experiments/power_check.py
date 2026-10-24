"""Power and size check for the registered rules of E14 (spread slope) and E17 (spread x total interaction).
Null world: the real games with the estimated |spread| effect subtracted from r. Each replicate resamples whole game days within the
discovery and confirmation periods, adds a KNOWN effect, and applies the registered rule. Reports how often each rule fires at effect 0 (size)
and at each injected effect (power). python -m experiments.power_check"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import e14_blowout_under as e14                       # noqa: E402
from experiments.e14_blowout_under import DayBoot                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports' / 'experiments' / 'POWER_CHECK'
N_REP, N_INNER = 300, 300
SLOPE_EFFECTS = [0.0, 0.05, 0.10, 0.15, 0.18, 0.25]            # points of r per spread point
PROD_EFFECTS = [0.0, 0.005, 0.010, 0.015, 0.020]               # points of r per (spread point x total point)


def p2(b):
    return max(2 * min((b <= 0).sum(), (b >= 0).sum()), 1) / (len(b) + 1)


def fit(db, a, t, y, mask, seasons, prod):
    cols = [np.ones(len(a)), a] + ([t, a * t] + [(seasons == s).astype(float) for s in sorted(set(seasons[mask]))[1:]] if prod else [])
    beta, lo, hi, boots = db.ols(np.column_stack(cols), y, mask)
    i = 3 if prod else 1
    return float(beta[i]), float(lo[i]), float(hi[i]), p2(boots[:, i])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = e14.load_frame()
    df = df[df['seasonYear'].isin(e14.DISCOVERY + e14.CONFIRM)].reset_index(drop=True)
    disc = df['seasonYear'].isin(e14.DISCOVERY).to_numpy()
    df['T'] = df['odds_total'] - df.loc[disc, 'odds_total'].mean()
    a0, T0, r0, S0 = df['abs_spread'].to_numpy(float), df['T'].to_numpy(float), df['r'].to_numpy(float), df['seasonYear'].to_numpy()
    # null world: remove the fitted |spread| main effect and any product term from the real data
    X = np.column_stack([np.ones(len(df)), a0, T0, a0 * T0] + [(S0 == s).astype(float) for s in sorted(set(S0))[1:]])
    beta = np.linalg.lstsq(X, r0, rcond=None)[0]
    r_null = r0 - beta[1] * (a0 - a0.mean()) - beta[3] * (a0 * T0 - (a0 * T0).mean())
    day = pd.factorize(pd.to_datetime(df['gameDate']))[0]
    groups = {p: [np.flatnonzero((day == d) & (disc == (p == 'd'))) for d in np.unique(day[disc == (p == 'd')])] for p in ('d', 'c')}
    rng = np.random.default_rng(20261023)
    res = {'slope': {str(e): {k: 0 for k in ('registered', 'confirmation_only', 'pooled')} for e in SLOPE_EFFECTS},
           'product': {str(e): {k: 0 for k in ('registered', 'confirmation_only', 'pooled')} for e in PROD_EFFECTS}}
    for rep in range(N_REP):
        idx, dayid, inst = [], [], 0
        for p in ('d', 'c'):
            for g in rng.integers(0, len(groups[p]), len(groups[p])):
                idx.append(groups[p][g])
                dayid.append(np.full(len(groups[p][g]), inst))
                inst += 1
        idx, dayid = np.concatenate(idx), np.concatenate(dayid)
        a, t, y0, s = a0[idx], T0[idx], r_null[idx], S0[idx]
        dm = np.isin(s, e14.DISCOVERY)
        cm = ~dm
        allm = np.ones(len(a), bool)
        db = DayBoot(dayid, n_boot=N_INNER, seed=1000 + rep)
        for e in SLOPE_EFFECTS:
            y = y0 + e * (a - a0.mean())
            d_, c_, p_ = fit(db, a, t, y, dm, s, False), fit(db, a, t, y, cm, s, False), fit(db, a, t, y, allm, s, False)
            res['slope'][str(e)]['registered'] += int(d_[0] > 0 and c_[0] > 0 and c_[1] > 0)       # E14 rule, positive direction
            res['slope'][str(e)]['confirmation_only'] += int(c_[1] > 0)
            res['slope'][str(e)]['pooled'] += int(p_[1] > 0)
        for e in PROD_EFFECTS:
            y = y0 + e * a * t
            d_, c_, p_ = fit(db, a, t, y, dm, s, True), fit(db, a, t, y, cm, s, True), fit(db, a, t, y, allm, s, True)
            res['product'][str(e)]['registered'] += int(np.sign(d_[0]) == np.sign(c_[0]) and c_[3] < 0.05)   # E17 rule
            res['product'][str(e)]['confirmation_only'] += int(c_[3] < 0.05)
            res['product'][str(e)]['pooled'] += int(p_[3] < 0.05)
        if rep % 50 == 0:
            print('rep', rep, flush=True)
    for fam in res.values():
        for e, d in fam.items():
            for k in d:
                d[k] = d[k] / N_REP
    res['n_rep'], res['n_inner'] = N_REP, N_INNER
    res['removed_effects'] = {'spread_slope': float(beta[1]), 'product': float(beta[3])}
    # E16 analytic power: coefficient on the overtime probability
    e16 = json.loads((ROOT / 'reports' / 'experiments' / 'E16_overtime' / 'results.json').read_text())
    c = e16['step3']['p_ot']['all games']['confirmation']
    se = (c['hi'] - c['lo']) / 3.92
    b = e16['step4']['buckets']
    # overtime probability is fixed by the discovery logistic: p(s) = expit(b0 + b1 s)
    b0, b1 = e16['step1']['intercept'], e16['step1']['slope_per_spread_point']
    pr = lambda s: 1 / (1 + np.exp(-(b0 + b1 * s)))
    dP = pr(b[0]['mean_abs_spread']) - pr(b[-1]['mean_abs_spread'])
    from scipy.stats import norm
    out16 = {'se_coefficient_on_p_ot': se, 'p_ot_range_0_3_to_12plus': float(dP)}
    for gap in (0.8, 2.0, 5.0):
        gamma = gap / dP
        out16[f'power_for_gap_{gap}_points'] = float(norm.cdf(gamma / se - 1.96))
    res['e16_p_ot_regression'] = out16
    (OUT / 'results.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
