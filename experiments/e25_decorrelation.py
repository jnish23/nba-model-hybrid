"""E25 step 2 - validity checks on the regenerated runs, then the edge-slope / blend diagnostic (see PREREGISTRATION.md, E25).
Run after experiments.e25_regen. python -m experiments.e25_decorrelation"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import metrics as M                                   # noqa: E402
from experiments.e01_residual import devig_open_prob                   # noqa: E402
from experiments.e25_regen import load_games_no_odds                   # noqa: E402
from experiments.lib import OUT, TEST_SEASONS, split_fold              # noqa: E402
from team.data_paths import latest_training_set                        # noqa: E402

NAME = 'E25_decorrelation'
RUNS = {'with_odds': 'E25_with_odds', 'residual': 'E25_residual', 'no_odds': 'E25_no_odds'}
DISC = ['2019-20', '2020-21', '2021-22']
CONF = ['2022-23', '2023-24', '2024-25', '2025-26']
N_BOOT = 1000
# reports/audit_fixes/walk_forward_no_odds.md, "new: neutral params": (ML log loss, spread MAE, total MAE)
AUDIT_NO_ODDS = {'2019-20': (0.6534, 10.49, 15.63), '2020-21': (0.6673, 11.29, 15.78), '2021-22': (0.6770, 11.31, 15.24),
                 '2022-23': (0.6736, 10.31, 14.88), '2023-24': (0.6668, 11.12, 15.08), '2024-25': (0.6190, 11.29, 14.93),
                 '2025-26': (0.6130, 11.52, 15.62)}
TOL = (0.03, 0.5, 0.5)
MARKETS = ['spread', 'total', 'ml']


# ------------------------------------------------------------------ validity

def validity(games):
    out = {}
    feats = json.loads((OUT / RUNS['no_odds'] / 'features.json').read_text())
    bad = sorted({f for fold in feats.values() for k in fold.values() for f in k if 'odds' in f.lower() or 'close' in f.lower()})
    out['1_no_market_features'] = {'pass': not bad, 'offending': bad}

    ids = {k: set(pd.read_csv(OUT / v / 'predictions.csv', usecols=['gameId'], dtype=str)['gameId']) for k, v in RUNS.items()}
    out['2_same_games'] = {'pass': ids['no_odds'] == ids['with_odds'] and ids['residual'] <= ids['with_odds'], 'n': {k: len(v) for k, v in ids.items()}}

    m, df = load_games_no_odds(latest_training_set())
    temporal, proxy = {}, {}
    lp = logit(np.clip(devig_open_prob(df), 1e-4, 1 - 1e-4))
    df = df.assign(_open_logodds=lp)
    for s in TEST_SEASONS:
        mt, calib, test, _ = split_fold(df, s)
        temporal[s] = bool(max(mt['gameDate'].max(), calib['gameDate'].max()) < test['gameDate'].min())
        fs = sorted({f for k in feats[s].values() for f in k})
        corr = {}
        for line in ('odds_total', 'odds_home_spread', '_open_logodds'):
            c = mt[fs].corrwith(mt[line]).abs()
            corr[line] = {'max_abs_corr': round(float(c.max()), 3), 'feature': c.idxmax(), 'over_0.9': c[c > 0.9].round(3).to_dict()}
        proxy[s] = corr
    out['3_temporal'] = {'pass': all(temporal.values()), 'per_fold': temporal}

    nd = games[games['seasonYear'].isin(TEST_SEASONS)]
    rows, ok = {}, True
    for s, g in nd.groupby('seasonYear'):
        p = M.clip_p(g['no_odds_ml_prob'])
        ll = float(-(g['target_home_win'] * np.log(p) + (1 - g['target_home_win']) * np.log(1 - p)).mean())
        sm = float((g['target_point_diff'] - g['no_odds_spread_pred']).abs().mean())
        tm = float((g['target_total_pts'] - g['no_odds_total_pred']).abs().mean())
        ref = AUDIT_NO_ODDS[s]
        gaps = (ll - ref[0], sm - ref[1], tm - ref[2])
        within = all(abs(d) <= t for d, t in zip(gaps, TOL))
        ok &= within
        rows[s] = {'ml_logloss': round(ll, 4), 'spread_mae': round(sm, 3), 'total_mae': round(tm, 3),
                   'audit': ref, 'gap': [round(x, 4) for x in gaps], 'within_tolerance': within}
    out['4_metrics_vs_audit'] = {'pass': ok, 'per_season': rows}
    out['5_market_proxy_corr'] = proxy
    return out


# ------------------------------------------------------------------ data

def load():
    base = None
    for tag, run in RUNS.items():
        p = pd.read_csv(OUT / run / 'predictions.csv', dtype={'gameId': str})
        cols = {'ml_prob_raw': f'{tag}_ml_raw', 'ml_prob': f'{tag}_ml_prob', 'spread_pred': f'{tag}_spread_pred', 'total_pred': f'{tag}_total_pred'}
        if base is None:
            base = p.drop(columns=[c for c in cols if c in p] + ['sigma_spread', 'sigma_total'], errors='ignore')
        base = base.merge(p[['gameId'] + list(cols)].rename(columns=cols), on='gameId', how='inner')
    base['gameDate'] = pd.to_datetime(base['gameDate'])
    base['regular'] = base['gameId'].str[2] == '2'
    base['L_ml'] = logit(np.clip(devig_open_prob(base), 1e-4, 1 - 1e-4))
    base['Lc_ml'] = logit(np.clip(base['close_home_prob'], 1e-4, 1 - 1e-4))
    return base


def frame(g, market, tag, close=False):
    """(e, r, L, y) arrays for one model and market; NaN rows dropped by the caller."""
    if market == 'total':
        L = g['close_total' if close else 'odds_total'].to_numpy(float)
        y = g['target_total_pts'].to_numpy(float)
        return g[f'{tag}_total_pred'].to_numpy(float) - L, y - L, L, y
    if market == 'spread':
        L = -g['close_home_spread' if close else 'odds_home_spread'].to_numpy(float)
        y = g['target_point_diff'].to_numpy(float)
        return g[f'{tag}_spread_pred'].to_numpy(float) - L, y - L, L, y
    L = g['Lc_ml' if close else 'L_ml'].to_numpy(float)
    y = g['target_home_win'].to_numpy(float)
    return logit(np.clip(g[f'{tag}_ml_raw'].to_numpy(float), 1e-4, 1 - 1e-4)) - L, y, L, y


def common_rows(g, market, close=False):
    ok = np.ones(len(g), bool)
    for tag in RUNS:
        e, r, L, _ = frame(g, market, tag, close)
        ok &= np.isfinite(e) & np.isfinite(r) & np.isfinite(L)
    return g[ok].reset_index(drop=True)


# ------------------------------------------------------------------ estimators

def logit_fit(y, X, off, iters=25):
    """Logistic regression with an offset (Newton). X has the regressors (include a ones column for an intercept)."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(off + X @ b)
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def beta(market, e, r, L):
    if market == 'ml':
        return float(logit_fit(r, np.column_stack([np.ones_like(e), e]), L)[1])
    ec, rc = e - e.mean(), r - r.mean()
    return float((ec * rc).sum() / (ec * ec).sum())


def k_fit(market, e, r, L):
    if market == 'ml':
        return float(logit_fit(r, e[:, None], L)[0])
    return float((e * r).sum() / (e * e).sum())


def loss(market, r, L, pred_shift):
    """Per-game loss of L + shift (shift = 0 is the line itself)."""
    if market == 'ml':
        p = M.clip_p(expit(L + pred_shift))
        return -(r * np.log(p) + (1 - r) * np.log(1 - p))
    return (r - pred_shift) ** 2


def day_draws(dates, n=N_BOOT):
    codes, uniq = pd.factorize(pd.Series(dates).dt.normalize())
    groups = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(M.RNG_SEED)
    for _ in range(n):
        yield np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def ci(a):
    a = np.asarray(a, float)
    return [float(np.quantile(a, .025)), float(np.quantile(a, .975))]


def p_two(a):
    a = np.asarray(a, float)
    return float(min(1.0, 2 * min((a <= 0).mean(), (a >= 0).mean())))


# ------------------------------------------------------------------ analysis

def slopes(g, market, close=False):
    """beta per model with day-bootstrap CIs, plus paired differences vs with_odds, for one period frame g."""
    arrs = {t: frame(g, market, t, close) for t in RUNS}
    est = {t: beta(market, a[0], a[1], a[2]) for t, a in arrs.items()}
    boots = {t: [] for t in RUNS}
    for idx in day_draws(g['gameDate']):
        for t, a in arrs.items():
            boots[t].append(beta(market, a[0][idx], a[1][idx], a[2][idx]))
    out = {t: {'beta': est[t], 'ci': ci(boots[t])} for t in RUNS}
    for t in ('no_odds', 'residual'):
        d = np.array(boots[t]) - np.array(boots['with_odds'])
        out[f'{t}_minus_with_odds'] = {'diff': est[t] - est['with_odds'], 'ci': ci(d), 'p': p_two(d)}
    if market != 'ml':
        for t, a in arrs.items():
            e, r, _, _ = a
            out[t]['err_corr'] = float(np.corrcoef(r - e, r)[0, 1])      # corr(actual - model, actual - line)
            out[t]['edge_sd'] = float(e.std())
    return out


def blend(g, market, close=False):
    """k fitted on discovery (no intercept), applied to confirmation; per-game loss difference vs the line."""
    d, c = g[g['seasonYear'].isin(DISC)], g[g['seasonYear'].isin(CONF)].reset_index(drop=True)
    out = {}
    for t in RUNS:
        e_d, r_d, L_d, _ = frame(d, market, t, close)
        e_c, r_c, L_c, _ = frame(c, market, t, close)
        k = k_fit(market, e_d, r_d, L_d)
        diff = loss(market, r_c, L_c, k * e_c) - loss(market, r_c, L_c, 0.0)
        boots = [diff[idx].mean() for idx in day_draws(c['gameDate'])]
        out[t] = {'k_discovery': k, 'line_loss': float(loss(market, r_c, L_c, 0.0).mean()), 'blend_minus_line': float(diff.mean()),
                  'ci': ci(boots), 'p': p_two(boots), 'diff_series': diff}
    for a, b in (('no_odds', 'with_odds'), ('no_odds', 'residual')):
        dd = out[a]['diff_series'] - out[b]['diff_series']
        boots = [dd[idx].mean() for idx in day_draws(c['gameDate'])]
        out[f'{a}_minus_{b}'] = {'diff': float(dd.mean()), 'ci': ci(boots), 'p': p_two(boots)}
    for t in RUNS:
        out[t].pop('diff_series')
    return out


def walk_forward_k(g, market):
    out = {}
    for t in RUNS:
        rows = {}
        for i, s in enumerate(TEST_SEASONS[1:], 1):
            prev, cur = g[g['seasonYear'].isin(TEST_SEASONS[:i])], g[g['seasonYear'] == s]
            e_p, r_p, L_p, _ = frame(prev, market, t)
            e_c, r_c, L_c, _ = frame(cur, market, t)
            k = k_fit(market, e_p, r_p, L_p)
            rows[s] = {'k': round(k, 4), 'blend_minus_line': float((loss(market, r_c, L_c, k * e_c) - loss(market, r_c, L_c, 0.0)).mean())}
        out[t] = rows
    return out


def deciles(g, market, tag):
    e, r, _, _ = frame(g, market, tag)
    q = pd.qcut(e, 10, labels=False, duplicates='drop')
    return pd.DataFrame({'e': e, 'r': r, 'q': q}).groupby('q').agg(mean_edge=('e', 'mean'), mean_resid=('r', 'mean'), n=('r', 'size')).round(3).to_dict('index')


def holm(ps):
    order = sorted(ps, key=ps.get)
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


def main():
    games = load()
    res = {'validity': validity(games)}
    res['validity_all_pass'] = all(v['pass'] for k, v in res['validity'].items() if 'pass' in v)
    for market in MARKETS:
        g = common_rows(games, market)
        mres = {'n': len(g)}
        for lab, seas in (('discovery', DISC), ('confirmation', CONF)):
            mres[lab] = slopes(g[g['seasonYear'].isin(seas)].reset_index(drop=True), market)
        mres['per_season_beta'] = {s: {t: round(beta(market, *frame(gs, market, t)[:3]), 4) for t in RUNS}
                                   for s, gs in g.groupby('seasonYear')}
        mres['blend'] = blend(g, market)
        mres['walk_forward_k'] = walk_forward_k(g, market)
        mres['deciles_confirmation'] = {t: deciles(g[g['seasonYear'].isin(CONF)], market, t) for t in RUNS}
        gr = g[g['regular']].reset_index(drop=True)
        mres['regular_season_only'] = {'confirmation': slopes(gr[gr['seasonYear'].isin(CONF)].reset_index(drop=True), market),
                                       'blend': blend(gr, market)}
        gc = common_rows(games, market, close=True)
        mres['vs_close'] = {'n': len(gc), 'confirmation': slopes(gc[gc['seasonYear'].isin(CONF)].reset_index(drop=True), market, close=True),
                            'blend': blend(gc, market, close=True)}
        res[market] = mres
    ps = {mk: res[mk]['blend']['no_odds']['p'] for mk in MARKETS}
    adj = holm(ps)
    dec = {}
    for mk in MARKETS:
        conf = res[mk]['confirmation']
        b = res[mk]['blend']['no_odds']
        dec[mk] = {
            'edges_informative': {t: conf[t]['ci'][0] > 0 for t in RUNS},
            'edges_overstated': {t: conf[t]['ci'][1] < 1 for t in RUNS},
            'H_DECOR': bool(conf['no_odds_minus_with_odds']['ci'][0] > 0 and b['ci'][1] < 0),
            'blend_p': b['p'], 'blend_p_holm': adj[mk]}
    res['decisions'] = dec
    out = OUT / NAME
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(json.dumps({'validity_all_pass': res['validity_all_pass'], 'decisions': dec}, indent=1))


if __name__ == '__main__':
    main()
