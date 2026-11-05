"""E28 - residual screen of the tracking / fatigue families. Registered in reports/experiments/PREREGISTRATION.md ("E28") before any
residual or mechanism regression was computed.

For each family F1-F5 and each target (total, margin, home win):
  (a) residual of the deployed-form out-of-sample prediction   (find 2019-20..2021-22)
  (b) residual against the opening line                        (find 2013-14..2021-22; home win from 2019-20)
  (c) mechanism: the team's own 3P%, FT%, rim FG%, pace and points per 100 possessions (find 2013-14..2021-22)
Confirmation 2022-23..2025-26 for all. python -m experiments.e28_screen  ->  reports/experiments/E28_tracking/screen.{md,json}
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import features_tracking as ft          # noqa: E402
from experiments import metrics as M                      # noqa: E402
from experiments.e28_reference import deployed_form       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports' / 'experiments' / 'E28_tracking'
VARIANT = ROOT / 'data' / 'processed' / 'experiments' / 'e28_training_set.csv'
MECH = ROOT / 'data' / 'processed' / 'experiments' / 'e28_mechanism.csv'
N_BOOT = 1000
FIND_A = ['2019-20', '2020-21', '2021-22']
FIND_B = ['2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19'] + FIND_A
ARCHIVE = FIND_B[:6]
CONF = ['2022-23', '2023-24', '2024-25', '2025-26']
FAMS = ['F1', 'F2', 'F3', 'F4', 'F5']
TARGETS = ['total', 'margin', 'win']
MECH_OUTCOMES = {'3P%': 'mech_3p_pct', 'FT%': 'mech_ft_pct', 'rim FG%': 'mech_rim_pct', 'pace': 'mech_pace', 'points/100 poss': 'mech_ppp'}
RIDGE = 1e-9
FIND_P = 0.10
BH_Q = 0.10
MECH_P = 0.05 / 3


# ------------------------------------------------------------------ bootstrap and regression
class Boot:
    """Resampling of whole game days: draw b gives every row the count of its day in that draw."""

    def __init__(self, dates, n_boot=None, seed=M.RNG_SEED):
        n_boot = N_BOOT if n_boot is None else n_boot
        self.codes, uniq = pd.factorize(pd.to_datetime(pd.Series(np.asarray(dates))))
        nd = len(uniq)
        rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(rng.integers(0, nd, nd), minlength=nd) for _ in range(n_boot)]).astype(np.float32)

    def row_weights(self, b):
        return self.W[b][self.codes].astype(float)


def ols_fit(X, y, w):
    Xw = X * w[:, None]
    return np.linalg.solve(X.T @ Xw + RIDGE * np.eye(X.shape[1]) * max(1.0, w.sum()), Xw.T @ y)


def logit_fit(X, y, off, w, beta0=None, iters=25, tol=1e-9):
    beta = np.zeros(X.shape[1]) if beta0 is None else beta0.copy()
    for _ in range(iters):
        p = expit(X @ beta + off)
        H = X.T @ (X * (p * (1 - p) * w)[:, None]) + RIDGE * np.eye(X.shape[1]) * max(1.0, w.sum())
        step = np.linalg.solve(H, X.T @ ((y - p) * w))
        beta += step
        if np.abs(step).max() < tol:
            break
    return beta


def fit_boot(X, y, off, kind, dates, n_newton=3):
    """Full-sample fit and bootstrap draws of all coefficients (logistic draws: Newton steps from the full-sample fit)."""
    boot = Boot(dates)
    n = len(y)
    ones = np.ones(n)
    if kind == 'ols':
        beta = ols_fit(X, y, ones)
        draws = np.stack([ols_fit(X, y, boot.row_weights(b)) for b in range(len(boot.W))])
    else:
        beta = logit_fit(X, y, off, ones)
        draws = np.stack([logit_fit(X, y, off, boot.row_weights(b), beta0=beta, iters=n_newton) for b in range(len(boot.W))])
    return beta, draws


def wald(beta, draws):
    """Joint Wald test with the bootstrap covariance; df = numerical rank of the covariance."""
    if len(beta) == 1:
        v = float(np.var(draws[:, 0], ddof=1))
        return float(beta[0] ** 2 / v), 1, float(chi2.sf(beta[0] ** 2 / v, 1))
    cov = np.cov(draws.T)
    ev, evec = np.linalg.eigh(cov)
    keep = ev > 1e-8 * ev.max()
    proj = evec[:, keep].T @ beta
    stat = float(np.sum(proj ** 2 / ev[keep]))
    return stat, int(keep.sum()), float(chi2.sf(stat, int(keep.sum())))


def season_dummies(seasons, levels):
    return np.column_stack([(np.asarray(seasons) == s).astype(float) for s in levels])


def bh(pvals):
    """Benjamini-Hochberg q-values (dict in, dict out)."""
    keys = list(pvals)
    p = np.array([pvals[k] for k in keys])
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in zip(range(m, 0, -1), order[::-1]):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return dict(zip(keys, q))


# ------------------------------------------------------------------ frames
def feature_frame():
    v = pd.read_csv(VARIANT, low_memory=False)
    v['gameId'] = v['gameId'].astype(str).str.zfill(10)
    v['gameDate'] = pd.to_datetime(v['gameDate'])
    return v


def frame_a(v):
    g = deployed_form()
    g = g[['gameId', 'total_pred', 'spread_pred', 'ml_stack']]
    f = v.merge(g, on='gameId', how='inner')
    f['y_total'] = f['target_total_pts'] - f['total_pred']
    f['y_margin'] = f['target_point_diff'] - f['spread_pred']
    f['y_win'] = f['target_home_win'].astype(float)
    f['off_win'] = logit(np.clip(f['ml_stack'].to_numpy(float), 1e-4, 1 - 1e-4))
    return f


def frame_b(v):
    from core import odds_clean as oc
    f = v.copy()
    bad = ~f['odds_total'].between(*oc.TOTAL_RANGE) | (f['odds_home_spread'].abs() > oc.MAX_ABS_SPREAD) | f['odds_total'].isna() | f['odds_home_spread'].isna()
    f = f[~bad].copy()
    f['y_total'] = f['target_total_pts'] - f['odds_total']
    f['y_margin'] = f['target_point_diff'] + f['odds_home_spread']
    f['y_win'] = f['target_home_win'].astype(float)
    op = pd.read_csv(ROOT / 'reports' / 'experiments' / 'E25_with_odds' / 'predictions.csv', dtype={'gameId': str}, usecols=['gameId', 'open_home_prob'])
    f = f.merge(op, on='gameId', how='left')
    f['off_win'] = logit(np.clip(f['open_home_prob'].to_numpy(float), 1e-4, 1 - 1e-4))
    return f


def family_X(f, family, target):
    cols = {}
    for n in ft.FAMILIES[family]:
        if n == 'one_side_tired':
            v = f['trk_one_side_tired']
            cols[n] = v.abs() if target == 'total' else v
        elif target == 'total':
            cols[n] = f[f'trk_{n}_home'] + f[f'trk_{n}_away']
        else:
            cols[n] = f[f'diff_trk_{n}']
    return pd.DataFrame(cols)


# ------------------------------------------------------------------ one test
def run_test(f, family, target, find_seasons, conf_seasons, include_playoffs=False, conf_extra=True, controls=(), alt_conf_y=None):
    """controls: extra (standardised) columns of `f` kept in every fit but outside the tested block (exploratory use only).
    alt_conf_y: {name: column} of other residuals; the find-period signal is applied to each in the confirmation seasons."""
    kind = 'logit' if target == 'win' else 'ols'
    X = family_X(f, family, target)
    y = f[f'y_{target}'].to_numpy(float)
    off = f['off_win'].to_numpy(float) if target == 'win' else np.zeros(len(f))
    C = f[list(controls)].astype(float) if len(controls) else pd.DataFrame(index=f.index)
    ok = X.notna().all(axis=1).to_numpy() & np.isfinite(y) & np.isfinite(off) & C.notna().all(axis=1).to_numpy()
    if not include_playoffs:
        ok &= (f['gameId'].str[:3] == '002').to_numpy()
    S = f['seasonYear'].to_numpy()
    fi = ok & np.isin(S, find_seasons)
    ci = ok & np.isin(S, conf_seasons)
    res = {'family': family, 'target': target, 'n_find': int(fi.sum()), 'n_conf': int(ci.sum()), 'n_dropped_missing_find': int((~X.notna().all(axis=1).to_numpy() & np.isin(S, find_seasons)).sum())}
    mu, sd = X[fi].mean(), X[fi].std()
    use = [c for c in X.columns if sd[c] > 0]
    res['columns'] = use
    Z = ((X[use] - mu[use]) / sd[use]).to_numpy(float)
    Cz = ((C - C[fi].mean()) / C[fi].std()).to_numpy(float) if len(controls) else np.zeros((len(f), 0))
    levels_f = [s for s in find_seasons if (S[fi] == s).any()]
    Df = np.column_stack([season_dummies(S[fi], levels_f), Z[fi], Cz[fi]])
    ns = len(levels_f)
    t0 = time.time()
    beta, draws = fit_boot(Df, y[fi], off[fi], kind, f['gameDate'].to_numpy()[fi])
    k_ = len(use)
    bf, Bf = beta[ns:ns + k_], draws[:, ns:ns + k_]
    w_stat, df_, p_find = wald(bf, Bf)
    res.update({'wald_find': w_stat, 'df': df_, 'p_find': p_find})
    res['coefs'] = {c: {'beta': float(bf[i]), 'lo': float(np.quantile(Bf[:, i], .025)), 'hi': float(np.quantile(Bf[:, i], .975))} for i, c in enumerate(use)}
    sig_f = Z[fi] @ bf
    s_sd = float(sig_f.std())
    sig = (Z @ bf) / s_sd
    levels_c = [s for s in conf_seasons if (S[ci] == s).any()]
    Dc = np.column_stack([season_dummies(S[ci], levels_c), sig[ci], Cz[ci]])
    gb, gdraws = fit_boot(Dc, y[ci], off[ci], kind, f['gameDate'].to_numpy()[ci])
    g_idx = len(levels_c)
    g, G = float(gb[g_idx]), gdraws[:, g_idx]
    res.update({'gamma': g, 'gamma_lo': float(np.quantile(G, .025)), 'gamma_hi': float(np.quantile(G, .975)),
                'p_one': float((1 + (G <= 0).sum()) / (len(G) + 1)), 'signal_sd_find_units': s_sd})
    per = {}
    for s in levels_c:
        m = S[ci] == s
        Ds = np.column_stack([np.ones(m.sum()), sig[ci][m], Cz[ci][m]])
        b = ols_fit(Ds, y[ci][m], np.ones(m.sum())) if kind == 'ols' else logit_fit(Ds, y[ci][m], off[ci][m], np.ones(m.sum()))
        per[s] = float(b[-1])
    res['gamma_by_season'] = per
    if conf_extra:
        Dcz = np.column_stack([season_dummies(S[ci], levels_c), Z[ci], Cz[ci]])
        cb, cd = fit_boot(Dcz, y[ci], off[ci], kind, f['gameDate'].to_numpy()[ci])
        res['wald_conf_p'] = wald(cb[len(levels_c):len(levels_c) + k_], cd[:, len(levels_c):len(levels_c) + k_])[2]
    for name, col in (alt_conf_y or {}).items():
        ya = f[col].to_numpy(float)
        cj = ci & np.isfinite(ya)
        Da = np.column_stack([season_dummies(S[cj], levels_c), sig[cj], Cz[cj]])
        ab, ad = fit_boot(Da, ya[cj], off[cj], kind, f['gameDate'].to_numpy()[cj])
        res.setdefault('alt', {})[name] = {'n': int(cj.sum()), 'gamma': float(ab[g_idx]), 'lo': float(np.quantile(ad[:, g_idx], .025)), 'hi': float(np.quantile(ad[:, g_idx], .975)),
                                           'p_one': float((1 + (ad[:, g_idx] <= 0).sum()) / (len(ad) + 1))}
    res['secs'] = round(time.time() - t0, 1)
    return res


def archive_only_p(f, family, target):
    """Wald p on the six archive seasons alone (sensitivity; decides nothing)."""
    kind = 'logit' if target == 'win' else 'ols'
    X = family_X(f, family, target)
    y = f[f'y_{target}'].to_numpy(float)
    ok = X.notna().all(axis=1).to_numpy() & np.isfinite(y) & (f['gameId'].str[:3] == '002').to_numpy() & np.isin(f['seasonYear'].to_numpy(), ARCHIVE)
    sd = X[ok].std()
    use = [c for c in X.columns if sd[c] > 0]
    Z = ((X[use] - X[ok][use].mean()) / sd[use]).to_numpy(float)
    S = f['seasonYear'].to_numpy()[ok]
    lv = [s for s in ARCHIVE if (S == s).any()]
    D = np.column_stack([season_dummies(S, lv), Z[ok]])
    beta, draws = fit_boot(D, y[ok], np.zeros(ok.sum()), kind, f['gameDate'].to_numpy()[ok])
    return {'n': int(ok.sum()), 'p': wald(beta[len(lv):], draws[:, len(lv):])[2]}


# ------------------------------------------------------------------ mechanism
def mech_frame(v):
    m = pd.read_csv(MECH, dtype={'gameId': str})
    m['gameId'] = m['gameId'].str.zfill(10)
    rows = []
    base = v[['gameId', 'gameDate', 'seasonYear', 'teamId_home', 'teamId_away', 'trk_one_side_tired']].copy()
    for side, other in (('home', 'away'), ('away', 'home')):
        r = base[['gameId', 'gameDate', 'seasonYear']].copy()
        r['teamId'] = base[f'teamId_{side}'].astype('int64')
        r['is_home'] = float(side == 'home')
        for fam in ft.FAMILIES:
            for n in ft.FAMILIES[fam]:
                if n == 'one_side_tired':
                    r['own_one_side_tired'] = base['trk_one_side_tired'] * (1 if side == 'home' else -1)
                else:
                    r[f'own_{n}'] = v[f'trk_{n}_{side}']
                    r[f'opp_{n}'] = v[f'trk_{n}_{other}']
        rows.append(r)
    r = pd.concat(rows, ignore_index=True).merge(m, on=['gameId', 'teamId'], how='left')
    for c in ('mech_3p_pct', 'mech_ft_pct', 'mech_rim_pct', 'mech_ppp'):
        r[c] = r[c] * 100.0
    return r


def run_mech(r, family, outcome_col, find_seasons=FIND_B, conf_seasons=CONF, controls=()):
    names = ft.FAMILIES[family]
    own = [f'own_{n}' for n in names]
    opp = [f'opp_{n}' for n in names if n != 'one_side_tired']
    y = r[outcome_col].to_numpy(float)
    S = r['seasonYear'].to_numpy()
    X = r[own + opp]
    C = r[list(controls)].astype(float) if len(controls) else pd.DataFrame(index=r.index)
    ok = X.notna().all(axis=1).to_numpy() & np.isfinite(y) & (r['gameId'].str[:3] == '002').to_numpy() & C.notna().all(axis=1).to_numpy()
    fi, ci = ok & np.isin(S, find_seasons), ok & np.isin(S, conf_seasons)
    Cz = ((C - C[fi].mean()) / C[fi].std()).to_numpy(float) if len(controls) else np.zeros((len(r), 0))
    mu, sd = r.loc[fi, own].mean(), r.loc[fi, own].std()
    use_own = [c for c in own if sd[c] > 0]
    Zown = ((r[use_own] - mu[use_own]) / sd[use_own]).to_numpy(float)
    use_opp = [c for c in opp if c.replace('opp_', 'own_') in use_own]
    Zopp = ((r[use_opp] - mu[[c.replace('opp_', 'own_') for c in use_opp]].to_numpy()) / sd[[c.replace('opp_', 'own_') for c in use_opp]].to_numpy()).to_numpy(float)
    H = r['is_home'].to_numpy(float)
    lv_f = [s for s in find_seasons if (S[fi] == s).any()]
    D = np.column_stack([season_dummies(S[fi], lv_f), H[fi], Zown[fi], Zopp[fi], Cz[fi]])
    k = len(use_own)
    base_cols = len(lv_f) + 1
    beta, draws = fit_boot(D, y[fi], np.zeros(fi.sum()), 'ols', r['gameDate'].to_numpy()[fi])
    bo, Bo = beta[base_cols:base_cols + k], draws[:, base_cols:base_cols + k]
    w_stat, df_, p_find = wald(bo, Bo)
    sig_own_f = Zown[fi] @ bo
    s_sd = float(sig_own_f.std())
    s_own = (Zown @ bo) / s_sd
    s_opp = (Zopp @ beta[base_cols + k:base_cols + k + len(use_opp)]) / s_sd if len(use_opp) else np.zeros(len(r))
    lv_c = [s for s in conf_seasons if (S[ci] == s).any()]
    Dc = np.column_stack([season_dummies(S[ci], lv_c), H[ci], s_opp[ci], s_own[ci], Cz[ci]])
    gb, gd = fit_boot(Dc, y[ci], np.zeros(ci.sum()), 'ols', r['gameDate'].to_numpy()[ci])
    gi = len(lv_c) + 2
    G = gd[:, gi]
    return {'family': family, 'outcome': outcome_col, 'n_find': int(fi.sum()), 'n_conf': int(ci.sum()), 'df': df_, 'p_find': p_find,
            'coefs': {c.replace('own_', ''): {'beta': float(bo[i]), 'lo': float(np.quantile(Bo[:, i], .025)), 'hi': float(np.quantile(Bo[:, i], .975))} for i, c in enumerate(use_own)},
            'gamma': float(gb[gi]), 'gamma_lo': float(np.quantile(G, .025)), 'gamma_hi': float(np.quantile(G, .975)), 'p_one': float((1 + (G <= 0).sum()) / (len(G) + 1))}


# ------------------------------------------------------------------ tiers and report
def tiers(results):
    """BH over the 15 family x target confirmation p-values of one residual type; failed-find families enter with p = 1."""
    pv = {k: (r['p_one'] if (r['p_find'] < FIND_P and r['gamma'] > 0) else 1.0) for k, r in results.items()}
    q = bh(pv)
    for k, r in results.items():
        r['q'] = q[k]
        if r['p_find'] >= FIND_P:
            r['tier'] = 'Reject (fails find step)'
        elif r['gamma'] <= 0:
            r['tier'] = 'Reject (opposite sign in confirmation)'
        elif q[k] < BH_Q:
            r['tier'] = 'CANDIDATE'
        else:
            r['tier'] = 'Reject (not confirmed, q >= 0.10)'
    return results


def table(res, kind):
    rows = []
    for k, r in res.items():
        rows.append({'family': r['family'], 'target': r['target'], 'n find': r['n_find'], 'n conf': r['n_conf'], 'df': r['df'], 'find Wald p': round(r['p_find'], 4),
                     'gamma': round(r['gamma'], 3), 'gamma 95% CI': f"[{r['gamma_lo']:.2f}, {r['gamma_hi']:.2f}]", 'one-sided p': round(r['p_one'], 4),
                     'q (BH)': round(r['q'], 3), 'conf Wald p': round(r.get('wald_conf_p', np.nan), 4), 'tier': r['tier']})
    return pd.DataFrame(rows)


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(out) + '\n'


def coef_table(res):
    rows = []
    for k, r in res.items():
        for c, d in r['coefs'].items():
            rows.append({'residual': k.split('|')[0], 'family': r['family'], 'target': r['target'], 'feature': c, 'per 1 SD': round(d['beta'], 4), '95% CI': f"[{d['lo']:.3f}, {d['hi']:.3f}]"})
    return pd.DataFrame(rows)


def main(quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    v = feature_frame()
    fa, fb = frame_a(v), frame_b(v)
    results = {'a': {}, 'b': {}}
    for rtype, fr in (('a', fa), ('b', fb)):
        for fam in FAMS:
            for tg in TARGETS:
                find = FIND_A if (rtype == 'a' or tg == 'win') else FIND_B
                key = f'{rtype}|{fam}|{tg}'
                results[rtype][key] = run_test(fr, fam, tg, find, CONF)
                r = results[rtype][key]
                print(key, 'n', r['n_find'], r['n_conf'], 'p_find', round(r['p_find'], 4), 'gamma', round(r['gamma'], 3), 'p1', round(r['p_one'], 4), f"{r['secs']}s", flush=True)
        results[rtype] = tiers(results[rtype])
    sens = {'archive_only': {f'{fam}|{tg}': archive_only_p(fb, fam, tg) for fam in FAMS for tg in ('total', 'margin')}}
    sens['playoffs_included'] = {}
    for rtype, fr in (('a', fa), ('b', fb)):
        for fam in FAMS:
            for tg in TARGETS:
                find = FIND_A if (rtype == 'a' or tg == 'win') else FIND_B
                r = run_test(fr, fam, tg, find, CONF, include_playoffs=True, conf_extra=False)
                sens['playoffs_included'][f'{rtype}|{fam}|{tg}'] = {k: r[k] for k in ('n_find', 'n_conf', 'p_find', 'gamma', 'p_one')}
    mech_df = mech_frame(v)
    mech = {}
    for fam in FAMS:
        for lab, col in MECH_OUTCOMES.items():
            mech[f'{fam}|{lab}'] = run_mech(mech_df, fam, col)
            r = mech[f'{fam}|{lab}']
            print('mech', fam, lab, 'p_find', round(r['p_find'], 4), 'gamma', round(r['gamma'], 3), 'p1', round(r['p_one'], 4), flush=True)
    for k, r in mech.items():
        r['find_pass'] = bool(r['p_find'] < FIND_P)
        r['replicates'] = bool(r['gamma'] > 0 and r['p_one'] < MECH_P)
    story = {}
    for fam in ('F1', 'F2'):
        story[fam] = bool(any(mech[f'{fam}|{lab}']['find_pass'] and mech[f'{fam}|{lab}']['replicates'] for lab in ('3P%', 'FT%', 'pace')))
    (OUT / 'screen.json').write_text(json.dumps({'results': results, 'sensitivity': sens, 'mechanism': mech, 'fatigue_story_supported': story}, indent=1, default=float), encoding='utf-8')
    write_report(results, sens, mech, story, v)


def write_report(results, sens, mech, story, v):
    L = ['# E28 - residual screen of the tracking / fatigue families', '',
         'Registered in `reports/experiments/PREREGISTRATION.md` ("E28") before any residual was computed. Built by `experiments/e28_screen.py`; '
         'raw numbers in `screen.json`. Find: (a) 2019-20..2021-22, (b) total and margin 2013-14..2021-22 (the first six from the archive lines), home win 2019-20..2021-22; '
         'confirmation 2022-23..2025-26. Regular season only. `gamma` = slope of the confirmation residual on the find-period signal standardised to the find-period SD: points of margin or total (log-odds for the home win) per SD of the signal; full replication of the find fit would give `signal_sd_find_units` (about 0.4-0.5 points for the margin signals, 1.5 for a|F1|total). Day-cluster bootstrap, 1,000 draws.', '']
    for rtype, title in (('a', '(a) residual of the deployed-form out-of-sample prediction'), ('b', '(b) residual against the opening line')):
        L += [f'## {title}', '', md(table(results[rtype], rtype))]
        cand = [k for k, r in results[rtype].items() if r['tier'] == 'CANDIDATE']
        L += [f'Candidates: {cand if cand else "none"}', '']
    L += ['## (c) mechanism: own team\'s shooting, pace and efficiency in game t', '']
    rows = [{'family': r['family'], 'outcome': k.split('|')[1], 'n find': r['n_find'], 'df': r['df'], 'find Wald p': round(r['p_find'], 4), 'gamma': round(r['gamma'], 3),
             'gamma 95% CI': f"[{r['gamma_lo']:.2f}, {r['gamma_hi']:.2f}]", 'one-sided p': round(r['p_one'], 4), 'replicates (p < 0.0167)': r['replicates']} for k, r in mech.items()]
    L += [md(pd.DataFrame(rows)), f'**Fatigue story (F1, F2 move 3P%, FT% or pace, rule registered):** {story}', '']
    L += ['## Coefficients in plain units (find period; points per 1 SD for total and margin, log-odds per 1 SD for home win)', '', md(coef_table(results['a'])), md(coef_table(results['b']))]
    L += ['## Mechanism coefficients (find period; percentage points / possessions per 1 SD of the own-team feature)', '']
    mr = [{'family': r['family'], 'outcome': k.split('|')[1], 'feature': c, 'per 1 SD': round(d['beta'], 4), '95% CI': f"[{d['lo']:.3f}, {d['hi']:.3f}]"} for k, r in mech.items() for c, d in r['coefs'].items()]
    L += [md(pd.DataFrame(mr))]
    L += ['## Sensitivity (decides nothing)', '', 'Wald p on the six archive seasons alone, (b) total and margin:', '', md(pd.DataFrame([{'test': k, **d} for k, d in sens['archive_only'].items()])),
          'Playoffs and play-in added:', '', md(pd.DataFrame([{'test': k, **{a: (round(b, 4) if isinstance(b, float) else b) for a, b in d.items()}} for k, d in sens['playoffs_included'].items()]))]
    (OUT / 'screen.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
