"""
Pure functions for steps S5, S6, S7 and S16 of reports/props/audit_2026-11/WP6_WP7_experiment_plan.md (sections 6b = C, 7c, 7d). No file or database access here;
the scripts scripts/a11_k_estimation.py and scripts/a11_rule_eval.py load the inputs and call these.

Notation (plan C.1): one row per (game, player, slot T, stat) of `market_bench.compare`, g = p_model - p_market, r = hit - p_market.

Estimators (plan C.2), least squares per stat, date-clustered bootstrap intervals (same resample counts as props.live.monitor):
  E0  r = b g                                 (through the origin; the frozen method)
  E1  r = a + b g
  E2  r = a+ 1[g >= 0] + a- 1[g < 0] + b g     (one slope per stat, an intercept per side the model leans to)
  E3  r = a + b clip(g, -c, c)                (c shared by the bet stats, profile least squares over C_GRID)
  E4  E1 per slot (reported only)
Every slope is then clipped by rules.clip_k (floor 0, cap 0.5). When the clip binds, the intercept(s) are RE-FITTED by least squares with the slope fixed at the
clipped value (a = mean(r) - b_clipped * mean(x), per intercept cell); that is the constrained least-squares optimum. Unclipped fits are unchanged by this.

Candidate rules (plan C.4): p* for both sides of every quote, then EV, side and selection through the same code (`add_ev_pstar` mirrors rules.add_ev; rules.select).
  C0 p_mkt + b0 g       C1 p_mkt + b1 g       C2 clip(p_mkt + a + b1 g)       C3 clip(p_mkt + a^{side(g)} + b2 g)       C4 clip(p_mkt + a + b3 clip(g, -c, c))
  P0 clip(p_mkt + a) with a from E1 (C2's intercept)       P00 p_mkt          clip = [0.01, 0.99]
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from props.betting import evaluate as be, rules, settle
from props.live import monitor as mon

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Fixed constants of the plan (written before any data is computed)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

BET_STATS = ('rebounds', 'assists', 'threes', 'ra')                # config/props_frozen_rule.json "stats"
PRIMARY = {'e': 0.03, 'time': 'earliest', 'min_books': 3}           # C.4: e = 0.03, earliest slot, 3 books
N_BOOT = 2000                                                       # A.3.2
SEED = 20261112                                                     # A.3.2
N_COMPARISONS = 23                                                  # A.5
ALPHA = 0.05
ALPHA_BONF = 0.05 / N_COMPARISONS                                   # 99.78% intervals
C_GRID = tuple(round(0.02 + 0.01 * i, 2) for i in range(11))        # E3: {0.02, 0.03, ..., 0.12}
P_CLIP = (0.01, 0.99)                                               # C2-C4, P0
FLAT_DEC = 1.0 + 100.0 / 110.0                                      # flat -110
TRIED_RULES = 60                                                    # C.6 / A.5: 37 + 23
LAST_ALLOWED_DATE = '2025-06-30'                                    # A.1
HOLDOUT_SEASON = '2025-26'
CANDIDATES = ('C0', 'C1', 'C2', 'C3', 'C4', 'P0', 'P00')
CAND_ESTIMATOR = {'C0': 'E0', 'C1': 'E1', 'C2': 'E1', 'C3': 'E2', 'C4': 'E3', 'P0': 'E1', 'P00': None}
EV_EDGES = [0.03, 0.04, 0.05, 0.07, np.inf]
EV_LABELS = ['3-4%', '4-5%', '5-7%', '7%+']

# C.5 keep rule (amended 2026-11-12: +0.25 points in items 2-3, so also item 4). ROI in fractions: 0.0025 = 0.25 points.
C1_WORSE_POINT = -0.005                                             # item 1: point below -0.5 points AND 95% upper bound below 0
ADOPT_MARGIN = 0.0025                                               # items 2-4 (amended, was 0.005)
C3_MIN_LO = -0.01                                                   # items 3-4: 95% lower bound at least -1.0 point
CAL_MAX_SE = 2.0                                                    # item 2: calibration by side within 2 clustered SE
# 7c (item 2 and 6; unchanged by the amendment)
KELLY = {'fraction': 0.25, 'max_bet': 0.02, 'bankroll': 100.0, 'max_game': 0.05}
CAP_ROI_TOL = 0.005
KELLY_SHARPE_MULT = 1.10
KELLY_ROI_TOL = 0.005
# 7d (item 2 and 5; amended bound: unadjusted 95%)
MINUTES_SPLIT = 20.0


class ExcludedSeasonError(SystemExit):
    """Raised for 2025-26 or later, or a date range ending after 2025-06-30 (plan A.1)."""


def check_season(season):
    s = str(season)
    if not re.fullmatch(r'\d{4}-\d{2}', s):
        raise ValueError(f'season {s!r} is not of the form YYYY-YY')
    if s >= HOLDOUT_SEASON:
        raise ExcludedSeasonError(f'season {s} refused: {HOLDOUT_SEASON} and later seasons are excluded from every rule experiment (plan A.1)')
    return s


def check_seasons(seasons):
    return [check_season(s) for s in seasons]


def check_dates(dates, what='rows'):
    d = pd.Series(np.asarray(dates)).dropna().astype(str).str[:10]
    if len(d) and d.max() > LAST_ALLOWED_DATE:
        raise ExcludedSeasonError(f'{what} reach {d.max()}, after {LAST_ALLOWED_DATE}: refused (plan A.1)')


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def xy_frame(cmp):
    """market_bench.compare rows (+ game_date) -> stat, game_date, minutes_before, g, r."""
    d = cmp[['stat', 'game_date', 'minutes_before', 'p_model', 'p_market', 'hit']].dropna().copy()
    d['game_date'] = d['game_date'].astype(str).str[:10]
    d['g'] = d['p_model'].to_numpy(float) - d['p_market'].to_numpy(float)
    d['r'] = d['hit'].to_numpy(float) - d['p_market'].to_numpy(float)
    return d.reset_index(drop=True)


def _sums(x, y, date_codes, n_dates, cell_codes, n_cells):
    """(n_dates, n_cells, 6) sums of (1, x, y, xx, xy, yy) per date and cell."""
    flat = date_codes * n_cells + cell_codes
    T = np.empty((n_dates * n_cells, 6))
    for j, v in enumerate((np.ones_like(x), x, y, x * x, x * y, y * y)):
        T[:, j] = np.bincount(flat, weights=v, minlength=n_dates * n_cells)
    return T.reshape(n_dates, n_cells, 6)


def _centred(T):
    n, sx, sy, sxx, sxy, syy = (T[..., i] for i in range(6))
    nn = np.where(n > 0, n, 1.0)
    return n, sx, sy, np.where(n > 0, sxx - sx * sx / nn, 0.0), np.where(n > 0, sxy - sx * sy / nn, 0.0), np.where(n > 0, syy - sy * sy / nn, 0.0)


def _group_slopes(T, cell_group, n_groups, intercept=True):
    """Slope per group of cells (common slope, one intercept per cell when `intercept`), and the pooled slope over all cells. T: (..., K, 6)."""
    M = np.zeros((len(cell_group), n_groups))
    M[np.arange(len(cell_group)), cell_group] = 1.0
    if intercept:
        _, _, _, sxx, sxy, _ = _centred(T)
    else:
        sxx, sxy = T[..., 3], T[..., 4]
    with np.errstate(divide='ignore', invalid='ignore'):
        b = (sxy @ M) / (sxx @ M)
        pooled = sxy.sum(-1) / sxx.sum(-1)
    return b, pooled


def _intercepts(T, b_cell):
    n, sx, sy = T[..., 0], T[..., 1], T[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(n > 0, sy / np.where(n > 0, n, 1.0) - b_cell * sx / np.where(n > 0, n, 1.0), np.nan)


def _ssr(T, b_cell):
    """Residual sum of squares per cell at slope b_cell with the least-squares intercept of each cell."""
    _, _, _, sxx, sxy, syy = _centred(T)
    return syy - 2.0 * b_cell * sxy + b_cell * b_cell * sxx


def _q(draws, alpha):
    draws = np.asarray(draws, float)
    if not np.isfinite(draws).any():
        return np.nan, np.nan
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return lo, hi


def fit_estimator(xy, est, stats=None, c=None, n_boot=N_BOOT, seed=SEED, alpha=ALPHA, c_grid=C_GRID, profile_stats=BET_STATS):
    """
    Fit one estimator on an xy_frame. Returns dict(estimator, c, by_stat: DataFrame, pooled: dict(b, lo, hi), profile: DataFrame or None).
    by_stat columns: stat (and slot for E4), n, b_raw, b_lo, b_hi, b (= rules.clip_k(b_raw)), clipped, a_raw / a (re-fitted at the clipped slope) with a_lo, a_hi
    (E1, E3, E4), a_pos_raw, a_pos, a_neg_raw, a_neg with intervals (E2). pooled: slope pooled over the stats in `stats` (per-cell intercepts for E1-E4).
    E3: c is chosen on `profile_stats` (the bet stats present) by minimum total residual sum of squares over c_grid with a, b re-fitted per stat (unconstrained), unless `c` is given.
    """
    if est not in ('E0', 'E1', 'E2', 'E3', 'E4'):
        raise ValueError(est)
    d = xy if stats is None else xy[xy['stat'].isin(stats)]
    d = d.reset_index(drop=True)
    st = sorted(d['stat'].unique())
    sidx = d['stat'].map({s: i for i, s in enumerate(st)}).to_numpy()
    codes, n_dates = mon._codes(d['game_date'])
    counts = mon._boot_counts(n_dates, n_boot, seed)
    g, r = d['g'].to_numpy(float), d['r'].to_numpy(float)
    profile = None
    if est == 'E3':
        pst = [s for s in st if s in set(profile_stats)] or st
        pm = np.isin(d['stat'].to_numpy(), pst)
        if c is None:
            rows = []
            pidx = d.loc[pm, 'stat'].map({s: i for i, s in enumerate(pst)}).to_numpy()
            pcodes, pn = mon._codes(d.loc[pm, 'game_date'])
            for cc in c_grid:
                Tc = _sums(np.clip(g[pm], -cc, cc), r[pm], pcodes, pn, pidx, len(pst)).sum(0)
                b, _ = _group_slopes(Tc, np.arange(len(pst)), len(pst))
                rows.append({'c': cc, 'ssr': float(_ssr(Tc, b).sum())})
            profile = pd.DataFrame(rows)
            c = float(profile.loc[profile['ssr'].idxmin(), 'c'])            # ties: the first (smallest) c
        x = np.clip(g, -c, c)
    else:
        x = g
    if est == 'E2':
        cell = sidx * 2 + (g < 0).astype(int)                                 # cell 2s = side g >= 0 (a+), 2s + 1 = g < 0 (a-)
        n_cells, cell_group = 2 * len(st), np.repeat(np.arange(len(st)), 2)
    elif est == 'E4':
        slots = sorted(d['minutes_before'].unique())
        cell = sidx * len(slots) + d['minutes_before'].map({t: i for i, t in enumerate(slots)}).to_numpy()
        n_cells = len(st) * len(slots)
        cell_group = np.arange(n_cells)
    else:
        cell, n_cells, cell_group = sidx, len(st), np.arange(len(st))
    T = _sums(x, r, codes, n_dates, cell, n_cells)
    Tp = T.sum(0)
    Tb = (counts @ T.reshape(n_dates, -1)).reshape(n_boot, n_cells, 6)
    icpt = est != 'E0'
    n_groups = int(cell_group.max()) + 1
    b_pt, pooled_pt = _group_slopes(Tp, cell_group, n_groups, icpt)
    b_bt, pooled_bt = _group_slopes(Tb, cell_group, n_groups, icpt)
    b_lo, b_hi = _q(b_bt, alpha)
    p_lo, p_hi = _q(pooled_bt, alpha)
    b_clip = np.array([rules.clip_k(v) if np.isfinite(v) else 0.0 for v in b_pt])
    out_rows = []
    if icpt:
        a_raw = _intercepts(Tp, b_pt[cell_group])
        a_fit = _intercepts(Tp, b_clip[cell_group])
        a_bt = _intercepts(Tb, b_bt[:, cell_group])
        a_lo, a_hi = _q(a_bt, alpha)
    for gi in range(n_groups):
        cells = np.flatnonzero(cell_group == gi)
        if est == 'E4':
            si, ti = divmod(gi, len(slots))
            row = {'stat': st[si], 'slot': int(slots[ti])}
        else:
            row = {'stat': st[gi]}
        row.update({'n': int(Tp[cells, 0].sum()), 'b_raw': float(b_pt[gi]), 'b_lo': float(b_lo[gi]), 'b_hi': float(b_hi[gi]), 'b': float(b_clip[gi]),
                    'clipped': bool(np.isfinite(b_pt[gi]) and b_clip[gi] != b_pt[gi])})
        if est == 'E2':
            for j, side in ((cells[0], 'pos'), (cells[1], 'neg')):
                row[f'n_{side}'] = int(Tp[j, 0])
                row[f'a_{side}_raw'], row[f'a_{side}'] = float(a_raw[j]), float(a_fit[j]) if np.isfinite(a_fit[j]) else 0.0
                row[f'a_{side}_lo'], row[f'a_{side}_hi'] = float(a_lo[j]), float(a_hi[j])
        elif icpt:
            j = cells[0]
            row['a_raw'], row['a'], row['a_lo'], row['a_hi'] = float(a_raw[j]), float(a_fit[j]), float(a_lo[j]), float(a_hi[j])
        out_rows.append(row)
    return {'estimator': est, 'c': c if est == 'E3' else None, 'by_stat': pd.DataFrame(out_rows),
            'pooled': {'b': float(pooled_pt), 'lo': float(p_lo), 'hi': float(p_hi), 'stats': st}, 'profile': profile}


def params_from_fit(fit):
    """JSON-ready parameters of a fit: {'b': {stat: b}, 'a': ..., 'a_pos': ..., 'a_neg': ..., 'c': ..., 'b_slot': {'stat|slot': b}}."""
    t = fit['by_stat']
    est = fit['estimator']
    if est == 'E4':
        return {'b_slot': {f"{r['stat']}|{int(r['slot'])}": r['b'] for _, r in t.iterrows()}, 'a_slot': {f"{r['stat']}|{int(r['slot'])}": r['a'] for _, r in t.iterrows()}}
    p = {'b': dict(zip(t['stat'], t['b'].astype(float)))}
    if est in ('E1', 'E3'):
        p['a'] = dict(zip(t['stat'], t['a'].astype(float)))
    if est == 'E2':
        p['a_pos'] = dict(zip(t['stat'], t['a_pos'].astype(float)))
        p['a_neg'] = dict(zip(t['stat'], t['a_neg'].astype(float)))
    if est == 'E3':
        p['c'] = float(fit['c'])
    return p


def candidate_params(cand, params_by_est):
    """The parameters each candidate uses (plan C.4). params_by_est: {'E0': params_from_fit(...), ...}."""
    e = params_by_est
    if cand == 'C0':
        return {'b': e['E0']['b']}
    if cand == 'C1':
        return {'b': e['E1']['b']}
    if cand == 'C2':
        return {'b': e['E1']['b'], 'a': e['E1']['a']}
    if cand == 'C3':
        return {'b': e['E2']['b'], 'a_pos': e['E2']['a_pos'], 'a_neg': e['E2']['a_neg']}
    if cand == 'C4':
        return {'b': e['E3']['b'], 'a': e['E3']['a'], 'c': e['E3']['c']}
    if cand == 'P0':
        return {'a': e['E1']['a']}
    if cand == 'P00':
        return {}
    if cand == 'C1_E4':
        return {'b_slot': e['E4']['b_slot']}
    raise ValueError(cand)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# p*, EV, selection
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _map(q, d):
    return q['stat'].map(d or {}).fillna(0.0).to_numpy(float)          # a stat without a parameter gets 0 (as rules.add_ev's fillna(0) for k)


def p_star(cand, q, params):
    """p* (probability of the OVER) for every quote row of q (columns stat, consensus_fair_over, p_mod, minutes_before)."""
    pm = q['consensus_fair_over'].to_numpy(float)
    g = q['p_mod'].to_numpy(float) - pm
    lo, hi = P_CLIP
    if cand in ('C0', 'C1'):
        return rules.shrunk(pm, q['p_mod'].to_numpy(float), _map(q, params['b']))
    if cand == 'C1_E4':
        key = q['stat'].astype(str) + '|' + q['minutes_before'].astype(int).astype(str)
        return rules.shrunk(pm, q['p_mod'].to_numpy(float), key.map(params['b_slot']).fillna(0.0).to_numpy(float))
    if cand == 'C2':
        return np.clip(pm + _map(q, params['a']) + _map(q, params['b']) * g, lo, hi)
    if cand == 'C3':
        a = np.where(g >= 0, _map(q, params['a_pos']), _map(q, params['a_neg']))
        return np.clip(pm + a + _map(q, params['b']) * g, lo, hi)
    if cand == 'C4':
        c = float(params['c'])
        return np.clip(pm + _map(q, params['a']) + _map(q, params['b']) * np.clip(g, -c, c), lo, hi)
    if cand == 'P0':
        return np.clip(pm + _map(q, params['a']), lo, hi)
    if cand == 'P00':
        return pm.copy()
    raise ValueError(cand)


def add_ev_pstar(q, ps):
    """rules.add_ev with p* given instead of k: the same EV formula (rules.ev_over_under), side choice, best_ev and dec_price, for both sides of every quote."""
    q = q.copy()
    q['p_star'] = np.asarray(ps, float)
    q['ev_over'], q['ev_under'] = rules.ev_over_under(q['p_star'], q['push'], q['over_dec'], q['under_dec'])
    q['best_side'] = np.where(q['ev_over'] >= q['ev_under'], 'over', 'under')
    q['best_ev'] = np.maximum(q['ev_over'], q['ev_under'])
    q['dec_price'] = np.where(q['best_side'] == 'over', q['over_dec'], q['under_dec'])
    return q


def select_candidate(q, cand, params, primary=PRIMARY):
    """p* -> EV for both sides -> side -> rules.select: the whole path is recomputed for every candidate (never patched after selection)."""
    base = q.drop(columns=[c for c in ('p_star', 'ev_over', 'ev_under', 'best_side', 'best_ev', 'dec_price') if c in q.columns])
    return rules.select(add_ev_pstar(base, p_star(cand, base, params)), primary['e'], primary['time'], primary['min_books'])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Prices and settlement
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

MARKET_KEY = ['game_id', 'player_id', 'stat', 'minutes_before', 'line']


def market_prices(q):
    """The price columns of scripts/props_phase5_dev.market_level: per (game, player, stat, T, line) the median and best bettable price per side."""
    g = q.groupby(MARKET_KEY)
    return g.agg(over_med=('over_dec', 'median'), under_med=('under_dec', 'median'), over_best=('over_dec', 'max'), under_best=('under_dec', 'max')).reset_index()


def price_and_settle(sel, m):
    """
    The selected bets (chosen on best-price EV) settled at the price of the selected quote ('best'), at the median bettable price of the SAME market, side and slot ('med', plan A.3.1,
    as props_phase5_dev.decompose) and at flat -110. Adds stake_<t> (1 for win/loss, 0 for push/void), status_<t> and profit_<price>_<t> for every settlement treatment t.
    """
    s = sel.drop(columns=[c for c in sel.columns if c.startswith(('profit_', 'status_', 'stake_'))] + [c for c in ('over_med', 'under_med', 'dec_med') if c in sel.columns])
    s = s.merge(m[MARKET_KEY + ['over_med', 'under_med']], on=MARKET_KEY, how='left', validate='many_to_one')
    s['dec_best'] = s['dec_price'].astype(float)
    s['dec_med'] = np.where(s['best_side'] == 'over', s['over_med'], s['under_med']).astype(float)
    s['dec_flat'] = FLAT_DEC
    for t in settle.TREATMENTS:
        for name in ('best', 'med', 'flat'):
            prof, status = settle.settle(s['best_side'].to_numpy(), s['line'], s[f'dec_{name}'], s['y'], s['played'], s['lbl_minutes'], s['expected_minutes'], s['book'].to_numpy(), t)
            s[f'profit_{name}_{t}'] = prof
        s[f'status_{t}'] = status
        s[f'stake_{t}'] = np.isin(status, ('win', 'loss')).astype(float)
    if 'units' not in s.columns:
        s['units'] = 1.0
    return s


def roi(b, price='med', treatment='stands'):
    st = (b['units'] * b[f'stake_{treatment}']).sum()
    return float((b['units'] * b[f'profit_{price}_{treatment}']).sum() / st) if st > 0 else np.nan


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Paired cluster bootstrap
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PairedBoot:
    """
    One set of cluster resample counts (props.live.monitor._boot_counts: n_boot x G multinomial, seed) over a FIXED universe of clusters (all game dates of the season), shared by every rule
    added. A rule's replicate ROI is (W @ profit_per_cluster) / (W @ stake_per_cluster); a difference of two rules is the difference of their replicate ROIs on the same resample, so rules with
    different bet sets are compared as ROI = sum(profit)/sum(stake) per rule, never as a mean of per-bet differences.
    """

    def __init__(self, universe, n_boot=N_BOOT, seed=SEED):
        self.universe = pd.Index(sorted(pd.unique(pd.Series(np.asarray(universe)).astype(str))))
        self.W = mon._boot_counts(len(self.universe), n_boot, seed)
        self.n_boot, self.seed = n_boot, seed
        self.num, self.den = {}, {}

    def add(self, name, cluster, profit, stake):
        codes = self.universe.get_indexer(pd.Series(np.asarray(cluster)).astype(str))
        if (codes < 0).any():
            raise ValueError(f'{name}: clusters outside the universe')
        G = len(self.universe)
        self.num[name] = np.bincount(codes, weights=np.asarray(profit, float), minlength=G)
        self.den[name] = np.bincount(codes, weights=np.asarray(stake, float), minlength=G)

    def add_bets(self, name, b, price='med', treatment='stands', cluster='game_date'):
        self.add(name, b[cluster], b['units'] * b[f'profit_{price}_{treatment}'], b['units'] * b[f'stake_{treatment}'])

    def draws(self, name):
        bd = self.W @ self.den[name]
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(bd > 0, (self.W @ self.num[name]) / np.where(bd > 0, bd, 1.0), np.nan)

    def point(self, name):
        s = self.den[name].sum()
        return float(self.num[name].sum() / s) if s > 0 else np.nan

    def roi(self, name, alpha=ALPHA):
        lo, hi = _q(self.draws(name), alpha)
        return {'point': self.point(name), 'lo': float(lo), 'hi': float(hi), 'stakes': float(self.den[name].sum())}

    def diff(self, a, b, alpha=ALPHA):
        d = self.draws(a) - self.draws(b)
        lo, hi = _q(d, alpha)
        return {'point': self.point(a) - self.point(b), 'lo': float(lo), 'hi': float(hi), 'se': float(np.nanstd(d, ddof=1))}


def subset_diff(boot_universe, cluster, profit, stake, mask_a, mask_b, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """ROI(mask_a) - ROI(mask_b) on the same resamples (7d: under bets with expected minutes < 20 against >= 20)."""
    pb = PairedBoot(boot_universe, n_boot, seed)
    cl, pr, st = np.asarray(cluster), np.asarray(profit, float), np.asarray(stake, float)
    pb.add('a', cl[mask_a], pr[mask_a], st[mask_a])
    pb.add('b', cl[mask_b], pr[mask_b], st[mask_b])
    return pb, pb.diff('a', 'b', alpha)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calibration, EV buckets
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calibration_by_side(b, treatment='stands'):
    """props.live.monitor.calibration (expected wins under p* against actual, date-clustered SE) for overs and unders separately. within_2se = |diff| <= 2 SE."""
    p = pd.DataFrame({'stake': b[f'stake_{treatment}'].to_numpy(float), 'win': (b[f'status_{treatment}'] == 'win').astype(float).to_numpy(),
                      'game_date': b['game_date'].astype(str).to_numpy(), 'side': b['best_side'].to_numpy()})
    po, pk, pmod = b['p_star'].to_numpy(float), b['consensus_fair_over'].to_numpy(float), b['p_mod'].to_numpy(float)
    over = p['side'].to_numpy() == 'over'
    p['p_star_side'] = np.where(over, po, 1 - po)
    p['p_market_side'] = np.where(over, pk, 1 - pk)
    p['p_model_side'] = np.where(over, pmod, 1 - pmod)
    out = {}
    for side in ('over', 'under'):
        c = mon.calibration(p[p['side'] == side])
        if c.get('n', 0):
            c['within_2se'] = bool(np.isfinite(c['diff_se']) and abs(c['diff']) <= CAL_MAX_SE * c['diff_se'])
        else:
            c['within_2se'] = False
        out[side] = c
    return out


def ev_bucket(best_ev):
    return pd.cut(pd.Series(np.asarray(best_ev, float)), EV_EDGES, labels=EV_LABELS, right=False)


def bucket_roi(b, label='7%+', price='med', treatment='stands'):
    sub = b[ev_bucket(b['best_ev']).to_numpy() == label]
    return roi(sub, price, treatment) if len(sub) else np.nan


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# C.5 keep rule
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _clause(item, text, value, threshold, holds, reached=True):
    return {'item': item, 'clause': text, 'value': value, 'threshold': threshold, 'holds': bool(holds) if reached else None, 'reached': reached}


def c5_decide(diffs, cal, bucket7, stop_slopes):
    """
    The keep rule of plan C.5, items 1-4 and 6, in order. Inputs (2024-25, median-price ROI, paired date-clustered):
      diffs[(A, B)] = {'point', 'lo', 'hi'} for ROI(A) - ROI(B), for every pair needed (C1-C0, C2-C1, C2-P0, C3-X, C4-X for X the pick so far);
      cal = calibration_by_side(C2 bets) = {'over': {..., 'within_2se'}, 'under': {...}};
      bucket7[cand] = realised median-price ROI of the 7%+ claimed-EV bucket;
      stop_slopes[estimator] = {'b', 'lo', 'hi'}: the pooled slope with intercept of each estimator on the scoring season (E0 uses E1's: E0 has no intercept).
    Returns dict(pick, estimator, stop, clauses=[...]). Item 2 is evaluated only when C1 is the pick after item 1 ('C2 replaces C1').
    """
    cl = []
    d = diffs[('C1', 'C0')]
    a1, b1 = d['point'] < C1_WORSE_POINT, d['hi'] < 0
    cl.append(_clause(1, 'C1 - C0: point below -0.5 points', d['point'], C1_WORSE_POINT, a1))
    cl.append(_clause(1, 'C1 - C0: 95% upper bound below 0', d['hi'], 0.0, b1))
    pick = 'C0' if (a1 and b1) else 'C1'
    cl.append(_clause(1, f'=> C1 clearly worse: {a1 and b1}; pick {pick}', None, None, not (a1 and b1)))
    if pick == 'C1':
        d21, d2p = diffs[('C2', 'C1')], diffs[('C2', 'P0')]
        x1, x2 = d21['point'] >= ADOPT_MARGIN, d2p['point'] >= ADOPT_MARGIN
        xo, xu = bool(cal['over'].get('within_2se')), bool(cal['under'].get('within_2se'))
        cl.append(_clause(2, 'C2 - C1: point at least +0.25 points', d21['point'], ADOPT_MARGIN, x1))
        cl.append(_clause(2, 'C2 - P0: point at least +0.25 points', d2p['point'], ADOPT_MARGIN, x2))
        cl.append(_clause(2, 'C2 calibration, overs: |actual - expected wins| <= 2 clustered SE', _z(cal['over']), CAL_MAX_SE, xo))
        cl.append(_clause(2, 'C2 calibration, unders: |actual - expected wins| <= 2 clustered SE', _z(cal['under']), CAL_MAX_SE, xu))
        if x1 and x2 and xo and xu:
            pick = 'C2'
        cl.append(_clause(2, f'=> pick {pick}', None, None, pick == 'C2'))
    else:
        cl.append(_clause(2, 'C2 replaces C1: not reached (C1 is not the pick)', None, None, False, reached=False))
    for item, cand in ((3, 'C3'), (4, 'C4')):
        d = diffs[(cand, pick)]
        y1, y2 = d['point'] >= ADOPT_MARGIN, d['lo'] >= C3_MIN_LO
        cl.append(_clause(item, f'{cand} - {pick}: point at least +0.25 points', d['point'], ADOPT_MARGIN, y1))
        cl.append(_clause(item, f'{cand} - {pick}: 95% lower bound at least -1.0 point', d['lo'], C3_MIN_LO, y2))
        ok = y1 and y2
        if cand == 'C4':
            v, ref = bucket7.get('C4', np.nan), bucket7.get(pick, np.nan)
            y3 = bool(np.isfinite(v) and np.isfinite(ref) and v >= ref)
            cl.append(_clause(4, f'C4 7%+ bucket median ROI at least {pick}\'s ({ref if np.isfinite(ref) else "n/a"})', v, ref, y3))
            ok = ok and y3
        if ok:
            pick = cand
        cl.append(_clause(item, f'=> pick {pick}', None, None, pick == cand))
    est = CAND_ESTIMATOR[pick]
    s = stop_slopes['E1' if est == 'E0' else est]
    stop = bool(np.isfinite(s['lo']) and np.isfinite(s['hi']) and s['lo'] <= 0 <= s['hi']) or not (np.isfinite(s['lo']) and np.isfinite(s['hi']))
    cl.append(_clause(6, f'stop: pooled slope with intercept of {est} on the scoring season, 95% interval [{s["lo"]:.4f}, {s["hi"]:.4f}] includes 0', s['b'], 0.0, stop))
    return {'pick': pick, 'estimator': est, 'stop': stop, 'clauses': cl}


def _z(c):
    return float(c['diff'] / c['diff_se']) if c.get('n', 0) and c.get('diff_se', 0) > 0 else np.nan


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 7c: caps, Kelly, metrics, keep rules
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def kelly_units(sel, fraction=KELLY['fraction'], max_bet=KELLY['max_bet'], bankroll=KELLY['bankroll'], max_game=KELLY['max_game'], ev_col='best_ev', dec_col='dec_price'):
    """
    Plan 7c item 2: f* = EV / (d - 1) at the best price (the selected quote's price and EV); stake = min(fraction * f*, max_bet) * B; then the total stake per GAME is capped at
    max_game * B, the lower-EV bets of the game cut first (the marginal bet is trimmed to the room left, the bets after it get 0). B fixed (no compounding). Returns a units array.
    """
    ev = np.asarray(sel[ev_col], float)
    d = np.asarray(sel[dec_col], float)
    f = ev / (d - 1.0)
    s = np.clip(np.minimum(fraction * f, max_bet), 0.0, None) * bankroll
    o = pd.DataFrame({'game_id': np.asarray(sel['game_id']), 'ev': ev, 's': s, 'pos': np.arange(len(s))}).sort_values(['game_id', 'ev', 'pos'], ascending=[True, False, True], kind='mergesort')
    cum_before = o.groupby('game_id', sort=False)['s'].cumsum() - o['s']
    room = max_game * bankroll
    o['u'] = np.clip(room - cum_before, 0.0, None).clip(upper=o['s'])
    out = np.empty(len(s))
    out[o['pos'].to_numpy()] = o['u'].to_numpy()
    return out


def daily_metrics(b, price='med', treatment='stands'):
    """Date-level Sharpe = mean / SD (ddof 1) of daily profit per unit staked over dates with a stake; max drawdown of cumulative daily profit (from 0) / total stake; stakes."""
    st = b['units'] * b[f'stake_{treatment}']
    pr = b['units'] * b[f'profit_{price}_{treatment}']
    d = pd.DataFrame({'date': b['game_date'].astype(str).to_numpy(), 'p': pr.to_numpy(float), 's': st.to_numpy(float)}).groupby('date').sum().sort_index()
    d = d[d['s'] > 0]
    total = float(d['s'].sum())
    out = {'roi': float(d['p'].sum() / total) if total > 0 else np.nan, 'n_stakes': int(((b[f'stake_{treatment}'] > 0) & (b['units'] > 0)).sum()),
           'stake_units': total, 'n_dates': int(len(d))}
    rd = d['p'] / d['s']
    sd = float(rd.std(ddof=1)) if len(rd) > 1 else np.nan
    out['sharpe'] = float(rd.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else np.nan
    cum = np.concatenate([[0.0], d['p'].cumsum().to_numpy()])
    dd = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else np.nan
    out['max_drawdown'] = dd
    out['dd_over_stake'] = dd / total if total > 0 else np.nan
    return out


def _ge(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and a >= b)


def c7_cap_decide(name, m25, u25, m24, u24):
    """7c item 6, a cap: Sharpe >= uncapped's and median ROI no more than 0.5 points below (2024-25, point), and 2023-24 Sharpe no lower than uncapped."""
    c = [_clause('7c', f'{name}: 2024-25 Sharpe at least uncapped ({u25["sharpe"]:.4f})', m25['sharpe'], u25['sharpe'], _ge(m25['sharpe'], u25['sharpe'])),
         _clause('7c', f'{name}: 2024-25 median ROI no more than 0.5 points below uncapped ({u25["roi"]:.4f})', m25['roi'], u25['roi'] - CAP_ROI_TOL, _ge(m25['roi'], u25['roi'] - CAP_ROI_TOL)),
         _clause('7c', f'{name}: 2023-24 Sharpe no lower than uncapped ({u24["sharpe"]:.4f})', m24['sharpe'], u24['sharpe'], _ge(m24['sharpe'], u24['sharpe']))]
    return {'adopt': all(x['holds'] for x in c), 'clauses': c}


def kelly_sharpe_bar(flat_sharpe):
    """'at least 1.10 times flat's': flat + 0.10 |flat| (= 1.10 x flat when flat > 0; for a negative flat Sharpe 1.10 x flat would be a LOWER bar than flat, which the plan cannot mean)."""
    return flat_sharpe + (KELLY_SHARPE_MULT - 1.0) * abs(flat_sharpe)


def c7_kelly_decide(k25, f25, k24, f24):
    """7c item 6, Kelly: Sharpe >= 1.10 x flat's, max drawdown / total stake <= flat's, median ROI no more than 0.5 points below flat's (2024-25), 2023-24 not reversing the Sharpe order."""
    bar = kelly_sharpe_bar(f25['sharpe'])
    c = [_clause('7c', f'Kelly: 2024-25 Sharpe at least 1.10 x flat ({f25["sharpe"]:.4f})', k25['sharpe'], bar, _ge(k25['sharpe'], bar)),
         _clause('7c', 'Kelly: 2024-25 max drawdown / total stake at most flat\'s', k25['dd_over_stake'], f25['dd_over_stake'], _ge(f25['dd_over_stake'], k25['dd_over_stake'])),
         _clause('7c', f'Kelly: 2024-25 median ROI no more than 0.5 points below flat ({f25["roi"]:.4f})', k25['roi'], f25['roi'] - KELLY_ROI_TOL, _ge(k25['roi'], f25['roi'] - KELLY_ROI_TOL)),
         _clause('7c', f'Kelly: 2023-24 Sharpe not below flat ({f24["sharpe"]:.4f}) (ordering not reversed)', k24['sharpe'], f24['sharpe'], _ge(k24['sharpe'], f24['sharpe']))]
    return {'adopt': all(x['holds'] for x in c), 'clauses': c}


def d7_decide(pooled, per_season):
    """7d item 5 (amended): pooled ROI(<20) - ROI(>=20) has an UNADJUSTED 95% lower bound above 0, and its point estimate is positive in each season."""
    c = [_clause('7d', 'pooled under <20 minus >=20: unadjusted 95% lower bound above 0', pooled['lo'], 0.0, bool(np.isfinite(pooled['lo']) and pooled['lo'] > 0))]
    for s, d in sorted(per_season.items()):
        c.append(_clause('7d', f'{s}: point estimate positive', d['point'], 0.0, bool(np.isfinite(d['point']) and d['point'] > 0)))
    return {'keep': all(x['holds'] for x in c), 'clauses': c}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# p_market variant (7a choice) and the candidate rule file
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

CURRENT_PMARKET = ('multiplicative', 'all_mean')


def pmarket_from_choice(choice):
    """choice.json of scripts/a11_devig_calibration.py (keys devig_method, consensus_set, passed, reason) -> (devig, set). Not passed or absent -> the current method."""
    if not choice or not choice.get('passed'):
        return CURRENT_PMARKET
    return (str(choice['devig_method']), str(choice['consensus_set']))


def apply_pmarket(cons, bets, variant):
    """
    Replace the consensus by a 7a variant table (columns game_id, player_id, minutes_before, stat, snapshot_ts_utc, consensus_line, consensus_fair_over, n_at_line).
    Bets keep only quotes of the same snapshot that sit on the variant's consensus line; consensus_fair_over and n_at_line come from the variant. Returns (cons, bets, n_bets_dropped).
    """
    key = ['game_id', 'player_id', 'minutes_before', 'stat']
    v = variant.copy()
    v['game_id'] = v['game_id'].astype(str)
    v['player_id'] = v['player_id'].astype(int)
    cols = key + ['snapshot_ts_utc', 'consensus_line', 'consensus_fair_over', 'n_at_line']
    b = bets.copy()
    b['game_id'] = b['game_id'].astype(str)
    nb = b.drop(columns=['consensus_fair_over', 'n_at_line']).merge(v[cols], on=key + ['snapshot_ts_utc'], how='inner', validate='many_to_one')
    nb = nb[nb['line'] == nb['consensus_line']].drop(columns=['consensus_line']).reset_index(drop=True)
    keep = [c for c in cons.columns if c in v.columns] or cols
    return v[list(dict.fromkeys(keep + cols))].reset_index(drop=True), nb[bets.columns], int(len(bets) - len(nb))


def guard_output(path, repo_root):
    """Refuse any output under <repo>/config/ and any file named like the frozen rule."""
    p = Path(path).resolve()
    cfg = (Path(repo_root) / 'config').resolve()
    if p == cfg or cfg in p.parents:
        raise SystemExit(f'refused: {path} is under config/ (plan C.6: config/props_frozen_rule.json and config/ are never written)')
    if p.name in ('props_frozen_rule.json', 'phase5_revised_rule.frozen_rule.json', 'props_live.json'):
        raise SystemExit(f'refused: {path} has the name of a frozen or live config file')
    return p


CANDIDATE_RULE_KEYS = ('stats', 'k_by_stat', 'primary', 'developed_on', 'tried_rules', 'recalibration', 'pooled_slopes', 'code_commit', 'features_db_sha256', 'n_comparisons',
                       'estimator', 'candidate', 'model_version', 'pmarket')


def candidate_rule(cand, params, stats, developed_on, pooled_slopes, code_commit, features_db_sha256, n_comparisons, model_version, pmarket, extra=None):
    """The candidate rule dict (plan C.6): the frozen file's keys plus estimator, intercept_by_stat (C2-C4), c (C4), developed_on, tried_rules = 60, features_db_sha256, model_version, code_commit."""
    out = {'stats': list(stats), 'k_by_stat': {s: float(params['b'].get(s, 0.0)) for s in stats}, 'primary': dict(PRIMARY), 'developed_on': list(developed_on), 'tried_rules': TRIED_RULES,
           'recalibration': 'none', 'pooled_slopes': pooled_slopes, 'code_commit': code_commit, 'features_db_sha256': features_db_sha256, 'n_comparisons': int(n_comparisons),
           'estimator': CAND_ESTIMATOR[cand], 'candidate': cand, 'model_version': model_version, 'pmarket': {'devig_method': pmarket[0], 'consensus_set': pmarket[1]}}
    if cand in ('C2', 'C4'):
        out['intercept_by_stat'] = {s: float(params['a'].get(s, 0.0)) for s in stats}
    if cand == 'C3':
        out['intercept_by_stat'] = {s: {'a_pos': float(params['a_pos'].get(s, 0.0)), 'a_neg': float(params['a_neg'].get(s, 0.0))} for s in stats}
    if cand == 'C4':
        out['c'] = float(params['c'])
    if cand in ('C2', 'C3', 'C4'):
        out['p_star_clip'] = list(P_CLIP)
    if extra:
        out.update(extra)
    return out


def write_rule_file(path, rule, repo_root, overwrite=False):
    """Write the candidate rule JSON and its sha256 sidecar (<path>.sha256, 'hash  name'). Refuses config/ and an existing file unless overwrite."""
    p = guard_output(path, repo_root)
    if p.exists() and not overwrite:
        raise FileExistsError(f'{p} exists: a candidate rule file is written once (use a new name)')
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(rule, indent=2, default=float).encode('utf-8')
    p.write_bytes(data)
    h = hashlib.sha256(data).hexdigest()
    side = p.with_name(p.name + '.sha256')
    side.write_text(f'{h}  {p.name}\n', encoding='utf-8')
    return p, side, h
