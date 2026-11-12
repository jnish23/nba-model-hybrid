"""E32 - the live rule's historical analogue (R0) and the "own view" rules (R1, R2), measured on saved predictions.

Pre-registered in reports/audit_pregame_2026-11/BRIEF.md (section E32), copied verbatim to
reports/experiments/E32_live_rule/REGISTRATION.md before any outcome was computed. No retraining, no network.

Run from the worktree root:  C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe -m experiments.e32_live_rule
Reads  reports/experiments/E25_with_odds/predictions.csv (production checkout, read-only: the worktree only holds fold_info.json),
       data/nba_stats.db (worktree snapshot, read-only) for real closing prices and the forward-season readout.
Writes reports/experiments/E32_live_rule/ (results.json, decision.json, tables.md, CSVs) and copies the summary to the A5 folder.
"""
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from team.betting import (calculate_implied_prob, cover_probabilities, cover_probability,      # noqa: E402
                          get_spread_bet, get_total_bet)
from core.odds_clean import american_to_payout, american_to_prob, load_clean_odds             # noqa: E402

OUT = ROOT / 'reports' / 'experiments' / 'E32_live_rule'
A5 = ROOT / 'reports' / 'audit_pregame_2026-11' / 'A5_E32_live_rule'
PRED = Path('C:/Users/jorda/nba-model-hybrid/reports/experiments/E25_with_odds/predictions.csv')
DB = ROOT / 'data' / 'nba_stats.db'
N_BOOT, SEED, THR = 1000, 42, 0.05
BE110 = calculate_implied_prob(-110)          # 0.5238
WIN110 = 100.0 / 110.0
SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
MARKETS = ['spread', 'total', 'ml']
ARMS = ['R0', 'R1', 'R2']
SGN = {'spread': 1.0, 'total': -1.0}          # edge on side +1 at line L = model + SGN * L (home / over); result likewise
FWD_SIGMA = {'spread': 14.286969463981409, 'total': 18.58954304144289}   # calibration_meta_with_odds_cand_20261019.json
CLIP = 1e-6                                   # experiments/metrics.clip_p


# ------------------------------------------------------------------------------------------ settlement
def settle(side, result):
    """side +1 = home / over, -1 = away / under; result = (margin + L) or (total - L) at the settled line.
    Returns 1 win, -1 loss, 0 push."""
    r = np.asarray(side, float) * np.asarray(result, float)
    return np.where(r > 0, 1, np.where(r < 0, -1, 0))


def profit110(outcome):
    return np.where(outcome == 1, WIN110, np.where(outcome == -1, -1.0, 0.0))


def profit0(outcome):
    return np.asarray(outcome, float)


def settle_ml(side, home_win, q_side):
    """Moneyline at fair price q_side (probability, zero vig): win (1-q)/q, loss -1. Returns (outcome, profit0)."""
    won = np.where(np.asarray(side) > 0, home_win, 1 - home_win).astype(int)
    out = np.where(won == 1, 1, -1)
    return out, np.where(won == 1, (1 - q_side) / q_side, -1.0)


def unit_checks():
    """Hand-made cases for the settle code and the rule code. Writes unit_checks.txt; raises on failure."""
    log = []

    def check(name, cond):
        log.append(f"{'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            raise AssertionError(name)

    # spread: home line -3 (whole), margin +3 -> push both sides; -3.5 -> home loses, away wins; margin +4 -> home wins
    check('spread whole line push (home margin 3, line -3)', settle(+1, 3 + -3) == 0 and settle(-1, 3 + -3) == 0)
    check('spread half line, home margin 3, line -3.5: home loses / away wins', settle(+1, 3 - 3.5) == -1 and settle(-1, 3 - 3.5) == 1)
    check('spread home margin 4 at -3: home wins / away loses', settle(+1, 4 - 3) == 1 and settle(-1, 4 - 3) == -1)
    check('spread away (+3.5) dog, home wins by 3: away covers', settle(-1, 3 + 3.5) == -1 and settle(+1, 3 + 3.5) == 1)  # line +3.5 is home line: home covers
    # total: result = total - line, side +1 = over
    check('total push 220 vs 220', settle(+1, 0) == 0 and settle(-1, 0) == 0)
    check('total 221 vs 220.5: over wins, under loses', settle(+1, 0.5) == 1 and settle(-1, 0.5) == -1)
    check('total 219 vs 220: under wins', settle(-1, 219 - 220) == 1)
    check('profit at -110: win 0.9091, loss -1, push 0',
          abs(profit110(np.array([1]))[0] - 0.909090909) < 1e-8 and profit110(np.array([-1]))[0] == -1 and profit110(np.array([0]))[0] == 0)
    check('profit at zero vig: win +1, loss -1, push 0', list(profit0(np.array([1, -1, 0]))) == [1.0, -1.0, 0.0])
    o, p = settle_ml(np.array([1, 1, -1, -1]), np.array([1, 0, 1, 0]), np.array([0.4, 0.4, 0.4, 0.4]))
    check('ML fair price 0.4 home bet: win +1.5 / loss -1; away bet at 0.4: home won -> loss, home lost -> +1.5',
          np.allclose(p, [1.5, -1, -1, 1.5]) and list(o) == [1, -1, -1, 1])
    # cover probabilities, hand arithmetic (math.erf), whole and half lines
    nd = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))                 # noqa: E731
    e, sd = 3.0, 13.4
    w, l = nd((e - 0.5) / sd), nd((-e - 0.5) / sd)
    check('cover_probabilities push-aware on a whole line (hand formula)',
          abs(float(cover_probabilities([e], [sd], lines=[-3.0])[0]) - w / (w + l)) < 1e-12)
    check('cover_probabilities half line = Phi(e/sd)', abs(float(cover_probabilities([e], [sd], lines=[-3.5])[0]) - nd(e / sd)) < 1e-12)
    check('cover_probabilities agrees with team.betting.cover_probability',
          abs(float(cover_probabilities([2.2], [12.9], lines=[-4.0])[0]) - cover_probability(2.2, 12.9, line=-4.0)) < 1e-12)
    # arms on a hand-made game: model margin +5, open home line -1 (own +4), close -6.5 (model vs close -1.5: CROSSED)
    d = pd.DataFrame({'gameId': [1], 'gameDate': ['2024-01-01'], 'seasonYear': ['2023-24'], 'model': [5.0], 'sigma': [13.4],
                      'open': [-1.0], 'later': [-6.5], 'result': [10.0]})
    check('hand game crossed: R0 no bet (edge 0.021), R1 no bet (crossed)', len(build_ou(d, 'spread', 'R0')) == 0 and len(build_ou(d, 'spread', 'R1')) == 0)
    r2 = build_ou(d, 'spread', 'R2')
    check('hand game crossed: R2 bets home at -6.5, edge ~0.093, margin 10 + -6.5 > 0 wins',
          len(r2) == 1 and r2.side.iloc[0] == 1 and r2.outcome.iloc[0] == 1 and abs(r2.edge.iloc[0] - (nd(4 / 13.4) - BE110)) < 1e-9)
    d2 = d.assign(later=-2.5)                                     # model vs close +2.5: not crossed, R1 selects on the open (whole line -1)
    r1 = build_ou(d2, 'spread', 'R1')
    w, l = nd(3.5 / 13.4), nd(-4.5 / 13.4)
    check('hand game not crossed: R1 bets home, open edge push-aware', len(r1) == 1 and abs(r1.edge.iloc[0] - (w / (w + l) - BE110)) < 1e-9)
    # total: model 230, open 224 (own +6, over), close 228 (not crossed)
    d3 = pd.DataFrame({'gameId': [2], 'gameDate': ['2024-01-02'], 'seasonYear': ['2023-24'], 'model': [230.0], 'sigma': [18.5],
                       'open': [224.5], 'later': [228.5], 'result': [240.0]})
    t = build_ou(d3, 'total', 'R1')
    check('total hand game: R1 over at 228.5, 240 > 228.5 wins; the market moved up, so the bet follows the move (fade = -1)',
          len(t) == 1 and t.side.iloc[0] == 1 and t.outcome.iloc[0] == 1 and t.fade.iloc[0] == -1)
    # ML hand game: model 0.70, open 0.55 (own +0.7), close 0.60: R0 edge 0.10 -> home; R1 open edge 0.15, close not crossed -> home;
    # R2: fair = expit(logit(0.6)+own) edge > 0.05
    m = pd.DataFrame({'gameId': [3], 'gameDate': ['2024-01-03'], 'seasonYear': ['2023-24'], 'p': [0.70], 'o': [0.55], 'c': [0.60], 'home_win': [1]})
    check('ML hand game: R0, R1, R2 all bet home', all(len(build_ml(m, a)) == 1 and build_ml(m, a).side.iloc[0] == 1 for a in ARMS))
    check('ML hand game zero-vig profit at fair 0.60: +0.6667', abs(build_ml(m, 'R0').profit0.iloc[0] - (0.4 / 0.6)) < 1e-12)
    m2 = m.assign(c=0.80)                                          # close passed the model's 0.70: R0 bets away (0.10 edge), R1 (open side home) is crossed
    check('ML crossed (close 0.80 > model 0.70 > open 0.55): R0 bets away, R1 no bet',
          len(build_ml(m2, 'R0')) == 1 and build_ml(m2, 'R0').side.iloc[0] == -1 and len(build_ml(m2, 'R1')) == 0)
    open(OUT / 'unit_checks.txt', 'w').write('\n'.join(log) + '\n')
    return log


# ------------------------------------------------------------------------------------------ arms (spread / total)
def build_ou(d, mk, arm, thr=THR, later_col='later'):
    """Bets of one arm for spread or total. d: columns model (home margin / total), sigma, open, later (line), result
    (home margin / total points); gameId, gameDate, seasonYear. Home / over is side +1; line L is the HOME line for the spread.
    Priced and settled at the later line (arm REF: at the open, the old backtest basis)."""
    sg = SGN[mk]
    M, S, Lo, Lc, R = (d[c].to_numpy(float) for c in ('model', 'sigma', 'open', later_col, 'result'))
    own = M + sg * Lo                                   # point edge against the open, side +1
    e_c = M + sg * Lc                                   # point edge against the later line, side +1
    if arm == 'R0':
        side, line = np.sign(e_c), Lc
        pw = cover_probabilities(np.abs(e_c), S, lines=Lc)
        sel = (pw - BE110) > thr
        edge = pw - BE110
    elif arm == 'R1':
        side, line = np.sign(own), Lc
        pe_open = cover_probabilities(np.abs(own), S, lines=Lo) - BE110
        sel = (pe_open > thr) & (own != 0) & (e_c != 0) & (np.sign(e_c) == np.sign(own))
        pw = cover_probabilities(np.abs(e_c), S, lines=Lc)
        edge = pe_open
    elif arm == 'R2':
        side, line = np.sign(own), Lc
        pw = cover_probabilities(np.abs(own), S, lines=Lc)
        edge = pw - BE110
        sel = (edge > thr) & (own != 0)
    elif arm == 'REF':                                  # the backtest basis: edge and settlement both at the open
        side, line = np.sign(own), Lo
        pw = cover_probabilities(np.abs(own), S, lines=Lo)
        edge = pw - BE110
        sel = (edge > thr) & (own != 0)
    else:
        raise ValueError(arm)
    return _finish_ou(d, sg, sel, side, line, R, pw, edge, own, Lo, Lc)


def _finish_ou(d, sg, sel, side, line, R, pw, edge, own, Lo, Lc):
    res = R + sg * line
    outcome = settle(side, res)
    move_dir = sg * (Lo - Lc)                           # > 0: the market moved toward side +1 between open and close
    fade = np.where(move_dir == 0, 0, np.where(side * move_dir < 0, 1, -1))   # 1 fades the move, -1 follows it, 0 no move
    b = pd.DataFrame({'gameId': d['gameId'].to_numpy(), 'gameDate': d['gameDate'].to_numpy(), 'seasonYear': d['seasonYear'].to_numpy(),
                      'side': side.astype(int), 'open': Lo, 'later': Lc, 'settle_line': line, 'own_pts': own, 'edge': edge, 'p_win': pw,
                      'fade': fade, 'outcome': outcome, 'profit110': profit110(outcome), 'profit0': profit0(outcome)})
    return b[sel].reset_index(drop=True)


def cross_bets(d, mk, mode):
    """Bets for the crossing readout (c): selection = edge against the open > 5% (the R1 selection), settled at the close.
    mode 'open_side' bets the open's side, 'close_side' the side the model now takes against the close."""
    sg = SGN[mk]
    M, S, Lo, Lc, R = (d[c].to_numpy(float) for c in ('model', 'sigma', 'open', 'later', 'result'))
    own, e_c = M + sg * Lo, M + sg * Lc
    pe_open = cover_probabilities(np.abs(own), S, lines=Lo) - BE110
    sel = (pe_open > THR) & (own != 0) & (e_c != 0)
    crossed = np.sign(e_c) != np.sign(own)
    side = np.sign(own) if mode == 'open_side' else np.sign(e_c)
    pw = cover_probabilities(np.abs(own if mode == 'open_side' else e_c), S, lines=Lc)
    b = _finish_ou(d, sg, sel, side, Lc, R, pw, pe_open, own, Lo, Lc)
    b['crossed'] = crossed[sel]
    return b


# ------------------------------------------------------------------------------------------ arms (moneyline, probabilities)
def build_ml(d, arm, thr=THR, p_clip=CLIP, with_real=False):
    """d: p (model P home), o (open fair), c (close fair), home_win, gameId, gameDate, seasonYear (+ real_home_ml, real_away_ml)."""
    p = np.clip(d['p'].to_numpy(float), p_clip, 1 - p_clip)
    o, c = d['o'].to_numpy(float), d['c'].to_numpy(float)
    own = logit(p) - logit(o)
    if arm == 'R0':
        side = np.sign(p - c)
        sel = np.abs(p - c) > thr
        pside = np.where(side > 0, p, 1 - p)
        edge = np.abs(p - c)
    elif arm == 'R1':
        side = np.sign(p - o)
        sel = (np.abs(p - o) > thr) & (side != 0) & (np.sign(p - c) == side)
        pside = np.where(side > 0, p, 1 - p)
        edge = np.abs(p - o)
    elif arm == 'R2':
        fair = expit(logit(np.clip(c, CLIP, 1 - CLIP)) + own)
        side = np.sign(own)
        sel = np.abs(fair - c) > thr
        pside = np.where(side > 0, fair, 1 - fair)
        edge = np.abs(fair - c)
    elif arm == 'REF':                                  # opening probability as the price: the old backtest basis
        side = np.sign(p - o)
        sel = np.abs(p - o) > thr
        pside = np.where(side > 0, p, 1 - p)
        edge = np.abs(p - o)
        c = o
    else:
        raise ValueError(arm)
    price = c
    q_side = np.where(side > 0, price, 1 - price)
    hw = d['home_win'].to_numpy(int)
    outcome, p0 = settle_ml(side, hw, q_side)
    move = c - o                                          # > 0: market moved toward home
    fade = np.where(move == 0, 0, np.where(side * move < 0, 1, -1))
    b = pd.DataFrame({'gameId': d['gameId'].to_numpy(), 'gameDate': d['gameDate'].to_numpy(), 'seasonYear': d['seasonYear'].to_numpy(),
                      'side': side.astype(int), 'open_prob': o, 'later_prob': price, 'q_side': q_side, 'edge': edge, 'p_win': pside,
                      'fade': fade, 'outcome': outcome, 'profit0': p0, 'profit110': np.nan})
    if 'real_home_ml' in d.columns:
        rh, ra = d['real_home_ml'].to_numpy(float), d['real_away_ml'].to_numpy(float)
        rp = np.where(side > 0, rh, ra)
        pay = american_to_payout(rp)
        b['real_price'] = rp
        b['profit_real'] = np.where(np.isfinite(rp), np.where(outcome == 1, pay, -1.0), np.nan)
    return b[sel].reset_index(drop=True)


# ------------------------------------------------------------------------------------------ bootstrap
class DayBoot:
    """Day bootstrap: resample the dates of `dates` (1,000 draws, seed 42); statistics are linear in per-date sums."""

    def __init__(self, dates, n_boot=N_BOOT, seed=SEED):
        self.u, self.inv = np.unique(np.asarray(dates), return_inverse=True)
        nd = len(self.u)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, nd, size=(n_boot, nd))
        self.W = np.stack([np.bincount(r, minlength=nd) for r in idx]).astype(float)

    def per_date(self, bets, cols):
        """Per-date sums of the given columns of `bets` (NaN -> 0), aligned to self.u; shape (n_dates, len(cols))."""
        if len(bets) == 0:
            return np.zeros((len(self.u), len(cols)))
        g = bets.assign(**{c: bets[c].fillna(0.0) for c in cols}).groupby('gameDate')[cols].sum()
        return g.reindex(self.u).fillna(0.0).to_numpy()


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [np.nan, np.nan]


def prep(b):
    """Add the count columns used by per-date sums."""
    b = b.copy()
    b['n'] = 1.0
    b['W'] = (b['outcome'] == 1).astype(float)
    b['L'] = (b['outcome'] == -1).astype(float)
    b['P'] = (b['outcome'] == 0).astype(float)
    if 'profit_real' in b.columns:
        b['pr'] = b['profit_real'].fillna(0.0)
        b['nr'] = b['profit_real'].notna().astype(float)
    else:
        b['pr'] = 0.0
        b['nr'] = 0.0
    b['p110'] = b['profit110'].fillna(0.0)
    b['p0'] = b['profit0']
    b['fadeb'] = (b['fade'] == 1).astype(float)
    b['follow'] = (b['fade'] == -1).astype(float)
    b['pw'] = b['p_win']
    return b


COLS = ['n', 'W', 'L', 'P', 'p110', 'p0', 'pr', 'nr', 'fadeb', 'follow', 'pw']


def _ratio(num, den):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def summarize(b, boot, ml=False):
    """Point estimates and day-bootstrap 95% CIs of one arm's bets. b: bets (any length) of one market/arm."""
    b = prep(b)
    s = b[COLS].sum()
    S = boot.per_date(b, COLS)
    T = boot.W @ S
    n, W, L, P, p110, p0, pr, nr, fadeb, follow, pw = (T[:, i] for i in range(len(COLS)))
    out = {'bets': int(s['n']), 'W': int(s['W']), 'L': int(s['L']), 'P': int(s['P']),
           'hit': float(s['W'] / (s['W'] + s['L'])) if (s['W'] + s['L']) else np.nan,
           'hit_ci': pct(_ratio(W, W + L)),
           'roi0': float(s['p0'] / s['n']) if s['n'] else np.nan, 'roi0_ci': pct(_ratio(p0, n)),
           'fade_share': float(s['fadeb'] / s['n']) if s['n'] else np.nan,
           'follow_share': float(s['follow'] / s['n']) if s['n'] else np.nan,
           'nomove_share': float(1 - (s['fadeb'] + s['follow']) / s['n']) if s['n'] else np.nan,
           'mean_p_win': float(s['pw'] / s['n']) if s['n'] else np.nan}
    if not ml:
        out['roi110'] = float(s['p110'] / s['n']) if s['n'] else np.nan
        out['roi110_ci'] = pct(_ratio(p110, n))
    else:
        out['mean_q'] = float(b['q_side'].mean()) if len(b) else np.nan
        if nr.max() > 0:
            out['n_real'] = int(s['nr'])
            out['roi_real'] = float(s['pr'] / s['nr']) if s['nr'] else np.nan
            out['roi_real_ci'] = pct(_ratio(pr, nr))
    return out


def per_season(b, boots_by_season, ml=False):
    rows = []
    for s in SEASONS:
        bs = b[b['seasonYear'] == s]
        r = summarize(bs, boots_by_season[s], ml=ml)
        r['season'] = s
        rows.append(r)
    return rows


def paired_diff(bx, b0, ml=False, season_rows=None):
    """Arm minus R0 ROI (-110; ML zero vig) bootstrapped over days on the union of the two arms' bet days."""
    dates = np.union1d(bx['gameDate'].unique(), b0['gameDate'].unique())
    if len(bx) == 0 or len(b0) == 0:
        return {'diff': np.nan, 'ci': [np.nan, np.nan], 'union_days': 0}
    boot = DayBoot(dates)
    col = 'p0' if ml else 'p110'
    px, p0 = prep(bx), prep(b0)
    Sx, S0 = boot.per_date(px, ['n', col]), boot.per_date(p0, ['n', col])
    Tx, T0 = boot.W @ Sx, boot.W @ S0
    d = _ratio(Tx[:, 1], Tx[:, 0]) - _ratio(T0[:, 1], T0[:, 0])
    pe = float(px[col].sum() / len(px) - p0[col].sum() / len(p0))
    return {'diff': pe, 'ci': pct(d), 'union_days': int(len(dates)), 'n_boot_valid': int(np.isfinite(d).sum())}


# ------------------------------------------------------------------------------------------ edge slopes
def _ols_boot(X, y, boot, inv_d):
    """OLS with intercept, point estimate + day-bootstrap percentile CI. X: (n, k) regressors, inv_d: row -> boot date index."""
    A = np.column_stack([np.ones(len(X)), X])
    k = A.shape[1]
    pts = np.linalg.lstsq(A, y, rcond=None)[0]
    nd = len(boot.u)
    # per-date sufficient statistics
    XtX = np.zeros((nd, k, k))
    Xty = np.zeros((nd, k))
    np.add.at(XtX, inv_d, A[:, :, None] * A[:, None, :])
    np.add.at(Xty, inv_d, A * y[:, None])
    TX = np.einsum('bd,dij->bij', boot.W, XtX)
    Ty = boot.W @ Xty
    est = np.linalg.solve(TX + 1e-9 * np.eye(k), Ty[:, :, None])[:, :, 0]
    return pts, [pct(est[:, j]) for j in range(k)], est


def logit_offset_fit(A, y, off, w=None, iters=30):
    """Weighted logistic regression with a fixed offset (Newton / IRLS). A includes the intercept column."""
    w = np.ones(len(y)) if w is None else w
    beta = np.zeros(A.shape[1])
    for _ in range(iters):
        eta = off + A @ beta
        mu = expit(eta)
        g = A.T @ (w * (y - mu))
        H = (A * (w * mu * (1 - mu))[:, None]).T @ A + 1e-9 * np.eye(A.shape[1])
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def _logit_boot(X, y, off, boot, inv_d):
    A = np.column_stack([np.ones(len(X)), X])
    pts = logit_offset_fit(A, y, off)
    est = np.zeros((boot.W.shape[0], A.shape[1]))
    for i in range(boot.W.shape[0]):
        est[i] = logit_offset_fit(A, y, off, w=boot.W[i][inv_d])
    return pts, [pct(est[:, j]) for j in range(A.shape[1])]


def edge_slopes(frames):
    """Descriptive readout (a). frames: dict market -> prepared game frame. Returns list of rows."""
    rows = []
    for mk in ('spread', 'total'):
        d = frames[mk]
        sg = SGN[mk]
        boot = DayBoot(d['gameDate'])
        M, Lo, Lc, R = (d[c].to_numpy(float) for c in ('model', 'open', 'later', 'result'))
        e_o, e_c = M + sg * Lo, M + sg * Lc
        y_c, y_o = R + sg * Lc, R + sg * Lo
        specs = [('result-close ~ own + (open-close)', y_c, np.column_stack([e_o, e_c - e_o]), ['own', 'open-close']),
                 ('result-close ~ model-close (single)', y_c, e_c[:, None], ['model-close']),
                 ('result-open ~ own + (close-open)', y_o, np.column_stack([e_o, e_o - e_c]), ['own', 'close-open']),
                 ('result-open ~ model-open (single)', y_o, e_o[:, None], ['model-open'])]
        for name, y, X, labels in specs:
            pts, cis, _ = _ols_boot(X, y, boot, boot.inv)
            for j, lab in enumerate(['intercept'] + labels):
                rows.append({'market': mk, 'model': name, 'term': lab, 'coef': float(pts[j]), 'lo': cis[j][0], 'hi': cis[j][1], 'n': len(d)})
    d = frames['ml']
    boot = DayBoot(d['gameDate'])
    p = np.clip(d['p'].to_numpy(float), CLIP, 1 - CLIP)
    o, c = d['o'].to_numpy(float), d['c'].to_numpy(float)
    y = d['home_win'].to_numpy(float)
    lp, lo_, lc = logit(p), logit(np.clip(o, CLIP, 1 - CLIP)), logit(np.clip(c, CLIP, 1 - CLIP))
    specs = [('home win ~ offset logit(close) + own + (open-close)', lc, np.column_stack([lp - lo_, lo_ - lc]), ['own', 'open-close']),
             ('home win ~ offset logit(close) + (model-close) (single)', lc, (lp - lc)[:, None], ['model-close']),
             ('home win ~ offset logit(open) + own + (close-open)', lo_, np.column_stack([lp - lo_, lc - lo_]), ['own', 'close-open']),
             ('home win ~ offset logit(open) + (model-open) (single)', lo_, (lp - lo_)[:, None], ['model-open'])]
    for name, off, X, labels in specs:
        pts, cis = _logit_boot(X, y, off, boot, boot.inv)
        for j, lab in enumerate(['intercept'] + labels):
            rows.append({'market': 'ml', 'model': name, 'term': lab, 'coef': float(pts[j]), 'lo': cis[j][0], 'hi': cis[j][1], 'n': len(d)})
    return rows


# ------------------------------------------------------------------------------------------ data
def load_games():
    p = pd.read_csv(PRED)
    p['gameDate'] = p['gameDate'].astype(str)
    clean = load_clean_odds(str(DB))
    clean['gid'] = clean['gameId'].astype(int)
    real = clean[['gid', 'close_home_ml', 'close_away_ml', 'close_n_books_ml']].rename(
        columns={'close_home_ml': 'real_home_ml', 'close_away_ml': 'real_away_ml'})
    p = p.merge(real, left_on='gameId', right_on='gid', how='left').drop(columns='gid')
    frames = {}
    base = dict(gameId=p['gameId'], gameDate=p['gameDate'], seasonYear=p['seasonYear'])
    for mk, mcol, scol, lo, lc, rcol in (('spread', 'spread_pred', 'sigma_spread', 'odds_home_spread', 'close_home_spread', 'target_point_diff'),
                                         ('total', 'total_pred', 'sigma_total', 'odds_total', 'close_total', 'target_total_pts')):
        f = pd.DataFrame({**base, 'model': p[mcol], 'sigma': p[scol], 'open': p[lo], 'later': p[lc], 'result': p[rcol]})
        frames[mk] = f.dropna(subset=['model', 'sigma', 'open', 'later', 'result']).reset_index(drop=True)
    f = pd.DataFrame({**base, 'p': p['ml_prob'], 'o': p['open_home_prob'], 'c': p['close_home_prob'], 'home_win': p['target_home_win'],
                      'real_home_ml': p['real_home_ml'], 'real_away_ml': p['real_away_ml']})
    frames['ml'] = f.dropna(subset=['p', 'o', 'c', 'home_win']).reset_index(drop=True)
    return p, frames


def check_against_production_rule(frames):
    """R0 selection and side must equal team.betting.get_spread_bet / get_total_bet applied row by row (the production rule)."""
    log = []
    for mk, fn in (('spread', get_spread_bet), ('total', get_total_bet)):
        d = frames[mk]
        b0 = build_ou(d, mk, 'R0')
        got = dict(zip(b0['gameId'], b0['side']))
        mism = 0
        for r in d.itertuples():
            side, _ = fn(r.model, r.later, std=r.sigma)
            want = {'home': 1, 'over': 1, 'away': -1, 'under': -1}.get(side)
            if got.get(r.gameId) != want:
                mism += 1
        log.append(f'{mk}: R0 vs team.betting.get_{mk}_bet at the close line, {len(d)} games, {len(b0)} bets, mismatches = {mism}')
        assert mism == 0
    return log


# ------------------------------------------------------------------------------------------ report helpers
def fmt_ci(v, ci, pctf=True, d=1):
    if v is None or not np.isfinite(v):
        return 'n/a'
    if pctf:
        return f'{100 * v:.{d}f}% [{100 * ci[0]:.{d}f}, {100 * ci[1]:.{d}f}]'
    return f'{v:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]'


def md_table(rows, cols, heads):
    out = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join(['---'] * len(heads)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(out)


def arm_row(mk, arm, s, ml=False):
    r = {'market': mk, 'arm': arm, 'bets': s['bets'], 'WLP': f"{s['W']}-{s['L']}-{s['P']}",
         'hit': fmt_ci(s['hit'], s['hit_ci'], d=1), 'roi0': fmt_ci(s['roi0'], s['roi0_ci']),
         'fade': f"{100 * s['fade_share']:.0f}%", 'nomove': f"{100 * s['nomove_share']:.0f}%",
         'mean_p': f"{100 * s['mean_p_win']:.1f}%"}
    r['roi110'] = fmt_ci(s['roi110'], s['roi110_ci']) if not ml else 'n/a (ML)'
    if ml:
        r['mean_q'] = f"{100 * s['mean_q']:.1f}%"
        r['real'] = (fmt_ci(s['roi_real'], s['roi_real_ci']) + f" (n={s['n_real']})") if 'roi_real' in s else 'n/a'
    return r


def jclean(o):
    if isinstance(o, dict):
        return {k: jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ------------------------------------------------------------------------------------------ forward (readout d)
def forward_readout():
    """This season's games with a real 11 am line: predictions_log_v2 (model + the line it logged at run time) joined with the
    open (nba_odds_opening consensus) and the close (last pre-game snapshot) from core.odds_clean.load_clean_odds."""
    con = sqlite3.connect(f'file:{DB.as_posix()}?mode=ro', uri=True)
    pl = pd.read_sql("SELECT * FROM predictions_log_v2 WHERE model_version = 'v1.0.0_hybrid' AND status = 'COMPLETED'", con)
    snaps = pd.read_sql("SELECT fetched_at_et, game_date, home_team, line_type, book, home_spread, total, in_game FROM odds_snapshots", con)
    con.close()
    clean = load_clean_odds(str(DB))
    clean = clean[clean['gameDate'] >= '2026-10-01']
    g = pl.merge(clean, left_on='game_id', right_on='gameId', how='left', suffixes=('', '_c'))
    info = {'completed_games': int(len(pl)), 'matched_to_clean_odds': int(g['gameId'].notna().sum())}
    # sanity: the logged line vs the first consensus snapshot at/after 11:00 ET on game day, and the opening snapshots vs nba_odds_opening
    cur = snaps[(snaps.line_type == 'current') & (snaps.in_game == 0)].copy()
    cur['t'] = pd.to_datetime(cur['fetched_at_et'])
    cur = cur[cur['t'] >= pd.to_datetime(cur['game_date']) + pd.Timedelta(hours=11)]
    first = cur.sort_values('t').groupby(['game_date', 'home_team', 'book'], as_index=False).first()
    cons = first.groupby(['game_date', 'home_team'], as_index=False)[['home_spread', 'total']].median()
    gg = g.merge(cons, left_on=['game_date', 'home_team'], right_on=['game_date', 'home_team'], how='left')
    info['logged_vs_first_snapshot_after_11et'] = {
        'n': int(gg['home_spread'].notna().sum()),
        'spread_mean_abs_diff': float((gg['odds_home_spread'] - gg['home_spread']).abs().mean()),
        'spread_share_equal': float(((gg['odds_home_spread'] - gg['home_spread']).abs() < 1e-6).mean()),
        'total_mean_abs_diff': float((gg['odds_total'] - gg['total']).abs().mean()),
        'total_share_equal': float(((gg['odds_total'] - gg['total']).abs() < 1e-6).mean())}
    opn = snaps[snaps.line_type == 'opening'].groupby(['game_date', 'home_team'], as_index=False)[['home_spread', 'total']].median()
    og = g.merge(opn, on=['game_date', 'home_team'], how='left', suffixes=('', '_osnap'))
    info['open_snapshot_vs_nba_odds_opening'] = {
        'n': int(og['home_spread'].notna().sum()),
        'spread_share_equal': float(((og['open_home_spread'] - og['home_spread']).abs() < 1e-6).mean()),
        'spread_mean_abs_diff': float((og['open_home_spread'] - og['home_spread']).abs().mean()),
        'total_mean_abs_diff': float((og['open_total'] - og['total']).abs().mean())}
    base = dict(gameId=g['game_id'], gameDate=g['game_date'].astype(str), seasonYear='2026-27')
    frames = {}
    for mk, mcol, lo, l11, lc, res in (('spread', 'spread_pred', 'open_home_spread', 'odds_home_spread', 'close_home_spread', None),
                                       ('total', 'total_pred', 'open_total', 'odds_total', 'close_total', None)):
        result = (g['home_score'] - g['away_score']) if mk == 'spread' else (g['home_score'] + g['away_score'])
        f = pd.DataFrame({**base, 'model': g[mcol], 'sigma': FWD_SIGMA[mk], 'open': g[lo], 'later': g[l11], 'close': g[lc], 'result': result})
        frames[mk] = f.dropna(subset=['model', 'open', 'later', 'close', 'result']).reset_index(drop=True)
    ph = american_to_prob(g['odds_home_ml'].to_numpy(float))
    pa = american_to_prob(g['odds_away_ml'].to_numpy(float))
    f = pd.DataFrame({**base, 'p': g['ml_prob'], 'o': g['open_home_prob'], 'c': ph / (ph + pa), 'close': g['close_home_prob'],
                      'home_win': (g['home_score'] > g['away_score']).astype(int)})
    frames['ml'] = f.dropna(subset=['p', 'o', 'c', 'close', 'home_win']).reset_index(drop=True)
    info['games_with_open_11am_close'] = {k: int(len(v)) for k, v in frames.items()}
    res_rows, bets_out = [], {}
    for mk in MARKETS:
        d = frames[mk]
        boot = DayBoot(d['gameDate'])
        ml = mk == 'ml'
        for arm in ARMS:
            b = build_ml(d, arm) if ml else build_ou(d, mk, arm)
            # CLV to the close
            if len(b):
                dd = d.set_index('gameId').loc[b['gameId']]
                if ml:
                    b['clv'] = b['side'].to_numpy() * (dd['close'].to_numpy() - dd['c'].to_numpy())
                else:
                    b['clv'] = b['side'].to_numpy() * SGN[mk] * (dd['later'].to_numpy() - dd['close'].to_numpy())
            else:
                b['clv'] = []
            s = summarize(b, boot, ml=ml)
            pb = b.assign(n=1.0, clv=b['clv'].astype(float))
            S = boot.per_date(pb, ['n', 'clv'])
            T = boot.W @ S
            s['clv_mean'] = float(b['clv'].mean()) if len(b) else np.nan
            s['clv_ci'] = pct(_ratio(T[:, 1], T[:, 0]))
            s['clv_pos_share'] = float((b['clv'] > 0).mean()) if len(b) else np.nan
            s['clv_zero_share'] = float((b['clv'] == 0).mean()) if len(b) else np.nan
            s['market'], s['arm'], s['days'] = mk, arm, int(len(boot.u))
            res_rows.append(s)
            bets_out[(mk, arm)] = b
    info['arms'] = res_rows
    return info, bets_out, frames


# ------------------------------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'bets').mkdir(exist_ok=True)
    A5.mkdir(parents=True, exist_ok=True)
    reg = (OUT / 'REGISTRATION.md').read_text(encoding='utf-8')
    assert 'sha256 of the section text' in reg, 'REGISTRATION.md must exist before any outcome is computed'
    ucheck = unit_checks()
    p, frames = load_games()
    prod_check = check_against_production_rule(frames)
    results = {'unit_checks': ucheck, 'production_rule_check': prod_check, 'n_games_file': int(len(p)),
               'games_per_market': {k: int(len(v)) for k, v in frames.items()}}
    games_file_by_season = p.groupby('seasonYear').size().to_dict()
    results['games_file_by_season'] = games_file_by_season
    results['games_market_by_season'] = {k: v.groupby('seasonYear').size().to_dict() for k, v in frames.items()}
    results['ml_real_price_games'] = {'with_real_close_prices': int(frames['ml']['real_home_ml'].notna().sum()),
                                      'of': int(len(frames['ml'])),
                                      'by_season': frames['ml'].groupby('seasonYear')['real_home_ml'].apply(lambda s: int(s.notna().sum())).to_dict()}

    boots = {mk: DayBoot(frames[mk]['gameDate']) for mk in MARKETS}
    sboots = {mk: {s: DayBoot(frames[mk].loc[frames[mk]['seasonYear'] == s, 'gameDate']) for s in SEASONS} for mk in MARKETS}
    bets = {}
    summ, season_tabs, arm_tables = {}, {}, []
    for mk in MARKETS:
        ml = mk == 'ml'
        for arm in ARMS + ['REF']:
            b = build_ml(frames[mk], arm) if ml else build_ou(frames[mk], mk, arm)
            bets[(mk, arm)] = b
            b.to_csv(OUT / 'bets' / f'{mk}_{arm}.csv', index=False)
            s = summarize(b, boots[mk], ml=ml)
            s['per_season'] = per_season(b, sboots[mk], ml=ml)
            summ[(mk, arm)] = s
    results['arms'] = {f'{mk}/{arm}': v for (mk, arm), v in summ.items()}

    # sensitivity (descriptive): exclude the 2019-20 bubble season; ML clip at 0.02 / 0.98
    sens = {}
    for mk in MARKETS:
        ml = mk == 'ml'
        d = frames[mk][frames[mk]['seasonYear'] != '2019-20']
        bt = DayBoot(d['gameDate'])
        for arm in ARMS:
            b = build_ml(d, arm) if ml else build_ou(d, mk, arm)
            s = summarize(b, bt, ml=ml)
            sens[f'{mk}/{arm}/ex_2019-20'] = {k: s[k] for k in s if k in ('bets', 'hit', 'hit_ci', 'roi0', 'roi0_ci', 'roi110', 'roi110_ci')}
    for arm in ARMS:
        b = build_ml(frames['ml'], arm, p_clip=0.02)
        s = summarize(b, boots['ml'], ml=True)
        sens[f'ml/{arm}/clip_0.02'] = {k: s[k] for k in ('bets', 'hit', 'hit_ci', 'roi0', 'roi0_ci')}
    results['sensitivity'] = sens

    # ---- decisions
    dec = {'H1': {}, 'H2': {}, 'H3': {}, 'notes': ['ML has no -110 ROI in the registration: H1 and H2 use the zero-vig ROI for ML; H3 uses the zero-vig hit rate (literal rule) for every market.']}
    for mk in MARKETS:
        ml = mk == 'ml'
        key = 'roi0' if ml else 'roi110'
        s0 = summ[(mk, 'R0')]
        hi = s0[key + '_ci'][1]
        dec['H1'][mk] = {'metric': 'roi0 (zero vig)' if ml else 'roi at -110', 'R0_roi': s0[key], 'ci': s0[key + '_ci'],
                         'ci_entirely_below_0': bool(hi < 0), 'confirmed': bool(hi < 0)}
        for arm in ('R1', 'R2'):
            pd_ = paired_diff(bets[(mk, arm)], bets[(mk, 'R0')], ml=ml)
            wins, rows = 0, []
            for sx, s0s in zip(summ[(mk, arm)]['per_season'], summ[(mk, 'R0')]['per_season']):
                a, c = sx[key], s0s[key]
                higher = bool(np.isfinite(a) and np.isfinite(c) and a > c)
                wins += higher
                rows.append({'season': sx['season'], 'arm_roi': a, 'R0_roi': c, 'arm_bets': sx['bets'], 'R0_bets': s0s['bets'], 'higher': higher})
            ci_pos = bool(np.isfinite(pd_['ci'][0]) and pd_['ci'][0] > 0)
            dec['H2'][f'{mk}/{arm}'] = {'metric': 'roi0 (zero vig)' if ml else 'roi at -110', 'diff_vs_R0': pd_['diff'], 'ci': pd_['ci'],
                                        'union_bet_days': pd_['union_days'], 'ci_excludes_0': bool(ci_pos or (np.isfinite(pd_['ci'][1]) and pd_['ci'][1] < 0)),
                                        'ci_lower_above_0': ci_pos, 'seasons_higher': int(wins), 'seasons': rows,
                                        'better_than_R0': bool(ci_pos and wins >= 5)}
        for arm in ARMS:
            s = summ[(mk, arm)]
            dec['H3'][f'{mk}/{arm}'] = {'hit_rate': s['hit'], 'ci': s['hit_ci'], 'lower_above_50': bool(s['hit_ci'][0] > 0.5),
                                        'expected_to_pay_on_zero_fee_venue': bool(s['hit_ci'][0] > 0.5),
                                        'roi0': s['roi0'], 'roi0_ci': s['roi0_ci']}
    (OUT / 'decision.json').write_text(json.dumps(jclean(dec), indent=1))

    # ---- (a) edge slopes
    slopes = edge_slopes(frames)
    pd.DataFrame(slopes).to_csv(OUT / 'edge_slopes.csv', index=False)
    results['edge_slopes'] = slopes

    # ---- (b) R2 sweep
    sweep = []
    for mk in MARKETS:
        ml = mk == 'ml'
        for t in range(0, 11):
            thr = t / 100.0
            b = build_ml(frames[mk], 'R2', thr=thr) if ml else build_ou(frames[mk], mk, 'R2', thr=thr)
            s = summarize(b, boots[mk], ml=ml)
            sweep.append({'market': mk, 'threshold_pts': t, 'bets': s['bets'], 'W': s['W'], 'L': s['L'], 'P': s['P'], 'hit': s['hit'],
                          'hit_lo': s['hit_ci'][0], 'hit_hi': s['hit_ci'][1], 'roi0': s['roi0'], 'roi0_lo': s['roi0_ci'][0], 'roi0_hi': s['roi0_ci'][1],
                          'roi110': s.get('roi110'), 'roi110_lo': s['roi110_ci'][0] if 'roi110_ci' in s else None,
                          'roi110_hi': s['roi110_ci'][1] if 'roi110_ci' in s else None})
    pd.DataFrame(sweep).to_csv(OUT / 'sweep_R2.csv', index=False)
    results['sweep_R2'] = sweep

    # ---- (c) crossing
    cross = {}
    for mk in MARKETS:
        d = frames[mk]
        if mk == 'ml':
            p_ = np.clip(d['p'].to_numpy(float), CLIP, 1 - CLIP)
            o_, c_ = d['o'].to_numpy(float), d['c'].to_numpy(float)
            own = logit(p_) - logit(o_)
            valid = (np.sign(p_ - c_) != 0) & (own != 0)
            crossed_all = np.sign(p_ - c_) != np.sign(own)
            sel_open = np.abs(p_ - o_) > THR
            crossed_sel = crossed_all[sel_open & valid]
            share_all, share_sel = float(crossed_all[valid].mean()), float(crossed_sel.mean())
            # open-side and close-side bets on the crossed games with an open edge > 5 points
            sel = sel_open & valid & crossed_all
            sub = d[sel].reset_index(drop=True)
            side_o = np.sign(own[sel])
            side_c = np.sign((p_ - c_)[sel])
            res = {}
            for nm, side in (('open_side_crossed', side_o), ('close_side_crossed', side_c)):
                q = np.where(side > 0, c_[sel], 1 - c_[sel])
                out_, p0_ = settle_ml(side, sub['home_win'].to_numpy(int), q)
                b = pd.DataFrame({'gameDate': sub['gameDate'], 'seasonYear': sub['seasonYear'], 'outcome': out_, 'profit0': p0_, 'profit110': np.nan,
                                  'fade': 0, 'p_win': 0.0, 'q_side': q})
                res[nm] = summarize(b, boots[mk], ml=True)
            res['open_side_not_crossed(R1)'] = summ[(mk, 'R1')]
        else:
            sg = SGN[mk]
            M, Lo, Lc = d['model'].to_numpy(float), d['open'].to_numpy(float), d['later'].to_numpy(float)
            own, e_c = M + sg * Lo, M + sg * Lc
            valid = (own != 0) & (e_c != 0)
            crossed_all = np.sign(e_c) != np.sign(own)
            bo = cross_bets(d, mk, 'open_side')
            S = d['sigma'].to_numpy(float)
            pe_open = cover_probabilities(np.abs(own), S, lines=Lo) - BE110
            sel_open = pe_open > THR
            share_all, share_sel = float(crossed_all[valid].mean()), float(crossed_all[valid & sel_open].mean())
            res = {}
            bo_c = bo[bo['crossed']]
            res['open_side_crossed'] = summarize(bo_c, boots[mk])
            bc = cross_bets(d, mk, 'close_side')
            res['close_side_crossed'] = summarize(bc[bc['crossed']], boots[mk])
            res['open_side_not_crossed(R1)'] = summ[(mk, 'R1')]
        cross[mk] = {'share_crossed_all_games': share_all, 'share_crossed_open_edge_over_5pct': share_sel,
                     'n_games': int(valid.sum()), 'n_open_edge_over_5pct': int((valid & sel_open).sum()), 'results': res}
    results['crossing'] = cross

    # ---- (e) bets per season
    per_year = []
    for mk in MARKETS:
        for arm in ARMS + ['REF']:
            b = bets[(mk, arm)]
            for s in SEASONS:
                n = int((b['seasonYear'] == s).sum())
                ng = int((frames[mk]['seasonYear'] == s).sum())
                gf = int(games_file_by_season[s])
                per_year.append({'market': mk, 'arm': arm, 'season': s, 'bets': n, 'games_in_sample': ng, 'games_in_file': gf,
                                 'bets_scaled_to_file': n * gf / ng if ng else np.nan})
    pyd = pd.DataFrame(per_year)
    pyd.to_csv(OUT / 'bets_per_season.csv', index=False)
    results['bets_per_season'] = per_year

    # ---- (d) forward season
    fwd_info, fwd_bets, fwd_frames = forward_readout()
    for (mk, arm), b in fwd_bets.items():
        b.to_csv(OUT / 'bets' / f'forward_{mk}_{arm}.csv', index=False)
    results['forward'] = fwd_info

    results = jclean(results)
    (OUT / 'results.json').write_text(json.dumps(results, indent=1, default=str))
    write_tables(results, summ, dec, bets)
    print('done')


def write_tables(results, summ, dec, bets):
    L = []
    L.append('# E32 tables (generated by experiments/e32_live_rule.py)\n')
    L.append('## Arms, pooled 2019-20..2025-26 (day-bootstrap 95% CI)\n')
    for mk in MARKETS:
        ml = mk == 'ml'
        rows = [arm_row(mk, arm, summ[(mk, arm)], ml) for arm in ARMS + ['REF']]
        if ml:
            L.append(f'### {mk}  (games: {results["games_per_market"][mk]})\n')
            L.append(md_table(rows, ['arm', 'bets', 'WLP', 'hit', 'roi0', 'real', 'fade', 'nomove', 'mean_p', 'mean_q'],
                              ['arm', 'bets', 'W-L-P', 'hit rate', 'ROI zero vig', 'ROI real close price', 'fades move', 'no move', 'mean model P(win)', 'mean fair price']))
        else:
            L.append(f'### {mk}  (games: {results["games_per_market"][mk]})\n')
            L.append(md_table(rows, ['arm', 'bets', 'WLP', 'hit', 'roi110', 'roi0', 'fade', 'nomove', 'mean_p'],
                              ['arm', 'bets', 'W-L-P', 'hit rate', 'ROI -110', 'ROI zero vig', 'fades move', 'no move', 'mean model P(win)']))
        L.append('')
    L.append('## Per season\n')
    for mk in MARKETS:
        ml = mk == 'ml'
        for arm in ARMS:
            L.append(f'### {mk} {arm}\n')
            rows = []
            for r in summ[(mk, arm)]['per_season']:
                rows.append({'season': r['season'], 'bets': r['bets'], 'WLP': f"{r['W']}-{r['L']}-{r['P']}", 'hit': f"{100 * r['hit']:.1f}%" if np.isfinite(r['hit']) else 'n/a',
                             'roi': (fmt_ci(r['roi0'], r['roi0_ci']) if ml else fmt_ci(r['roi110'], r['roi110_ci'])) if r['bets'] else 'n/a',
                             'roi0': fmt_ci(r['roi0'], r['roi0_ci']) if r['bets'] else 'n/a'})
            L.append(md_table(rows, ['season', 'bets', 'WLP', 'hit', 'roi'] + ([] if ml else ['roi0']),
                              ['season', 'bets', 'W-L-P', 'hit rate', 'ROI zero vig' if ml else 'ROI -110'] + ([] if ml else ['ROI zero vig'])))
            L.append('')
    L.append('## Decisions\n')
    L.append('```json\n' + json.dumps(jclean(dec), indent=1) + '\n```\n')
    (OUT / 'tables.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
