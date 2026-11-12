"""Own metric / bet / settle code for the A2 audit (does not import experiments.metrics or team.betting)."""
import math
import numpy as np
import pandas as pd

BE110 = 110.0 / 210.0          # break-even of -110 = 0.52381


def Phi(x):
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(x / math.sqrt(2.0)))


def implied(odds):
    o = np.asarray(odds, float)
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def payout(odds):
    o = np.asarray(odds, float)
    return np.where(o < 0, 100.0 / -o, o / 100.0)


def logloss(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p):
    return float(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean())


def cover_prob(edge_abs, sigma, line):
    """P(bet wins | it settles). Half-point lines: Phi(e/s). Whole-number lines: win/(win+loss) with a 0.5 continuity correction."""
    e = np.asarray(edge_abs, float)
    s = np.asarray(sigma, float)
    half = Phi(e / s)
    ln = np.asarray(line, float)
    whole = np.isfinite(ln) & (ln == np.round(ln))
    w, l = Phi((e - 0.5) / s), Phi((-e - 0.5) / s)
    return np.where(whole, w / (w + l), half)


def settle_spread_home(margin, home_line):
    """+1 home covers, -1 away covers, 0 push. home_line is the home team's handicap (negative = home favourite)."""
    r = np.asarray(margin, float) + np.asarray(home_line, float)
    return np.sign(r)


def settle_total_over(total, line):
    return np.sign(np.asarray(total, float) - np.asarray(line, float))


def make_bets(df, thr=0.05, line_sp='odds_home_spread', line_tot='odds_total', ml_cols=('odds_home_ml', 'odds_away_ml'),
              prob_col='ml_prob', pred_sp='spread_pred', pred_tot='total_pred', sig_sp='sigma_spread', sig_tot='sigma_total',
              markets=('ml', 'spread', 'total'), price=-110.0, settle_sp=None, settle_tot=None, settle_ml_odds=None):
    """Return a bet table. Bets are CHOSEN against (line_sp, line_tot, ml_cols) and SETTLED against (settle_sp, settle_tot) when given
    (default: same lines). Pushes are dropped (stake refunded). Missing line -> no bet."""
    out = []
    base = df[['gameId', 'gameDate', 'seasonYear']]
    margin, total, hw = df['target_point_diff'].to_numpy(float), df['target_total_pts'].to_numpy(float), df['target_home_win'].to_numpy(float)
    if 'ml' in markets:
        p = df[prob_col].to_numpy(float)
        oh, oa = df[ml_cols[0]].to_numpy(float), df[ml_cols[1]].to_numpy(float)
        eh, ea = p - implied(oh), (1 - p) - implied(oa)
        home = eh >= ea
        edge = np.where(home, eh, ea)
        ok = np.isfinite(edge) & (edge > thr)
        so_h = oh if settle_ml_odds is None else settle_ml_odds[0]
        so_a = oa if settle_ml_odds is None else settle_ml_odds[1]
        px = np.where(home, so_h, so_a)
        ok &= np.isfinite(px)
        won = np.where(home, hw == 1, hw == 0)
        prof = np.where(won, payout(px) * 100, -100.0)
        d = base[ok].copy()
        d['market'] = 'ml'; d['side'] = np.where(home, 'home', 'away')[ok]; d['edge'] = edge[ok]; d['price'] = px[ok]
        d['won'] = won[ok]; d['profit'] = prof[ok]; d['push'] = False
        out.append(d)
    for mk, pred, ln_col, sg, act, sett in (('spread', pred_sp, line_sp, sig_sp, margin, settle_sp), ('total', pred_tot, line_tot, sig_tot, total, settle_tot)):
        if mk not in markets:
            continue
        ln = df[ln_col].to_numpy(float)
        pr = df[pred].to_numpy(float)
        s = df[sg].to_numpy(float)
        pe = pr + ln if mk == 'spread' else pr - ln          # >0: home covers / over
        edge = cover_prob(np.abs(pe), s, ln) - BE110
        first = pe > 0
        sl = ln if sett is None else np.asarray(sett, float)
        r = act + sl if mk == 'spread' else act - sl         # >0: home covers / over
        ok = np.isfinite(pe) & np.isfinite(sl) & (edge > thr)
        push = ok & (r == 0)
        won = np.where(first, r > 0, r < 0)
        sel = ok & ~push
        d = base[sel].copy()
        d['market'] = mk; d['side'] = np.where(first, 'home' if mk == 'spread' else 'over', 'away' if mk == 'spread' else 'under')[sel]
        d['edge'] = edge[sel]; d['price'] = price; d['won'] = won[sel]
        d['profit'] = np.where(won[sel], 100.0 / 1.1, -100.0); d['push'] = False
        d['n_push_refunded'] = 0
        out.append(d)
        out.append(pd.DataFrame({'_pushes_' + mk: [int(push.sum())]}))
    bets = pd.concat([o for o in out if 'market' in o], ignore_index=True)
    bets['gameDate'] = pd.to_datetime(bets['gameDate'])
    pushes = {list(o.columns)[0][8:]: int(o.iloc[0, 0]) for o in out if 'market' not in o}
    return bets, pushes


def day_boot_stat(bets, n=1000, seed=42, by='gameDate'):
    """Day-bootstrap (resample dates, 1,000 draws, seed 42). Returns dict roi, hit, with 95% CIs (percent)."""
    b = bets.sort_values('gameDate')
    days, inv = np.unique(b[by].to_numpy(), return_inverse=True)
    nd = len(days)
    prof = np.bincount(inv, weights=b['profit'].to_numpy(float), minlength=nd)
    wins = np.bincount(inv, weights=b['won'].to_numpy(float), minlength=nd)
    cnt = np.bincount(inv, minlength=nd).astype(float)
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, nd, size=(n, nd))
    roi = prof[dr].sum(1) / (100.0 * cnt[dr].sum(1)) * 100
    hit = wins[dr].sum(1) / cnt[dr].sum(1) * 100
    q = lambda a: (float(np.quantile(a, .025)), float(np.quantile(a, .975)))
    return {'bets': int(len(b)), 'wins': int(b['won'].sum()), 'losses': int(len(b) - b['won'].sum()), 'hit': float(b['won'].mean() * 100),
            'hit_ci': q(hit), 'roi': float(b['profit'].sum() / (100.0 * len(b)) * 100), 'roi_ci': q(roi)}


def paired_day_boot(frame, fn_a, fn_b, n=1000, seed=42):
    """Mean of (a - b) per game with day-bootstrap CI. fn_a/fn_b: per-game loss Series aligned to frame."""
    d = (fn_a - fn_b).to_numpy(float)
    days, inv = np.unique(frame['gameDate'].to_numpy(), return_inverse=True)
    nd = len(days)
    s = np.bincount(inv, weights=d, minlength=nd)
    c = np.bincount(inv, minlength=nd).astype(float)
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, nd, size=(n, nd))
    bs = s[dr].sum(1) / c[dr].sum(1)
    return float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
