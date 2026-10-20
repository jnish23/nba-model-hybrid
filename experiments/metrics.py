"""Metrics, paired comparisons and bootstrap CIs shared by all experiments.

All bootstraps resample GAME DAYS (not games) with replacement, because games on one night share
information (injury news, schedule effects) and bets in a day are not independent.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from team import betting as bt

N_BOOT = 1000
RNG_SEED = 20261017
BREAK_EVEN_110 = bt.calculate_implied_prob(-110)


# ------------------------------------------------------------------ point metrics

def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def reliability(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    rows = [{'bin': b, 'n': int((idx == b).sum()), 'mean_pred': p[idx == b].mean(), 'mean_obs': y[idx == b].mean()}
            for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows)


def clip_p(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)


def game_losses(df, ml_col='ml_prob', spread_col='spread_pred', total_col='total_pred'):
    """Per-game losses, indexed by gameId, for paired comparisons."""
    out = pd.DataFrame(index=df['gameId'].values)
    out['seasonYear'] = df['seasonYear'].values
    out['gameDate'] = pd.to_datetime(df['gameDate']).values
    if ml_col in df:
        p, y = clip_p(df[ml_col]), df['target_home_win'].to_numpy(float)
        out['ll'] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        out['brier'] = (p - y) ** 2
    if spread_col in df:
        e = df['target_point_diff'].to_numpy(float) - df[spread_col].to_numpy(float)
        out['ae_spread'], out['se_spread'] = np.abs(e), e ** 2
    if total_col in df:
        e = df['target_total_pts'].to_numpy(float) - df[total_col].to_numpy(float)
        out['ae_total'], out['se_total'] = np.abs(e), e ** 2
    return out


def core_metrics(df):
    """Pooled primary metrics for one prediction frame."""
    d = df.dropna(subset=['ml_prob'])
    y, p = d['target_home_win'].to_numpy(float), clip_p(d['ml_prob'])
    m = {'games': len(d),
         'ml_logloss': float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
         'ml_brier': float(((p - y) ** 2).mean()), 'ml_ece': ece(y, p), 'ml_acc': float(((p > .5) == y).mean()),
         'ml_distinct_probs': int(pd.Series(np.round(d['ml_prob'], 6)).nunique())}
    for tag, pred, act in (('spread', 'spread_pred', 'target_point_diff'), ('total', 'total_pred', 'target_total_pts')):
        s = df.dropna(subset=[pred])
        e = s[act].to_numpy(float) - s[pred].to_numpy(float)
        m.update({f'{tag}_mae': float(np.abs(e).mean()), f'{tag}_rmse': float(np.sqrt((e ** 2).mean())),
                  f'{tag}_bias': float(e.mean())})
    return m


def per_season(df, fn=core_metrics):
    return pd.DataFrame({s: fn(g) for s, g in df.groupby('seasonYear')}).T


# ------------------------------------------------------------------ paired comparison

def _day_index(dates):
    codes, uniques = pd.factorize(pd.to_datetime(dates))
    return codes, len(uniques)


def paired(base, exp, col, lower_is_better=True, n_boot=N_BOOT, subset=None):
    """
    Compare per-game loss `col` of an experiment with the baseline on the games both have.
    Returns pooled mean difference (exp - base; negative = experiment better when lower_is_better),
    95% day-bootstrap CI, per-season differences, and how many seasons improved.
    """
    j = base[[col, 'seasonYear', 'gameDate']].join(exp[[col]], rsuffix='_exp', how='inner').dropna()
    if subset is not None:
        j = j[j.index.isin(subset)]
    d = (j[col + '_exp'] - j[col]).to_numpy()
    codes, nd = _day_index(j['gameDate'])
    sums = np.bincount(codes, weights=d, minlength=nd)
    cnts = np.bincount(codes, minlength=nd)
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, nd, size=(n_boot, nd))
    boots = sums[draws].sum(1) / cnts[draws].sum(1)
    seas = j.assign(d=d).groupby('seasonYear')['d'].mean()
    better = (seas < 0) if lower_is_better else (seas > 0)
    return {'n': len(j), 'base_mean': float(j[col].mean()), 'diff': float(d.mean()), 'ci_lo': float(np.quantile(boots, .025)), 'ci_hi': float(np.quantile(boots, .975)),
            'seasons_improved': int(better.sum()), 'seasons': int(len(seas)), 'per_season': seas.round(5).to_dict(),
            'ci_excludes_zero_in_favor': bool((np.quantile(boots, .975) < 0) if lower_is_better else (np.quantile(boots, .025) > 0))}


# ------------------------------------------------------------------ bets

def _payout(odds):
    odds = np.asarray(odds, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(odds < 0, 100.0 / -odds, odds / 100.0)


def _imp(odds):
    odds = np.asarray(odds, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(odds < 0, -odds / (-odds + 100), 100 / (odds + 100))


def make_bets(df, thr=0.05, markets=('ml', 'spread', 'total'), rule='edge', flat_points=1.0,
              sigma_spread=None, sigma_total=None):
    """
    One row per bet. rule='edge': bet when P(win) - break-even > thr (team/betting.py).
    rule='flat': bet when |pred - line| > flat_points (spread/total only; ML uses the edge rule).
    Opening lines/prices (the model inputs) are used for settlement, as in the walk-forward backtests.
    Columns: gameId, gameDate, seasonYear, market, side, edge, price, line, close_line, won, push, profit (per $100).
    """
    frames = []
    base = df[['gameId', 'gameDate', 'seasonYear']].copy()
    if 'ml' in markets:
        p = df['ml_prob'].to_numpy(float)
        oh, oa = df['odds_home_ml'].to_numpy(float), df['odds_away_ml'].to_numpy(float)
        eh, ea = p - _imp(oh), (1 - p) - _imp(oa)
        home = eh >= ea
        edge = np.where(home, eh, ea)
        ok = np.isfinite(edge) & (edge > thr)
        price = np.where(home, oh, oa)
        win_home = df['target_home_win'].to_numpy(float) == 1
        won = np.where(home, win_home, ~win_home)
        d = base[ok].copy().assign(market='ml', side=np.where(home, 'home', 'away')[ok], edge=edge[ok], price=price[ok],
                                   line=np.nan, won=won[ok], push=False)
        d['profit'] = np.where(d['won'], _payout(d['price']) * 100, -100.0)
        d['close_prob'] = np.where(home, df['close_home_prob'].to_numpy(float), 1 - df['close_home_prob'].to_numpy(float))[ok] if 'close_home_prob' in df else np.nan
        d['open_prob'] = _imp(d['price'])
        d['close_line'] = np.nan
        frames.append(d)
    for mk, pred, line_col, act, sig_col, close_col in (
            ('spread', 'spread_pred', 'odds_home_spread', 'target_point_diff', 'sigma_spread', 'close_home_spread'),
            ('total', 'total_pred', 'odds_total', 'target_total_pts', 'sigma_total', 'close_total')):
        if mk not in markets:
            continue
        pr, ln = df[pred].to_numpy(float), df[line_col].to_numpy(float)
        sig = df[sig_col].to_numpy(float) if sig_col in df else np.full(len(df), bt.DEFAULT_SPREAD_STD if mk == 'spread' else bt.DEFAULT_TOTAL_STD)
        override = sigma_spread if mk == 'spread' else sigma_total
        if override is not None:
            sig = np.asarray(override, float) if np.ndim(override) else np.full(len(df), float(override))
        pe = pr + ln if mk == 'spread' else pr - ln
        edge = norm.cdf(np.abs(pe) / sig) - BREAK_EVEN_110
        ok = np.isfinite(pe) & (np.abs(pe) > flat_points if rule == 'flat' else edge > thr)
        first = pe > 0                      # home / over
        a = df[act].to_numpy(float)
        res = (a + ln) if mk == 'spread' else (a - ln)       # >0: home covers / over hits
        push = res == 0
        won = np.where(first, res > 0, res < 0)
        sel = ok & ~push
        d = base[sel].copy().assign(
            market=mk, side=np.where(first, 'home' if mk == 'spread' else 'over', 'away' if mk == 'spread' else 'under')[sel],
            edge=edge[sel], price=-110.0, line=ln[sel], won=won[sel], push=False)
        d['profit'] = np.where(d['won'], 100 / 1.1, -100.0)
        cl = df[close_col].to_numpy(float) if close_col in df else np.full(len(df), np.nan)
        d['close_line'] = cl[sel]
        d['close_prob'] = np.nan
        d['open_prob'] = np.nan
        frames.append(d)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out['gameDate'] = pd.to_datetime(out['gameDate'])
    return out


def clv_points(bets):
    """Closing line value in points for spread/total bets (positive = the line moved toward our side)."""
    b = bets[bets['market'].isin(['spread', 'total'])].dropna(subset=['close_line'])
    sign = np.where(b['side'].isin(['home']), 1.0, np.where(b['side'] == 'away', -1.0, np.where(b['side'] == 'over', -1.0, 1.0)))
    # home spread bet at line s: better if the close is lower (more negative)  -> clv = s - close
    # away: -(s - close); over at total t: better if close is higher -> close - t; under: t - close
    clv = np.where(b['market'] == 'spread', sign * (b['line'] - b['close_line']),
                   np.where(b['side'] == 'over', b['close_line'] - b['line'], b['line'] - b['close_line']))
    return pd.Series(clv, index=b.index)


def max_drawdown(profit_in_time_order):
    cum = np.cumsum(np.asarray(profit_in_time_order, float)) / 100.0
    if len(cum) == 0:
        return 0.0
    return float((np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum).max())


def bet_metrics(bets, n_boot=N_BOOT):
    """bets, hit rate, ROI (profit/stake) and max drawdown (in $100 units) with day-bootstrap 95% CIs."""
    if len(bets) == 0:
        return {'bets': 0}
    b = bets.sort_values('gameDate').reset_index(drop=True)
    codes, nd = _day_index(b['gameDate'])
    prof = b['profit'].to_numpy(float)
    won = b['won'].to_numpy(float)
    sums = np.bincount(codes, weights=prof, minlength=nd)
    wins = np.bincount(codes, weights=won, minlength=nd)
    cnts = np.bincount(codes, minlength=nd).astype(float)
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, nd, size=(n_boot, nd))
    roi_b = sums[draws].sum(1) / (100 * cnts[draws].sum(1)) * 100
    hit_b = wins[draws].sum(1) / cnts[draws].sum(1) * 100
    by_day = [prof[codes == i] for i in range(nd)]
    dd_b = np.array([max_drawdown(np.concatenate([by_day[i] for i in row])) for row in draws[:min(n_boot, 300)]])
    q = lambda a: (float(np.quantile(a, .025)), float(np.quantile(a, .975)))
    return {'bets': len(b), 'hit_rate': float(won.mean() * 100), 'hit_ci': q(hit_b), 'roi': float(prof.sum() / (100 * len(b)) * 100),
            'roi_ci': q(roi_b), 'max_drawdown': max_drawdown(prof), 'dd_ci': q(dd_b)}


def roi_diff_ci(bets_a, bets_b, n_boot=N_BOOT):
    """Day-bootstrap CI for ROI(a) - ROI(b) (bets of two rules on the same period)."""
    days = pd.to_datetime(pd.concat([bets_a['gameDate'], bets_b['gameDate']])).drop_duplicates().sort_values().reset_index(drop=True)
    m = {d: i for i, d in enumerate(days)}
    nd = len(days)

    def agg(b):
        idx = b['gameDate'].map(m).to_numpy()
        return np.bincount(idx, weights=b['profit'].to_numpy(float), minlength=nd), np.bincount(idx, minlength=nd).astype(float)
    sa, ca = agg(bets_a)
    sb, cb = agg(bets_b)
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, nd, size=(n_boot, nd))
    ra = sa[draws].sum(1) / np.maximum(ca[draws].sum(1), 1) ; rb = sb[draws].sum(1) / np.maximum(cb[draws].sum(1), 1)
    diff = ra - rb
    return {'roi_diff': float(sa.sum() / max(ca.sum(), 1) - sb.sum() / max(cb.sum(), 1)),
            'ci': (float(np.quantile(diff, .025)), float(np.quantile(diff, .975)))}
