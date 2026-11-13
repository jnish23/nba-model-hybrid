"""
Forward-test monitor: pure functions (pandas / numpy / scipy only, DataFrames in and out, no database access) behind scripts/props_forward_monitor.py.

Conventions
- A stake is a bet settled win or loss. A void (player did not play) and a push refund and are not stakes. ROI is profit per unit staked.
- Three prices: median (the primary view), best (read as "value if a limit order at that price fills") and flat -110.
- Every interval is a bootstrap over game dates (the cluster), 2.5th to 97.5th percentile. Bets on the same night share the same market conditions, so they are not independent.
- CLV is `clv_prob` x 100 (points of consensus fair probability of the bet side, close minus at the bet time; positive means the market moved toward the bet).
- The decision rule of reports/props/forward_test_preregistration.md is `sequential_status`.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

STAKE_STATUSES = ('win', 'loss')
EV_EDGES = [-np.inf, 0.03, 0.04, 0.05, 0.07, np.inf]
EV_LABELS = ['<3%', '3-4%', '4-5%', '5-7%', '7%+']
MIN_EDGES = [-np.inf, 20, 28, 34, np.inf]
MIN_LABELS = ['<20', '20-28', '28-34', '34+']
LOW_N = 50
STAT_COLUMN = {'rebounds': 'rebounds', 'assists': 'assists', 'threes': 'threes'}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Cluster bootstrap
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _codes(cluster):
    codes, uniques = pd.factorize(pd.Series(np.asarray(cluster)), sort=True)
    return codes, len(uniques)


def _boot_counts(n_groups, n_boot, seed):
    """(n_boot, n_groups) resample counts of the clusters; one multinomial draw per replicate, so a bootstrap statistic is a matrix product."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_boot).astype(float)


def _ratio_se(num_g, den_g):
    """Cluster-robust SE of sum(num) / sum(den) (linearisation)."""
    den = den_g.sum()
    if den <= 0:
        return np.nan
    r = num_g.sum() / den
    return float(np.sqrt(np.sum((num_g - r * den_g) ** 2)) / den)


def _cluster_ratio(num, den, cluster, n_boot, seed, alpha):
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    cl = np.asarray(cluster)[ok]
    num, den = num[ok], den[ok]
    out = {'point': np.nan, 'lo': np.nan, 'hi': np.nan, 'se': np.nan, 'p_le0': np.nan, 'n': int((den > 0).sum()) if len(den) else 0, 'n_clusters': 0}
    if len(num) == 0 or den.sum() <= 0:
        return out
    codes, g = _codes(cl)
    num_g = np.bincount(codes, weights=num, minlength=g)
    den_g = np.bincount(codes, weights=den, minlength=g)
    keep = den_g > 0                                   # clusters with no stake carry no information about the ratio
    num_g, den_g = num_g[keep], den_g[keep]
    g = len(num_g)
    out.update(point=float(num_g.sum() / den_g.sum()), se=_ratio_se(num_g, den_g), n_clusters=int(g))
    counts = _boot_counts(g, n_boot, seed)
    bn, bd = counts @ num_g, counts @ den_g
    ratio = np.where(bd > 0, bn / np.where(bd > 0, bd, 1.0), np.nan)
    ratio = ratio[np.isfinite(ratio)]
    out['lo'], out['hi'] = (float(x) for x in np.percentile(ratio, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    out['p_le0'] = float(np.mean(ratio <= 0))
    return out


def cluster_bootstrap_roi(profit, stake, cluster, n_boot=2000, seed=0, alpha=0.05):
    """
    ROI = sum(profit) / sum(stake) with a date-clustered bootstrap. Returns dict(point, lo, hi, se, p_le0, n, n_clusters):
    p_le0 is the share of bootstrap replicates with ROI <= 0; se is the analytic cluster-robust standard error.
    """
    return _cluster_ratio(profit, stake, cluster, n_boot, seed, alpha)


def cluster_bootstrap_mean(values, cluster, n_boot=2000, seed=0, alpha=0.05):
    """Mean of `values` (NaN dropped) with a clustered bootstrap interval and the analytic clustered SE. Same return keys as cluster_bootstrap_roi."""
    values = np.asarray(values, float)
    return _cluster_ratio(values, np.where(np.isfinite(values), 1.0, 0.0), cluster, n_boot, seed, alpha)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing the settled record
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _side_prob(side, p_over):
    return np.where(np.asarray(side) == 'over', p_over, 1.0 - p_over)


def prepare_bets(bets, results):
    """
    prop_bets joined to prop_results (settled bets only). Adds: stake (1 for win/loss), win, profit_median / profit_best / profit_flat (0 for non-stakes), p_market_side, p_star_side,
    p_model_side (probability of the bet side), hit_over (1/0 when the player played and the stat did not land on the line, else NaN), clv_points.
    """
    res_cols = ['bet_id', 'status_stands', 'profit_stands_best', 'profit_stands_median', 'profit_stands_flat', 'clv_prob', 'line_move_favour', 'close_line', 'y', 'played', 'minutes_actual']
    res = results[[c for c in res_cols if c in results.columns]]
    d = bets.merge(res, on='bet_id', how='inner', validate='one_to_one').copy()
    d['stake'] = d['status_stands'].isin(STAKE_STATUSES).astype(float)
    d['win'] = (d['status_stands'] == 'win').astype(float)
    for name in ('best', 'median', 'flat'):
        d[f'profit_{name}'] = np.where(d['stake'] > 0, d[f'profit_stands_{name}'].astype(float), 0.0)
    d['p_market_side'] = _side_prob(d['side'], d['p_over_market'].astype(float))
    d['p_star_side'] = _side_prob(d['side'], d['p_over_star'].astype(float))
    d['p_model_side'] = _side_prob(d['side'], d['p_over_model'].astype(float))
    played = d['played'].fillna(0).astype(float) > 0
    y = d['y'].astype(float)
    d['hit_over'] = np.where(played & y.notna() & (y != d['line'].astype(float)), (y > d['line'].astype(float)).astype(float), np.nan)
    d['clv_points'] = 100.0 * d['clv_prob'].astype(float)
    return d


def add_segments(d):
    """Adds the derived segment columns: slot, stat_side, ev_bucket, minutes_bucket, week (Monday of the game's week)."""
    d = d.copy()
    d['slot'] = 'T-' + d['minutes_before'].astype(int).astype(str)
    d['stat_side'] = d['stat'].astype(str) + ' ' + d['side'].astype(str)
    d['ev_bucket'] = pd.Categorical(pd.cut(d['ev_best'].astype(float), EV_EDGES, labels=EV_LABELS, right=False), categories=EV_LABELS, ordered=True)
    d['minutes_bucket'] = pd.Categorical(pd.cut(d['expected_minutes'].astype(float), MIN_EDGES, labels=MIN_LABELS, right=False), categories=MIN_LABELS, ordered=True)
    dates = pd.to_datetime(d['game_date'])
    d['week'] = (dates - pd.to_timedelta(dates.dt.weekday, unit='D')).dt.strftime('%Y-%m-%d')
    return d


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Segment table
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _segment_row(g, n_boot, seed, min_n):
    stakes = g['stake'].to_numpy()
    n_st = int(stakes.sum())
    row = {'n_bets': len(g), 'n_stakes': n_st, 'wins': int(g['win'].sum())}
    on = stakes > 0
    row['win_rate'] = float(g['win'].sum() / n_st) if n_st else np.nan
    row['mean_p_market'] = float(g.loc[on, 'p_market_side'].mean()) if n_st else np.nan
    row['mean_p_star'] = float(g.loc[on, 'p_star_side'].mean()) if n_st else np.nan
    for name in ('median', 'best'):
        r = cluster_bootstrap_roi(g[f'profit_{name}'], stakes, g['game_date'], n_boot, seed)
        row[f'roi_{name}'], row[f'roi_{name}_lo'], row[f'roi_{name}_hi'] = r['point'], r['lo'], r['hi']
    row['roi_flat'] = float(g['profit_flat'].sum() / n_st) if n_st else np.nan
    c = cluster_bootstrap_mean(g['clv_points'], g['game_date'], n_boot=200, seed=seed)
    row['clv_mean'], row['clv_n'], row['clv_se'] = c['point'], c['n'], c['se']
    row['low_n'] = bool(n_st < min_n)
    return row


def segment_table(bets, by, n_boot=2000, seed=0, min_n=LOW_N, include_all=False):
    """
    One row per segment of `prepare_bets` output. `by`: a column name or list of names (derived ones: slot, stat_side, ev_bucket, minutes_bucket, week).
    Columns: n_bets, n_stakes (win/loss only), wins, win_rate, mean_p_market and mean_p_star (of the bet side, over stakes), ROI at median and best price with date-clustered intervals,
    ROI flat -110, CLV mean in points with n and clustered SE, low_n (n_stakes < min_n).
    """
    d = add_segments(bets)
    cols = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, g in d.groupby(cols, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(cols, key)), **_segment_row(g, n_boot, seed, min_n)})
    out = pd.DataFrame(rows)
    if include_all:
        allrow = {c: 'ALL' for c in cols}
        out = pd.concat([pd.DataFrame([{**allrow, **_segment_row(d, n_boot, seed, min_n)}]), out], ignore_index=True)
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Information slope, with and without an intercept
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _slope_estimates(T):
    """
    T: (..., K, 5) sums of (n, x, y, xx, xy) per stat. Returns three arrays (..., K+1): slope through the origin, slope with a per-stat intercept, mean intercept;
    the last column is the pooled estimate (the pooled intercept version removes each stat's own means, so a stat-specific bias cannot leak into the slope).
    """
    n, sx, sy, sxx, sxy = (T[..., i] for i in range(5))
    with np.errstate(divide='ignore', invalid='ignore'):
        sxx_c = sxx - sx ** 2 / n
        sxy_c = sxy - sx * sy / n
        b_origin = sxy / sxx
        b_icpt = sxy_c / sxx_c
        a_icpt = sy / n - b_icpt * sx / n
        N, SX, SY = n.sum(-1), sx.sum(-1), sy.sum(-1)
        p_origin = sxy.sum(-1) / sxx.sum(-1)
        p_icpt = sxy_c.sum(-1) / sxx_c.sum(-1)
        p_a = SY / N - p_icpt * SX / N
    return (np.concatenate([b_origin, p_origin[..., None]], -1), np.concatenate([b_icpt, p_icpt[..., None]], -1), np.concatenate([a_icpt, p_a[..., None]], -1))


def information_slope(df, n_boot=2000, seed=0, alpha=0.05):
    """
    Regress (hit_over - p_market) on (p_model - p_market): (a) through the origin, the quantity the frozen k came from; (b) with an intercept per stat, which absorbs a market-wide over-bias.
    df columns: stat, game_date, hit_over, p_market, p_model (rows with a missing value are dropped). One row per stat plus 'pooled' (pooled slopes use the stat-specific intercepts in (b)).
    Intervals are date-clustered bootstraps. `over_bias` = mean(hit_over - p_market): positive when overs hit more often than the market's fair probability says.
    """
    d = df[['stat', 'game_date', 'hit_over', 'p_market', 'p_model']].dropna().copy()
    cols = ['stat', 'n', 'n_dates', 'over_bias', 'slope_origin', 'so_lo', 'so_hi', 'slope_icpt', 'si_lo', 'si_hi', 'intercept', 'a_lo', 'a_hi', 'p_icpt_le0']
    if d.empty:
        return pd.DataFrame(columns=cols)
    stats = sorted(d['stat'].unique())
    k = len(stats)
    x = (d['p_model'] - d['p_market']).to_numpy(float)
    y = (d['hit_over'] - d['p_market']).to_numpy(float)
    codes, g = _codes(d['game_date'])
    sidx = d['stat'].map({s: i for i, s in enumerate(stats)}).to_numpy()
    T = np.zeros((g, k, 5))
    for j, v in enumerate((np.ones_like(x), x, y, x * x, x * y)):
        np.add.at(T[..., j], (codes, sidx), v)
    point = _slope_estimates(T.sum(0))
    counts = _boot_counts(g, n_boot, seed)
    boot = _slope_estimates((counts @ T.reshape(g, k * 5)).reshape(n_boot, k, 5))
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []
    for i, name in enumerate(stats + ['pooled']):
        m = np.ones(len(d), bool) if name == 'pooled' else (sidx == i)
        row = {'stat': name, 'n': int(m.sum()), 'n_dates': int(pd.Series(codes[m]).nunique()), 'over_bias': float(y[m].mean())}   # y = hit_over - p_market
        for key, (est, b) in {'o': (point[0], boot[0]), 'i': (point[1], boot[1])}.items():
            lo, hi = np.nanpercentile(b[:, i], [lo_q, hi_q]) if np.isfinite(b[:, i]).any() else (np.nan, np.nan)
            if key == 'o':
                row['slope_origin'], row['so_lo'], row['so_hi'] = float(est[i]), float(lo), float(hi)
            else:
                row['slope_icpt'], row['si_lo'], row['si_hi'] = float(est[i]), float(lo), float(hi)
                row['p_icpt_le0'] = float(np.nanmean(b[:, i] <= 0))
        a_lo, a_hi = np.nanpercentile(boot[2][:, i], [lo_q, hi_q]) if np.isfinite(boot[2][:, i]).any() else (np.nan, np.nan)
        row['intercept'], row['a_lo'], row['a_hi'] = float(point[2][i]), float(a_lo), float(a_hi)
        rows.append(row)
    return pd.DataFrame(rows)[cols]


def slope_frame_from_bets(prep):
    """Slope input from the settled bets: one row per bet (the bets are selected, so this is the slope on the bet set, not on the market)."""
    d = prep.rename(columns={'p_over_market': 'p_market', 'p_over_model': 'p_model'})
    return d[['stat', 'game_date', 'hit_over', 'p_market', 'p_model']]


def market_frame(signals, outcomes, stats=None):
    """
    One row per market (game, player, stat) at its earliest evaluated slot, from prop_signals, joined to outcomes (game_id, player_id, stat, y, played).
    Returns stat, game_date, hit_over, p_market, p_model, line, slot_minutes. hit_over is NaN when the player did not play, the outcome is unknown or the stat landed on the line.
    """
    s = signals.copy()
    if stats is not None:
        s = s[s['stat'].isin(stats)]
    s = s.sort_values('slot_minutes', ascending=False).drop_duplicates(['game_id', 'player_id', 'stat'])
    o = outcomes[['game_id', 'player_id', 'stat', 'y', 'played']].drop_duplicates(['game_id', 'player_id', 'stat'])
    m = s.merge(o, on=['game_id', 'player_id', 'stat'], how='left')
    played = m['played'].fillna(0).astype(float) > 0
    y = m['y'].astype(float)
    line = m['line'].astype(float)
    m['hit_over'] = np.where(played & y.notna() & (y != line), (y > line).astype(float), np.nan)
    return m[['game_id', 'player_id', 'stat', 'game_date', 'slot_minutes', 'line', 'p_market', 'p_model', 'hit_over']]


def outcomes_from_box(box, keys):
    """
    box: game_id, player_id, minutes, rebounds, assists, threes (one row per player per finished game). keys: game_id, player_id, stat to resolve. Returns keys with y and played;
    keys whose game is not in the box scores are dropped (not yet played or not yet synced); a player missing from a game that is present did not play.
    """
    k = keys[['game_id', 'player_id', 'stat']].drop_duplicates()
    k = k[k['game_id'].isin(set(box['game_id']))]
    b = box.copy()
    b['ra'] = b['rebounds'] + b['assists']
    b['pra'] = b['ra'] + b.get('points', 0)
    m = k.merge(b, on=['game_id', 'player_id'], how='left')
    played = (m['minutes'].fillna(0) > 0)
    y = np.full(len(m), np.nan)
    for stat in ('rebounds', 'assists', 'threes', 'ra', 'pra', 'points'):
        if stat in m.columns:
            sel = (m['stat'] == stat).to_numpy()
            y[sel] = m.loc[sel, stat].to_numpy(float)
    m['y'] = np.where(played, y, np.nan)
    m['played'] = played.astype(int)
    return m[['game_id', 'player_id', 'stat', 'y', 'played']]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calibration and CLV break-even
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calibration(prep):
    """
    On settled stakes: expected wins under p* (the rule's probability of the bet side) against actual wins, with the date-clustered SE of the difference, plus the Brier score of p*, p_model and
    p_market for the bet side. Returns dict. A negative `diff` means the bets won less often than p* says.
    """
    d = prep[prep['stake'] > 0]
    if d.empty:
        return {'n': 0}
    codes, g = _codes(d['game_date'])
    resid = d['win'].to_numpy() - d['p_star_side'].to_numpy()
    resid_g = np.bincount(codes, weights=resid, minlength=g)
    diff = float(resid.sum())
    se = float(np.sqrt(np.sum(resid_g ** 2)))
    out = {'n': int(len(d)), 'actual_wins': float(d['win'].sum()), 'expected_wins_pstar': float(d['p_star_side'].sum()), 'diff': diff, 'diff_se': se,
           'z': diff / se if se > 0 else np.nan, 'mean_p_star': float(d['p_star_side'].mean()), 'mean_p_market': float(d['p_market_side'].mean()),
           'mean_p_model': float(d['p_model_side'].mean()), 'win_rate': float(d['win'].mean())}
    for name, col in (('pstar', 'p_star_side'), ('pmodel', 'p_model_side'), ('pmarket', 'p_market_side')):
        out[f'brier_{name}'] = float(np.mean((d['win'].to_numpy() - d[col].to_numpy()) ** 2))
    return out


def clv_breakeven(prep):
    """
    Break-even CLV from the record's own prices. A bet at decimal price d wins with the probability the close implies, p_close = p_bet + CLV, if the close is right.
    ROI = p_close * d - 1, so ROI is zero when CLV = 1/d - p_bet (the fair probability of the bet side at the bet time). Mean over bets that have a CLV value, at the median, best and flat prices,
    next to the mean CLV actually earned and the ROI that CLV implies (mean of (p_bet + CLV) * d - 1). Returns dict (points of probability, ROI in fraction).
    """
    d = prep[prep['clv_prob'].notna() & (prep['stake'] > 0)]
    if d.empty:
        return {'n': 0}
    out = {'n': int(len(d)), 'clv_mean_points': float(d['clv_points'].mean())}
    for name, col in (('median', 'dec_median'), ('best', 'dec_best'), ('flat', 'dec_flat')):
        price = d[col].astype(float)
        out[f'breakeven_clv_{name}_points'] = float(100 * (1.0 / price - d['p_market_side']).mean())
        out[f'clv_implied_roi_{name}'] = float(((d['p_market_side'] + d['clv_prob']) * price - 1.0).mean())
        out[f'mean_price_{name}'] = float(price.mean())
    out['realised_roi_median_same_bets'] = float(d['profit_median'].sum() / d['stake'].sum())
    out['realised_roi_best_same_bets'] = float(d['profit_best'].sum() / d['stake'].sum())
    return out


def fill_summary(prep, fills):
    """
    Summary of prop_fill_checks (bet_id, book_still_at_or_better_next, best_still_at_or_better_next, minutes_price_available, max_best_price_before_tip) for the settled bets:
    share of 0/1 columns, quantiles of the numeric ones, and best-price ROI split by best_still_at_or_better_next. Returns dict.
    """
    f = fills.drop_duplicates('bet_id')
    d = prep.merge(f, on='bet_id', how='inner', suffixes=('', '_fill'))
    out = {'n': int(len(d))}
    if d.empty:
        return out
    for c in ('book_still_at_or_better_next', 'best_still_at_or_better_next'):
        if c in d:
            v = pd.to_numeric(d[c], errors='coerce')
            out[f'share_{c}'] = float(v.mean())
            out[f'n_{c}'] = int(v.notna().sum())
    for c in ('minutes_price_available', 'max_best_price_before_tip'):
        if c in d:
            v = pd.to_numeric(d[c], errors='coerce').dropna()
            if len(v):
                out[f'{c}_q25_q50_q75'] = tuple(float(x) for x in np.percentile(v, [25, 50, 75]))
    if 'max_best_price_before_tip' in d:
        v = pd.to_numeric(d['max_best_price_before_tip'], errors='coerce')
        out['share_max_best_price_at_or_above_bet'] = float((v >= d['dec_best'] - 1e-9)[v.notna()].mean()) if v.notna().any() else np.nan
    if 'best_still_at_or_better_next' in d:
        split = {}
        flag = pd.to_numeric(d['best_still_at_or_better_next'], errors='coerce')
        for val in (1, 0):
            g = d[flag == val]
            if g['stake'].sum() > 0:
                split[int(val)] = {'n_stakes': int(g['stake'].sum()), 'roi_best': float(g['profit_best'].sum() / g['stake'].sum())}
        out['best_roi_by_best_still_next'] = split
    return out


CONFIG_COLUMNS = ['model_version', 'model_sha256', 'rule_sha256', 'roster_rule']


def config_segments(bets):
    """
    Distinct (model_version, model_sha256, rule_sha256, roster_rule) among the bets with counts, first and last game date and the slots used. The forward test is valid only while
    this has ONE row for the test bets; a second row means a model, rule or roster-rule change, which starts a new numbered segment.
    """
    cols = [c for c in CONFIG_COLUMNS if c in bets.columns]
    d = bets.copy()
    d[cols] = d[cols].fillna('(blank)')
    g = d.groupby(cols, sort=True).agg(n_bets=('bet_id', 'size'), first_date=('game_date', 'min'), last_date=('game_date', 'max'),
                                       slots=('minutes_before', lambda s: ','.join(f'T-{int(k)}' for k in sorted(s.unique(), reverse=True))))
    return g.reset_index()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Sample size and the sequential decision rule
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def per_date_sums(profit, stake, cluster):
    """Per game date: stakes, profit."""
    d = pd.DataFrame({'c': np.asarray(cluster), 'p': np.asarray(profit, float), 's': np.asarray(stake, float)}).dropna()
    g = d.groupby('c').agg(stakes=('s', 'sum'), profit=('p', 'sum'))
    return g[g['stakes'] > 0]


def required_dates(profit, stake, cluster, half_width, z=1.96):
    """
    Game dates needed for the 95% half-width of the clustered ROI to reach `half_width` (in ROI units, e.g. 0.02), from this record's own per-date variance:
    SE^2 = sum_g (profit_g - ROI * stake_g)^2 / (sum_g stake_g)^2, so with G dates of the same mix, SE = sqrt(mean(resid^2)) / (mean(stake) * sqrt(G)).
    Returns dict(dates, stakes, mean_stakes_per_date, roi, sd_resid_per_date, half_width_now, n_dates_now).
    """
    g = per_date_sums(profit, stake, cluster)
    roi = g['profit'].sum() / g['stakes'].sum()
    resid = g['profit'] - roi * g['stakes']
    msq = float(np.mean(resid ** 2))
    mean_s = float(g['stakes'].mean())
    dates = (z * np.sqrt(msq) / (mean_s * half_width)) ** 2
    return {'dates': float(dates), 'stakes': float(dates * mean_s), 'mean_stakes_per_date': mean_s, 'roi': float(roi), 'sd_resid_per_date': float(np.sqrt(msq)),
            'half_width_now': float(z * np.sqrt(msq) / (mean_s * np.sqrt(len(g)))), 'n_dates_now': int(len(g)), 'target': float(half_width)}


def conditional_power(theta_hat, se_now, t, theta_alt, z_crit=1.96):
    """
    Probability that the final test (ROI lower bound > 0, i.e. Z > z_crit) passes, given the estimate at information fraction t = stakes / horizon and a true ROI of theta_alt for the rest.
    B-value form: Z_final ~ N(B_t + mu (1 - t), 1 - t) with B_t = Z_t sqrt(t), mu = theta_alt / SE_final, SE_final = se_now sqrt(t).
    """
    if not np.isfinite(theta_hat) or not np.isfinite(se_now) or se_now <= 0 or t <= 0:
        return np.nan
    if t >= 1:
        return float(theta_hat - z_crit * se_now > 0)
    b_t = (theta_hat / se_now) * np.sqrt(t)
    mu = theta_alt / (se_now * np.sqrt(t))
    return float(1.0 - norm.cdf((z_crit - b_t - mu * (1.0 - t)) / np.sqrt(1.0 - t)))


def _cutoff_date(cum_target, g):
    """First game date whose cumulative stakes reach cum_target (g indexed by sorted date with column 'stakes'), or None."""
    cum = g['stakes'].cumsum()
    hit = cum[cum >= cum_target - 1e-9]
    return None if hit.empty else hit.index[0]


def sequential_status(profit, stake, cluster, horizon_stakes, horizon_date=None, today=None, alt_roi=0.03, cp_floor=0.10, looks=(1 / 3, 2 / 3), n_boot=4000, seed=20261112, min_dates=30):
    """
    The pre-registered rule on the median-price ROI of the forward-test bets (profit/stake per bet; cluster = game date as 'YYYY-MM-DD').
    - Horizon: the first game date by which `horizon_stakes` settled stakes have accumulated, or `horizon_date` (reached when `today` >= horizon_date), whichever comes first. The final test uses
      the data through that date: PASS when the lower bound of the 95% date-clustered bootstrap interval of the ROI is above 0 (and there are at least `min_dates` game dates), otherwise FAIL.
    - Interim looks (futility only, never for success): at each look fraction of the horizon, on the data through the first game date at which cumulative stakes reach that fraction, STOP-FUTILITY when
      the conditional power of passing at the horizon, taking a true ROI of `alt_roi` for the remaining data, is below `cp_floor`.
    - Otherwise CONTINUE.
    Returns dict(status, reason, n_stakes, n_dates, roi, lo, hi, p_le0, t, cp, looks=[...]).
    """
    d = pd.DataFrame({'c': np.asarray(cluster).astype(str), 'p': np.asarray(profit, float), 's': np.asarray(stake, float)}).dropna()
    d = d[d['s'] > 0]
    g = d.groupby('c').agg(stakes=('s', 'sum'), profit=('p', 'sum')).sort_index()
    n_stakes = float(g['stakes'].sum()) if len(g) else 0.0
    base = {'n_stakes': int(n_stakes), 'n_dates': int(len(g)), 'horizon_stakes': int(horizon_stakes), 'horizon_date': horizon_date, 'looks': []}

    def roi_on(sub):
        mask = d['c'].isin(sub.index)
        r = cluster_bootstrap_roi(d.loc[mask, 'p'], d.loc[mask, 's'], d.loc[mask, 'c'], n_boot=n_boot, seed=seed)
        return r

    stake_cut = _cutoff_date(horizon_stakes, g) if len(g) else None
    date_reached = horizon_date is not None and today is not None and str(today) >= str(horizon_date)
    final_due = stake_cut is not None or date_reached
    if final_due:
        cut = stake_cut
        if date_reached and (cut is None or str(horizon_date) < str(cut)):
            cut = str(horizon_date)
        sub = g[g.index <= cut]
        r = roi_on(sub)
        enough = len(sub) >= min_dates
        passed = bool(np.isfinite(r['lo']) and r['lo'] > 0 and enough)
        reason = (f'horizon reached ({int(sub["stakes"].sum())} stakes through {cut}): lower bound {r["lo"]:+.4f} ' + ('> 0' if passed else
                  ('<= 0' if enough else f'and only {len(sub)} game dates (< {min_dates})')))
        return {**base, 'status': 'PASS' if passed else 'FAIL', 'reason': reason, 'cutoff': cut, 'n_stakes': int(sub['stakes'].sum()), 'n_dates': int(len(sub)), 'roi': r['point'], 'lo': r['lo'],
                'hi': r['hi'], 'p_le0': r['p_le0'], 't': float(sub['stakes'].sum() / horizon_stakes), 'cp': np.nan}
    # interim looks, futility only
    for frac in looks:
        cut = _cutoff_date(frac * horizon_stakes, g) if len(g) else None
        if cut is None:
            continue
        sub = g[g.index <= cut]
        mask = d['c'].isin(sub.index)
        num_g = sub['profit'].to_numpy()
        den_g = sub['stakes'].to_numpy()
        roi = float(num_g.sum() / den_g.sum())
        se = _ratio_se(num_g, den_g)
        t = float(den_g.sum() / horizon_stakes)
        cp = conditional_power(roi, se, t, alt_roi)
        base['looks'].append({'fraction': float(frac), 'cutoff': cut, 'n_stakes': int(den_g.sum()), 'roi': roi, 'se': se, 't': t, 'cp': cp})
        if np.isfinite(cp) and cp < cp_floor:
            r = roi_on(sub)
            return {**base, 'status': 'STOP-FUTILITY', 'reason': f'look at {frac:.2f} of the horizon: conditional power {cp:.3f} < {cp_floor:.2f} (true ROI {alt_roi:+.3f} assumed for the rest)', 'cutoff': cut,
                    'roi': roi, 'lo': r['lo'], 'hi': r['hi'], 'p_le0': r['p_le0'], 't': t, 'cp': cp}
    roi = float(g['profit'].sum() / g['stakes'].sum()) if n_stakes > 0 else np.nan
    r = cluster_bootstrap_roi(d['p'], d['s'], d['c'], n_boot=n_boot, seed=seed) if n_stakes > 0 else {'lo': np.nan, 'hi': np.nan, 'p_le0': np.nan}
    last_cp = base['looks'][-1]['cp'] if base['looks'] else np.nan
    return {**base, 'status': 'CONTINUE', 'reason': 'horizon not reached, no futility trigger', 'roi': roi, 'lo': r['lo'], 'hi': r['hi'], 'p_le0': r['p_le0'], 't': n_stakes / horizon_stakes,
            'cp': last_cp}
