"""
Pure pandas/numpy logic of the props dashboard (no Streamlit): pricing modes, settlement profit, the Board filters, ROI with game-clustered intervals, cumulative curves
and breakdowns. Kept free of Streamlit so it can be unit-tested and reused by the data exporter. Compatible with the dashboard environment (pandas 2.2, numpy 1.26).
"""
import numpy as np
import pandas as pd

FLAT_110 = 1.0 + 100.0 / 110.0
PRICE_MODES = {
    'median': 'Generic: median bettable price',
    'best': 'Best available price',
    'flat': 'Flat -110',
}
TREATMENTS = {
    'stands': 'Bet stands once the player plays',
    'flagged': 'Overs void on early exit at DraftKings / Fanatics',
    'all': 'Overs void on early exit at every book',
}
STAT_LABEL = {'rebounds': 'Rebounds', 'assists': 'Assists', 'threes': '3-pointers', 'ra': 'Reb + Ast', 'points': 'Points', 'pr': 'Pts + Reb', 'pa': 'Pts + Ast', 'pra': 'Pts + Reb + Ast'}


def dec_col(mode):
    return {'median': 'dec_median', 'best': 'dec_best', 'flat': 'dec_flat'}[mode]


def ev_col(mode):
    return {'median': 'ev_median', 'best': 'ev_best', 'flat': 'ev_flat'}[mode]


def american(dec):
    """Decimal odds -> American string (e.g. 1.909 -> -110)."""
    dec = np.asarray(dec, dtype=float)
    out = np.where(dec >= 2.0, np.round((dec - 1.0) * 100.0), -np.round(100.0 / np.maximum(dec - 1.0, 1e-9)))
    return np.array([f'{int(v):+d}' if np.isfinite(v) else '' for v in out])


def profit(status, dec):
    """Profit per unit staked: win pays dec - 1, loss costs 1, push and void are 0."""
    status, dec = np.asarray(status), np.asarray(dec, dtype=float)
    return np.where(status == 'win', dec - 1.0, np.where(status == 'loss', -1.0, 0.0))


def with_profit(df, mode, treatment):
    """Adds `profit` and `stake` (1 for a settled win/loss, else 0) for the chosen pricing mode and settlement treatment."""
    st = df[f'status_{treatment}'].to_numpy()
    out = df.copy()
    out['profit'] = profit(st, out[dec_col(mode)].to_numpy())
    out['stake'] = np.isin(st, ['win', 'loss']).astype(float)
    return out


def side_prob(df, col_over):
    """Probability of the bet's own side from an over probability column."""
    p = df[col_over].to_numpy(dtype=float)
    return np.where(df['side'].to_numpy() == 'over', p, 1.0 - p)


SORTS = {'ev': 'Highest EV at the chosen price', 'ev_now': 'Highest EV now (upcoming games)', 'tip': 'Tip time, then EV'}


def started(df, now):
    """True for bets whose game has already tipped (reference tip before `now`)."""
    return (pd.to_datetime(df['tip_ref_utc'], utc=True) <= now).to_numpy()


def board(df, mode='median', min_ev=0.0, stats=None, sides=None, top_n=15, sort='ev', upcoming_only=False, now=None, cur=None):
    """
    The rule-qualified bets of one slate, filtered and ranked by expected value AT THE CHOSEN PRICE. Returns (shown, hidden_count). The bets themselves are the frozen rule's
    (selected at the best price); at a worse price some are no longer worth taking, which is what `min_ev` exposes. `sort` is one of SORTS ('ev_now' uses the odds-now frame `cur`;
    bets without a current quote go last); `upcoming_only` drops games that have already tipped (needs `now`).
    """
    d = df
    if stats:
        d = d[d['stat'].isin(stats)]
    if sides:
        d = d[d['side'].isin(sides)]
    if upcoming_only and now is not None:
        d = d[~started(d, now)]
    d = d[d[ev_col(mode)] >= min_ev]
    if sort == 'ev_now' and cur is not None and len(cur):
        ev_now = d[['bet_id']].merge(cur[['bet_id', 'ev_best_now']], on='bet_id', how='left')['ev_best_now'].to_numpy()
        d = d.assign(_ev_now=ev_now).sort_values('_ev_now', ascending=False, na_position='last').drop(columns='_ev_now')
    elif sort == 'tip':
        d = d.assign(_tip=pd.to_datetime(d['tip_ref_utc'], utc=True)).sort_values(['_tip', ev_col(mode)], ascending=[True, False]).drop(columns='_tip')
    else:
        d = d.sort_values(ev_col(mode), ascending=False)
    return d.head(top_n), max(len(d) - top_n, 0)


def results_for_display(d, results_now, mode, treatment):
    """
    Result, Actual and Units per bet for the Board: the official settlement where the bet is settled, else the provisional grade from the finished game's box score (marked
    '(prov.)'), else blank. Returns a DataFrame aligned with `d`.
    """
    official = d[f'status_{treatment}'].to_numpy(dtype=object) if f'status_{treatment}' in d else np.full(len(d), np.nan, dtype=object)
    actual = d['y'].to_numpy(dtype=float)
    dec = d[dec_col(mode)].to_numpy(dtype=float)
    status = np.where(pd.isna(official), '', official).astype(object)
    units = np.where(np.isin(status, ['win', 'loss', 'push', 'void']), profit(status.astype(str), dec), np.nan)
    if results_now is not None and len(results_now):
        r = d[['bet_id']].merge(results_now[['bet_id', 'provisional', 'actual']], on='bet_id', how='left')
        prov = r['provisional'].to_numpy(dtype=object)
        use = (status == '') & ~pd.isna(prov)
        status = np.where(use, [f'{p} (prov.)' for p in prov], status)
        actual = np.where(use, r['actual'].to_numpy(dtype=float), actual)
        pv = np.array([p if isinstance(p, str) else '' for p in prov], dtype=object)
        units = np.where(use, profit(pv.astype(str), dec), units)
    return pd.DataFrame({'Result': status, 'Actual': actual, 'Units': units})


def result_kind(result):
    """'win' / 'loss' / 'other' / '' for a Result cell (official or provisional)."""
    r = str(result)
    return 'win' if r.startswith('win') else 'loss' if r.startswith('loss') else ('other' if r else '')


EV_BANDS = [(-np.inf, 0.0, 'below 0%'), (0.0, 0.03, '0 to 3%'), (0.03, 0.06, '3 to 6%'), (0.06, 0.10, '6 to 10%'), (0.10, np.inf, '10% and up')]


def ev_band(ev):
    """Label of the EV band of each value (EV as a fraction)."""
    ev = np.asarray(ev, dtype=float)
    out = np.full(len(ev), '', dtype=object)
    for lo, hi, name in EV_BANDS:
        out[(ev >= lo) & (ev < hi)] = name
    return out


def cluster_bootstrap_ratio(num, den, clusters, n_boot=1000, seed=0, alpha=0.05):
    """Point estimate and (1 - alpha) interval of sum(num)/sum(den) resampling whole clusters (games)."""
    codes, uniq = pd.factorize(pd.Series(np.asarray(clusters)))
    g = len(uniq)
    sn = np.bincount(codes, weights=np.asarray(num, float), minlength=g)
    sd = np.bincount(codes, weights=np.asarray(den, float), minlength=g)
    if sd.sum() <= 0 or g < 2:
        return np.nan, np.nan, np.nan
    w = np.random.default_rng(seed).multinomial(g, np.full(g, 1.0 / g), size=n_boot).astype(float)
    denom = w @ sd
    draws = np.where(denom > 0, (w @ sn) / np.where(denom > 0, denom, 1.0), np.nan)
    return sn.sum() / sd.sum(), float(np.nanquantile(draws, alpha / 2)), float(np.nanquantile(draws, 1 - alpha / 2))


def roi_summary(df, mode='median', treatment='stands', n_boot=1000):
    """Bets, win rate and ROI per unit staked with a game-clustered 95% interval."""
    d = with_profit(df, mode, treatment)
    stakes = d['stake'].sum()
    if stakes == 0:
        return {'bets': 0, 'win_rate': np.nan, 'roi': np.nan, 'lo': np.nan, 'hi': np.nan, 'units': 0.0, 'player_games': 0}
    p, lo, hi = cluster_bootstrap_ratio(d['profit'], d['stake'], d['game_id'], n_boot)
    wins = (d[f'status_{treatment}'] == 'win').sum()
    return {'bets': int(stakes), 'win_rate': float(wins / stakes), 'roi': p, 'lo': lo, 'hi': hi, 'units': float(d['profit'].sum()),
            'player_games': int(d.groupby(['game_id', 'player_id']).ngroups)}


def cumulative(df, mode='median', treatment='stands'):
    """Daily cumulative units with an approximate 95% band (independent-bets variance; wider in truth because bets on one game are correlated)."""
    d = with_profit(df, mode, treatment)
    d = d[d['stake'] > 0]
    daily = d.groupby('game_date').agg(units=('profit', 'sum'), var=('profit', lambda s: float(((s - s.mean()) ** 2).sum()) if len(s) > 1 else 0.0), n=('profit', 'size')).sort_index()
    out = pd.DataFrame({'game_date': daily.index, 'cum_units': daily['units'].cumsum().to_numpy(), 'n': daily['n'].cumsum().to_numpy()})
    se = np.sqrt(daily['var'].cumsum().to_numpy())
    out['lo'], out['hi'] = out['cum_units'] - 1.96 * se, out['cum_units'] + 1.96 * se
    return out


def breakdown(df, by, mode='median', treatment='stands'):
    """ROI by a column (stat, side, book, decision time...), without intervals (they are on the summary tiles); sorted by bets."""
    d = with_profit(df, mode, treatment)
    g = d.groupby(by).agg(bets=('stake', 'sum'), units=('profit', 'sum'))
    g['roi'] = np.where(g['bets'] > 0, g['units'] / g['bets'].where(g['bets'] > 0), np.nan)
    return g.reset_index().sort_values('bets', ascending=False)


def noise_note(n_bets):
    """The standard error of ROI after n unit bets is about 1/sqrt(n): a plain-language reminder against over-reading short records."""
    if n_bets < 1:
        return 'No settled bets yet.'
    se = 100.0 / np.sqrt(n_bets)
    return f'After {int(n_bets):,} settled bets the standard error of ROI is about {se:.1f} points: records shorter than a few thousand bets cannot tell a 3% edge from luck.'
