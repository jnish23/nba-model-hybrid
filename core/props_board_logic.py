"""
Pure pandas/numpy logic of the props dashboard (no Streamlit): pricing modes, settlement profit, the Board filters, ROI with date-clustered bootstrap intervals, cumulative curves
with a design-effect band, breakdowns, model-version tables, diagnostics and the freshness-banner decision. Kept free of Streamlit so it can be unit-tested and reused by the data exporter. Compatible with the dashboard environment (pandas 2.2, numpy 1.26).
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


NOW_COLS = {'median': ('dec_median_now', 'ev_median_now'), 'best': ('dec_best_now', 'ev_best_now'), 'flat': (None, 'ev_flat_now')}
NOW_FRAME_COLS = ['bet_id', 'line_now', 'dec_median_now', 'dec_best_now', 'ev_median_now', 'ev_best_now', 'ev_flat_now', 'still_qualifies']


def with_odds_now(t, d, cur, mode, now):
    """
    Live pending view: the price, line and EV at the latest snapshot next to the recorded bet, and whether it would still qualify. `cur` may be None or empty (no upcoming game has a quote:
    the file then has only a header, or lacks columns). A game that has already tipped has no price to take ('game started'); an upcoming game without a quote says 'no quote'.
    """
    have = cur if cur is not None and len(cur) else pd.DataFrame(columns=NOW_FRAME_COLS)
    for col in NOW_FRAME_COLS:
        if col not in have.columns:
            have = have.assign(**{col: np.nan})
    c = d[['bet_id']].merge(have[NOW_FRAME_COLS], on='bet_id', how='left')
    dec_n, ev_n = NOW_COLS[mode]
    dec = c[dec_n].to_numpy(float) if dec_n else np.where(c['ev_flat_now'].notna(), FLAT_110, np.nan)
    gone = started(d, now)
    t = t.copy()
    t['Price now'] = np.where(np.isfinite(dec), american(np.nan_to_num(dec, nan=2.0)), np.where(gone, 'game started', 'no quote'))
    t['Line now'] = c['line_now'].to_numpy(float)
    t['EV now'] = c[ev_n].to_numpy(float)
    t['Still bet?'] = np.where(c['still_qualifies'].isna(), '', np.where(c['still_qualifies'] == 1, 'yes', 'no'))
    return t


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


def day_record(d, results):
    """
    Record of the bets in `d` from their Board results (`results`: results_for_display(d, ...), aligned with `d`), counting exactly what the Board colours green / red / grey.
    Returns {'all' | 'over' | 'under': {'won', 'lost', 'other' (push or void), 'open' (not yet graded)}, 'units': profit of the graded bets}; provisional grades count.
    """
    kinds = np.array([result_kind(r) for r in results['Result']], dtype=object)
    sides = d['side'].to_numpy()
    out = {}
    for name, m in (('all', np.ones(len(d), dtype=bool)), ('over', sides == 'over'), ('under', sides == 'under')):
        out[name] = {'won': int((m & (kinds == 'win')).sum()), 'lost': int((m & (kinds == 'loss')).sum()),
                     'other': int((m & (kinds == 'other')).sum()), 'open': int((m & (kinds == '')).sum())}
    out['units'] = float(np.nansum(results['Units'].to_numpy(dtype=float)))
    return out


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


def _cluster_key(d, cluster=None):
    """Column used as the resampling cluster: `cluster` if given and present, else game_date (all bets of a slate move together), else game_id."""
    if cluster is not None and cluster in d.columns:
        return cluster
    return 'game_date' if 'game_date' in d.columns else 'game_id'


def date_cluster_roi(df, mode='median', treatment='stands', n_boot=1000, seed=0, alpha=0.05, cluster=None):
    """
    ROI (sum profit / sum stakes; pushes and voids are not stakes) with a (1 - alpha) interval from a date-clustered bootstrap: whole game_date clusters are resampled with replacement,
    because bets on one player-game, one game and one slate are correlated (59% of live bets share a player-game with another bet). Deterministic for a given `seed`.
    Returns {'roi', 'lo', 'hi', 'stakes', 'clusters'}; the interval is NaN with fewer than two dates.
    """
    d = with_profit(df, mode, treatment)
    d = d[d['stake'] > 0]
    if d.empty:
        return {'roi': np.nan, 'lo': np.nan, 'hi': np.nan, 'stakes': 0, 'clusters': 0}
    key = d[_cluster_key(d, cluster)]
    p, lo, hi = cluster_bootstrap_ratio(d['profit'], d['stake'], key, n_boot, seed, alpha)
    return {'roi': p, 'lo': lo, 'hi': hi, 'stakes': int(d['stake'].sum()), 'clusters': int(key.nunique())}


def independent_roi_interval(df, mode='median', treatment='stands', n_boot=1000, seed=0, alpha=0.05):
    """The interval a naive bootstrap gives, resampling single bets as if independent. Only a comparison for `date_cluster_roi`; it is too narrow on clustered bets."""
    d = with_profit(df, mode, treatment)
    d = d[d['stake'] > 0]
    if d.empty:
        return {'roi': np.nan, 'lo': np.nan, 'hi': np.nan, 'stakes': 0}
    p, lo, hi = cluster_bootstrap_ratio(d['profit'], d['stake'], np.arange(len(d)), n_boot, seed, alpha)
    return {'roi': p, 'lo': lo, 'hi': hi, 'stakes': int(d['stake'].sum())}


def roi_summary(df, mode='median', treatment='stands', n_boot=1000, cluster=None):
    """Bets, win rate and ROI per unit staked with a date-clustered 95% interval (`cluster` overrides the clustering column; default game_date, game_id if the frame has no dates)."""
    d = with_profit(df, mode, treatment)
    stakes = d['stake'].sum()
    if stakes == 0:
        return {'bets': 0, 'win_rate': np.nan, 'roi': np.nan, 'lo': np.nan, 'hi': np.nan, 'units': 0.0, 'player_games': 0, 'clusters': 0}
    r = date_cluster_roi(df, mode, treatment, n_boot, cluster=cluster)
    wins = (d[f'status_{treatment}'] == 'win').sum()
    return {'bets': int(stakes), 'win_rate': float(wins / stakes), 'roi': r['roi'], 'lo': r['lo'], 'hi': r['hi'], 'units': float(d['profit'].sum()),
            'player_games': int(d.groupby(['game_id', 'player_id']).ngroups), 'clusters': r['clusters']}


def design_effect(df, mode='median', treatment='stands', cluster=None):
    """
    Variance inflation of the summed profit when bets of one date are correlated: (sum over dates of the squared date total of the profit residuals) / (sum of squared residuals), with the
    usual G/(G-1) small-sample factor. About 1 for independent bets, floored at 1. Estimated on the whole record, so it is a single number, not a function of time.
    """
    d = with_profit(df, mode, treatment)
    d = d[d['stake'] > 0]
    if len(d) < 3:
        return 1.0
    key = d[_cluster_key(d, cluster)]
    g = key.nunique()
    resid = d['profit'] - d['profit'].mean()
    iid = float((resid ** 2).sum())
    if g < 2 or iid <= 0:
        return 1.0
    clustered = float((resid.groupby(key).sum() ** 2).sum())
    return float(max(1.0, clustered / iid * g / (g - 1)))


def cumulative(df, mode='median', treatment='stands', deff='auto'):
    """
    Daily cumulative units with an approximate 95% band. Design choice: the band is the independent-bets standard error (squared profit residuals about the overall mean, summed to
    each date) multiplied by sqrt(design effect), the design effect being estimated once from the data by `design_effect` (same-date clustering). It is NOT a date-block bootstrap: it
    keeps the sqrt(n) growth of a random walk and widens it by how much more the daily totals vary than independent bets would. `deff`: 'auto' (estimate), None or 1 (the old
    independent-bets band), or a number. The factor used is stored in `out.attrs['design_effect']`.
    """
    d = with_profit(df, mode, treatment)
    d = d[d['stake'] > 0]
    k = design_effect(df, mode, treatment) if isinstance(deff, str) and deff == 'auto' else (1.0 if deff is None else float(deff))
    resid2 = (d['profit'] - (d['profit'].mean() if len(d) else 0.0)) ** 2
    daily = d.assign(_r2=resid2).groupby('game_date').agg(units=('profit', 'sum'), var=('_r2', 'sum'), n=('profit', 'size')).sort_index()
    out = pd.DataFrame({'game_date': daily.index, 'cum_units': daily['units'].cumsum().to_numpy(), 'n': daily['n'].cumsum().to_numpy()})
    se = np.sqrt(daily['var'].cumsum().to_numpy() * k)
    out['lo'], out['hi'] = out['cum_units'] - 1.96 * se, out['cum_units'] + 1.96 * se
    out.attrs['design_effect'] = k
    return out


def breakdown(df, by, mode='median', treatment='stands'):
    """ROI by a column (stat, side, book, decision time...), without intervals (they are on the summary tiles); sorted by bets."""
    d = with_profit(df, mode, treatment)
    g = d.groupby(by).agg(bets=('stake', 'sum'), units=('profit', 'sum'))
    g['roi'] = np.where(g['bets'] > 0, g['units'] / g['bets'].where(g['bets'] > 0), np.nan)
    return g.reset_index().sort_values('bets', ascending=False)


def noise_note(n_bets, interval=None, n_dates=None):
    """
    Plain-language reminder against over-reading short records. With `interval` = (lo, hi), the date-clustered 95% interval of ROI (fractions) and `n_dates` the number of dates, the note
    quotes that interval and says the independent-bets figure (about 1/sqrt(n)) is too small because bets cluster. Without it only the independent-bets floor is given, labelled as such.
    """
    if n_bets < 1:
        return 'No settled bets yet.'
    floor = 100.0 / np.sqrt(n_bets)
    if interval is not None and n_dates is not None and np.isfinite(interval[0]) and np.isfinite(interval[1]):
        lo, hi = 100 * interval[0], 100 * interval[1]
        half = (hi - lo) / 2.0
        zero = 'The interval includes zero, so the record cannot yet tell an edge from luck.' if lo <= 0 <= hi else 'The interval excludes zero.'
        return (f'After {int(n_bets):,} settled bets on {int(n_dates)} dates the date-clustered 95% interval for ROI is {lo:+.1f}% to {hi:+.1f}% (half-width about {half:.1f} points). '
                f'Treating bets as independent would give only about +/-{1.96 * floor:.1f} points: too narrow, because bets on one player-game or one slate move together. {zero}')
    return (f'After {int(n_bets):,} settled bets the standard error of ROI is about {floor:.1f} points if the bets were independent. They are not (bets on one player-game or slate move together), '
            f'so the true error is larger; records shorter than a few thousand bets cannot tell a 3% edge from luck.')


# Model versions, diagnostics and the freshness banner. All pure: inputs are the exported tables (bets.csv, slot_log.csv, meta.json) and a clock passed in by the caller.
SLOT_LABEL = {240: 'T-4h', 60: 'T-1h', 5: 'T-5m'}
SLOT_ORDER = ['T-4h', 'T-1h', 'T-5m']
MIN_STAKES = 50                                  # a cell with fewer settled stakes than this is flagged as too small to read (kept, not hidden)
EV_CLAIM_EDGES = [0.03, 0.04, 0.05, 0.07]
EV_CLAIM_LABELS = ['3-4%', '4-5%', '5-7%', '7%+']
RESTORED_PROVENANCE = 'restored from dashboard export 2026-11 (not re-derivable)'
LIVE_PROVENANCE = 'recorded live'
UNKNOWN_PROVENANCE = 'unknown (older export)'
STALE_AFTER_MIN = 45
FRESH_AHEAD_H = 8
FRESH_BEHIND_H = 4
SLOT_STATUSES = ['done', 'missed', 'no_quotes']


def short_hash(x, n=8):
    """First n characters of a hash or commit id; 'n/a' for None, NaN or an empty value (an export can carry a null)."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or not str(x).strip():
        return 'n/a'
    return str(x)[:n]


def slot_label(minutes_before):
    """240 -> 'T-4h', 60 -> 'T-1h', 5 -> 'T-5m'; anything else 'T-<n>m'; unreadable 'unknown'."""
    try:
        m = int(round(float(minutes_before)))
    except (TypeError, ValueError):
        return 'unknown'
    return SLOT_LABEL.get(m, f'T-{m}m')


def _col(df, name):
    return df[name].to_numpy(dtype=float) if name in df.columns else np.full(len(df), np.nan)


def _staked(df, treatment):
    """Rows that are settled stakes (win or loss) under the treatment, with a `slot` label added."""
    col = f'status_{treatment}'
    d = df[df[col].isin(['win', 'loss'])].copy() if col in df.columns else df.iloc[0:0].copy()
    if 'slot' not in d.columns:
        d['slot'] = d['minutes_before'].map(slot_label) if 'minutes_before' in d.columns else 'unknown'
    return d


def _mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _diag_row(g, treatment, n_boot, seed, min_stakes):
    """One diagnostics row from settled stakes `g`: win rate, mean market / shrunk probability of the bet side, ROI at median and best price with date-clustered intervals, CLV."""
    out = {'stakes': int(len(g)), 'win_rate': float((g[f'status_{treatment}'] == 'win').mean()) if len(g) else np.nan}
    if len(g) and 'side' in g.columns:
        out['p_market'] = _mean(side_prob(g, 'p_over_market')) if 'p_over_market' in g.columns else np.nan
        out['p_shrunk'] = _mean(side_prob(g, 'p_over_star')) if 'p_over_star' in g.columns else np.nan
    else:
        out['p_market'] = out['p_shrunk'] = np.nan
    for mode in ('median', 'best'):
        r = date_cluster_roi(g, mode, treatment, n_boot, seed) if len(g) else {'roi': np.nan, 'lo': np.nan, 'hi': np.nan}
        out[f'roi_{mode}'], out[f'roi_{mode}_lo'], out[f'roi_{mode}_hi'] = r['roi'], r['lo'], r['hi']
    clv = _col(g, 'clv_prob')
    out['clv_pts'] = _mean(clv) * 100 if np.isfinite(clv).any() else np.nan
    out['clv_n'] = int(np.isfinite(clv).sum())
    out['too_small'] = bool(len(g) < min_stakes)
    return out


DIAG_COLS = ['stakes', 'win_rate', 'p_market', 'p_shrunk', 'roi_median', 'roi_median_lo', 'roi_median_hi', 'roi_best', 'roi_best_lo', 'roi_best_hi', 'clv_pts', 'clv_n', 'too_small']


def diagnostic_table(df, by, treatment='stands', n_boot=300, seed=0, min_stakes=MIN_STAKES):
    """
    Settled stakes grouped by `by` (a column name or a list, e.g. 'side', 'stat', 'slot', ['side', 'slot']; 'slot' is derived from minutes_before when absent): stakes, win rate, mean market
    fair and mean shrunk probability of the bet side (compare both with the win rate), ROI at the median and at the best price with date-clustered 95% intervals, and mean CLV in
    points with its n (CLV exists only where the consensus line did not move). `too_small` marks groups with fewer than `min_stakes` stakes: kept in the table, to be shown greyed.
    """
    keys = [by] if isinstance(by, str) else list(by)
    d = _staked(df, treatment)
    rows = []
    if len(d) and all(k in d.columns for k in keys):
        for key, g in d.groupby(keys, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(keys, key)), **_diag_row(g, treatment, n_boot, seed, min_stakes)})
    out = pd.DataFrame(rows, columns=keys + DIAG_COLS)
    if len(out):
        order = {s: i for i, s in enumerate(SLOT_ORDER)}
        out = out.sort_values(keys, key=lambda s: s.map(order).fillna(len(order)) if s.name == 'slot' else s, kind='stable').reset_index(drop=True)
    return out


EV_CLAIM_COLS = ['ev_bucket', 'stakes', 'claimed_ev', 'roi_best', 'roi_best_lo', 'roi_best_hi', 'roi_median', 'roi_median_lo', 'roi_median_hi', 'too_small']


def ev_claim_table(df, treatment='stands', n_boot=300, seed=0, min_stakes=MIN_STAKES):
    """
    Claimed versus realised: settled stakes bucketed by the EV claimed at the best price when flagged (ev_best: 3-4%, 4-5%, 5-7%, 7%+; '<3%' only if such rows exist), with the mean claimed
    EV, realised ROI at the best and at the median price (date-clustered intervals) and the stake count. If the model's edge were real and priced right, realised ROI at the best price
    would rise with the claim; a flat or falling line is a warning.
    """
    d = _staked(df, treatment)
    ev = _col(d, 'ev_best')
    d = d[np.isfinite(ev)]
    ev = ev[np.isfinite(ev)]
    labels = ['<3%'] + EV_CLAIM_LABELS
    if d.empty:
        return pd.DataFrame(columns=EV_CLAIM_COLS)
    idx = np.digitize(ev, EV_CLAIM_EDGES)
    rows = []
    for i, lab in enumerate(labels):
        g = d[idx == i]
        if i == 0 and g.empty:
            continue
        r = _diag_row(g, treatment, n_boot, seed, min_stakes)
        rows.append({'ev_bucket': lab, 'stakes': r['stakes'], 'claimed_ev': _mean(ev[idx == i]), 'roi_best': r['roi_best'], 'roi_best_lo': r['roi_best_lo'], 'roi_best_hi': r['roi_best_hi'],
                     'roi_median': r['roi_median'], 'roi_median_lo': r['roi_median_lo'], 'roi_median_hi': r['roi_median_hi'], 'too_small': r['too_small']})
    return pd.DataFrame(rows, columns=EV_CLAIM_COLS)


def slot_summary(slot_log, now, days=7):
    """
    Decision slots of the last `days` days from slot_log.csv (columns slot_minutes, status, written_utc): done / missed / no_quotes per slot (T-4h, T-1h, T-5m) and in total, with the
    share done. Always returns one row per slot plus 'all', zeros if the log is empty or lacks the columns.
    """
    base = pd.DataFrame({'slot': SLOT_ORDER + ['all']})
    for s in SLOT_STATUSES:
        base[s] = 0
    if slot_log is not None and len(slot_log) and {'slot_minutes', 'status', 'written_utc'} <= set(slot_log.columns):
        t = pd.to_datetime(slot_log['written_utc'], utc=True, errors='coerce')
        s = slot_log[(t >= now - pd.Timedelta(days=days)) & (t <= now)].copy()
        s['slot'] = s['slot_minutes'].map(slot_label)
        counts = s.groupby(['slot', 'status']).size()
        for slot in SLOT_ORDER:
            for st_ in SLOT_STATUSES:
                base.loc[base['slot'] == slot, st_] = int(counts.get((slot, st_), 0))
        for st_ in SLOT_STATUSES:
            base.loc[base['slot'] == 'all', st_] = int(base.loc[base['slot'] != 'all', st_].sum())
    base['total'] = base[SLOT_STATUSES].sum(axis=1)
    base['done_share'] = np.where(base['total'] > 0, base['done'] / base['total'].where(base['total'] > 0), np.nan)
    return base


def line_move_summary(df):
    """
    How often the consensus line moved between the flag time and the T-5 close, and by how much in the bettor's favour (line_move_favour, in points: close line minus the line taken for an
    over, the reverse for an under, so positive = the bettor got a better line than the close). Only bets flagged before T-5 with a known close are counted. Rows: all, then per slot.
    CLV (clv_prob) is blank on moved lines, so it describes only the bets whose line did not move.
    """
    cols = ['group', 'bets_with_close', 'moved', 'share_moved', 'moved_in_favour', 'share_in_favour', 'mean_line_move_favour']
    if 'close_line' not in df.columns or 'minutes_before' not in df.columns:
        return pd.DataFrame(columns=cols)
    d = df[df['close_line'].notna() & (pd.to_numeric(df['minutes_before'], errors='coerce') > 5)].copy()
    d['slot'] = d['minutes_before'].map(slot_label)
    rows = []
    for name, g in [('all', d)] + [(s, d[d['slot'] == s]) for s in SLOT_ORDER[:2]]:
        mv = _col(g, 'line_move_favour')
        moved = np.isfinite(mv)
        n, m = len(g), int(moved.sum())
        fav = int((mv[moved] > 0).sum())
        rows.append({'group': name, 'bets_with_close': n, 'moved': m, 'share_moved': m / n if n else np.nan, 'moved_in_favour': fav, 'share_in_favour': fav / m if m else np.nan,
                     'mean_line_move_favour': float(mv[moved].mean()) if m else np.nan})
    return pd.DataFrame(rows, columns=cols)


def version_series(d):
    return d['model_version'].fillna('unknown').astype(str) if 'model_version' in d.columns else pd.Series('unknown', index=d.index)


def version_meta_table(meta, bets):
    """
    Model versions with their date span, bet count and provenance. Uses meta['model_versions'] (list of dicts: version, first_game_date, last_game_date, bets, provenance) when the export
    has it; otherwise derives the table from bets.csv (model_version, game_date) with provenance 'unknown (older export)'.
    """
    cols = ['version', 'first_game_date', 'last_game_date', 'bets', 'provenance']
    mv = meta.get('model_versions') if isinstance(meta, dict) else None
    if isinstance(mv, list) and mv and all(isinstance(r, dict) and 'version' in r for r in mv):
        t = pd.DataFrame(mv).reindex(columns=cols)
        t['provenance'] = t['provenance'].fillna(UNKNOWN_PROVENANCE)
        return t
    rows = []
    if bets is not None and len(bets):
        v = version_series(bets)
        dates = bets['game_date'].astype(str) if 'game_date' in bets.columns else pd.Series('', index=bets.index)
        for ver, g in bets.assign(_v=v, _d=dates).groupby('_v', sort=False):
            rows.append({'version': ver, 'first_game_date': g['_d'].min(), 'last_game_date': g['_d'].max(), 'bets': int(len(g)), 'provenance': UNKNOWN_PROVENANCE})
    return pd.DataFrame(rows, columns=cols).sort_values('first_game_date', kind='stable').reset_index(drop=True)


def restored_versions(version_table):
    """Versions whose provenance says the rows were restored from an export."""
    if version_table is None or version_table.empty:
        return []
    return [r['version'] for _, r in version_table.iterrows() if str(r['provenance']).startswith('restored')]


def version_summary(df, treatment='stands', n_boot=300, seed=0, min_stakes=MIN_STAKES):
    """One row per model_version (plus 'all' first): first/last game_date, bets recorded, then the diagnostics row (stakes, win rate, ROI median / best with intervals, CLV)."""
    rows = []
    v = version_series(df)
    staked = _staked(df, treatment)
    sv = version_series(staked)
    first = {}
    if 'game_date' in df.columns and len(df):
        first = df['game_date'].astype(str).groupby(v).min().to_dict()
    versions = sorted(v.unique(), key=lambda x: (first.get(x, ''), x))
    groups = [('all', df, staked)] + [(ver, df[v == ver], staked[sv == ver]) for ver in versions]
    for ver, g_all, g_st in groups:
        dates = g_all['game_date'].astype(str) if 'game_date' in g_all.columns and len(g_all) else pd.Series(dtype=str)
        rows.append({'model_version': ver, 'first_game_date': dates.min() if len(dates) else '', 'last_game_date': dates.max() if len(dates) else '', 'bets': int(len(g_all)),
                     **_diag_row(g_st, treatment, n_boot, seed, min_stakes)})
    return pd.DataFrame(rows)


def freshness_status(built_at_utc, bets, now, treatment='stands', stale_after_min=STALE_AFTER_MIN, ahead_h=FRESH_AHEAD_H, behind_h=FRESH_BEHIND_H):
    """
    Decision for the freshness banner. `built_at_utc`: meta.json built_at_utc; `bets`: the bets table; `now`: a tz-aware Timestamp. The data is stale when it was built more than
    `stale_after_min` minutes ago. A warning is due when it is stale AND an unsettled bet (no result under `treatment`) tips within the next `ahead_h` hours or the last `behind_h` hours
    (then the prices, "odds now" and results on screen matter and may be out of date). Returns {'age_min' (None if the build time is unreadable), 'stale', 'n_open_near', 'warn'}.
    """
    built = pd.to_datetime(built_at_utc, utc=True, errors='coerce') if built_at_utc is not None else pd.NaT
    if pd.isna(built):
        return {'age_min': None, 'stale': False, 'n_open_near': 0, 'warn': False}
    age = (now - built).total_seconds() / 60.0
    n_open = 0
    if bets is not None and len(bets) and 'tip_ref_utc' in bets.columns:
        tip = pd.to_datetime(bets['tip_ref_utc'], utc=True, errors='coerce')
        col = f'status_{treatment}'
        open_ = bets[col].isna() if col in bets.columns else (bets['y'].isna() if 'y' in bets.columns else pd.Series(True, index=bets.index))
        near = (tip >= now - pd.Timedelta(hours=behind_h)) & (tip <= now + pd.Timedelta(hours=ahead_h))
        n_open = int((open_ & near).sum())
    stale = bool(age > stale_after_min)
    return {'age_min': float(age), 'stale': stale, 'n_open_near': n_open, 'warn': bool(stale and n_open > 0)}


def age_text(age_min):
    """'38 min' / '2 h 05 min' for an age in minutes."""
    m = int(round(age_min))
    return f'{m} min' if m < 120 else f'{m // 60} h {m % 60:02d} min'


# Live panel (display only: the recorded bet and its settlement never change). Same stat definitions as props/live/scores.py STAT_VALUE, on core/live_feed box columns.
LIVE_STAT = {'points': lambda b: b['points'], 'rebounds': lambda b: b['rebounds'], 'assists': lambda b: b['assists'], 'threes': lambda b: b['threes'],
             'ra': lambda b: b['rebounds'] + b['assists'], 'pra': lambda b: b['points'] + b['rebounds'] + b['assists']}
LIVE_WINDOW_H = 5                                # a tipped game is looked up for this long; no NBA game runs longer
DEFAULT_MINUTES = 34.0
MIN_MINUTES_FOR_PACE = 5.0
LIVE_COLS = ['Game', 'Clock', 'Player', 'Bet', 'Now', 'Min', 'Status', 'On pace', 'kind']


def live_candidates(bets, now, window_h=LIVE_WINDOW_H):
    """Open bets (no official result yet) whose game tipped within the last `window_h` hours: the only ones worth asking the live feed about."""
    if bets is None or bets.empty:
        return bets
    tip = pd.to_datetime(bets['tip_ref_utc'], utc=True)
    open_ = bets['y'].isna() if 'y' in bets else pd.Series(True, index=bets.index)
    return bets[open_ & (tip <= now) & (tip > now - pd.Timedelta(hours=window_h))]


def period_label(period, clock):
    """(3, '4:12') -> 'Q3 4:12'; period 5 -> 'OT', 6 -> '2OT'."""
    if period is None or pd.isna(period):
        return ''
    p = int(period)
    name = f'Q{p}' if p <= 4 else ('OT' if p == 5 else f'{p - 4}OT')
    return f'{name} {clock}' if isinstance(clock, str) and clock else name


def live_status(side, line, current):
    """('hit' | 'busted' | 'alive', text). Over: hit once current >= line, else how many more it needs. Under: busted once current > line, else how many more it can absorb."""
    if current is None or pd.isna(current):
        return 'none', 'not in the box score yet'
    if side == 'over':
        return ('hit', 'hit') if current >= line else ('alive', f'needs {int(np.ceil(line - current))} more')
    return ('busted', 'busted') if current > line else ('alive', f'{int(np.floor(line - current))} more allowed')


def on_pace(current, minutes, expected_minutes):
    """current / minutes * expected minutes (34 when the bet has none; never below minutes already played); NaN before MIN_MINUTES_FOR_PACE minutes."""
    if current is None or pd.isna(current) or pd.isna(minutes) or minutes < MIN_MINUTES_FOR_PACE:
        return np.nan
    exp = expected_minutes if expected_minutes is not None and pd.notna(expected_minutes) and expected_minutes > 0 else DEFAULT_MINUTES
    return current / minutes * max(exp, minutes)


def live_view(bets, board, lines):
    """
    One display row per open bet in a live game. `board`: core/live_feed scoreboard (nba_game_id, status, period, clock, scores); `lines`: box rows of those games with an
    nba_game_id column. Bets whose game is not live on the board are left out; a bet whose player has no box row yet shows no current value.
    """
    if bets is None or bets.empty or board is None or board.empty:
        return pd.DataFrame(columns=LIVE_COLS)
    live = board[board['status'] == 'live'].dropna(subset=['nba_game_id'])
    d = bets.merge(live[['nba_game_id', 'period', 'clock', 'home_tricode', 'away_tricode', 'home_score', 'away_score']], left_on='game_id', right_on='nba_game_id')
    if d.empty:
        return pd.DataFrame(columns=LIVE_COLS)
    box = lines if lines is not None and len(lines) else pd.DataFrame(columns=['nba_game_id', 'player_id', 'minutes', 'points', 'rebounds', 'assists', 'threes'])
    box = box[['nba_game_id', 'player_id', 'minutes', 'points', 'rebounds', 'assists', 'threes']].rename(columns={'nba_game_id': 'game_id'})
    d = d.merge(box.astype({'player_id': 'int64'}), on=['game_id', 'player_id'], how='left')
    rows = []
    for _, r in d.iterrows():
        cur = LIVE_STAT[r['stat']](r) if r['stat'] in LIVE_STAT and pd.notna(r['minutes']) else np.nan
        kind, text = live_status(r['side'], r['line'], cur)
        rows.append({'Game': f"{r['away_tricode']} {int(r['away_score'])} @ {r['home_tricode']} {int(r['home_score'])}", 'Clock': period_label(r['period'], r['clock']),
                     'Player': r['player_name'], 'Bet': f"{r['side'].upper()} {STAT_LABEL.get(r['stat'], r['stat'])} {r['line']:g}",
                     'Now': cur, 'Min': r['minutes'], 'Status': text, 'On pace': on_pace(cur, r['minutes'], r.get('expected_minutes')), 'kind': kind})
    return pd.DataFrame(rows, columns=LIVE_COLS).sort_values(['Game', 'Player'], kind='stable').reset_index(drop=True)
