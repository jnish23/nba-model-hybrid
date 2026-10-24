"""E14 - do big-spread (blowout-prone) games go under the opening total? Registered in
reports/experiments/E14_blowout_under/PREREGISTRATION.md before any number was computed.
Market residual r = actual total - opening total; H1: slope of r on |opening spread| is negative (find on 2019-20..2021-22,
confirm on 2022-23..2025-26, 95% CI from resampling whole game days). python -m experiments.e14_blowout_under"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import metrics as M                       # noqa: E402
from experiments.e12_shopping import load_multibook, payout   # noqa: E402
from core import odds_clean as oc                           # noqa: E402
from core.odds_clean import load_clean_odds                 # noqa: E402
from team.data_paths import latest_training_set             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'nba_stats.db'
OUT = ROOT / 'reports' / 'experiments' / 'E14_blowout_under'
DISCOVERY = ['2019-20', '2020-21', '2021-22']
CONFIRM = ['2022-23', '2023-24', '2024-25', '2025-26']
EDGES = [0, 3, 6, 9, 12, np.inf]
LABELS = ['0-3', '3-6', '6-9', '9-12', '12+']
THRESHOLDS = [6, 7, 8, 9, 10, 11, 12, 13, 14]
MIN_DISC_BETS = 200
WIN_UNDER = 100 / 110
DROPPED = {}


class DayBoot:
    """Resampling of whole game days. Every statistic is built from per-day sums, so a draw is a weighted sum over days."""

    def __init__(self, dates, n_boot=M.N_BOOT, seed=M.RNG_SEED):
        self.codes, uniq = pd.factorize(pd.to_datetime(pd.Series(np.asarray(dates))))
        self.nd = len(uniq)
        rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(rng.integers(0, self.nd, self.nd), minlength=self.nd) for _ in range(n_boot)]).astype(float)

    def day_sum(self, v):
        return np.bincount(self.codes, weights=np.asarray(v, float), minlength=self.nd)

    def ratio(self, num, den, mask):
        """Point estimate and CI of sum(num)/sum(den) over masked rows."""
        m = np.asarray(mask, bool)
        n = self.day_sum(np.where(m, num, 0.0))
        d = self.day_sum(np.where(m, den, 0.0))
        pt = n.sum() / d.sum() if d.sum() else np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            b = (self.W @ n) / (self.W @ d)
        return pt, *np.nanquantile(b, [.025, .975])

    def ols(self, X, y, mask):
        """OLS of y on X (intercept already a column of X). Returns beta, lo, hi, and the bootstrap draws."""
        m = np.asarray(mask, bool)
        X, y = np.asarray(X, float)[m], np.asarray(y, float)[m]
        codes = self.codes[m]
        k = X.shape[1]
        xtx = np.zeros((self.nd, k, k))
        xty = np.zeros((self.nd, k))
        for i in range(k):
            xty[:, i] = np.bincount(codes, weights=X[:, i] * y, minlength=self.nd)
            for j in range(i, k):
                xtx[:, i, j] = xtx[:, j, i] = np.bincount(codes, weights=X[:, i] * X[:, j], minlength=self.nd)
        beta = np.linalg.solve(xtx.sum(0), xty.sum(0))
        A = np.einsum('bd,dij->bij', self.W, xtx)
        B = self.W @ xty
        boots = np.linalg.solve(A, B[..., None])[..., 0]
        lo, hi = np.quantile(boots, [.025, .975], axis=0)
        return beta, lo, hi, boots


def design(df, mask, controls=(), season_fe=False, seasons=None):
    """[1, |spread|, controls..., season dummies] restricted to rows in `mask` (full-length array, zeros outside)."""
    cols = [np.ones(len(df)), df['abs_spread'].to_numpy(float)] + [df[c].to_numpy(float) for c in controls]
    names = ['intercept', 'abs_spread'] + list(controls)
    if season_fe:
        ss = sorted(set(df.loc[mask, 'seasonYear'])) if seasons is None else seasons
        for s in ss[1:]:
            cols.append((df['seasonYear'] == s).to_numpy(float))
            names.append(f'season_{s}')
    return np.column_stack(cols), names


def slope(db, df, mask, y='r', controls=(), season_fe=False):
    X, names = design(df, mask, controls, season_fe)
    beta, lo, hi, boots = db.ols(X, df[y].to_numpy(float), mask)
    i = names.index('abs_spread')
    return {'slope': float(beta[i]), 'lo': float(lo[i]), 'hi': float(hi[i]), 'n': int(np.asarray(mask).sum())}, boots[:, i]


def load_frame():
    use = ['gameId', 'gameDate', 'seasonYear', 'odds_home_spread', 'odds_total', 'target_total_pts', 'target_point_diff']
    g = pd.read_csv(latest_training_set(), usecols=use, low_memory=False)
    g['gameId'] = g['gameId'].astype(str).str.zfill(10)
    g = g.dropna(subset=use).drop_duplicates('gameId')
    g = g[g['seasonYear'] >= '2013-14']
    # same plausibility bounds as core/odds_clean.py; added after the first run showed 2016-17 holds corrupt opening totals
    # (sd of r = 52 vs 18 in every other season). No game from 2019-20 on is affected.
    bad = ~g['odds_total'].between(*oc.TOTAL_RANGE) | (g['odds_home_spread'].abs() > oc.MAX_ABS_SPREAD)
    DROPPED.update(g[bad].groupby('seasonYear').size().to_dict())
    g = g[~bad]
    g['abs_spread'] = g['odds_home_spread'].abs()
    g['r'] = g['target_total_pts'] - g['odds_total']
    g['bucket'] = pd.cut(g['abs_spread'], EDGES, labels=LABELS, right=False)
    with sqlite3.connect(f'file:{DB}?mode=ro', uri=True) as c:
        mins = pd.read_sql('SELECT gameId, MAX(minutes) AS mx FROM team_game_full GROUP BY gameId', c)
    mins['gameId'] = mins['gameId'].astype(str).str.zfill(10)
    g = g.merge(mins, on='gameId', how='left')
    g['ot'] = g['mx'] > 240
    clean = load_clean_odds(DB)
    clean['gameId'] = clean['gameId'].astype(str).str.zfill(10)
    g = g.merge(clean[['gameId', 'open_total', 'close_total', 'close_home_spread']].drop_duplicates('gameId'), on='gameId', how='left')
    g['r_close'] = g['target_total_pts'] - g['close_total']
    g['close_move'] = g['close_total'] - g['odds_total']
    g['season_idx'] = g['seasonYear']
    return g.sort_values(['gameDate', 'gameId']).reset_index(drop=True)


def bucket_table(db, df, mask, label):
    rows = []
    for b in LABELS + ['10+']:
        sel = (df['bucket'] == b) if b != '10+' else (df['abs_spread'] >= 10)
        m = np.asarray(mask) & sel.to_numpy()
        nonpush = (df['r'] != 0).to_numpy()
        ur, ulo, uhi = db.ratio((df['r'] < 0).astype(float), nonpush.astype(float), m & nonpush)
        mr, mlo, mhi = db.ratio(df['r'], np.ones(len(df)), m)
        rows.append({'period': label, 'bucket': b, 'games': int(m.sum()), 'pushes': int((m & ~nonpush).sum()),
                     'under_rate': ur, 'under_lo': ulo, 'under_hi': uhi, 'mean_r': mr, 'mean_r_lo': mlo, 'mean_r_hi': mhi})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    db = DayBoot(df['gameDate'])
    S = df['seasonYear']
    masks = {'discovery': S.isin(DISCOVERY).to_numpy(), 'confirmation': S.isin(CONFIRM).to_numpy(),
             'registered (2019-20..2025-26)': S.isin(DISCOVERY + CONFIRM).to_numpy(), 'supplementary 2013-14..2018-19': (S < '2019-20').to_numpy()}
    reg = ~df['ot'].fillna(False).to_numpy()
    res = {'n_games_total': int(len(df)), 'ot_missing': int(df['ot'].isna().sum()), 'dropped_implausible_odds_by_season': {k: int(v) for k, v in DROPPED.items()}}

    # ---------------------------------------------------------- step 1: H1
    h1 = {}
    for k, m in masks.items():
        h1[k], _ = slope(db, df, m)
    h1['discovery_negative'] = h1['discovery']['slope'] < 0
    h1['confirmation_negative'] = h1['confirmation']['slope'] < 0
    h1['confirmation_ci_below_zero'] = h1['confirmation']['hi'] < 0
    h1['SUPPORTED'] = bool(h1['discovery_negative'] and h1['confirmation_negative'] and h1['confirmation_ci_below_zero'])
    res['h1'] = h1
    res['h1_by_season'] = {s: slope(db, df, (S == s).to_numpy())[0] for s in sorted(S.unique())}

    # ---------------------------------------------------------- step 2: buckets and controlled regression
    res['buckets'] = sum((bucket_table(db, df, masks[k], k) for k in ('discovery', 'confirmation', 'registered (2019-20..2025-26)', 'supplementary 2013-14..2018-19')), [])
    ctl = {}
    for k in ('discovery', 'confirmation', 'registered (2019-20..2025-26)', 'supplementary 2013-14..2018-19'):
        X, names = design(df, masks[k], controls=['odds_total'], season_fe=True)
        beta, lo, hi, _ = db.ols(X, df['r'].to_numpy(float), masks[k])
        ctl[k] = {n: {'coef': float(beta[i]), 'lo': float(lo[i]), 'hi': float(hi[i])} for i, n in enumerate(names)}
        ctl[k]['n'] = int(masks[k].sum())
    res['controlled'] = ctl
    res['controlled_confirm_ci_below_zero'] = bool(ctl['confirmation']['abs_spread']['hi'] < 0)
    # spread and total are positively correlated in the data?
    for k in ('registered (2019-20..2025-26)',):
        mm = masks[k]
        res['corr_abs_spread_total'] = float(np.corrcoef(df.loc[mm, 'abs_spread'], df.loc[mm, 'odds_total'])[0, 1])
        res['mean_abs_spread'] = float(df.loc[mm, 'abs_spread'].mean())
        res['share_10plus'] = float((df.loc[mm, 'abs_spread'] >= 10).mean())
        res['n_10plus'] = int((df.loc[mm, 'abs_spread'] >= 10).sum())

    # ---------------------------------------------------------- step 3: does the model absorb it?
    oos = []
    for p in sorted((ROOT / 'reports').glob('oos_predictions_20??-??.csv')):
        o = pd.read_csv(p, dtype={'game_id': str}, usecols=['game_id', 'total_pred', 'odds_total', 'odds_home_spread', 'home_score', 'away_score'])
        o['gameId'] = o['game_id'].str.zfill(10)
        oos.append(o)
    oos = pd.concat(oos).drop_duplicates('gameId')
    j = df.merge(oos[['gameId', 'total_pred', 'odds_total', 'odds_home_spread', 'home_score', 'away_score']], on='gameId', how='inner', suffixes=('', '_oos'))
    res['oos_checks'] = {'games': int(len(j)), 'odds_total_equal': float((np.abs(j['odds_total'] - j['odds_total_oos']) < 1e-9).mean()),
                         'spread_equal': float((np.abs(j['odds_home_spread'] - j['odds_home_spread_oos']) < 1e-9).mean()),
                         'actual_equal': float(((j['home_score'] + j['away_score']) == j['target_total_pts']).mean())}
    j['model_resid'] = j['target_total_pts'] - j['total_pred']
    j['model_minus_line'] = j['total_pred'] - j['odds_total']
    dbj = DayBoot(j['gameDate'])
    sj = j['seasonYear']

    def jslope(mask, y, controls=()):
        X, names = design(j, mask, controls)
        b, lo, hi, boots = dbj.ols(X, j[y].to_numpy(float), mask)
        i = names.index('abs_spread')
        return {'slope': float(b[i]), 'lo': float(lo[i]), 'hi': float(hi[i]), 'n': int(np.asarray(mask).sum())}
    s3 = {}
    for tag, m in (('2021-22..2025-26 (all OOS)', np.ones(len(j), bool)), ('2022-23..2025-26 (confirmation)', sj.isin(CONFIRM).to_numpy())):
        s3[tag] = {'model residual (actual - total_pred)': jslope(m, 'model_resid'), 'market residual (actual - opening total)': jslope(m, 'r'),
                   'model lean (total_pred - opening total)': jslope(m, 'model_minus_line')}
    res['step3_oos'] = s3
    res['step3_oos_by_season'] = {s: jslope((sj == s).to_numpy(), 'model_resid') for s in sorted(sj.unique())}
    brows = []
    for b in LABELS:
        m = (j['bucket'] == b).to_numpy()
        row = {'bucket': b, 'games': int(m.sum())}
        for col in ('model_resid', 'r', 'model_minus_line'):
            pt, lo, hi = dbj.ratio(j[col], np.ones(len(j)), m)
            row[col], row[col + '_lo'], row[col + '_hi'] = pt, lo, hi
        brows.append(row)
    res['step3_buckets'] = brows
    # walk-forward baseline (7 folds)
    bl = pd.read_csv(ROOT / 'reports' / 'experiments' / 'baseline' / 'predictions.csv', dtype={'gameId': str}, usecols=['gameId', 'total_pred'])
    bl['gameId'] = bl['gameId'].str.zfill(10)
    jb = df.merge(bl, on='gameId', how='inner')
    jb['model_resid'] = jb['target_total_pts'] - jb['total_pred']
    dbb = DayBoot(jb['gameDate'])
    X, names = design(jb, np.ones(len(jb), bool))
    b, lo, hi, _ = dbb.ols(X, jb['model_resid'].to_numpy(float), np.ones(len(jb), bool))
    res['step3_baseline_wf'] = {'n': int(len(jb)), 'slope': float(b[1]), 'lo': float(lo[1]), 'hi': float(hi[1]),
                                'by_season': {}}
    for s in sorted(jb['seasonYear'].unique()):
        m = (jb['seasonYear'] == s).to_numpy()
        X, _ = design(jb, m)
        b, lo, hi, _ = dbb.ols(X, jb['model_resid'].to_numpy(float), m)
        res['step3_baseline_wf']['by_season'][s] = {'slope': float(b[1]), 'lo': float(lo[1]), 'hi': float(hi[1]), 'n': int(m.sum())}

    # ---------------------------------------------------------- step 4: overtime split
    s4 = {'ot_rate_by_bucket': {}, 'slopes': {}}
    otok = df['ot'].notna().to_numpy()
    for b in LABELS:
        m = (df['bucket'] == b).to_numpy() & masks['registered (2019-20..2025-26)'] & otok
        s4['ot_rate_by_bucket'][b] = dict(zip(('rate', 'lo', 'hi'), db.ratio(df['ot'].fillna(False).astype(float), np.ones(len(df)), m)), n=int(m.sum()))
    ot = df['ot'].fillna(False).to_numpy()
    for per in ('discovery', 'confirmation', 'registered (2019-20..2025-26)'):
        pm = masks[per] & otok
        s4['slopes'][per] = {'all': slope(db, df, pm)[0], 'regulation only': slope(db, df, pm & ~ot)[0], 'overtime only': slope(db, df, pm & ot)[0],
                             'regulation only, controlled': None}
        X, names = design(df, pm & ~ot, controls=['odds_total'], season_fe=True)
        beta, lo, hi, _ = db.ols(X, df['r'].to_numpy(float), pm & ~ot)
        s4['slopes'][per]['regulation only, controlled'] = {'slope': float(beta[1]), 'lo': float(lo[1]), 'hi': float(hi[1]), 'n': int((pm & ~ot).sum())}
    # slope difference all - regulation (paired draws)
    pm = masks['confirmation'] & otok
    _, bo_all = slope(db, df, pm)
    _, bo_reg = slope(db, df, pm & ~ot)
    d = bo_all - bo_reg
    s4['confirmation_slope_all_minus_regulation'] = {'diff': float(slope(db, df, pm)[0]['slope'] - slope(db, df, pm & ~ot)[0]['slope']),
                                                     'lo': float(np.quantile(d, .025)), 'hi': float(np.quantile(d, .975))}
    # mean r for OT vs regulation games, by bucket (confirmation + discovery pooled)
    s4['mean_r_ot_vs_reg'] = {}
    for b in LABELS:
        m = (df['bucket'] == b).to_numpy() & masks['registered (2019-20..2025-26)'] & otok
        s4['mean_r_ot_vs_reg'][b] = {'regulation': db.ratio(df['r'], np.ones(len(df)), m & ~ot)[0], 'overtime': db.ratio(df['r'], np.ones(len(df)), m & ot)[0],
                                     'n_ot': int((m & ot).sum()), 'n_reg': int((m & ~ot).sum())}
    # regulation-only under rate by bucket (confirmation)
    s4['reg_buckets'] = []
    for per in ('discovery', 'confirmation'):
        pm = masks[per] & otok & ~ot
        s4['reg_buckets'] += bucket_table(db, df, pm, per + ' (regulation only)')
    res['step4'] = s4

    # ---------------------------------------------------------- step 5: opening vs closing
    cm = df['close_total'].notna().to_numpy()
    s5 = {'n_with_close': int(cm.sum()), 'by_bucket': [], 'slopes': {}}
    for b in LABELS:
        m = (df['bucket'] == b).to_numpy() & masks['registered (2019-20..2025-26)'] & cm
        row = {'bucket': b, 'games': int(m.sum())}
        for col in ('close_move', 'r', 'r_close'):
            row[col], row[col + '_lo'], row[col + '_hi'] = db.ratio(df[col].fillna(0), np.ones(len(df)), m)
        nonpush = (df['r_close'] != 0).to_numpy()
        row['under_rate_close'], row['under_rate_close_lo'], row['under_rate_close_hi'] = db.ratio((df['r_close'] < 0).astype(float), nonpush.astype(float), m & nonpush)
        nonpush0 = (df['r'] != 0).to_numpy()
        row['under_rate_open'], row['under_rate_open_lo'], row['under_rate_open_hi'] = db.ratio((df['r'] < 0).astype(float), nonpush0.astype(float), m & nonpush0)
        s5['by_bucket'].append(row)
    for per in ('discovery', 'confirmation', 'registered (2019-20..2025-26)'):
        pm = masks[per] & cm
        s5['slopes'][per] = {'close_total - open_total ~ |spread|': slope(db, df.assign(close_move=df['close_move'].fillna(0)), pm, y='close_move')[0],
                             'actual - open_total ~ |spread|': slope(db, df, pm)[0],
                             'actual - close_total ~ |spread|': slope(db, df.assign(r_close=df['r_close'].fillna(0)), pm, y='r_close')[0]}
    res['step5'] = s5
    both = df['open_total'].notna()
    res['open_total_check'] = {'games_compared': int(both.sum()),
                               'share_equal': float((np.abs(df.loc[both, 'open_total'] - df.loc[both, 'odds_total']) < 1e-9).mean()),
                               'mean_abs_diff': float(np.abs(df.loc[both, 'open_total'] - df.loc[both, 'odds_total']).mean())}

    # ---------------------------------------------------------- step 6: betting
    mats, _ = load_multibook()
    bt = df[['gameId']].copy()
    tots, under_odds = mats['tot'].reindex(bt['gameId']), mats['under'].reindex(bt['gameId'])
    best_line = tots.max(axis=1)
    best_price = []
    for (_, trow), (_, urow), bl_ in zip(tots.iterrows(), under_odds.iterrows(), best_line):
        if np.isnan(bl_):
            best_price.append(np.nan)
            continue
        q = urow[trow == bl_].dropna()
        best_price.append(q.max() if len(q) else -110.0)
    df['best_line'] = best_line.to_numpy()
    df['best_price'] = np.array(best_price, float)
    df['n_books'] = tots.notna().sum(axis=1).to_numpy()
    a = df['target_total_pts'].to_numpy(float)

    def under_profit(line, price=None):
        res_ = np.asarray(line, float) - a                          # >0 under wins
        pay = np.full(len(df), WIN_UNDER) if price is None else payout(price)
        return np.where(res_ > 0, pay, np.where(res_ < 0, -1.0, 0.0))
    df['p_open'] = under_profit(df['odds_total'])
    df['p_close'] = under_profit(df['close_total'].fillna(np.nan))
    df['p_best110'] = under_profit(df['best_line'])
    df['p_bestq'] = under_profit(df['best_line'], df['best_price'].fillna(-110.0))
    ones = np.ones(len(df))

    def roi(col, mask):
        mm = np.asarray(mask) & df[col].notna().to_numpy()
        pt, lo, hi = db.ratio(df[col].fillna(0), ones, mm)
        return {'roi': pt * 100, 'lo': lo * 100, 'hi': hi * 100, 'bets': int(mm.sum()),
                'win_rate': float(((df.loc[mm, col] > 0).sum()) / max(1, (df.loc[mm, col] != 0).sum()))}
    grid = []
    for X_ in THRESHOLDS:
        row = {'X': X_}
        for per in ('discovery', 'confirmation'):
            m = masks[per] & (df['abs_spread'] >= X_).to_numpy()
            row[per] = roi('p_open', m)
        grid.append(row)
    res['bet_grid'] = grid
    elig = [r for r in grid if r['discovery']['bets'] >= MIN_DISC_BETS]
    chosen = max(elig, key=lambda r: r['discovery']['roi'])['X']
    res['chosen_X'] = chosen
    s6 = {'X': chosen}
    allm = {'discovery': masks['discovery'], 'confirmation': masks['confirmation']}
    for per, pm in allm.items():
        sel = pm & (df['abs_spread'] >= chosen).to_numpy()
        s6[per] = {'under all games (control)': roi('p_open', pm),
                   f'under |spread|>={chosen}, opening @ -110': roi('p_open', sel),
                   f'under |spread|>={chosen}, closing total @ -110': roi('p_close', sel),
                   f'under |spread|>={chosen}, best-book line @ -110': roi('p_best110', sel),
                   f'under |spread|>={chosen}, best-book line @ quoted price': roi('p_bestq', sel)}
        # paired with the same games for the best-book comparison
        sb = sel & df['best_line'].notna().to_numpy()
        s6[per]['games with best-book data'] = int(sb.sum())
        s6[per][f'(same games) opening @ -110'] = roi('p_open', sb)
        s6[per][f'(same games) best-book @ quoted price'] = roi('p_bestq', sb)
    res['bet'] = s6

    # ---- EXPLORATORY, NOT REGISTERED: the mirror bet (over when |spread| >= X), same selection rule (X from discovery only)
    over_t, over_odds = mats['tot'].reindex(df['gameId']), mats['over'].reindex(df['gameId'])
    ob_line = over_t.min(axis=1)
    ob_price = []
    for (_, trow), (_, orow), bl_ in zip(over_t.iterrows(), over_odds.iterrows(), ob_line):
        if np.isnan(bl_):
            ob_price.append(np.nan)
            continue
        q = orow[trow == bl_].dropna()
        ob_price.append(q.max() if len(q) else -110.0)
    df['ob_line'], df['ob_price'] = ob_line.to_numpy(), np.array(ob_price, float)

    def over_profit(line, price=None):
        res_ = a - np.asarray(line, float)                          # >0 over wins
        pay = np.full(len(df), WIN_UNDER) if price is None else payout(price)
        return np.where(res_ > 0, pay, np.where(res_ < 0, -1.0, 0.0))
    df['po_open'] = over_profit(df['odds_total'])
    df['po_close'] = over_profit(df['close_total'])
    df['po_bestq'] = over_profit(df['ob_line'], df['ob_price'].fillna(-110.0))
    ogrid = []
    for X_ in THRESHOLDS:
        row = {'X': X_}
        for per in ('discovery', 'confirmation'):
            row[per] = roi('po_open', masks[per] & (df['abs_spread'] >= X_).to_numpy())
        ogrid.append(row)
    ochosen = max([r_ for r_ in ogrid if r_['discovery']['bets'] >= MIN_DISC_BETS], key=lambda r_: r_['discovery']['roi'])['X']
    o6 = {'X': ochosen, 'grid': ogrid}
    for per in ('discovery', 'confirmation'):
        sel = masks[per] & (df['abs_spread'] >= ochosen).to_numpy()
        o6[per] = {'over all games (control)': roi('po_open', masks[per]), f'over |spread|>={ochosen}, opening @ -110': roi('po_open', sel),
                   f'over |spread|>={ochosen}, closing total @ -110': roi('po_close', sel),
                   f'over |spread|>={ochosen}, best-book line @ quoted price': roi('po_bestq', sel)}
    # fixed round-number cut for context (10+, the user's "blowout" definition), also from the confirmation period only for reference
    o6['fixed_10plus_confirmation'] = roi('po_open', masks['confirmation'] & (df['abs_spread'] >= 10).to_numpy())
    o6['fixed_10plus_discovery'] = roi('po_open', masks['discovery'] & (df['abs_spread'] >= 10).to_numpy())
    res['bet_over_exploratory'] = o6
    # hit-rate context
    res['breakeven_110'] = float(M.BREAK_EVEN_110)

    # ---------------------------------------------------------- persist
    keep = ['gameId', 'gameDate', 'seasonYear', 'odds_home_spread', 'odds_total', 'close_total', 'target_total_pts', 'abs_spread', 'r', 'ot', 'best_line', 'best_price']
    df[keep].to_csv(OUT / 'e14_games.csv', index=False)
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else float(o)))
    print(json.dumps({k: res[k] for k in ('h1', 'chosen_X')}, indent=1, default=float))


if __name__ == '__main__':
    main()
