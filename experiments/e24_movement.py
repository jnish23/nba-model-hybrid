"""E24 - line movement (angle 8 of 8), historical part. Registered in reports/experiments/PREREGISTRATION.md ("E24") before any
relationship between movement and outcome was computed. Regular-season games only. python -m experiments.e24_movement"""
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import odds_clean as oc                                       # noqa: E402
from experiments import e14_blowout_under as e14                       # noqa: E402
from experiments.e14_blowout_under import DayBoot                       # noqa: E402
from experiments.e12_shopping import book_matrices, payout             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'nba_stats.db'
OUT = ROOT / 'reports' / 'experiments' / 'E24_movement'
DISC, CONF = e14.DISCOVERY, e14.CONFIRM
CUTS = [2, 3, 4, 5, 6]
WIN = 100 / 110


def load_mats(table, bad_windows=False):
    """Cleaned per-book matrices (rows = gameId) of one odds table, like experiments.e12_shopping.load_multibook."""
    with sqlite3.connect(DB) as conn:
        tab = oc._load_table(conn, table)
        games = oc._game_results(conn)
    oc._fill_team_ids([tab])
    key = ['gameDate', 'teamId_home', 'teamId_away']
    tab = tab[(tab['gameDate'] >= '2019-10-01') & ~tab['status'].str.contains('Postponed', na=False)]
    if bad_windows:
        for lo, hi in oc.CLOSING_BAD_WINDOWS:
            tab = tab[~tab['gameDate'].between(lo, hi)]
    tab = tab.drop_duplicates(subset=key, keep='last').reset_index(drop=True)
    mats = book_matrices(tab)
    ids = tab[key].merge(games[key + ['gameId']], on=key, how='left')['gameId']
    valid = (ids.notna() & ~ids.duplicated(keep='last')).to_numpy()
    ids = ids.astype(str).str.zfill(10)
    out = {}
    for k, m in mats.items():
        m = m.loc[valid].copy()
        m.index = ids[valid].values
        out[k] = m
    return out


def two_sided(b):
    return float(max(2 * min((b <= 0).sum(), (b >= 0).sum()), 1) / (len(b) + 1))


def fit(db, df, mask, y, xs, season_dummies=False):
    cols, names = [np.ones(len(df))], ['intercept']
    for x in xs:
        cols.append(df[x].to_numpy(float)), names.append(x)
    if season_dummies:
        for s in sorted(set(df.loc[mask, 'seasonYear']))[1:]:
            cols.append((df['seasonYear'] == s).to_numpy(float)), names.append(f'season_{s}')
    X = np.column_stack(cols)
    yv = df[y].to_numpy(float)
    m = np.asarray(mask) & np.isfinite(X).all(1) & np.isfinite(yv)
    beta, lo, hi, boots = db.ols(np.nan_to_num(X), np.nan_to_num(yv), m)
    out = {}
    for i, n in enumerate(names):
        if n.startswith('season_') or n == 'intercept':
            continue
        out[n] = {'coef': float(beta[i]), 'lo': float(lo[i]), 'hi': float(hi[i]), 'se': float((hi[i] - lo[i]) / 3.92), 'p_two_sided': two_sided(boots[:, i])}
    out['n'] = int(m.sum())
    return out


def main():
    logging.getLogger().setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as conn:
        odds = oc.load_clean_odds(DB)
    odds['gameId'] = odds['gameId'].astype(str).str.zfill(10)
    odds = odds[odds['gameId'].str.startswith('002')].copy()
    odds['gameDate'] = pd.to_datetime(odds['gameDate'])
    y_ = odds['gameDate'].dt.year.where(odds['gameDate'].dt.month >= 8, odds['gameDate'].dt.year - 1)
    odds['seasonYear'] = y_.astype(str) + '-' + ((y_ + 1) % 100).astype(str).str.zfill(2)
    df = odds[odds['seasonYear'].isin(DISC + CONF)].dropna(subset=['open_total', 'close_total', 'open_home_spread', 'close_home_spread']).copy()
    df = df[df['home_score'] + df['away_score'] > 100].reset_index(drop=True)
    df['actual_total'] = df['home_score'] + df['away_score']
    df['margin'] = df['home_score'] - df['away_score']
    df['move_total'] = df['close_total'] - df['open_total']
    df['move_home'] = -(df['close_home_spread'] - df['open_home_spread'])
    df['abs_move_home'] = df['move_home'].abs()
    df['err_total'] = df['actual_total'] - df['close_total']
    df['err_spread'] = df['margin'] + df['close_home_spread']
    df['err_open_total'] = df['actual_total'] - df['open_total']
    df['abs_spread'] = df['close_home_spread'].abs()
    S = df['seasonYear']
    disc, conf = S.isin(DISC).to_numpy(), S.isin(CONF).to_numpy()
    db = DayBoot(df['gameDate'])
    ones = np.ones(len(df))
    res = {'games': int(len(df)), 'games_by_season': {s: int((S == s).sum()) for s in DISC + CONF},
           'move_total_sd': float(df['move_total'].std()), 'move_home_sd': float(df['move_home'].std()), 'err_total_sd': float(df['err_total'].std()),
           'share_total_moved': float((df['move_total'] != 0).mean()), 'share_spread_moved': float((df['move_home'] != 0).mean())}

    # per-book alternative move
    mo, mc = load_mats('nba_odds_opening'), load_mats('nba_odds', bad_windows=True)
    ids = df['gameId']
    to, tc = mo['tot'].reindex(ids), mc['tot'].reindex(ids)
    pb = (tc.to_numpy() - to.to_numpy())
    df['move_total_perbook'] = np.where(np.isfinite(pb).any(1), np.nanmedian(np.where(np.isfinite(pb), pb, np.nan), axis=1), np.nan)
    res['perbook_available'] = int(df['move_total_perbook'].notna().sum())

    # ---- step 1: efficiency test
    s1 = {}
    q99 = df['move_total'].abs().quantile(0.99)
    trim = (df['move_total'].abs() <= q99).to_numpy()
    for per, pm in (('discovery', disc), ('confirmation', conf), ('registered pooled', disc | conf)):
        s1[per] = {'primary': fit(db, df, pm, 'err_total', ['move_total']), 'controls': fit(db, df, pm, 'err_total', ['move_total', 'abs_spread'], True),
                   'trimmed_1pct': fit(db, df, pm & trim, 'err_total', ['move_total']), 'per_book_move': fit(db, df, pm, 'err_total', ['move_total_perbook']),
                   'spread': fit(db, df, pm, 'err_spread', ['move_home']), 'open_error_total_vs_move': fit(db, df, pm, 'err_open_total', ['move_total'])}
    res['step1'] = s1
    d_, c_ = s1['discovery']['primary']['move_total'], s1['confirmation']['primary']['move_total']
    same = bool(np.sign(d_['coef']) == np.sign(c_['coef']))
    res['step1_reading'] = {'same_sign': same, 'confirmation_p_two_sided': c_['p_two_sided'], 'supported': bool(same and c_['p_two_sided'] < 0.05),
                            'direction': ('momentum' if c_['coef'] > 0 else 'overreaction'), 'confirmation_slope': c_['coef'], 'discovery_slope': d_['coef'],
                            'detectable_slope_conf_80pct': 2.8 * c_['se'], 'holm': 'pending: family of 8 incomplete'}

    # ---- step 2: cross-market
    q75 = float(df.loc[disc, 'abs_move_home'].quantile(0.75))
    df['big_move_home'] = (df['abs_move_home'] >= q75).astype(float)
    s2 = {'discovery_q75_abs_spread_move': q75}
    for per, pm in (('discovery', disc), ('confirmation', conf)):
        s2[per] = {'continuous': fit(db, df, pm, 'err_total', ['move_total', 'move_home', 'abs_move_home']), 'indicator': fit(db, df, pm, 'err_total', ['move_total', 'move_home', 'big_move_home'])}
    res['step2'] = s2

    # ---- step 3: model relevance
    oos = pd.concat([pd.read_csv(p, dtype={'game_id': str}, usecols=['game_id', 'total_pred']) for p in sorted((ROOT / 'reports').glob('oos_predictions_20??-??.csv'))])
    oos['gameId'] = oos['game_id'].str.zfill(10)
    j = df.merge(oos[['gameId', 'total_pred']].drop_duplicates('gameId'), on='gameId', how='inner')
    j['model_resid'] = j['actual_total'] - j['total_pred']
    dbj = DayBoot(j['gameDate'])
    res['step3'] = {'n': int(len(j)), 'model_resid_on_move_total': fit(dbj, j, np.ones(len(j), bool), 'model_resid', ['move_total']),
                    'model_resid_on_move_total_confirmation': fit(dbj, j, j['seasonYear'].isin(CONF).to_numpy(), 'model_resid', ['move_total'])}

    # ---- step 4: betting at the close, priced with the closing over/under quotes
    cons = df['close_total'].to_numpy(float)
    L, O, U = tc.to_numpy(float), mc['over'].reindex(ids).to_numpy(float), mc['under'].reindex(ids).to_numpy(float)
    act = df['actual_total'].to_numpy(float)
    line_o, price_o = np.full(len(df), np.nan), np.full(len(df), np.nan)
    line_u, price_u = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for i in range(len(df)):
        row = L[i]
        ok = np.isfinite(row)
        if not ok.any():
            continue
        dist = np.where(ok, np.abs(row - cons[i]), np.inf)
        best = np.flatnonzero(dist == dist.min())
        for arr, lo_, pr_, lg in ((O[i], line_o, price_o, i), (U[i], line_u, price_u, i)):
            cand = arr[best]
            k = best[np.nanargmax(payout(cand))] if np.isfinite(cand).any() else best[0]
            lo_[lg], pr_[lg] = row[k], arr[k] if np.isfinite(arr[k]) else -110.0

    def settle(line, price, side):
        r = (act - line) * (1 if side == 'over' else -1)
        pay = payout(np.nan_to_num(price, nan=-110.0))
        return np.where(r > 0, pay, np.where(r < 0, -1.0, 0.0))
    P = {('over', 'quoted'): settle(line_o, price_o, 'over'), ('under', 'quoted'): settle(line_u, price_u, 'under'),
         ('over', 'naive'): settle(cons, np.full(len(df), -110.0), 'over'), ('under', 'naive'): settle(cons, np.full(len(df), -110.0), 'under')}
    res['quote_check'] = {'mean_over_payout': float(np.nanmean(payout(price_o))), 'mean_under_payout': float(np.nanmean(payout(price_u))), 'share_with_quotes': float(np.isfinite(line_o).mean())}
    mv = df['move_total'].to_numpy(float)

    def roi(side_arr, m, kind):
        p = np.where(side_arr > 0, P[('over', kind)], P[('under', kind)])
        ok = m & np.isfinite(p)
        pt, lo, hi = db.ratio(np.where(ok, p, 0.0), ones, ok)
        return {'roi': pt * 100, 'lo': lo * 100, 'hi': hi * 100, 'bets': int(ok.sum())}
    grid = []
    for direction in ('with the move', 'against the move'):
        sign = np.sign(mv) * (1 if direction == 'with the move' else -1)
        for c in CUTS:
            sel = np.abs(mv) >= c
            row = {'direction': direction, 'cutoff': c}
            for per, pm in (('discovery', disc), ('confirmation', conf)):
                m = pm & sel & (sign != 0)
                row[per] = {'quoted': roi(sign, m, 'quoted'), 'naive': roi(sign, m, 'naive')}
            grid.append(row)
    elig = [g for g in grid if g['discovery']['quoted']['bets'] >= 200]
    bet = {'grid': grid}
    if elig:
        pick = max(elig, key=lambda g: g['discovery']['quoted']['roi'])
        bet['chosen'] = {'direction': pick['direction'], 'cutoff': pick['cutoff']}
        bet['chosen_result'] = {p: pick[p] for p in ('discovery', 'confirmation')}
    controls = {}
    for per, pm in (('discovery', disc), ('confirmation', conf)):
        controls[per] = {}
        for side in ('over', 'under'):
            for kind in ('quoted', 'naive'):
                p = P[(side, kind)]
                ok = pm & np.isfinite(p)
                pt, lo, hi = db.ratio(np.where(ok, p, 0.0), ones, ok)
                controls[per][f'{side}, all games, {kind}'] = {'roi': pt * 100, 'lo': lo * 100, 'hi': hi * 100, 'bets': int(ok.sum())}
    bet['controls'] = controls
    res['bet'] = bet

    # ---- step 5: sensitivity
    sens = {}
    for lab, m in (('2021-22..2025-26', S.isin(DISC[2:] + CONF).to_numpy()), ('2021-22 only', (S == '2021-22').to_numpy())):
        sens[lab] = fit(db, df, m, 'err_total', ['move_total'])
    near = np.zeros(len(df), bool)
    for lo_, hi_ in oc.CLOSING_BAD_WINDOWS:
        near |= df['gameDate'].between(pd.Timestamp(lo_) - pd.Timedelta(days=14), pd.Timestamp(hi_) + pd.Timedelta(days=14)).to_numpy()
    sens['excluding games within 14 days of the contamination windows (discovery)'] = fit(db, df, disc & ~near, 'err_total', ['move_total'])
    res['sensitivity'] = sens
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    print(json.dumps({'games': res['games'], 'move_sd': res['move_total_sd'], 'step1': res['step1_reading'], 'quote_check': res['quote_check'], 'chosen': bet.get('chosen')}, indent=1, default=float))


if __name__ == '__main__':
    main()
