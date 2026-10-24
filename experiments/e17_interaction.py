"""E17 - spread x total interaction (angle 3 of 8). Registered in reports/experiments/PREREGISTRATION.md ("E17") before any
number was computed. python -m experiments.e17_interaction"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import e14_blowout_under as e14                       # noqa: E402
from experiments.e14_blowout_under import DayBoot                       # noqa: E402
from experiments.e12_shopping import load_multibook, payout            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports' / 'experiments' / 'E17_interaction'
DISC, CONF = e14.DISCOVERY, e14.CONFIRM
SB_EDGES, SB_LABELS = [0, 5, 10, np.inf], ['0-5', '5-10', '10+']
TL = ['low total', 'mid total', 'high total']
WIN = 100 / 110
MIN_CELL_BETS = 150


def p2(boots):
    lo, hi = (boots <= 0).sum(), (boots >= 0).sum()
    return float(max(2 * min(lo, hi), 1) / (len(boots) + 1))


def reg(db, df, mask, y, tcol, seasons=None):
    """r ~ |spread| + T + |spread| x T + season dummies. Returns coefficient dicts and the draws."""
    a, t = df['abs_spread'].to_numpy(float), df[tcol].to_numpy(float)
    ss = sorted(set(df.loc[mask, 'seasonYear'])) if seasons is None else seasons
    cols = [np.ones(len(df)), a, t, a * t] + [(df['seasonYear'] == s).to_numpy(float) for s in ss[1:]]
    beta, lo, hi, boots = db.ols(np.column_stack(cols), df[y].to_numpy(float), mask)
    out = {}
    for i, n in enumerate(['intercept', 'abs_spread', 'T', 'abs_spread_x_T']):
        out[n] = {'coef': float(beta[i]), 'lo': float(lo[i]), 'hi': float(hi[i]), 'p_two_sided': p2(boots[:, i])}
    out['n'] = int(np.asarray(mask).sum())
    return out, boots


def ratio_diff(db, num, den, m1, m2):
    n1, d1 = db.day_sum(np.where(m1, num, 0.0)), db.day_sum(np.where(m1, den, 0.0))
    n2, d2 = db.day_sum(np.where(m2, num, 0.0)), db.day_sum(np.where(m2, den, 0.0))
    pt = n1.sum() / d1.sum() - n2.sum() / d2.sum()
    b = (db.W @ n1) / (db.W @ d1) - (db.W @ n2) / (db.W @ d2)
    return {'diff': float(pt), 'lo': float(np.quantile(b, .025)), 'hi': float(np.quantile(b, .975))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = e14.load_frame()
    db = DayBoot(df['gameDate'])
    S = df['seasonYear']
    disc, conf = S.isin(DISC).to_numpy(), S.isin(CONF).to_numpy()
    ones = np.ones(len(df))
    res = {}

    # ---- grid definitions (fixed from discovery)
    t_mean = float(df.loc[disc, 'odds_total'].mean())
    c1, c2 = (float(x) for x in np.quantile(df.loc[disc, 'odds_total'], [1 / 3, 2 / 3]))
    df['T'] = df['odds_total'] - t_mean
    df['terc'] = np.digitize(df['odds_total'], [c1, c2])
    df['sb'] = pd.cut(df['abs_spread'], SB_EDGES, labels=SB_LABELS, right=False)
    df['T_ws'] = df['odds_total'] - df.groupby('seasonYear')['odds_total'].transform('mean')
    q = df.groupby('seasonYear')['odds_total'].transform(lambda s: pd.Series(np.digitize(s, np.quantile(s, [1 / 3, 2 / 3])), index=s.index))
    df['terc_ws'] = q.astype(int)
    res['definitions'] = {'discovery_mean_total': t_mean, 'tercile_cutoffs': [c1, c2], 'abs_spread_edges': [0, 5, 10, 'inf']}
    nonpush = (df['r'] != 0).to_numpy()
    under = (df['r'] < 0).astype(float).to_numpy()

    def grid(tc, periods):
        rows = []
        for per, pm in periods:
            for ti in range(3):
                for si, sl in enumerate(SB_LABELS):
                    m = pm & (df[tc] == ti).to_numpy() & (df['sb'] == sl).to_numpy()
                    mr = db.ratio(df['r'], ones, m)
                    ur = db.ratio(under, nonpush.astype(float), m & nonpush)
                    rows.append({'period': per, 'total': TL[ti], 'spread': sl, 'games': int(m.sum()), 'share_of_period': float(m.sum() / pm.sum()),
                                 'mean_r': mr[0], 'mean_r_lo': mr[1], 'mean_r_hi': mr[2], 'under_rate': ur[0], 'under_lo': ur[1], 'under_hi': ur[2]})
        return rows
    res['grid'] = grid('terc', [('discovery', disc), ('confirmation', conf)])
    res['grid_within_season_terciles'] = grid('terc_ws', [('discovery', disc), ('confirmation', conf)])
    res['tercile_shares'] = {per: [float(((df['terc'] == i).to_numpy() & pm).sum() / pm.sum()) for i in range(3)] for per, pm in (('discovery', disc), ('confirmation', conf))}

    # ---- step 2: primary regression
    s2 = {}
    boots_c = None
    for per, pm in (('discovery', disc), ('confirmation', conf), ('registered pooled', disc | conf)):
        s2[per], b = reg(db, df, pm, 'r', 'T')
        if per == 'confirmation':
            boots_c = b
        tv = df.loc[pm, 'T']
        slopes = {}
        for pc in (16, 50, 84):
            tp = float(np.percentile(tv, pc))
            d = b[:, 1] + b[:, 3] * tp
            pt = s2[per]['abs_spread']['coef'] + s2[per]['abs_spread_x_T']['coef'] * tp
            slopes[f'T at {pc}th percentile ({tp + t_mean:.1f} total)'] = {'slope': float(pt), 'lo': float(np.quantile(d, .025)), 'hi': float(np.quantile(d, .975))}
        s2[per]['spread_slope_at_total_percentiles'] = slopes
    ci_ = s2['confirmation']['abs_spread_x_T']
    iqr = float(np.subtract(*np.percentile(df.loc[conf, 'T'], [75, 25])))
    se = (ci_['hi'] - ci_['lo']) / 3.92
    s2['confirmation']['detectable'] = {'se_product': se, 'iqr_total': iqr, 'min_detectable_slope_change_over_iqr': 2.8 * se * iqr}
    cell_hi = (df['terc'] == 2).to_numpy() & (df['sb'] == '10+').to_numpy()
    cell_lo = (df['terc'] == 0).to_numpy() & (df['sb'] == '10+').to_numpy()
    s2['contrast_10plus_top_minus_bottom_tercile'] = {per: ratio_diff(db, df['r'], ones, pm & cell_hi, pm & cell_lo) for per, pm in (('discovery', disc), ('confirmation', conf))}
    sp = {per: s2[per]['abs_spread_x_T']['coef'] for per in ('discovery', 'confirmation')}
    same = bool(np.sign(sp['discovery']) == np.sign(sp['confirmation']))
    res['step2'] = s2
    res['step2_reading'] = {'same_sign_both_periods': same, 'confirmation_p_two_sided': s2['confirmation']['abs_spread_x_T']['p_two_sided'],
                            'example_sign_positive_in_confirmation': bool(sp['confirmation'] > 0),
                            'H_INT_supported_raw': bool(same and s2['confirmation']['abs_spread_x_T']['p_two_sided'] < 0.05), 'holm': 'pending: family of 8 incomplete'}
    # within-season centered version
    ws = {}
    for per, pm in (('discovery', disc), ('confirmation', conf)):
        ws[per] = reg(db, df, pm, 'r', 'T_ws')[0]['abs_spread_x_T']
    res['step2_sensitivity_within_season_total'] = ws
    # no COVID seasons
    sens = {}
    for lab, m in (('2021-22 only', (S == '2021-22').to_numpy()), ('pooled 2021-22..2025-26', S.isin(['2021-22'] + CONF).to_numpy())):
        sens[lab] = reg(db, df, m, 'r', 'T')[0]['abs_spread_x_T']
    res['sensitivity_no_covid_seasons'] = sens

    # ---- step 3: model test
    oos = pd.concat([pd.read_csv(p, dtype={'game_id': str}, usecols=['game_id', 'total_pred']) for p in sorted((ROOT / 'reports').glob('oos_predictions_20??-??.csv'))])
    oos['gameId'] = oos['game_id'].str.zfill(10)
    j = df.merge(oos[['gameId', 'total_pred']].drop_duplicates('gameId'), on='gameId', how='inner')
    j['model_resid'] = j['target_total_pts'] - j['total_pred']
    j['model_lean'] = j['total_pred'] - j['odds_total']
    dbj = DayBoot(j['gameDate'])
    s3 = {}
    for lab, m in (('2021-22..2025-26 (all OOS)', np.ones(len(j), bool)), ('confirmation 2022-23..2025-26', j['seasonYear'].isin(CONF).to_numpy())):
        s3[lab] = {y: reg(dbj, j, m, y, 'T')[0] for y in ('model_resid', 'model_lean', 'r')}
    res['step3'] = s3
    mc = s3['confirmation 2022-23..2025-26']['model_resid']['abs_spread_x_T']
    res['step3_reading'] = {'residual_product_coef': mc['coef'], 'p_two_sided': mc['p_two_sided'],
                            'same_sign_as_market_confirmation': bool(np.sign(mc['coef']) == np.sign(sp['confirmation'])),
                            'layer_b_supported': bool(np.sign(mc['coef']) == np.sign(sp['confirmation']) and mc['p_two_sided'] < 0.05)}
    res['layer_c_run'] = bool(res['step2_reading']['H_INT_supported_raw'] and res['step3_reading']['layer_b_supported'])

    # ---- step 5: betting by cell and side
    mats, _ = load_multibook()
    tots, over_odds, under_odds = (mats[k].reindex(df['gameId']) for k in ('tot', 'over', 'under'))
    a_ = df['target_total_pts'].to_numpy(float)

    def best(lines_, odds_, pick_min):
        bl = (lines_.min(axis=1) if pick_min else lines_.max(axis=1)).to_numpy()
        pr = np.full(len(df), np.nan)
        for i, (lrow, orow, b) in enumerate(zip(lines_.to_numpy(), odds_.to_numpy(), bl)):
            if np.isnan(b):
                continue
            qq = orow[lrow == b]
            qq = qq[~np.isnan(qq)]
            pr[i] = qq.max() if len(qq) else -110.0
        return bl, pr
    ob_line, ob_pr = best(tots, over_odds, True)
    ub_line, ub_pr = best(tots, under_odds, False)

    def prof(line, side, price=None):
        r_ = (a_ - np.asarray(line, float)) * (1 if side == 'over' else -1)
        pay = np.full(len(df), WIN) if price is None else payout(np.nan_to_num(price, nan=-110.0))
        return np.where(r_ > 0, pay, np.where(r_ < 0, -1.0, 0.0))
    P = {'over_open': prof(df['odds_total'], 'over'), 'over_close': prof(df['close_total'], 'over'), 'over_best': prof(ob_line, 'over', ob_pr),
         'under_open': prof(df['odds_total'], 'under'), 'under_close': prof(df['close_total'], 'under'), 'under_best': prof(ub_line, 'under', ub_pr)}
    ok = {'close': df['close_total'].notna().to_numpy(), 'best': ~np.isnan(ob_line)}

    def roi(key, m):
        m = m & (ok['close'] if key.endswith('close') else ok['best'] if key.endswith('best') else True)
        pt, lo, hi = db.ratio(P[key], ones, m)
        return {'roi': pt * 100, 'lo': lo * 100, 'hi': hi * 100, 'bets': int(m.sum())}
    cells = []
    for ti in range(3):
        for sl in SB_LABELS:
            cm = (df['terc'] == ti).to_numpy() & (df['sb'] == sl).to_numpy()
            for side in ('over', 'under'):
                cells.append({'total': TL[ti], 'spread': sl, 'side': side, 'discovery': roi(f'{side}_open', disc & cm), 'confirmation': roi(f'{side}_open', conf & cm), '_cm': cm})
    elig = [c for c in cells if c['discovery']['bets'] >= MIN_CELL_BETS]
    pick = max(elig, key=lambda c: c['discovery']['roi'])
    bet = {'chosen': {k: pick[k] for k in ('total', 'spread', 'side')}, 'cells': [{k: v for k, v in c.items() if k != '_cm'} for c in cells]}
    for per, pm in (('discovery', disc), ('confirmation', conf)):
        sel = pm & pick['_cm']
        s = pick['side']
        bet[per] = {f'{s} chosen cell, opening @ -110': roi(f'{s}_open', sel), f'{s} chosen cell, closing @ -110': roi(f'{s}_close', sel), f'{s} chosen cell, best book': roi(f'{s}_best', sel),
                    'over all games (control)': roi('over_open', pm), 'under all games (control)': roi('under_open', pm)}
    res['bet'] = bet

    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    print(json.dumps({'defs': res['definitions'], 'shares': res['tercile_shares'], 'reading': res['step2_reading'],
                      'prod': {k: res['step2'][k]['abs_spread_x_T'] for k in ('discovery', 'confirmation')}, 'layer_b': res['step3_reading'], 'chosen': bet['chosen']}, indent=1, default=float))


if __name__ == '__main__':
    main()
