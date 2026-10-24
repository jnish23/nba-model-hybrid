"""E16 - overtime and close finishes (angle 2 of 8). Registered in reports/experiments/PREREGISTRATION.md ("E16") before any
number was computed. Reuses the E14 frame, bootstrap and betting settlement. python -m experiments.e16_overtime"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import e14_blowout_under as e14                       # noqa: E402
from experiments.e14_blowout_under import DayBoot, LABELS               # noqa: E402
from experiments.e12_shopping import load_multibook, payout            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports' / 'experiments' / 'E16_overtime'
DISC, CONF = e14.DISCOVERY, e14.CONFIRM
QS = [0.5, 0.6, 0.7, 0.8, 0.9]
WIN = 100 / 110


def pval_pos(boots):
    """One-sided bootstrap p for 'coefficient > 0' (share of draws <= 0, floored at 1/1001)."""
    return float((1 + (boots <= 0).sum()) / (len(boots) + 1))


def coef(db, df, mask, y, xcol, controls=('abs_spread', 'odds_total')):
    cols = [np.ones(len(df)), df[xcol].to_numpy(float)] + [df[c].to_numpy(float) for c in controls]
    X = np.column_stack(cols)
    beta, lo, hi, boots = db.ols(X, df[y].to_numpy(float), mask)
    return {'coef': float(beta[1]), 'lo': float(lo[1]), 'hi': float(hi[1]), 'p_one_sided': pval_pos(boots[:, 1]), 'n': int(np.asarray(mask).sum())}, boots[:, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = e14.load_frame()
    db = DayBoot(df['gameDate'])
    S = df['seasonYear']
    disc, conf = S.isin(DISC).to_numpy(), S.isin(CONF).to_numpy()
    reg_all = S.isin(DISC + CONF).to_numpy()
    ot = df['ot'].fillna(False).to_numpy()
    ones = np.ones(len(df))
    res = {}

    # ---- step 1: overtime logistic on discovery, frozen
    lr = LogisticRegression(C=1e9, max_iter=1000).fit(df.loc[disc, ['abs_spread']], ot[disc].astype(int))
    df['p_ot'] = lr.predict_proba(df[['abs_spread']])[:, 1]
    res['step1'] = {'intercept': float(lr.intercept_[0]), 'slope_per_spread_point': float(lr.coef_[0][0])}
    edges = np.unique(np.quantile(df.loc[disc, 'p_ot'], np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(df['p_ot'], edges[1:-1])
    cal = []
    for per, pm in (('discovery (in-sample)', disc), ('confirmation (out-of-sample)', conf)):
        for b in range(len(edges) - 1):
            m = pm & (bins == b)
            if not m.any():
                continue
            obs, lo, hi = db.ratio(ot.astype(float), ones, m)
            cal.append({'period': per, 'bin': b + 1, 'games': int(m.sum()), 'predicted': float(df.loc[m, 'p_ot'].mean()), 'observed': obs, 'lo': lo, 'hi': hi,
                        'gap_beyond_interval_pp': float(max(0, max(df.loc[m, 'p_ot'].mean() - hi, lo - df.loc[m, 'p_ot'].mean())) * 100)})
    res['calibration'] = cal
    res['calibration_passes_2pp'] = bool(all(c['gap_beyond_interval_pp'] <= 2 for c in cal if c['period'].startswith('confirmation')))

    # ---- step 2: close-finish probability
    sigma = float((df.loc[disc, 'target_point_diff'] + df.loc[disc, 'odds_home_spread']).std())
    sp = df['odds_home_spread'].to_numpy(float)
    df['p_close'] = norm.cdf((3.5 + sp) / sigma) - norm.cdf((-3.5 + sp) / sigma)
    res['sigma_margin_discovery'] = sigma

    # ---- step 3: primary and secondary market tests
    s3 = {}
    boots_store = {}
    for xcol in ('p_ot', 'p_close'):
        for lab, pm_ in (('all games', np.ones(len(df), bool)), ('regulation only', ~ot)):
            for per, m in (('discovery', disc), ('confirmation', conf)):
                r_, b_ = coef(db, df, m & pm_, 'r', xcol)
                s3.setdefault(xcol, {}).setdefault(lab, {})[per] = r_
                boots_store[(xcol, lab, per)] = b_
    # all-minus-regulation difference with shared draws (confirmation)
    for xcol in ('p_ot', 'p_close'):
        d = boots_store[(xcol, 'all games', 'confirmation')] - boots_store[(xcol, 'regulation only', 'confirmation')]
        s3[xcol]['difference_all_minus_regulation_confirmation'] = {'diff': float(s3[xcol]['all games']['confirmation']['coef'] - s3[xcol]['regulation only']['confirmation']['coef']),
                                                                    'lo': float(np.quantile(d, .025)), 'hi': float(np.quantile(d, .975))}
    a, g = s3['p_ot']['all games']['confirmation'], s3['p_ot']['regulation only']['confirmation']
    d = s3['p_ot']['difference_all_minus_regulation_confirmation']
    pos_both = s3['p_ot']['all games']['discovery']['coef'] > 0 and a['coef'] > 0
    all_ok = bool(pos_both and a['p_one_sided'] < 0.05)
    reg_zero = bool(g['lo'] <= 0 <= g['hi']) or bool(abs(g['coef']) < 0.5 * abs(a['coef']) and d['lo'] > 0)
    res['step3'] = s3
    res['step3_reading'] = {'H_OT_supported_raw': all_ok, 'positive_in_both_periods': bool(pos_both), 'confirmation_p_one_sided': a['p_one_sided'],
                            'regulation_only_includes_zero_or_small': reg_zero, 'overtime_is_mechanism': bool(all_ok and reg_zero),
                            'holm': 'pending: family of 8 incomplete'}
    # sensitivity without 2019-20 / 2020-21
    sens = {}
    for lab, m in (('discovery = 2021-22 only', (S == '2021-22').to_numpy()), ('pooled 2021-22..2025-26', S.isin(['2021-22'] + CONF).to_numpy())):
        sens[lab] = {'p_ot all games': coef(db, df, m, 'r', 'p_ot')[0], 'p_ot regulation only': coef(db, df, m & ~ot, 'r', 'p_ot')[0],
                     'abs_spread slope all games': e14.slope(db, df, m)[0], 'abs_spread slope regulation only': e14.slope(db, df, m & ~ot)[0]}
    res['sensitivity_no_covid_seasons'] = sens

    # ---- step 4: size sanity check
    rows = []
    pm = reg_all
    mu_ot = db.ratio(df['r'], ones, pm & ot)[0]
    mu_reg = db.ratio(df['r'], ones, pm & ~ot)[0]
    for b in LABELS:
        m = pm & (df['bucket'] == b).to_numpy()
        row = {'bucket': b, 'games': int(m.sum()), 'mean_abs_spread': float(df.loc[m, 'abs_spread'].mean())}
        for key, num, mm in (('ot_share', ot.astype(float), m), ('mean_r_ot', df['r'], m & ot), ('mean_r_reg', df['r'], m & ~ot), ('mean_r_all', df['r'], m)):
            row[key], row[key + '_lo'], row[key + '_hi'] = db.ratio(num, ones, mm)
        rows.append(row)
    res['step4'] = {'buckets': rows, 'mu_ot': mu_ot, 'mu_reg': mu_reg, 'ot_minus_reg_mean_r': mu_ot - mu_reg,
                    'ot_share_overall': float(ot[pm].mean()),
                    'implied_gap_0_3_vs_12plus': (rows[0]['ot_share'] - rows[-1]['ot_share']) * (mu_ot - mu_reg),
                    'observed_mean_r_gap_0_3_minus_12plus': rows[0]['mean_r_all'] - rows[-1]['mean_r_all'],
                    'observed_mean_r_gap_regulation': rows[0]['mean_r_reg'] - rows[-1]['mean_r_reg'],
                    'se_two_groups_of_2000': float(np.sqrt(2 * df.loc[reg_all, 'r'].var() / 2000)), 'sd_r': float(df.loc[reg_all, 'r'].std())}

    # ---- step 5: decomposition with refitted logistic per draw
    codes = db.codes

    def decomp(fit_mask, mu_mask):
        idx_fit = np.flatnonzero(fit_mask)
        x_fit, y_fit = df.loc[fit_mask, ['abs_spread']].to_numpy(), ot[fit_mask].astype(int)
        out = []
        for b in range(db.W.shape[0]):
            w = db.W[b][codes]
            wf = w[idx_fit]
            if wf.sum() == 0:
                continue
            lg = LogisticRegression(C=1e9, max_iter=200).fit(x_fit, y_fit, sample_weight=wf)
            p = lg.predict_proba(x_fit)[:, 1]
            dpds = float(np.average(p * (1 - p), weights=wf) * lg.coef_[0][0])
            wm = w * mu_mask
            mo = np.sum(wm * ot * df['r'].to_numpy()) / np.sum(wm * ot)
            mr = np.sum(wm * ~ot * df['r'].to_numpy()) / np.sum(wm * ~ot)
            out.append(((mo - mr) * dpds, dpds, mo - mr))
        return np.array(out)

    def point(fit_mask, mu_mask):
        lg = LogisticRegression(C=1e9, max_iter=1000).fit(df.loc[fit_mask, ['abs_spread']], ot[fit_mask].astype(int))
        p = lg.predict_proba(df.loc[fit_mask, ['abs_spread']])[:, 1]
        dpds = float((p * (1 - p)).mean() * lg.coef_[0][0])
        mo = df.loc[mu_mask & ot, 'r'].mean()
        mr = df.loc[mu_mask & ~ot, 'r'].mean()
        return (mo - mr) * dpds, dpds, mo - mr
    dec = {}
    for lab, fm, mm in (('discovery logistic, pooled mean r (registered)', disc, reg_all), ('confirmation logistic and mean r', conf, conf), ('discovery logistic and mean r', disc, disc)):
        arr = decomp(fm, mm.astype(float))
        pt = point(fm, mm)
        dec[lab] = {'ot_contribution_to_slope': {'point': float(pt[0]), 'lo': float(np.quantile(arr[:, 0], .025)), 'hi': float(np.quantile(arr[:, 0], .975))},
                    'dp_ds': {'point': float(pt[1]), 'lo': float(np.quantile(arr[:, 1], .025)), 'hi': float(np.quantile(arr[:, 1], .975))},
                    'mu_ot_minus_mu_reg': {'point': float(pt[2]), 'lo': float(np.quantile(arr[:, 2], .025)), 'hi': float(np.quantile(arr[:, 2], .975))}}
    dec['direct slopes, registered seasons'] = {'all games': e14.slope(db, df, reg_all)[0], 'regulation only': e14.slope(db, df, reg_all & ~ot)[0]}
    res['step5'] = dec

    # ---- step 6: model test
    oos = pd.concat([pd.read_csv(p, dtype={'game_id': str}, usecols=['game_id', 'total_pred']) for p in sorted((ROOT / 'reports').glob('oos_predictions_20??-??.csv'))])
    oos['gameId'] = oos['game_id'].str.zfill(10)
    j = df.merge(oos[['gameId', 'total_pred']].drop_duplicates('gameId'), on='gameId', how='inner')
    j['model_resid'] = j['target_total_pts'] - j['total_pred']
    j['model_lean'] = j['total_pred'] - j['odds_total']
    dbj = DayBoot(j['gameDate'])
    s6 = {}
    for lab, m in (('2021-22..2025-26', np.ones(len(j), bool)), ('confirmation 2022-23..2025-26', j['seasonYear'].isin(CONF).to_numpy())):
        s6[lab] = {y: coef(dbj, j, m, y, 'p_ot')[0] for y in ('model_resid', 'model_lean', 'r')}
    res['step6'] = s6
    res['step6_supported'] = bool(s6['confirmation 2022-23..2025-26']['model_resid']['p_one_sided'] < 0.05 and s6['confirmation 2022-23..2025-26']['model_resid']['coef'] > 0)

    # ---- step 8: betting
    mats, _ = load_multibook()
    tots, over_odds, under_odds = (mats[k].reindex(df['gameId']) for k in ('tot', 'over', 'under'))
    a_ = df['target_total_pts'].to_numpy(float)

    def best(lines_, odds_, pick_min):
        bl = (lines_.min(axis=1) if pick_min else lines_.max(axis=1)).to_numpy()
        pr = np.full(len(df), np.nan)
        for i, (lrow, orow, b) in enumerate(zip(lines_.to_numpy(), odds_.to_numpy(), bl)):
            if np.isnan(b):
                continue
            q = orow[lrow == b]
            q = q[~np.isnan(q)]
            pr[i] = q.max() if len(q) else -110.0
        return bl, pr
    ob_line, ob_pr = best(tots, over_odds, True)
    ub_line, ub_pr = best(tots, under_odds, False)

    def prof(line, side, price=None):
        res_ = (a_ - np.asarray(line, float)) * (1 if side == 'over' else -1)
        pay = np.full(len(df), WIN) if price is None else payout(np.nan_to_num(price, nan=-110.0))
        return np.where(res_ > 0, pay, np.where(res_ < 0, -1.0, 0.0))
    P = {'over_open': prof(df['odds_total'], 'over'), 'over_close': prof(df['close_total'], 'over'), 'over_best': prof(ob_line, 'over', ob_pr),
         'under_open': prof(df['odds_total'], 'under'), 'under_close': prof(df['close_total'], 'under'), 'under_best': prof(ub_line, 'under', ub_pr)}
    ok_close = df['close_total'].notna().to_numpy()
    ok_best = ~np.isnan(ob_line)

    def roi(key, m):
        m = m & (ok_close if key.endswith('close') else ok_best if key.endswith('best') else True)
        pt, lo, hi = db.ratio(P[key], ones, m)
        return {'roi': pt * 100, 'lo': lo * 100, 'hi': hi * 100, 'bets': int(m.sum())}
    pot = df['p_ot'].to_numpy()
    cuts_hi = {q: float(np.quantile(pot[disc], q)) for q in QS}
    cuts_lo = {q: float(np.quantile(pot[disc], 1 - q)) for q in QS}
    grid = []
    for q in QS:
        row = {'q': q, 'cutoff_p_ot': cuts_hi[q]}
        for per, pm_ in (('discovery', disc), ('confirmation', conf)):
            row[per] = roi('over_open', pm_ & (pot >= cuts_hi[q]))
        grid.append(row)
    qsel = max([r_ for r_ in grid if r_['discovery']['bets'] >= 200], key=lambda r_: r_['discovery']['roi'])['q']
    bet = {'q_chosen': qsel, 'cutoff': cuts_hi[qsel], 'grid': grid}
    for per, pm_ in (('discovery', disc), ('confirmation', conf)):
        sel = pm_ & (pot >= cuts_hi[qsel])
        bet[per] = {'over all games (control)': roi('over_open', pm_), 'over opening @ -110': roi('over_open', sel), 'over closing @ -110': roi('over_close', sel), 'over best book': roi('over_best', sel)}
    mgrid = []
    for q in QS:
        row = {'q': q, 'cutoff_p_ot': cuts_lo[q]}
        for per, pm_ in (('discovery', disc), ('confirmation', conf)):
            row[per] = roi('under_open', pm_ & (pot <= cuts_lo[q]))
        mgrid.append(row)
    qm = max([r_ for r_ in mgrid if r_['discovery']['bets'] >= 200], key=lambda r_: r_['discovery']['roi'])['q']
    mirror = {'q_chosen': qm, 'cutoff': cuts_lo[qm], 'grid': mgrid}
    for per, pm_ in (('discovery', disc), ('confirmation', conf)):
        sel = pm_ & (pot <= cuts_lo[qm])
        mirror[per] = {'under opening @ -110': roi('under_open', sel), 'under closing @ -110': roi('under_close', sel), 'under best book': roi('under_best', sel)}
    res['bet_registered_over'], res['bet_mirror_exploratory'] = bet, mirror

    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    print(json.dumps({'step1': res['step1'], 'cal_pass': res['calibration_passes_2pp'], 'sigma': sigma, 'reading': res['step3_reading'],
                      'p_ot_conf': s3['p_ot']['all games']['confirmation'], 'p_ot_conf_reg': s3['p_ot']['regulation only']['confirmation']}, indent=1, default=float))


if __name__ == '__main__':
    main()
