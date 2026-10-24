"""E23 - favourite on a back-to-back (angle 7 of 8). Registered in reports/experiments/PREREGISTRATION.md ("E23") before any number was
computed. Regular-season games only. python -m experiments.e23_b2b"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import e14_blowout_under as e14                       # noqa: E402
from experiments.e14_blowout_under import DayBoot                       # noqa: E402
from experiments.e19_tanking import fit_flags, e14_ratio_diff           # noqa: E402
from experiments.totals_bets import settlement_profits                  # noqa: E402
from team.data_paths import latest_training_set                         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports' / 'experiments' / 'E23_b2b'
DISC, CONF = e14.DISCOVERY, e14.CONFIRM
THRESHOLDS = [8, 6, 10]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = e14.load_frame()
    df = df[df['seasonYear'].isin(DISC + CONF) & df['gameId'].str.startswith('002')]
    ts = pd.read_csv(latest_training_set(), usecols=['gameId', 'is_b2b_home', 'is_b2b_away', 'days_rest_home', 'days_rest_away'], low_memory=False)
    ts['gameId'] = ts['gameId'].astype(str).str.zfill(10)
    df = df.merge(ts.drop_duplicates('gameId'), on='gameId', how='left')
    res = {'games_before_dropping_pickems': int(len(df))}
    df = df[df['odds_home_spread'] != 0].dropna(subset=['is_b2b_home', 'is_b2b_away']).reset_index(drop=True)
    res['games'] = int(len(df))
    home_fav = (df['odds_home_spread'] < 0).to_numpy()
    df['FB'] = np.where(home_fav, df['is_b2b_home'], df['is_b2b_away']).astype(float)
    df['UB'] = np.where(home_fav, df['is_b2b_away'], df['is_b2b_home']).astype(float)
    fav_margin = np.where(home_fav, df['target_point_diff'], -df['target_point_diff']).astype(float)
    S = df['seasonYear']
    disc, conf = S.isin(DISC).to_numpy(), S.isin(CONF).to_numpy()
    df['T_c'] = df['odds_total'] - df.loc[disc, 'odds_total'].mean()
    db = DayBoot(df['gameDate'])
    ones = np.ones(len(df))
    nonpush = (df['r'] != 0).to_numpy()
    under = (df['r'] < 0).astype(float).to_numpy()

    # ---- blowout quantities
    sigma = float((df.loc[disc, 'target_point_diff'] + df.loc[disc, 'odds_home_spread']).std())
    res['sigma_margin_discovery'] = sigma
    mu_fav = df['abs_spread'].to_numpy(float)                                       # expected favourite margin
    p_fav_bl = 1 - norm.cdf((14.5 - mu_fav) / sigma)
    p_any_bl = p_fav_bl + norm.cdf((-14.5 - mu_fav) / sigma)
    df['bl_obs'] = (np.abs(df['target_point_diff']) >= 15).astype(float)
    df['favbl_obs'] = (fav_margin >= 15).astype(float)
    df['bl_diff'] = df['bl_obs'] - p_any_bl
    df['favbl_diff'] = df['favbl_obs'] - p_fav_bl
    df['bl_implied'], df['favbl_implied'] = p_any_bl, p_fav_bl

    # ---- step 1: cells
    cells, counts = [], {}
    for thr in THRESHOLDS:
        for side, col in (('favourite', 'FB'), ('underdog', 'UB')):
            for flag in (1.0, 0.0):
                for hi in (False, True):
                    sb = (df['abs_spread'] >= thr).to_numpy() == hi
                    cm = (df[col].to_numpy() == flag) & sb
                    for per, pm in (('discovery', disc), ('confirmation', conf)):
                        m = pm & cm
                        mr = db.ratio(df['r'], ones, m) if m.sum() else (np.nan,) * 3
                        ur = db.ratio(under, nonpush.astype(float), m & nonpush) if m.sum() else (np.nan,) * 3
                        bo = db.ratio(df['bl_obs'], ones, m) if m.sum() else (np.nan,) * 3
                        bd = db.ratio(df['bl_diff'], ones, m) if m.sum() else (np.nan,) * 3
                        fd = db.ratio(df['favbl_diff'], ones, m) if m.sum() else (np.nan,) * 3
                        cells.append({'threshold': thr, 'side': side, 'b2b': bool(flag), 'spread_bucket': f'>= {thr}' if hi else f'< {thr}', 'period': per, 'games': int(m.sum()),
                                      'mean_r': mr[0], 'mean_r_lo': mr[1], 'mean_r_hi': mr[2], 'under_rate': ur[0], 'under_lo': ur[1], 'under_hi': ur[2],
                                      'blowout_obs': bo[0], 'blowout_implied': float(df.loc[m, 'bl_implied'].mean()) if m.sum() else np.nan, 'blowout_diff': bd[0], 'blowout_diff_lo': bd[1], 'blowout_diff_hi': bd[2],
                                      'fav_blowout_obs': float(df.loc[m, 'favbl_obs'].mean()) if m.sum() else np.nan, 'fav_blowout_implied': float(df.loc[m, 'favbl_implied'].mean()) if m.sum() else np.nan,
                                      'fav_blowout_diff': fd[0], 'fav_blowout_diff_lo': fd[1], 'fav_blowout_diff_hi': fd[2]})
    res['cells'] = cells

    # ---- step 2: primary regression
    s2 = {}
    for per, pm in (('discovery', disc), ('confirmation', conf)):
        s2[per] = fit_flags(db, df, pm, 'r', ['FB', 'UB'])
    res['step2'] = s2
    res['step2_reading'] = {'discovery_wald_p': s2['discovery']['wald_p'], 'confirmation_wald_p': s2['confirmation']['wald_p'],
                            'H_B2B_supported_raw': bool(s2['discovery']['wald_p'] < 0.10 and s2['confirmation']['wald_p'] < 0.05), 'holm': 'pending: family of 8 incomplete'}
    sens = {}
    for lab, m in (('2021-22..2025-26', S.isin(DISC[2:] + CONF).to_numpy()),):
        sens[lab] = fit_flags(db, df, m, 'r', ['FB', 'UB'])
    res['sensitivity'] = sens

    # ---- step 3: blowout mechanism
    s3 = {'lpm': {}, 'cell': {}}
    for per, pm in (('discovery', disc), ('confirmation', conf)):
        s3['lpm'][per] = {y: fit_flags(db, df, pm, y, ['FB', 'UB']) for y in ('favbl_obs', 'bl_obs')}
        m = pm & (df['FB'].to_numpy() == 1) & (df['abs_spread'].to_numpy() >= 8)
        s3['cell'][per] = {'games': int(m.sum()), 'fav_blowout_diff': dict(zip(('diff', 'lo', 'hi'), db.ratio(df['favbl_diff'], ones, m))) if m.sum() else None,
                           'fav_blowout_obs': float(df.loc[m, 'favbl_obs'].mean()) if m.sum() else None, 'fav_blowout_implied': float(df.loc[m, 'favbl_implied'].mean()) if m.sum() else None}
        if m.sum():
            mn = pm & (df['FB'].to_numpy() == 0) & (df['abs_spread'].to_numpy() >= 8)
            s3['cell'][per]['vs_rested_favourites_diff'] = e14_ratio_diff(db, df['favbl_diff'], ones, m, mn)
    d_, c_ = s3['cell']['discovery'], s3['cell']['confirmation']
    supp = bool(d_['fav_blowout_diff'] and c_['fav_blowout_diff'] and d_['fav_blowout_diff']['diff'] < 0 and c_['fav_blowout_diff']['diff'] < 0 and c_['fav_blowout_diff']['hi'] < 0)
    res['step3'] = s3
    res['step3_reading'] = {'mechanism_supported': supp}

    # ---- step 4: model test
    oos = pd.concat([pd.read_csv(p, dtype={'game_id': str}, usecols=['game_id', 'total_pred']) for p in sorted((ROOT / 'reports').glob('oos_predictions_20??-??.csv'))])
    oos['gameId'] = oos['game_id'].str.zfill(10)
    j = df.merge(oos[['gameId', 'total_pred']].drop_duplicates('gameId'), on='gameId', how='inner')
    j['model_resid'] = j['target_total_pts'] - j['total_pred']
    j['model_lean'] = j['total_pred'] - j['odds_total']
    dbj = DayBoot(j['gameDate'])
    s4 = {'n': int(len(j)), 'models': {}, 'cells': []}
    for lab, mm in (('2021-22..2025-26', np.ones(len(j), bool)), ('confirmation 2022-23..2025-26', j['seasonYear'].isin(CONF).to_numpy())):
        s4['models'][lab] = {y: fit_flags(dbj, j, mm, y, ['FB', 'UB']) for y in ('model_resid', 'model_lean', 'r')}
        for side, col in (('favourite', 'FB'), ('underdog', 'UB')):
            for flag in (1.0, 0.0):
                for hi in (False, True):
                    m = mm & (j[col].to_numpy() == flag) & ((j['abs_spread'].to_numpy() >= 8) == hi)
                    if m.sum():
                        s4['cells'].append({'period': lab, 'side': side, 'b2b': bool(flag), 'spread_bucket': '>= 8' if hi else '< 8', 'games': int(m.sum()),
                                            'model_resid': dict(zip(('mean', 'lo', 'hi'), dbj.ratio(j['model_resid'], np.ones(len(j)), m)))})
    res['step4'] = s4
    res['layer_b_supported'] = bool(s4['models']['confirmation 2022-23..2025-26']['model_resid']['wald_p'] < 0.05)
    res['layer_c_run'] = bool(res['step2_reading']['H_B2B_supported_raw'] and res['layer_b_supported'])

    # ---- step 6: betting
    P, ok = settlement_profits(df)

    def roi(key, m):
        m = m & (ok['close'] if key.endswith('close') else ok['best'] if key.endswith('best') else True)
        pt, lo, hi = db.ratio(P[key], ones, m)
        return {'roi': pt * 100, 'lo': lo * 100, 'hi': hi * 100, 'bets': int(m.sum())}
    cand = []
    for side, col in (('favourite', 'FB'), ('underdog', 'UB')):
        for hi in (False, True):
            cm = (df[col].to_numpy() == 1) & ((df['abs_spread'].to_numpy() >= 8) == hi)
            for bs in ('over', 'under'):
                cand.append({'cell': f'{side} on back-to-back, |spread| {">= 8" if hi else "< 8"}', 'side': bs, 'cm': cm, 'discovery': roi(f'{bs}_open', disc & cm), 'confirmation': roi(f'{bs}_open', conf & cm)})
    elig = [c for c in cand if c['discovery']['bets'] >= 100]
    bet = {'cells': [{k: v for k, v in c.items() if k != 'cm'} for c in cand]}
    if elig:
        pick = max(elig, key=lambda c: c['discovery']['roi'])
        bet['chosen'] = {'cell': pick['cell'], 'side': pick['side']}
        for per, pm in (('discovery', disc), ('confirmation', conf)):
            sel = pm & pick['cm']
            s_ = pick['side']
            bet[per] = {f'{s_} chosen cell, opening @ -110': roi(f'{s_}_open', sel), f'{s_} chosen cell, closing @ -110': roi(f'{s_}_close', sel), f'{s_} chosen cell, best book': roi(f'{s_}_best', sel),
                        'over, all games (control)': roi('over_open', pm), 'under, all games (control)': roi('under_open', pm)}
    res['bet'] = bet
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    c8 = [c for c in cells if c['threshold'] == 8 and c['side'] == 'favourite']
    print(json.dumps({'games': res['games'], 'sigma': sigma, 'cell_counts_fav_b2b_ge8': {c['period']: c['games'] for c in c8 if c['b2b'] and c['spread_bucket'] == '>= 8'},
                      'step2': res['step2_reading'], 'step3': res['step3_reading'], 'layer_b': res['layer_b_supported']}, indent=1, default=float))


if __name__ == '__main__':
    main()
