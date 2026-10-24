"""E19 - late-season tanking and locked seeds (angle 5 of 8). Registered in reports/experiments/PREREGISTRATION.md ("E19") before any
number was computed. Regular-season games only. python -m experiments.e19_tanking"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import e14_blowout_under as e14                       # noqa: E402
from experiments.e14_blowout_under import DayBoot                       # noqa: E402
from experiments.totals_bets import settlement_profits                  # noqa: E402
from team.data_paths import latest_training_set                         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'nba_stats.db'
OUT = ROOT / 'reports' / 'experiments' / 'E19_tanking'
DISC, CONF = e14.DISCOVERY, e14.CONFIRM
SEASONS = DISC + CONF
EAST = {'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS'}
LATE_N, TANK_N, CLEAR = 70, 5, 6.0


def build_flags():
    with sqlite3.connect(f'file:{DB}?mode=ro', uri=True) as c:
        tg = pd.read_sql("SELECT seasonYear, teamId, teamTricode, gameId, gameDate, WL FROM team_game_full WHERE gameId LIKE '002%'", c)
    tg['gameId'] = tg['gameId'].astype(str).str.zfill(10)
    tg['date'] = pd.to_datetime(tg['gameDate'])
    latest = tg[tg['seasonYear'] == '2025-26'].drop_duplicates('teamId').set_index('teamId')['teamTricode']
    conf = {t: ('E' if latest.get(t) in EAST else 'W') for t in tg['teamId'].unique() if t in latest.index}
    rows = []
    for season, sg in tg[tg['seasonYear'].isin(SEASONS)].groupby('seasonYear'):
        teams = sorted(sg['teamId'].unique())
        dates = np.array(sorted(sg['date'].unique()))
        ti = {t: i for i, t in enumerate(teams)}
        di = {d: i for i, d in enumerate(dates)}
        W = np.zeros((len(dates), len(teams)))
        G = np.zeros((len(dates), len(teams)))
        for t, d, wl in zip(sg['teamId'], sg['date'], sg['WL']):
            G[di[d], ti[t]] += 1
            W[di[d], ti[t]] += wl == 'W'
        cw = np.vstack([np.zeros(len(teams)), np.cumsum(W, 0)[:-1]])     # record before each date
        cg = np.vstack([np.zeros(len(teams)), np.cumsum(G, 0)[:-1]])
        conf_idx = {k: [ti[t] for t in teams if conf.get(t) == k] for k in ('E', 'W')}
        for i, d in enumerate(dates):
            w, g = cw[i], cg[i]
            wp = np.where(g > 0, w / np.maximum(g, 1), 0.5)
            tank = rankdata(wp, method='min') <= TANK_N
            locked = np.zeros(len(teams), bool)
            for k, idx in conf_idx.items():
                idx = np.array(idx)
                order = idx[np.lexsort((-w[idx], -wp[idx]))]                  # win% desc, wins desc
                diff = (w[order] - (g[order] - w[order])) / 2.0                 # (W-L)/2
                for r, t in enumerate(order):
                    below = diff[r] - diff[r + 1] if r + 1 < len(order) else np.inf
                    above = diff[r - 1] - diff[r] if r > 0 else np.inf
                    locked[t] = (r < 10) and below > CLEAR and (r == 0 or above > CLEAR)
            for j, t in enumerate(teams):
                rows.append((season, t, d, cg[i, j] + 1, bool(tank[j]), bool(locked[j])))
    st = pd.DataFrame(rows, columns=['seasonYear', 'teamId', 'date', 'game_no', 'tank', 'locked'])
    gm = tg[tg['seasonYear'].isin(SEASONS)].merge(st, on=['seasonYear', 'teamId', 'date'], how='left')
    out = gm.groupby('gameId').agg(gn_max=('game_no', 'max'), gn_min=('game_no', 'min'), T=('tank', 'max'), K=('locked', 'max'), n=('teamId', 'size')).reset_index()
    out['L'] = out['gn_max'] >= LATE_N
    return out[out['n'] == 2]


def p_wald(boots, beta):
    S = np.cov(boots, rowvar=False)
    w = float(beta @ np.linalg.solve(S, beta))
    return w, float(chi2.sf(w, len(beta)))


def fit_flags(db, df, mask, y, flag_cols, inter_cols=None, season_dummies=True):
    """OLS y ~ 1 + a + total_c + flags + flags x a + season dummies. Returns coef table and joint Wald on the flag block."""
    a = df['abs_spread'].to_numpy(float)
    tc = df['T_c'].to_numpy(float)
    cols, names = [np.ones(len(df)), a, tc], ['intercept', 'abs_spread', 'total_c']
    for f in flag_cols:
        cols.append(df[f].to_numpy(float)), names.append(f)
    for f in flag_cols:
        cols.append(df[f].to_numpy(float) * a), names.append(f'{f}_x_abs_spread')
    ss = sorted(set(df.loc[mask, 'seasonYear']))
    if season_dummies:
        for s in ss[1:]:
            cols.append((df['seasonYear'] == s).to_numpy(float)), names.append(f'season_{s}')
    beta, lo, hi, boots = db.ols(np.column_stack(cols), df[y].to_numpy(float), mask)
    idx = [names.index(f) for f in flag_cols] + [names.index(f'{f}_x_abs_spread') for f in flag_cols]
    w, p = p_wald(boots[:, idx], beta[idx])
    tab = {}
    for i, n in enumerate(names):
        if n.startswith('season_') or n == 'intercept':
            continue
        b = boots[:, i]
        tab[n] = {'coef': float(beta[i]), 'lo': float(lo[i]), 'hi': float(hi[i]), 'p_two_sided': float(max(2 * min((b <= 0).sum(), (b >= 0).sum()), 1) / (len(b) + 1))}
    return {'coefs': tab, 'wald': w, 'wald_p': p, 'df': len(idx), 'n': int(np.asarray(mask).sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = e14.load_frame()
    scope = {}
    df_all = df[df['seasonYear'].isin(SEASONS)]
    scope['games_registered_seasons_all_types'] = int(len(df_all))
    scope['playoff_and_playin_games'] = int((~df_all['gameId'].str.startswith('002')).sum())
    db_all = DayBoot(df['gameDate'])
    S_all = df['seasonYear']
    rs_all = df['gameId'].str.startswith('002').to_numpy()
    m_reg = S_all.isin(SEASONS).to_numpy()
    scope['e14_slope_regular_season_only'] = {k: e14.slope(db_all, df, m & rs_all)[0] for k, m in (('discovery', S_all.isin(DISC).to_numpy()), ('confirmation', S_all.isin(CONF).to_numpy()), ('registered', m_reg))}
    scope['e14_slope_all_game_types'] = {k: e14.slope(db_all, df, m)[0] for k, m in (('discovery', S_all.isin(DISC).to_numpy()), ('confirmation', S_all.isin(CONF).to_numpy()), ('registered', m_reg))}

    df = df[df['seasonYear'].isin(SEASONS) & df['gameId'].str.startswith('002')].reset_index(drop=True)
    fl = build_flags()
    df = df.merge(fl[['gameId', 'gn_max', 'gn_min', 'T', 'K', 'L']], on='gameId', how='left')
    res = {'scope': scope, 'games_with_flags': int(df['L'].notna().sum()), 'games': int(len(df))}
    df = df.dropna(subset=['L']).reset_index(drop=True)
    for f in ('L', 'T', 'K'):
        df[f] = df[f].astype(bool)
    # sanity: own game number against the training set column
    ts = pd.read_csv(latest_training_set(), usecols=['gameId', 'season_game_number_home', 'season_game_number_away'], low_memory=False)
    ts['gameId'] = ts['gameId'].astype(str).str.zfill(10)
    chk = df.merge(ts, on='gameId', how='left')
    gm = np.maximum(chk['season_game_number_home'], chk['season_game_number_away'])
    res['game_number_check'] = {'share_equal_to_training_set_max': float((gm == chk['gn_max']).mean()), 'share_within_1': float((np.abs(gm - chk['gn_max']) <= 1).mean())}
    df['LT'], df['LK'] = df['L'] & df['T'], df['L'] & df['K']
    df['LO'] = df['L'] & ~df['T'] & ~df['K']
    S = df['seasonYear']
    disc, conf = S.isin(DISC).to_numpy(), S.isin(CONF).to_numpy()
    df['T_c'] = df['odds_total'] - df.loc[disc, 'odds_total'].mean()
    db = DayBoot(df['gameDate'])
    ones = np.ones(len(df))
    nonpush = (df['r'] != 0).to_numpy()
    under = (df['r'] < 0).astype(float).to_numpy()
    df['notL'] = ~df['L']
    for c in ('L', 'T', 'K', 'LT', 'LK', 'LO', 'notL'):
        df[c + '_f'] = df[c].astype(float)

    # ---- step 1: counts and descriptives
    groups = {'not late': 'notL', 'late, other (LO)': 'LO', 'late, tanker (LT)': 'LT', 'late, locked (LK)': 'LK'}
    s1 = []
    for per, pm in (('discovery', disc), ('confirmation', conf)):
        for lab, col in groups.items():
            m = pm & df[col].to_numpy()
            mr = db.ratio(df['r'], ones, m) if m.sum() else (np.nan,) * 3
            ur = db.ratio(under, nonpush.astype(float), m & nonpush) if m.sum() else (np.nan,) * 3
            s1.append({'period': per, 'group': lab, 'games': int(m.sum()), 'mean_r': mr[0], 'lo': mr[1], 'hi': mr[2], 'under_rate': ur[0], 'under_lo': ur[1], 'under_hi': ur[2]})
    cnt = {}
    for per, pm in (('discovery', disc), ('confirmation', conf)):
        cnt[per] = {k: int((pm & df[k].to_numpy()).sum()) for k in ('L', 'T', 'K', 'LT', 'LK', 'LO')}
        cnt[per]['games'] = int(pm.sum())
        cnt[per]['LT_and_LK'] = int((pm & df['LT'].to_numpy() & df['LK'].to_numpy()).sum())
    res['counts'], res['step1'] = cnt, s1
    res['counts_by_season'] = {s: {k: int((S.eq(s).to_numpy() & df[k].to_numpy()).sum()) for k in ('L', 'T', 'K', 'LT', 'LK')} | {'games': int(S.eq(s).sum())} for s in SEASONS}

    # ---- step 2: primary regression and model B
    A, Bm = ['L', 'T', 'K'], ['LT', 'LK', 'LO']
    for c in A + Bm:
        df[c + '_f'] = df[c].astype(float)
    flagA, flagB = [c + '_f' for c in A], [c + '_f' for c in Bm]
    s2 = {'A': {}, 'B': {}}
    for per, pm in (('discovery', disc), ('confirmation', conf)):
        s2['A'][per] = fit_flags(db, df, pm, 'r', flagA)
        s2['B'][per] = fit_flags(db, df, pm, 'r', flagB)
    res['step2'] = s2
    res['step2_reading'] = {'discovery_wald_p': s2['A']['discovery']['wald_p'], 'confirmation_wald_p': s2['A']['confirmation']['wald_p'],
                            'H_TANK_supported_raw': bool(s2['A']['discovery']['wald_p'] < 0.10 and s2['A']['confirmation']['wald_p'] < 0.05), 'holm': 'pending: family of 8 incomplete'}
    # sensitivity
    sens = {}
    for lab, m in (('2021-22..2025-26', S.isin(SEASONS[2:]).to_numpy()), ('2023-24..2025-26 (participation-policy era)', S.isin(SEASONS[4:]).to_numpy())):
        sens[lab] = fit_flags(db, df, m, 'r', flagA)
        sens[lab]['counts'] = {k: int((m & df[k].to_numpy()).sum()) for k in ('L', 'T', 'K', 'LT', 'LK')}
    res['sensitivity'] = sens

    # ---- step 3: line movement
    df['move'] = df['close_total'] - df['odds_total']
    cm = df['close_total'].notna().to_numpy()
    mv = []
    for per, pm in (('discovery', disc), ('confirmation', conf), ('registered', disc | conf)):
        base_m = pm & cm & df['notL'].to_numpy()
        for lab, col in groups.items():
            m = pm & cm & df[col].to_numpy()
            if not m.any():
                continue
            row = {'period': per, 'group': lab, 'games': int(m.sum())}
            for key, val in (('move', df['move'].fillna(0)), ('abs_move', df['move'].abs().fillna(0)), ('r_open', df['r']), ('r_close', df['r_close'].fillna(0))):
                row[key], row[key + '_lo'], row[key + '_hi'] = db.ratio(val, ones, m)
                if lab != 'not late':
                    d = e14_ratio_diff(db, val, ones, m, base_m)
                    row[key + '_vs_not_late'] = d
            mv.append(row)
    res['step3'] = mv

    # ---- step 4: model test
    oos = pd.concat([pd.read_csv(p, dtype={'game_id': str}, usecols=['game_id', 'total_pred']) for p in sorted((ROOT / 'reports').glob('oos_predictions_20??-??.csv'))])
    oos['gameId'] = oos['game_id'].str.zfill(10)
    j = df.merge(oos[['gameId', 'total_pred']].drop_duplicates('gameId'), on='gameId', how='inner')
    j['model_resid'] = j['target_total_pts'] - j['total_pred']
    j['model_lean'] = j['total_pred'] - j['odds_total']
    dbj = DayBoot(j['gameDate'])
    s4 = {'n': int(len(j)), 'models': {}, 'groups': []}
    for lab, mm in (('2021-22..2025-26', np.ones(len(j), bool)), ('confirmation 2022-23..2025-26', j['seasonYear'].isin(CONF).to_numpy())):
        s4['models'][lab] = {y: fit_flags(dbj, j, mm, y, flagA) for y in ('model_resid', 'model_lean', 'r')}
        for g_, col in groups.items():
            m = mm & j[col].to_numpy()
            if m.sum():
                s4['groups'].append({'period': lab, 'group': g_, 'games': int(m.sum()), **{k: dict(zip(('mean', 'lo', 'hi'), dbj.ratio(j[k], np.ones(len(j)), m))) for k in ('model_resid', 'r')}})
    res['step4'] = s4
    res['layer_b_supported'] = bool(s4['models']['confirmation 2022-23..2025-26']['model_resid']['wald_p'] < 0.05)
    res['layer_c_run'] = bool(res['step2_reading']['H_TANK_supported_raw'] and res['layer_b_supported'])

    # ---- EXPLORATORY (not registered): is the late-season level a step at game 70 or a gradual drift through the season?
    edges_gn = [0, 20, 40, 60, 70, 100]
    labels_gn = ['games 1-19', '20-39', '40-59', '60-69', '70+']
    df['gn_bin'] = pd.cut(df['gn_max'], edges_gn, labels=labels_gn, right=False)
    gnr = []
    for per, pm in (('discovery', disc), ('confirmation', conf), ('registered', disc | conf)):
        for lab in labels_gn:
            m = pm & (df['gn_bin'] == lab).to_numpy()
            mr = db.ratio(df['r'], ones, m)
            ur = db.ratio(under, nonpush.astype(float), m & nonpush)
            gnr.append({'period': per, 'bin': lab, 'games': int(m.sum()), 'mean_r': mr[0], 'lo': mr[1], 'hi': mr[2], 'under_rate': ur[0], 'under_lo': ur[1], 'under_hi': ur[2]})
    res['exploratory_game_number_bins'] = gnr
    # the late-season difference by season (registered seasons), late minus not late
    res['exploratory_late_minus_notlate_by_season'] = {s: e14_ratio_diff(db, df['r'], ones, S.eq(s).to_numpy() & df['L'].to_numpy(), S.eq(s).to_numpy() & df['notL'].to_numpy()) if (S.eq(s).to_numpy() & df['L'].to_numpy()).sum() > 20 else None for s in SEASONS}

    # ---- step 6: betting
    P, ok = settlement_profits(df)

    def roi(key, m):
        m = m & (ok['close'] if key.endswith('close') else ok['best'] if key.endswith('best') else True)
        pt, lo, hi = db.ratio(P[key], ones, m)
        return {'roi': pt * 100, 'lo': lo * 100, 'hi': hi * 100, 'bets': int(m.sum())}
    cand = []
    for lab, col in (('late, tanker (LT)', 'LT'), ('late, locked (LK)', 'LK'), ('late, any (L)', 'L')):
        for side in ('over', 'under'):
            gm_ = df[col].to_numpy()
            cand.append({'group': lab, 'side': side, 'col': col, 'discovery': roi(f'{side}_open', disc & gm_), 'confirmation': roi(f'{side}_open', conf & gm_)})
    elig = [c for c in cand if c['discovery']['bets'] >= 100]
    bet = {'cells': [{k: v for k, v in c.items() if k != 'col'} for c in cand]}
    if elig:
        pick = max(elig, key=lambda c: c['discovery']['roi'])
        bet['chosen'] = {'group': pick['group'], 'side': pick['side']}
        for per, pm in (('discovery', disc), ('confirmation', conf)):
            sel = pm & df[pick['col']].to_numpy()
            s_ = pick['side']
            bet[per] = {f'{s_} chosen group, opening @ -110': roi(f'{s_}_open', sel), f'{s_} chosen group, closing @ -110': roi(f'{s_}_close', sel), f'{s_} chosen group, best book': roi(f'{s_}_best', sel),
                        'over, all late games': roi('over_open', pm & df['L'].to_numpy()), 'under, all late games': roi('under_open', pm & df['L'].to_numpy())}
    res['bet'] = bet
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    print(json.dumps({'scope': scope, 'game_number_check': res['game_number_check'], 'counts': cnt, 'reading': res['step2_reading'], 'layer_b': res['layer_b_supported']}, indent=1, default=float))


def e14_ratio_diff(db, num, den, m1, m2):
    n1, d1 = db.day_sum(np.where(m1, num, 0.0)), db.day_sum(np.where(m1, den, 0.0))
    n2, d2 = db.day_sum(np.where(m2, num, 0.0)), db.day_sum(np.where(m2, den, 0.0))
    pt = n1.sum() / d1.sum() - n2.sum() / d2.sum()
    b = (db.W @ n1) / (db.W @ d1) - (db.W @ n2) / (db.W @ d2)
    return {'diff': float(pt), 'lo': float(np.quantile(b, .025)), 'hi': float(np.quantile(b, .975))}


if __name__ == '__main__':
    main()
