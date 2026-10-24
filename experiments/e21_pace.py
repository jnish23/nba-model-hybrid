"""E21 - pace mismatch (angle 6 of 8). Registered in reports/experiments/PREREGISTRATION.md ("E21") before any number was computed.
Regular-season games only. python -m experiments.e21_pace"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import e14_blowout_under as e14                       # noqa: E402
from experiments.e14_blowout_under import DayBoot                       # noqa: E402
from experiments.totals_bets import settlement_profits                  # noqa: E402
from team.data_paths import latest_training_set                         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'nba_stats.db'
OUT = ROOT / 'reports' / 'experiments' / 'E21_pace'
DISC, CONF = e14.DISCOVERY, e14.CONFIRM
QS = [0.5, 0.6, 0.7, 0.8, 0.9]


def team_paces():
    with sqlite3.connect(f'file:{DB}?mode=ro', uri=True) as c:
        tg = pd.read_sql("SELECT seasonYear, teamId, gameId, gameDate, minutes, fieldGoalsAttempted fga, freeThrowsAttempted fta, offensiveRebounds orb, turnovers tov, pace AS box_pace "
                         "FROM team_game_full WHERE gameId LIKE '002%'", c)
    tg['gameId'] = tg['gameId'].astype(str).str.zfill(10)
    tg['date'] = pd.to_datetime(tg['gameDate'])
    tg['pace48'] = (tg['fga'] + 0.44 * tg['fta'] - tg['orb'] + tg['tov']) * 240 / tg['minutes']
    tg = tg.sort_values(['seasonYear', 'teamId', 'date', 'gameId'])
    g = tg.groupby(['seasonYear', 'teamId'])['pace48']
    for w in (10, 20, 30):
        tg[f'roll{w}'] = g.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=10).mean())
    return tg


def build_frame():
    df = e14.load_frame()
    df = df[df['gameId'].str.startswith('002')].reset_index(drop=True)
    tg = team_paces()
    ids = pd.read_csv(latest_training_set(), usecols=['gameId', 'teamId_home', 'teamId_away'], low_memory=False)
    ids['gameId'] = ids['gameId'].astype(str).str.zfill(10)
    df = df.merge(ids.drop_duplicates('gameId'), on='gameId', how='left')
    key = tg[['gameId', 'teamId', 'pace48', 'roll10', 'roll20', 'roll30', 'box_pace']]
    for side in ('home', 'away'):
        k = key.rename(columns={c: f'{c}_{side}' for c in ('pace48', 'roll10', 'roll20', 'roll30', 'box_pace')})
        df = df.merge(k, left_on=['gameId', f'teamId_{side}'], right_on=['gameId', 'teamId'], how='left').drop(columns='teamId')
    df['actual_pace'] = (df['pace48_home'] + df['pace48_away']) / 2
    for w in (10, 20, 30):
        df[f'mean_pace{w}'] = (df[f'roll{w}_home'] + df[f'roll{w}_away']) / 2
        df[f'gap{w}'] = (df[f'roll{w}_home'] - df[f'roll{w}_away']).abs()
    df['mean_pace'], df['gap'] = df['mean_pace20'], df['gap20']
    return df


def pone_neg(b):
    return float((1 + (b >= 0).sum()) / (len(b) + 1))


def fit(db, df, mask, y, xcols, season_dummies=False, tcenter=None):
    cols, names = [np.ones(len(df))], ['intercept']
    for x in xcols:
        v = df[x].to_numpy(float) - (tcenter if x == 'odds_total' else 0.0)
        cols.append(v), names.append(x)
    if season_dummies:
        ss = sorted(set(df.loc[mask, 'seasonYear']))
        for s in ss[1:]:
            cols.append((df['seasonYear'] == s).to_numpy(float)), names.append(f'season_{s}')
    X = np.column_stack(cols)
    m = np.asarray(mask) & np.isfinite(X).all(1) & np.isfinite(df[y].to_numpy(float))
    beta, lo, hi, boots = db.ols(np.nan_to_num(X), np.nan_to_num(df[y].to_numpy(float)), m)
    out = {}
    for i, n in enumerate(names):
        if n.startswith('season_') or n == 'intercept':
            continue
        b = boots[:, i]
        out[n] = {'coef': float(beta[i]), 'lo': float(lo[i]), 'hi': float(hi[i]), 'se': float((hi[i] - lo[i]) / 3.92), 'p_one_sided_negative': pone_neg(b),
                  'p_two_sided': float(max(2 * min((b <= 0).sum(), (b >= 0).sum()), 1) / (len(b) + 1))}
    out['n'] = int(m.sum())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_frame()
    S = df['seasonYear']
    reg_seasons = DISC + CONF
    res = {'games_regular_season_all': int(len(df))}
    ok20 = df['gap20'].notna() & df['actual_pace'].notna()
    res['games_with_20g_pace'] = {s: int((ok20 & (S == s)).sum()) for s in sorted(S.unique())}
    res['formula_sanity'] = {'corr_game_pace_vs_box_pace': float(np.corrcoef(df.loc[ok20, 'actual_pace'], ((df.loc[ok20, 'box_pace_home'] + df.loc[ok20, 'box_pace_away']) / 2))[0, 1]),
                             'mean_gap_20': float(df.loc[ok20, 'gap20'].mean()), 'sd_gap_20': float(df.loc[ok20, 'gap20'].std())}
    db = DayBoot(df['gameDate'])
    disc, conf = S.isin(DISC).to_numpy(), S.isin(CONF).to_numpy()
    reg = S.isin(reg_seasons).to_numpy()
    supp = (S < '2019-20').to_numpy()
    allm = np.ones(len(df), bool)
    ot = df['ot'].fillna(False).to_numpy()
    df['T_c'] = df['odds_total'] - df.loc[disc, 'odds_total'].mean()
    tc = float(df.loc[disc, 'odds_total'].mean())

    # ---- step 1: mechanism
    s1 = {}
    periods = (('discovery', disc), ('confirmation', conf), ('registered pooled', reg), ('supplementary 2013-14..2018-19', supp), ('all 13 seasons', allm))
    for per, pm in periods:
        s1[per] = {'base': fit(db, df, pm, 'actual_pace', ['mean_pace', 'gap']), 'season_dummies': fit(db, df, pm, 'actual_pace', ['mean_pace', 'gap'], True),
                   'regulation_only': fit(db, df, pm & ~ot, 'actual_pace', ['mean_pace', 'gap'])}
    for w in (10, 30):
        s1['window_robustness'] = s1.get('window_robustness', {})
        for per, pm in (('discovery', disc), ('confirmation', conf)):
            s1['window_robustness'].setdefault(per, {})[f'window {w}'] = fit(db, df, pm, 'actual_pace', [f'mean_pace{w}', f'gap{w}'])
    res['step1'] = s1
    g_d, g_c = s1['discovery']['base']['gap'], s1['confirmation']['base']['gap']
    sup1 = bool(g_d['coef'] < 0 and g_c['coef'] < 0 and g_c['p_one_sided_negative'] < 0.05)
    res['step1_reading'] = {'supported': sup1, 'discovery_gap_coef': g_d['coef'], 'confirmation_gap_coef': g_c['coef'], 'confirmation_p_one_sided': g_c['p_one_sided_negative'],
                            'min_detectable_gap_coef_confirmation': 2.8 * g_c['se'], 'confirmation_mean_pace_coef': s1['confirmation']['base']['mean_pace']['coef']}

    # ---- step 2: market test (run regardless; gated status recorded)
    s2 = {}
    for per, pm in (('discovery', disc), ('confirmation', conf), ('registered pooled', reg), ('pooled 2021-22..2025-26 (no COVID seasons)', S.isin(reg_seasons[2:]).to_numpy())):
        s2[per] = {'base': fit(db, df, pm, 'r', ['gap', 'mean_pace', 'odds_total'], tcenter=tc), 'season_dummies': fit(db, df, pm, 'r', ['gap', 'mean_pace', 'odds_total'], True, tcenter=tc),
                   'regulation_only': fit(db, df, pm & ~ot, 'r', ['gap', 'mean_pace', 'odds_total'], tcenter=tc)}
    res['step2'] = s2
    m_d, m_c = s2['discovery']['base']['gap'], s2['confirmation']['base']['gap']
    sup2 = bool(m_d['coef'] < 0 and m_c['coef'] < 0 and m_c['p_one_sided_negative'] < 0.05)
    res['step2_reading'] = {'supported_raw': sup2, 'gated_on_step1': sup1, 'status': ('gated pass' if sup1 else 'exploratory and ungated'), 'confirmation_p_one_sided': m_c['p_one_sided_negative'],
                            'holm_input_p': (m_c['p_one_sided_negative'] if sup1 else 1.0), 'min_detectable_gap_coef_confirmation': 2.8 * m_c['se']}

    # ---- step 3: model test
    oos = pd.concat([pd.read_csv(p, dtype={'game_id': str}, usecols=['game_id', 'total_pred']) for p in sorted((ROOT / 'reports').glob('oos_predictions_20??-??.csv'))])
    oos['gameId'] = oos['game_id'].str.zfill(10)
    j = df.merge(oos[['gameId', 'total_pred']].drop_duplicates('gameId'), on='gameId', how='inner')
    j = j[j['gap'].notna() & j['mean_pace'].notna()].reset_index(drop=True)
    j['model_resid'] = j['target_total_pts'] - j['total_pred']
    j['model_lean'] = j['total_pred'] - j['odds_total']
    dbj = DayBoot(j['gameDate'])
    s3 = {'n': int(len(j)), 'models': {}}
    for lab, mm in (('2021-22..2025-26', np.ones(len(j), bool)), ('confirmation 2022-23..2025-26', j['seasonYear'].isin(CONF).to_numpy())):
        s3['models'][lab] = {y: fit(dbj, j, mm, y, ['gap', 'mean_pace', 'odds_total'], tcenter=tc) for y in ('model_resid', 'model_lean', 'r')}
    edges = np.quantile(j['gap'], np.linspace(0, 1, 6))
    edges[0], edges[-1] = -np.inf, np.inf
    j['gq'] = np.digitize(j['gap'], edges[1:-1])
    qrows = []
    for q in range(5):
        m = (j['gq'] == q).to_numpy()
        row = {'quintile': q + 1, 'games': int(m.sum()), 'gap_mean': float(j.loc[m, 'gap'].mean())}
        for k in ('model_resid', 'r', 'model_lean'):
            row[k], row[k + '_lo'], row[k + '_hi'] = dbj.ratio(j[k], np.ones(len(j)), m)
        qrows.append(row)
    s3['quintiles'] = qrows
    res['step3'] = s3
    mdl = s3['models']['confirmation 2022-23..2025-26']['model_resid']['gap']
    res['step3_reading'] = {'supported': bool(mdl['coef'] < 0 and mdl['p_one_sided_negative'] < 0.05), 'confirmation_coef': mdl['coef'], 'p_one_sided': mdl['p_one_sided_negative']}
    res['layer_c_gate_passed'] = bool(sup1 and sup2 and res['step3_reading']['supported'])

    # ---- step 5: betting
    d2 = df[df['gap'].notna()].reset_index(drop=True)
    db2 = DayBoot(d2['gameDate'])
    S2 = d2['seasonYear']
    dm2, cm2 = S2.isin(DISC).to_numpy(), S2.isin(CONF).to_numpy()
    ones = np.ones(len(d2))
    P, okm = settlement_profits(d2)

    def roi(key, m):
        m = m & (okm['close'] if key.endswith('close') else okm['best'] if key.endswith('best') else True)
        pt, lo, hi = db2.ratio(P[key], ones, m)
        return {'roi': pt * 100, 'lo': lo * 100, 'hi': hi * 100, 'bets': int(m.sum())}
    gap = d2['gap'].to_numpy()
    cuts = {q: float(np.quantile(gap[dm2], q)) for q in QS}
    grid = []
    for q in QS:
        row = {'q': q, 'cutoff_gap': cuts[q]}
        for per, pm in (('discovery', dm2), ('confirmation', cm2)):
            row[per] = roi('under_open', pm & (gap >= cuts[q]))
            row[per + '_over_mirror'] = roi('over_open', pm & (gap >= cuts[q]))
        grid.append(row)
    elig = [r for r in grid if r['discovery']['bets'] >= 200]
    bet = {'grid': grid}
    if elig:
        qsel = max(elig, key=lambda r: r['discovery']['roi'])['q']
        bet['q_chosen'], bet['cutoff'] = qsel, cuts[qsel]
        for per, pm in (('discovery', dm2), ('confirmation', cm2)):
            sel = pm & (gap >= cuts[qsel])
            bet[per] = {'under opening @ -110': roi('under_open', sel), 'under closing @ -110': roi('under_close', sel), 'under best book': roi('under_best', sel),
                        'under all games (control)': roi('under_open', pm), 'over, same games (exploratory mirror)': roi('over_open', sel)}
    res['bet'] = bet

    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    print(json.dumps({'counts': res['games_with_20g_pace'], 'sanity': res['formula_sanity'], 'step1': res['step1_reading'], 'step2': res['step2_reading'], 'step3': res['step3_reading'],
                      'gate': res['layer_c_gate_passed']}, indent=1, default=float))


if __name__ == '__main__':
    main()
