"""E28 exploratory follow-ups. NOT registered: written after the registered screen was read, to check two confounds the screen
exposed (the per-possession load features and the pace outcome both contain the previous game's pace) and whether the
opening-line candidates survive against the closing line. Nothing here changes a tier. python -m experiments.e28_explore"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import e28_screen as sc                     # noqa: E402
from experiments import features_tracking as ft             # noqa: E402

ROOT = sc.ROOT
OUT = sc.OUT


def pace_controls():
    with sqlite3.connect(f'file:{ROOT / "data" / "nba_stats.db"}?mode=ro', uri=True) as c:
        t = pd.read_sql('SELECT teamId, gameId, gameDate, pace FROM team_game_full', c)
    t['gameId'] = t['gameId'].astype(str).str.zfill(10)
    t['gameDate'] = pd.to_datetime(t['gameDate'])
    t = t.sort_values(['teamId', 'gameDate', 'gameId'], kind='stable').reset_index(drop=True)
    g = t.groupby('teamId', sort=False)['pace']
    t['prev_pace'] = g.shift(1)
    norm = g.transform(lambda s: s.ewm(span=ft.SPAN).mean().shift(1))
    t['pace_norm'] = norm.where(g.cumcount() >= ft.MIN_NORM)
    return t[['gameId', 'teamId', 'prev_pace', 'pace_norm']]


def add_game_controls(f, pc):
    f = f.copy()
    for side in ('home', 'away'):
        s = pc.rename(columns={'teamId': f'teamId_{side}', 'prev_pace': f'pp_{side}', 'pace_norm': f'pn_{side}'})
        f = f.merge(s, on=['gameId', f'teamId_{side}'], how='left')
    f['c_sum_prev_pace'] = f['pp_home'] + f['pp_away']
    f['c_sum_pace_norm'] = f['pn_home'] + f['pn_away']
    f['c_diff_prev_pace'] = f['pp_home'] - f['pp_away']
    f['c_diff_pace_norm'] = f['pn_home'] - f['pn_away']
    return f


def mech_controls(r, pc):
    r = r.merge(pc.rename(columns={'prev_pace': 'c_own_prev_pace', 'pace_norm': 'c_own_pace_norm'}), on=['gameId', 'teamId'], how='left')
    opp = r[['gameId', 'teamId', 'c_own_prev_pace', 'c_own_pace_norm']].rename(columns={'teamId': 'oppTeam', 'c_own_prev_pace': 'c_opp_prev_pace', 'c_own_pace_norm': 'c_opp_pace_norm'})
    r['_k'] = r.groupby('gameId').cumcount()
    o = opp.copy()
    o['_k'] = o.groupby('gameId').cumcount()
    o['_k'] = 1 - o['_k']
    r = r.merge(o.drop(columns='oppTeam'), on=['gameId', '_k'], how='left').drop(columns='_k')
    return r


def row(res):
    return {'n find': res['n_find'], 'find Wald p': round(res['p_find'], 4), 'gamma': round(res['gamma'], 3), 'gamma 95% CI': f"[{res['gamma_lo']:.2f}, {res['gamma_hi']:.2f}]", 'one-sided p': round(res['p_one'], 4)}


def main():
    v = sc.feature_frame()
    pc = pace_controls()
    fa, fb = sc.frame_a(v), sc.frame_b(v)
    fa, fb = add_game_controls(fa, pc), add_game_controls(fb, pc)
    reg = json.loads((OUT / 'screen.json').read_text())
    L = ['# E28 exploratory follow-ups (NOT registered)', '',
         'Written after the registered screen (`screen.md`) was read. They check two confounds the screen exposed and one practical question; they change no tier. '
         'Every number here is a forking-path number and would need its own registration before it could support a decision.', '']
    # X0: the code path with controls=() reproduces the registered numbers
    r0 = sc.run_test(fa, 'F1', 'total', sc.FIND_A, sc.CONF)
    old = reg['results']['a']['a|F1|total']
    L += ['## X0. Reproducibility of the registered numbers', '', f"a|F1|total now: Wald p {r0['p_find']:.6f}, gamma {r0['gamma']:.6f}; registered run: {old['p_find']:.6f}, {old['gamma']:.6f}. "
          f"{'Identical.' if abs(r0['p_find'] - old['p_find']) < 1e-9 and abs(r0['gamma'] - old['gamma']) < 1e-9 else 'DIFFERENT - investigate.'}", '']

    # X1: pace controls on the totals candidates and the margin
    rows = []
    for fam in ('F1', 'F3'):
        for tg, cc in (('total', ['c_sum_prev_pace', 'c_sum_pace_norm']), ('margin', ['c_diff_prev_pace', 'c_diff_pace_norm'])):
            for lab, ctrl in (('registered (no controls)', ()), ('+ previous-game pace and team pace norm', cc)):
                r = sc.run_test(fa, fam, tg, sc.FIND_A, sc.CONF, controls=ctrl, conf_extra=False)
                rows.append({'family': fam, 'target': tg, 'spec': lab, **row(r)})
    L += ['## X1. (a) model-residual tests with the pace of the previous game and the team pace norm as controls', '',
          'F1 divides distance by possessions, so a slow previous game raises `xload`, `dist_per_poss_last` and `rot_xload`; teams repeat their pace, so the total (and the pace outcome) moves with the previous '
          'game\'s pace whatever the legs did. F3 carries the previous game\'s pace directly. If the family only matters through pace, the controls remove it.', '', sc.md(pd.DataFrame(rows))]

    # X2: mechanism with pace controls
    mech = mech_controls(sc.mech_frame(v), pc)
    ctrl = ['c_own_prev_pace', 'c_opp_prev_pace', 'c_own_pace_norm', 'c_opp_pace_norm']
    rows = []
    for fam in ('F1', 'F2'):
        for lab, col in (('3P%', 'mech_3p_pct'), ('FT%', 'mech_ft_pct'), ('pace', 'mech_pace')):
            for spec, cc in (('registered', ()), ('+ pace controls', ctrl)):
                r = sc.run_mech(mech, fam, col, controls=cc)
                rows.append({'family': fam, 'outcome': lab, 'spec': spec, 'n find': r['n_find'], 'find Wald p': round(r['p_find'], 4), 'gamma': round(r['gamma'], 3),
                             'gamma 95% CI': f"[{r['gamma_lo']:.2f}, {r['gamma_hi']:.2f}]", 'one-sided p': round(r['p_one'], 4),
                             'rule (find p<0.10, gamma>0, p<0.0167)': bool(r['p_find'] < sc.FIND_P and r['gamma'] > 0 and r['p_one'] < sc.MECH_P)})
    L += ['## X2. Mechanism (c) for F1 and F2 with pace controls', '', 'Controls: own and opponent previous-game pace and prior EWMA pace.', '', sc.md(pd.DataFrame(rows))]

    # X3: (b) candidates against the closing line
    op = pd.read_csv(ROOT / 'reports' / 'experiments' / 'E25_with_odds' / 'predictions.csv', dtype={'gameId': str}, usecols=['gameId', 'close_total', 'close_home_spread'])
    fb2 = fb.merge(op, on='gameId', how='left')
    fb2['y_total_close'] = fb2['target_total_pts'] - fb2['close_total']
    fb2['y_margin_close'] = fb2['target_point_diff'] + fb2['close_home_spread']
    rows = []
    for fam, tg in (('F1', 'margin'), ('F3', 'margin'), ('F4', 'margin'), ('F5', 'total'), ('F4', 'total'), ('F1', 'total')):
        r = sc.run_test(fb2, fam, tg, sc.FIND_B, sc.CONF, conf_extra=False, alt_conf_y={'closing line': f'y_{tg}_close'})
        a = r['alt']['closing line']
        rows.append({'family': fam, 'target': tg, 'opening gamma': round(r['gamma'], 3), 'opening 95% CI': f"[{r['gamma_lo']:.2f}, {r['gamma_hi']:.2f}]",
                     'n (closing)': a['n'], 'closing gamma': round(a['gamma'], 3), 'closing 95% CI': f"[{a['lo']:.2f}, {a['hi']:.2f}]", 'closing one-sided p': round(a['p_one'], 4)})
    L += ['## X3. The (b) signals applied to the closing line (confirmation seasons)', '',
          'Same find-period signal (built from the opening-line residual), confirmation residual taken against the CLOSING line instead. '
          'A gamma near 0 against the close means the market had absorbed the information by the close; a bet on the opening number would then show positive closing-line value.', '', sc.md(pd.DataFrame(rows))]
    (OUT / 'explore.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
