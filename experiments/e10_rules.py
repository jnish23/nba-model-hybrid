"""E10 - bet-selection rule (see PREREGISTRATION.md), on the baseline predictions. python -m experiments.e10_rules"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import metrics as M                                                 # noqa: E402
from experiments.lib import OUT, TEST_SEASONS, label_calib_folds, load_games, load_run   # noqa: E402

NAME = 'E10_selection_rule'
GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10]
MIN_CALIB_BETS = 100


def cross_fitted_ml(calib):
    """Out-of-sample calibrated ML probabilities on the calibration set (isotonic fitted on the other time half)."""
    c = calib.sort_values('gameDate').copy()
    h = len(c) // 2
    p = np.empty(len(c))
    a, b = c.iloc[:h], c.iloc[h:]
    p[:h] = IsotonicRegression(out_of_bounds='clip').fit(b['ml_prob_raw'], b['target_home_win']).predict(a['ml_prob_raw'])
    p[h:] = IsotonicRegression(out_of_bounds='clip').fit(a['ml_prob_raw'], a['target_home_win']).predict(b['ml_prob_raw'])
    c['ml_prob'] = p
    return c


def roi(bets):
    return bets['profit'].sum() / (100 * len(bets)) * 100 if len(bets) else np.nan


def pick_threshold(calib, market):
    c = cross_fitted_ml(calib) if market == 'ml' else calib
    best, best_roi, table = 0.05, -np.inf, {}
    for t in GRID:
        b = M.make_bets(c, thr=t, markets=(market,))
        table[t] = (len(b), roi(b))
        if len(b) >= MIN_CALIB_BETS and roi(b) > best_roi:
            best, best_roi = t, roi(b)
    return best, table


def predicted_win(bets, df):
    """Average model probability that the selected side wins (for the calibration-of-selected-bets check)."""
    d = df.set_index('gameId')
    ps = []
    for r in bets.itertuples():
        row = d.loc[r.gameId]
        if r.market == 'ml':
            ps.append(row['ml_prob'] if r.side == 'home' else 1 - row['ml_prob'])
        else:
            sig = row['sigma_spread'] if r.market == 'spread' else row['sigma_total']
            pe = abs(row['spread_pred'] + row['odds_home_spread']) if r.market == 'spread' else abs(row['total_pred'] - row['odds_total'])
            ps.append(norm.cdf(pe / sig))
    return float(np.mean(ps)) if ps else np.nan


def main():
    base, cal = load_run('baseline')
    _, df = load_games()
    cal['fold'] = label_calib_folds(df, cal)
    thr_by = {}
    rules = {}
    # baseline rule (b): 5% edge in every market
    rules['(b) 5% edge [baseline]'] = M.make_bets(base, thr=0.05)
    # (a) flat 1.0 point for spread/total, ML keeps the 5% edge rule
    a_sp_tot = M.make_bets(base, rule='flat', flat_points=1.0, markets=('spread', 'total'))
    rules['(a) flat 1.0 pt'] = pd.concat([M.make_bets(base, thr=0.05, markets=('ml',)), a_sp_tot], ignore_index=True)
    # (c) per-fold threshold chosen on the calibration-set predictions
    parts = []
    for s in TEST_SEASONS:
        te, ca = base[base['seasonYear'] == s], cal[cal['fold'] == s]
        thr_by[s] = {}
        for mk in ('ml', 'spread', 'total'):
            t, tab = pick_threshold(ca, mk)
            thr_by[s][mk] = t
            parts.append(M.make_bets(te, thr=t, markets=(mk,)))
    rules['(c) walk-forward threshold'] = pd.concat(parts, ignore_index=True)
    lines = [f'# {NAME}\n', '## Thresholds chosen per fold for rule (c) (from calibration-set predictions only)\n',
             pd.DataFrame(thr_by).T.to_markdown(), '']
    decisions = {}
    for mk in ('ml', 'spread', 'total'):
        rows = []
        b0 = rules['(b) 5% edge [baseline]']
        b0 = b0[b0['market'] == mk]
        for rname, bets in rules.items():
            b = bets[bets['market'] == mk]
            bm = M.bet_metrics(b)
            pw = predicted_win(b, base)
            row = {'rule': rname, 'bets': bm['bets'], 'hit rate %': round(bm['hit_rate'], 2), 'hit 95% CI': f"[{bm['hit_ci'][0]:.1f}, {bm['hit_ci'][1]:.1f}]",
                   'mean model P(win)': round(pw * 100, 2), 'ROI %': round(bm['roi'], 2), 'ROI 95% CI': f"[{bm['roi_ci'][0]:.1f}, {bm['roi_ci'][1]:.1f}]",
                   'max DD (units)': round(bm['max_drawdown'], 1), 'DD 95% CI': f"[{bm['dd_ci'][0]:.0f}, {bm['dd_ci'][1]:.0f}]"}
            clv = M.clv_points(b)
            row['CLV pts'] = round(float(clv.mean()), 3) if len(clv) else None
            if rname != '(b) 5% edge [baseline]':
                d = M.roi_diff_ci(b, b0)
                row['ROI diff vs (b)'] = round(d['roi_diff'], 2)
                row['diff 95% CI'] = f"[{d['ci'][0]:.1f}, {d['ci'][1]:.1f}]"
                ok = d['roi_diff'] >= 1.0 and d['ci'][0] > 0 and abs(bm['hit_rate'] - pw * 100) <= 3.0
                row['KEEP'] = bool(ok)
                decisions[f'{mk}: {rname}'] = bool(ok)
            rows.append(row)
        lines += [f'## {mk.upper()}\n', pd.DataFrame(rows).to_markdown(index=False), '']
    # per-season bet counts / ROI for each rule
    for rname, bets in rules.items():
        g = bets.groupby(['seasonYear', 'market'])['profit'].agg(n='count', roi=lambda x: x.sum() / (100 * len(x)) * 100).unstack()
        g.columns = [f'{mk} {"bets" if stat == "n" else "ROI%"}' for stat, mk in g.columns]
        t = g[sorted(g.columns, key=lambda c: (['ml', 'spread', 'total'].index(c.split()[0]), c))]
        lines += [f'### per season: {rname}', t.round(1).to_markdown(), '']
    lines += ['## Decision (pre-registered: ROI diff >= +1.0 pt, day-bootstrap CI excluding 0, hit rate within 3 pp of mean model P(win))\n'] + \
        [f'- {k}: **{"KEEP" if v else "REJECT"}**' for k, v in decisions.items()]
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    (OUT / NAME / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'decision.json').write_text(json.dumps({'decision': decisions, 'thresholds': thr_by}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
