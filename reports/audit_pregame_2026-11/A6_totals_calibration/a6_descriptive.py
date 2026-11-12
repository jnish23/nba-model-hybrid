"""A6 descriptive questions (no decision): why is the tree model of the total worse than its own input line, and does a constant shift beat both?
Uses the same saved runs as E27 (E13 test predictions + baseline calibration blocks). Run from the worktree root:
  C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe reports/audit_pregame_2026-11/A6_totals_calibration/a6_descriptive.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.e27_candidate_form import BUCKETS, BUCKET_NAMES, Scorer, read_run, run_variants    # noqa: E402
from experiments.lib import TEST_SEASONS, label_calib_folds, load_games                              # noqa: E402

OUT = Path(__file__).resolve().parent


def main():
    m, games = load_games()
    base, calib = read_run('baseline')
    calib['fold'] = label_calib_folds(games, calib)
    e13, _ = read_run('E13_combined_winners', calib_name='baseline')
    fe, ve = run_variants(e13.reset_index(drop=True), calib, games)
    sc = Scorer(fe, 42)
    raw, c, line, act, S = fe['total_pred'].to_numpy(float), ve['c']['pred'], sc.line, sc.act, sc.S
    N = len(fe)
    res = {}

    # --- constant-shift forecasts, fitted fold by fold on the calibration block only
    l1, l2, l3 = (np.full(N, np.nan) for _ in range(3))
    consts = {}
    for s in TEST_SEASONS:
        cal = calib[calib['fold'] == s]
        ok = cal['odds_total'].notna() & cal['total_pred'].notna() & cal['target_total_pts'].notna()
        cal = cal[ok]
        d_line = float((cal['target_total_pts'] - cal['odds_total']).mean())
        d_pred = float((cal['target_total_pts'] - cal['total_pred']).mean())
        consts[s] = {'mean_actual_minus_line': d_line, 'mean_actual_minus_pred': d_pred, 'n_cal': int(len(cal))}
        mk = S == s
        l1[mk] = line[mk] + d_line          # line + mean (actual - line)
        l2[mk] = line[mk] + d_pred          # line + mean (actual - raw forecast)
        l3[mk] = raw[mk] + d_pred           # raw forecast + mean (actual - raw forecast): bias-only fix of the model
    res['constants_by_fold'] = consts
    cands = {'raw': raw, 'fix_c': c, 'line': line, 'line_plus_mean_actual_minus_line (L1)': l1, 'line_plus_mean_actual_minus_pred (L2)': l2, 'raw_plus_mean_actual_minus_pred (L3)': l3}
    res['levels'] = {k: sc.level(v)['pooled'] for k, v in cands.items()}
    res['mae_by_season'] = {k: {s: sc.level(v)[s]['mae'] for s in TEST_SEASONS} for k, v in cands.items()}
    pr = {}
    for k in ('line_plus_mean_actual_minus_line (L1)', 'line_plus_mean_actual_minus_pred (L2)', 'raw_plus_mean_actual_minus_pred (L3)'):
        for ref in ('raw', 'fix_c', 'line'):
            p = sc.paired(cands[k], cands[ref])
            pr[f'{k} minus {ref}'] = {'diff': p['diff'], 'lo': p['lo'], 'hi': p['hi'], 'seasons_first_better': p['seasons_a_better']}
    res['paired_mae'] = pr

    # --- where is the excess error
    gap = raw - line
    y = act - line
    ex_mae = np.abs(act - raw) - np.abs(act - line)
    ex_mse = (act - raw) ** 2 - (act - line) ** 2
    rows = []
    for name, (lo, hi) in zip(BUCKET_NAMES, BUCKETS):
        mk = (line >= lo) & (line < hi)
        rows.append({'bucket': name, 'n': int(mk.sum()), 'share_games': float(mk.mean()), 'mae_raw': float(np.abs(act - raw)[mk].mean()), 'mae_line': float(np.abs(act - line)[mk].mean()),
                     'mae_fix_c': float(np.abs(act - c)[mk].mean()),
                     'excess_mae_per_game': float(ex_mae[mk].mean()), 'contribution_to_pooled_excess_mae': float(ex_mae[mk].sum() / N),
                     'excess_mse_per_game': float(ex_mse[mk].mean()), 'contribution_to_pooled_excess_mse': float(ex_mse[mk].sum() / N),
                     'mean_gap_pred_minus_line': float(gap[mk].mean()), 'mean_actual_minus_line': float(y[mk].mean()),
                     'mean_E_gap2': float((gap[mk] ** 2).mean()), 'mean_2_gap_times_y': float((2 * gap[mk] * y[mk]).mean())})
    res['by_line_bucket'] = rows
    res['pooled_excess'] = {'mae': float(ex_mae.mean()), 'mse': float(ex_mse.mean())}
    ext = (line < 215) | (line >= 242)
    res['extremes_lt215_or_ge242'] = {'share_games': float(ext.mean()), 'contribution_to_pooled_excess_mae': float(ex_mae[ext].sum() / N), 'contribution_to_pooled_excess_mse': float(ex_mse[ext].sum() / N)}
    top = line >= 242
    res['above_242_only'] = {'share_games': float(top.mean()), 'contribution_to_pooled_excess_mae': float(ex_mae[top].sum() / N), 'contribution_to_pooled_excess_mse': float(ex_mse[top].sum() / N)}
    # deciles
    dec = pd.qcut(pd.Series(line), 10, labels=False, duplicates='drop').to_numpy()
    res['by_line_decile'] = [{'decile': int(d + 1), 'line_range': [float(line[dec == d].min()), float(line[dec == d].max())], 'n': int((dec == d).sum()),
                              'excess_mae_per_game': float(ex_mae[dec == d].mean()), 'mean_gap': float(gap[dec == d].mean()), 'mean_actual_minus_line': float(y[dec == d].mean()),
                              'mean_actual_minus_raw': float((act - raw)[dec == d].mean())} for d in range(10)]

    # --- does the model follow the line smoothly?
    def ols(xv, yv):
        X = np.column_stack([np.ones(len(xv)), xv])
        b = np.linalg.lstsq(X, yv, rcond=None)[0]
        return float(b[1]), float(b[0])
    res['follow_the_line'] = {'slope_pred_on_line': ols(line, raw)[0], 'slope_actual_on_line': ols(line, act)[0], 'sd_line': float(line.std()), 'sd_pred': float(raw.std()), 'sd_actual': float(act.std()),
                              'sd_gap_pred_minus_line': float(gap.std()), 'corr_gap_with_actual_minus_line': float(np.corrcoef(gap, y)[0, 1]),
                              'slope_actual_minus_line_on_gap': ols(gap, y)[0], 'corr_gap_with_line': float(np.corrcoef(gap, line)[0, 1]),
                              'slope_gap_on_line': ols(line, gap)[0], 'mean_E_gap2': float((gap ** 2).mean()), 'mean_2_gap_times_y': float((2 * gap * y).mean())}
    # share of the model's departures from the line explained by the line level itself (the part a smooth function of the line would remove)
    X = np.column_stack([np.ones(N), line, line ** 2])
    b = np.linalg.lstsq(X, gap, rcond=None)[0]
    res['follow_the_line']['share_of_gap_variance_explained_by_quadratic_in_line'] = float(1 - ((gap - X @ b) ** 2).sum() / ((gap - gap.mean()) ** 2).sum())
    hi = line >= 235
    res['lines_ge_235_contribution'] = {'share_games': float(hi.mean()), 'excess_mae': float(ex_mae[hi].sum() / N), 'excess_mse': float(ex_mse[hi].sum() / N)}
    res['high_line_by_season'] = {s: {'share_ge_235': float(hi[S == s].mean()), 'mean_line': float(line[S == s].mean()), 'mean_gap_when_ge_235': float(gap[(S == s) & hi].mean()) if ((S == s) & hi).any() else None,
                                      'mean_actual_minus_line_when_ge_235': float(y[(S == s) & hi].mean()) if ((S == s) & hi).any() else None} for s in TEST_SEASONS}
    (OUT / 'descriptive.json').write_text(json.dumps(res, indent=1, default=float))

    # markdown
    L = ['# A6 descriptive: why a tree model of the total trails its own input line', '', 'E13 test frame (8,899 games, 7 seasons); excess = raw model minus opening line; positive means the raw model is worse.', '',
         '## Pooled MAE and paired differences vs the raw model, fix (c) and the line', '', '| forecast | MAE | RMSE | bias |', '|---|---|---|---|']
    for k, v in res['levels'].items():
        L.append(f"| {k} | {v['mae']:.3f} | {v['rmse']:.3f} | {v['bias']:+.3f} |")
    L += ['', '| comparison (first minus second, MAE) | diff [95% CI] | seasons first better |', '|---|---|---|']
    for k, v in pr.items():
        L.append(f"| {k} | {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] | {v['seasons_first_better']}/7 |")
    L += ['', 'Per-season MAE: ' + '; '.join(f"{k}: " + ', '.join(f'{x:.2f}' for x in v.values()) for k, v in res['mae_by_season'].items()), '',
          '## Excess error by opening-total bucket', '', '| bucket | games | MAE raw | MAE line | MAE fix (c) | excess MAE per game | contribution to pooled excess MAE | excess MSE contribution | mean gap (pred - line) | mean actual - line |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['bucket']} | {r['n']} | {r['mae_raw']:.2f} | {r['mae_line']:.2f} | {r['mae_fix_c']:.2f} | {r['excess_mae_per_game']:+.2f} | {r['contribution_to_pooled_excess_mae']:+.3f} | {r['contribution_to_pooled_excess_mse']:+.2f} | {r['mean_gap_pred_minus_line']:+.2f} | {r['mean_actual_minus_line']:+.2f} |")
    pe = res['pooled_excess']
    L += ['', f"Pooled excess: MAE {pe['mae']:+.3f}, MSE {pe['mse']:+.2f}. Lines below 215 or at/above 242 (extremes) are {res['extremes_lt215_or_ge242']['share_games']:.1%} of games and carry "
          f"{res['extremes_lt215_or_ge242']['contribution_to_pooled_excess_mae']:+.3f} of the MAE excess ({res['extremes_lt215_or_ge242']['contribution_to_pooled_excess_mae'] / pe['mae']:.0%}) and "
          f"{res['extremes_lt215_or_ge242']['contribution_to_pooled_excess_mse']:+.2f} of the MSE excess ({res['extremes_lt215_or_ge242']['contribution_to_pooled_excess_mse'] / pe['mse']:.0%}). "
          f"Lines at/above 242 alone: {res['above_242_only']['share_games']:.1%} of games, {res['above_242_only']['contribution_to_pooled_excess_mse'] / pe['mse']:.0%} of the MSE excess.", '',
          f"Lines at/above 235: {res['lines_ge_235_contribution']['share_games']:.1%} of games, {res['lines_ge_235_contribution']['excess_mae'] / pe['mae']:.0%} of the MAE excess and {res['lines_ge_235_contribution']['excess_mse'] / pe['mse']:.0%} of the MSE excess.", '',
          'By season, games with a line at/above 235: ' + '; '.join(f"{s}: share {v['share_ge_235']:.1%}, mean gap {v['mean_gap_when_ge_235']:+.1f}, mean actual-line {v['mean_actual_minus_line_when_ge_235']:+.1f}" for s, v in res['high_line_by_season'].items()), '',
          '## Decile view (line deciles)', '', '| decile | line range | n | excess MAE/game | mean gap | mean actual - line | mean actual - raw |', '|---|---|---|---|---|---|---|']
    for r in res['by_line_decile']:
        L.append(f"| {r['decile']} | {r['line_range'][0]:.1f}-{r['line_range'][1]:.1f} | {r['n']} | {r['excess_mae_per_game']:+.2f} | {r['mean_gap']:+.2f} | {r['mean_actual_minus_line']:+.2f} | {r['mean_actual_minus_raw']:+.2f} |")
    f = res['follow_the_line']
    L += ['', '## Does the forecast follow the line?', '',
          f"- slope of the raw forecast on the line {f['slope_pred_on_line']:.3f} versus slope of the actual total on the line {f['slope_actual_on_line']:.3f}; sd line {f['sd_line']:.2f}, sd forecast {f['sd_pred']:.2f}, sd actual {f['sd_actual']:.2f}",
          f"- the forecast's gap from the line: sd {f['sd_gap_pred_minus_line']:.2f}; correlation with (actual - line) {f['corr_gap_with_actual_minus_line']:+.3f}; slope of (actual - line) on the gap {f['slope_actual_minus_line_on_gap']:+.3f}; correlation of the gap with the line level {f['corr_gap_with_line']:+.3f}, slope of the gap on the line {f['slope_gap_on_line']:+.3f}",
          f"- raw MSE minus line MSE = E[gap^2] - 2 E[gap x (actual - line)] = {f['mean_E_gap2']:.2f} - {f['mean_2_gap_times_y']:.2f} = {f['mean_E_gap2'] - f['mean_2_gap_times_y']:+.2f}",
          f"- a quadratic in the line level explains {f['share_of_gap_variance_explained_by_quadratic_in_line']:.1%} of the variance of the gap"]
    (OUT / 'descriptive.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
