"""
Phase 4 evaluation of one season's saved predictions against outcomes, with the plan's metrics, baselines and cluster-bootstrap intervals.

    python scripts/props_phase4_eval.py --version v2_val_2024-25 --season 2024-25 --out reports/props/phase4_eval_2024-25.md

Writes a markdown report and checks the pre-registered go/no-go criteria (reports/props/phase4_plan.md section 2.7) that can be evaluated from one run.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.models import data, evaluate as ev, outcome as oc, recalibrate  # noqa: E402
from props.models.pipeline import observed  # noqa: E402
from props.quality import md_table  # noqa: E402

CORE = ['points', 'rebounds', 'assists', 'threes', 'pra']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--season', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--features-db', default=None)
    ap.add_argument('--n-boot', type=int, default=ev.N_BOOT)
    ap.add_argument('--r2b', action='store_true', help='apply the frozen rolling recalibration R2b to the saved pmfs before evaluating')
    ap.add_argument('--r2b-carry-version', default=None, help='version whose previous-season fold supplies the carry-over rows')
    ap.add_argument('--r2b-carry-season', default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    frame = data.load_frame(a.features_db) if a.features_db else data.load_frame()
    rows, pmfs = ev.load_rows(a.version, a.season, frame)
    if a.r2b:                                                       # the frozen configuration: rolling recalibration R2b on top of the saved pmfs
        rec = recalibrate.RollingRecalibratorB()
        carry_rows, carry_pm = ev.load_rows_light(a.r2b_carry_version, a.r2b_carry_season) if a.r2b_carry_version else (None, None)
        new = {}
        for stat in pmfs:
            carry = None if carry_rows is None else (carry_rows['game_date'], observed(carry_rows, stat), oc.pmf_mean(carry_pm[stat]), recalibrate.game_group(carry_rows['game_id']))
            new[stat], _ = rec.apply(stat, pmfs[stat], rows['game_date'], observed(rows, stat), carry, groups=recalibrate.game_group(rows['game_id']))
        pmfs = new
        logging.info(f'R2b applied (carry-over from {a.r2b_carry_version} {a.r2b_carry_season})')
    rows['minutes_bucket'] = pd.cut(rows['minutes_mean'], [0, 15, 28, 60], labels=['under 15', '15-28', '28+']).astype(str)
    rows['month'] = pd.to_datetime(rows['game_date']).dt.strftime('%Y-%m')
    logging.info(f'{len(rows)} rows loaded')

    acc = ev.accuracy_table(rows, pmfs, frame, n_boot=a.n_boot)
    logging.info('accuracy done')
    bias_all = ev.bias_table(rows, pmfs, n_boot=a.n_boot)
    bias_bucket = ev.bias_table(rows, pmfs, stats=['points', 'rebounds', 'assists', 'threes', 'pra'], by='minutes_bucket', n_boot=a.n_boot)
    bias_month = ev.bias_table(rows, pmfs, stats=['points', 'pra'], by='month', n_boot=a.n_boot)
    bias_pos = ev.bias_table(rows, pmfs, stats=['points', 'rebounds', 'assists'], by='pos_group', n_boot=a.n_boot)
    logging.info('bias done')
    cal = ev.calibration_scores(rows, pmfs)
    byt = ev.by_decision_time(rows, pmfs, n_boot=a.n_boot)
    logging.info('calibration and decision-time done')

    # pre-registered go/no-go criteria that one run can answer
    c1 = acc.assign(ok=acc['diff_hi_game'] < 0)
    c2 = bias_all[bias_all['stat'].isin(['points', 'rebounds', 'assists'])]
    c2ok = c2['obs_over_pred_minus_1_%'].abs().le(2.0)
    c3 = cal.assign(ok=cal['ece'] <= 0.02)
    c4 = byt.assign(ok=byt['lo'] <= 0)                                   # not significantly worse: the interval does not lie entirely above 0
    verdict = [
        f"1. CRPS beats the best baseline with the 95% interval below 0 in all 8 stats: **{'PASS' if c1['ok'].all() else 'FAIL'}** ({int(c1['ok'].sum())}/8; failing: {c1.loc[~c1['ok'], 'stat'].tolist() or 'none'})",
        f"2. Mean bias within +/-2% for points, rebounds, assists: **{'PASS' if c2ok.all() else 'FAIL'}** (points {c2.loc[c2['stat'] == 'points', 'obs_over_pred_minus_1_%'].iloc[0]:+.1f}%, "
        f"rebounds {c2.loc[c2['stat'] == 'rebounds', 'obs_over_pred_minus_1_%'].iloc[0]:+.1f}%, assists {c2.loc[c2['stat'] == 'assists', 'obs_over_pred_minus_1_%'].iloc[0]:+.1f}%)",
        f"3. ECE of P(over) at ladder lines at most 0.02 in every stat: **{'PASS' if c3['ok'].all() else 'FAIL'}** (worst {c3['ece'].max():.4f} in {c3.loc[c3['ece'].idxmax(), 'stat']})",
        f"4. CRPS at T-5 not significantly worse than at T-240 (interval not entirely above 0) in every stat: **{'PASS' if c4['ok'].all() else 'FAIL'}** (failing: {c4.loc[~c4['ok'], 'stat'].tolist() or 'none'})",
    ]
    r = lambda d, n=4: md_table(d.round(n), n)
    md = [f'# Phase 4 evaluation: {a.season} ({a.version}{" + R2b" if a.r2b else ""})\n',
          f'{len(rows)} played player-game rows (three decision times each, pooled), {rows["game_id"].nunique()} games. Intervals: 95% bootstrap over whole games ({a.n_boot} draws); '
          'the date-clustered interval is the sensitivity check. Differences are paired (model minus best baseline; negative = model better).\n',
          '## Pre-registered go/no-go criteria\n', '\n'.join(f'- {v}' for v in verdict) + '\n',
          '## 1. Accuracy against the baselines (CRPS, lower is better)\n', r(acc[['stat', 'n', 'crps_model', 'crps_poisson', 'crps_normal', 'crps_empirical20', 'best_baseline', 'diff_vs_best', 'diff_lo_game', 'diff_hi_game',
                                                                                     'diff_lo_date', 'diff_hi_date', 'rel_improvement_%']]),
          f'The empirical baseline fell back to Poisson for {acc["empirical_fallback_share"].mean():.1%} of rows (fewer than {ev.MIN_HIST} prior games). The Normal baseline\'s one sd per stat is fitted on this season itself, which favours the baseline.\n',
          '## 2. Bias of the mean (observed / predicted - 1)\n', '### Overall\n', r(bias_all, 2), '### By projected minutes\n', r(bias_bucket, 2), '### By month\n', r(bias_month, 2), '### By position group\n', r(bias_pos, 2),
          '## 3. Calibration of P(X > line) at the ladder lines (raw, no recalibration)\n', r(cal), '## 4. Decision time (CRPS and the paired T-5 minus T-240 difference)\n', r(byt, 4)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(verdict))
    logging.info(f'wrote {a.out}')


if __name__ == '__main__':
    main()
