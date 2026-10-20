"""
Phase 4: decide the pre-registered remedies for the over-prediction on DEVELOPMENT folds only (reports/props/phase4_plan.md section 2.6).

    python scripts/props_phase4_remedies.py r2dev      # rolling recalibration on 2022-23 (no bias; must not hurt) and 2023-24 (bias; must help)
    python scripts/props_phase4_remedies.py r1dev --r1-version r1_2023-24    # season-position features removed, retrained on 2023-24

Adoption rules, fixed before any result was looked at. "Core stats" = points, rebounds, assists, threes, pra.
  R1: on 2023-24 the mean absolute bias over the core stats shrinks by at least one third AND no core stat's paired CRPS difference interval lies entirely above 0.
  R2: on 2023-24 the mean absolute bias over the core stats shrinks by at least one half AND on 2022-23 no core stat's paired CRPS difference interval lies entirely above 0
      (and the same non-harm condition holds on 2023-24).
Intervals are 95% game-clustered bootstrap intervals of the paired CRPS difference (remedy minus original).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.models import data, evaluate as ev, recalibrate  # noqa: E402
from props.models.pipeline import observed  # noqa: E402

CORE = ['points', 'rebounds', 'assists', 'threes', 'pra']


def verdict(cmp, need_shrink, label, check_bias=True):
    core = cmp[cmp['stat'].isin(CORE)]
    shrink = 1 - core['bias_B_%'].abs().mean() / core['bias_A_%'].abs().mean()
    harmed = core[core['lo'] > 0]['stat'].tolist()
    ok = (shrink >= need_shrink if check_bias else True) and not harmed
    print(f'  {label}: mean |bias| over core stats {core["bias_A_%"].abs().mean():.2f}% -> {core["bias_B_%"].abs().mean():.2f}% (shrink {shrink:.0%}, need {need_shrink:.0%}); '
          f'stats with CRPS significantly worse: {harmed or "none"} -> {"PASS" if ok else "FAIL"}')
    return ok


def r2dev(frame):
    rec = recalibrate.RollingRecalibrator()
    res = {}
    for season in ('2022-23', '2023-24'):
        rows, pm = ev.load_rows('v2_base', season, frame)
        after = {}
        for stat in pm:
            after[stat], _ = rec.apply(stat, pm[stat], rows['game_date'], observed(rows, stat))
        cmp = ev.compare_versions(rows, pm, rows, after)
        print(f'\nR2 on {season}: original (A) vs rolling-recalibrated (B)')
        print(cmp.round(4).to_string(index=False))
        res[season] = cmp
    print('\nPre-registered R2 verdict:')
    a = verdict(res['2023-24'], 0.5, '2023-24 (bias must shrink by half, no CRPS harm)')
    b = verdict(res['2022-23'], 0.0, '2022-23 (no harm)', check_bias=False)
    print(f'R2 ADOPTED on the dev folds: {a and b}')


def r1dev(frame, r1_version):
    rows_a, pm_a = ev.load_rows('v2_base', '2023-24', frame)
    rows_b, pm_b = ev.load_rows(r1_version, '2023-24', frame)
    cmp = ev.compare_versions(rows_a, pm_a, rows_b, pm_b)
    print('\nR1 on 2023-24: v2_base (A) vs season-position features removed (B)')
    print(cmp.round(4).to_string(index=False))
    print('\nPre-registered R1 verdict:')
    ok = verdict(cmp, 1 / 3, '2023-24 (bias must shrink by a third, no CRPS harm)')
    print(f'R1 ADOPTED on the dev folds: {ok}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['r2dev', 'r1dev'])
    ap.add_argument('--r1-version', default='r1_2023-24')
    a = ap.parse_args()
    fr = data.load_frame()
    r2dev(fr) if a.mode == 'r2dev' else r1dev(fr, a.r1_version)
