"""
Paired comparison of two saved versions on one season (labels only; light on memory).

    python scripts/props_phase4_compare.py --a v2_base --b v3_norefit_2023-24 --season 2023-24
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.models import calibrate, evaluate as ev, outcome as oc  # noqa: E402
from props.models.pipeline import observed  # noqa: E402

CORE = ['points', 'rebounds', 'assists', 'threes', 'pra']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--season', required=True)
    ap.add_argument('--n-boot', type=int, default=1000)
    a = ap.parse_args()
    ra, pa = ev.load_rows_light(a.a, a.season)
    rb, pb = ev.load_rows_light(a.b, a.season)
    cmp = ev.compare_versions(ra, pa, rb, pb, n_boot=a.n_boot)
    print(f'\n{a.a} (A) vs {a.b} (B), {a.season}   (B_minus_A < 0: B better)')
    print(cmp[['stat', 'crps_A', 'crps_B', 'B_minus_A', 'lo', 'hi', 'bias_A_%', 'bias_B_%']].round(4).to_string(index=False))
    core = cmp[cmp['stat'].isin(CORE)]
    print('mean |bias| over core stats: %.2f%% -> %.2f%% (shrink %.0f%%); core stats significantly worse: %s' % (
        core['bias_A_%'].abs().mean(), core['bias_B_%'].abs().mean(), 100 * (1 - core['bias_B_%'].abs().mean() / core['bias_A_%'].abs().mean()),
        core[core['lo'] > 0]['stat'].tolist() or 'none'))
    for lab, rows, pm in (('A', ra, pa), ('B', rb, pb)):
        vr = {s: np.mean((observed(rows, s) - oc.pmf_mean(pm[s])) ** 2) / np.mean(calibrate.pmf_sd(pm[s]) ** 2) for s in ('points', 'rebounds', 'pra')}
        print(lab, 'variance ratio (obs/pred):', {k: round(float(v), 3) for k, v in vr.items()})


if __name__ == '__main__':
    main()
