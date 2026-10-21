"""
Measured Monte Carlo noise in P(over) from the joint simulation (points, threes, combos): the same rows are simulated with different seeds at 4,000 and at
20,000 draws; the standard deviation of the seed-to-seed difference / sqrt(2) is the noise of one run, in probability points, at the ladder lines.

    python scripts/props_phase4_mc_noise.py --version v2_base --season 2023-24 --rows 3000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.models import calibrate, data, folds as folds_mod, outcome as oc, pipeline, registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--season', required=True)
    ap.add_argument('--rows', type=int, default=3000)
    ap.add_argument('--features-db', default=None)
    a = ap.parse_args()
    frame = data.load_frame(a.features_db) if a.features_db else data.load_frame()
    fold = folds_mod.make_fold(frame, a.season)
    bundle = registry.load_bundle(a.version, a.season)
    test = frame.iloc[fold.test_idx]
    played = test[(test['lbl_played'] == 1) & (test['lbl_minutes'] > 0) & (test['m_min_h15'] >= 20)]            # rotation players: the rows a bettor would price
    rows = played.sample(a.rows, random_state=0)
    final = dict(bundle, **bundle['refit']) if 'refit' in bundle else bundle
    if 'refit' in bundle:
        final['rates'].nb_k = bundle['rates'].nb_k
    runs = {}
    for n, seed in ((4000, 11), (4000, 12), (20000, 13), (20000, 14)):
        runs[(n, seed)] = pipeline.distributions(final, rows, bundle['families'], n, seed)
    out = []
    for stat in ('points', 'threes', 'pra', 'pr', 'rebounds', 'assists'):
        r = {'stat': stat}
        lines, _, _ = calibrate.over_prob_table(runs[(20000, 13)][stat])                                       # ONE set of lines for every run (a run's own mean would move the line)
        for n, (s1, s2) in ((4000, (11, 12)), (20000, (13, 14))):
            p1 = np.stack([oc.prob_over(runs[(n, s1)][stat], lines[:, j]) for j in range(lines.shape[1])], axis=1)
            p2 = np.stack([oc.prob_over(runs[(n, s2)][stat], lines[:, j]) for j in range(lines.shape[1])], axis=1)
            mid = (p1 > 0.2) & (p1 < 0.8)                                                                       # lines near the middle: where the bets are
            r[f'sd_pp_{n}'] = 100 * float(np.std((p1 - p2)[mid]) / np.sqrt(2))
        out.append(r)
    t = pd.DataFrame(out)
    print(t.round(3).to_string(index=False))
    print('\nstandard deviation of one run\'s P(over) in percentage points, for lines with P(over) between 0.2 and 0.8 (rebounds and assists are exact mixtures: ~0 by construction)')


if __name__ == '__main__':
    main()
