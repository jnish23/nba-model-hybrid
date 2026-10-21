"""
Phase 3 diagnostics on the saved development-fold predictions (never reads 2024-25 or 2025-26):
where the point-forecast bias and the combo under-dispersion come from.

    python scripts/props_phase3_diagnostics.py --version v1_base
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.models import calibrate, data, folds as folds_mod, outcome as oc, registry  # noqa: E402
from props.models.pipeline import observed  # noqa: E402

STATS = ['points', 'rebounds', 'assists', 'threes', 'pra']


def load(version, season, frame):
    idx, pmfs = registry.load_predictions(version, season)
    cols = ['game_id', 'player_id', 'minutes_before', 'lbl_minutes', 'm_min_h15', 'game_date']
    keep = [c for c in frame.columns if c.startswith(('y_', 'lbl_')) or c in ('m_min_h15', 'pos_group')]
    f = frame[['game_id', 'player_id', 'minutes_before'] + list(dict.fromkeys(keep))]
    played = idx.merge(f, on=['game_id', 'player_id', 'minutes_before'], how='left', validate='one_to_one')
    return played, pmfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--folds', nargs='*', default=list(folds_mod.DEV_TEST_SEASONS))
    args = ap.parse_args()
    frame = data.load_frame()
    out = []
    for s in args.folds:
        played, pmfs = load(args.version, s, frame)
        mu = {k: oc.pmf_mean(pmfs[k]) for k in STATS}
        sd = {k: calibrate.pmf_sd(pmfs[k]) for k in STATS}
        y = {k: observed(played, k) for k in STATS}
        ok = np.ones(len(played), bool)
        for k in STATS:
            ok &= ~np.isnan(y[k])
        bucket = pd.cut(played['m_min_h15'], [0, 15, 28, 60], labels=['<15', '15-28', '28+'])
        print(f'\n===== {s}: {ok.sum()} played rows =====')
        print(f"minutes: predicted {played['minutes_mean'][ok].mean():.3f}  observed {played['lbl_minutes'][ok].mean():.3f}")
        rows = []
        for key, grp in [('all', pd.Series(True, index=played.index))] + [(f'T-{t}', played['minutes_before'] == t) for t in (240, 60, 5)] + \
                         [(f'min {b}', bucket == b) for b in ('<15', '15-28', '28+')]:
            m = ok & grp.to_numpy()
            r = {'segment': key, 'n': int(m.sum()), 'min bias': (played['lbl_minutes'][m] - played['minutes_mean'][m]).mean()}
            for k in ('points', 'rebounds', 'assists', 'threes', 'pra'):
                r[f'{k} bias%'] = 100 * (y[k][m].mean() / mu[k][m].mean() - 1)
                r[f'{k} varR'] = np.mean((y[k][m] - mu[k][m]) ** 2) / np.mean(sd[k][m] ** 2)
            rows.append(r)
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        z = pd.DataFrame({k: (y[k][ok] - mu[k][ok]) / sd[k][ok] for k in ('points', 'rebounds', 'assists')})
        print('standardized-residual correlations (points, rebounds, assists):')
        print(z.corr().round(3).to_string())
        # how much correlation would close the combo variance gap?
        for combo, parts in (('pra', ('points', 'rebounds', 'assists')),):
            var_sum = sum(np.mean(sd[p][ok] ** 2) for p in parts)
            print(f'{combo}: mean predicted variance {np.mean(sd[combo][ok] ** 2):.2f}  sum of the parts {var_sum:.2f}  observed MSE {np.mean((y[combo][ok] - mu[combo][ok]) ** 2):.2f}')
        # calibration of the central interval: share of outcomes inside the central 80% and 50% predictive intervals
        for k in STATS:
            cdf = np.cumsum(pmfs[k][ok].astype(float), axis=1)
            yy = y[k][ok].astype(int)
            cap = oc.CAPS[k]
            hi = cdf[np.arange(len(yy)), np.minimum(yy, cap)]
            lo = np.where(yy > 0, cdf[np.arange(len(yy)), np.clip(yy - 1, 0, cap)], 0.0)
            # randomized PIT, fixed seed, so discrete stats are comparable with 0.1 / 0.9
            pit = lo + np.random.default_rng(3).random(len(yy)) * (hi - lo)
            out.append({'season': s, 'stat': k, 'PIT<.1': (pit < .1).mean(), 'PIT>.9': (pit > .9).mean(), 'PIT in .25-.75': ((pit > .25) & (pit < .75)).mean()})
    print('\nrandomized PIT (uniform would give 0.10 / 0.10 / 0.50):')
    print(pd.DataFrame(out).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
