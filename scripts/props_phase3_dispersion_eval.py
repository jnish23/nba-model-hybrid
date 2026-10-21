"""
Development-fold check of the dispersion calibration (Stage F0): fit the scaler on each fold's calibration window, apply it to the saved test pmfs, and
compare CRPS, variance ratio, PIT tails and over-probability log loss before and after, with and without the isotonic layer. Development folds only.

    python scripts/props_phase3_dispersion_eval.py --version v1_base
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.models import calibrate, data, dispersion, folds as folds_mod, outcome as oc, pipeline, registry  # noqa: E402
from props.models.pipeline import observed  # noqa: E402

STATS = list(oc.CAPS)


def pit_tails(pm, y, seed=3):
    cap = pm.shape[1] - 1
    cdf = np.cumsum(pm.astype(float), axis=1)
    hi = cdf[np.arange(len(y)), np.minimum(y, cap)]
    lo = np.where(y > 0, cdf[np.arange(len(y)), np.clip(y - 1, 0, cap)], 0.0)
    pit = lo + np.random.default_rng(seed).random(len(y)) * (hi - lo)
    return (pit < .1).mean(), (pit > .9).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--folds', nargs='*', default=list(folds_mod.DEV_TEST_SEASONS))
    ap.add_argument('--sims', type=int, default=4000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    frame = data.load_frame()
    chosen = json.loads((registry.version_dir(args.version) / 'manifest.json').read_text())['families']
    all_rows, c_rows = [], []
    for s in args.folds:
        fold = folds_mod.make_fold(frame, s)
        bundle = registry.load_bundle(args.version, s)
        cal = frame.iloc[fold.calib_idx]
        cal = cal[(cal['lbl_played'] == 1) & (cal['lbl_minutes'] > 0)]
        cres = pipeline.distributions(bundle, cal, chosen, args.sims, 1)
        cal_min = cres['minutes_mean']
        scaler = dispersion.DispersionScaler()
        for st in STATS:
            y = observed(cal, st)
            ok = ~np.isnan(y)
            scaler.fit(st, cres[st][ok], y[ok], cal_min[ok])
        c_rows += [{'season': s, 'stat': k[0], 'bucket': k[1], 'c': v} for k, v in scaler.c.items()]
        logging.info(f'{s}: calibration-window distributions ready')
        idx, pmfs = registry.load_predictions(args.version, s)
        keep = [c for c in frame.columns if c.startswith(('y_', 'lbl_'))]
        test = idx.merge(frame[['game_id', 'player_id', 'minutes_before'] + keep], on=['game_id', 'player_id', 'minutes_before'], how='left', validate='one_to_one')
        raw_iso = bundle['calibrator']                                       # isotonic fitted on the raw calibration pmfs (as saved)
        iso_w = calibrate.ThresholdCalibrator()                              # isotonic fitted on the widened calibration pmfs
        for st in STATS:
            y = observed(cal, st)
            ok = ~np.isnan(y)
            iso_w.fit(st, scaler.transform(st, cres[st][ok], cal_min[ok]), y[ok])
        for st in STATS:
            y = observed(test, st)
            ok = ~np.isnan(y)
            yy = y[ok].astype(int)
            raw = pmfs[st][ok]
            wide = scaler.transform(st, raw, test['minutes_mean'].to_numpy()[ok])
            for name, pm, cal_obj in (('raw', raw, None), ('raw+iso', raw, raw_iso), ('widen', wide, None), ('widen+iso', wide, iso_w)):
                rep = calibrate.calibration_report(st, pm, y[ok], cal_obj)
                ll = rep['logloss_cal'] if cal_obj is not None else rep['logloss_raw']
                ece = rep['ece_cal'] if cal_obj is not None else rep['ece_raw']
                lo_t, hi_t = pit_tails(pm, yy)
                all_rows.append({'season': s, 'stat': st, 'variant': name, 'crps': oc.crps(pm, yy).mean(), 'logloss': ll, 'ece': ece,
                                 'varR': np.mean((yy - oc.pmf_mean(pm)) ** 2) / np.mean(calibrate.pmf_sd(pm) ** 2), 'PIT<.1': lo_t, 'PIT>.9': hi_t})
        logging.info(f'{s}: evaluated')
        pd.DataFrame([{'stat': k[0], 'bucket': k[1], 'c': v} for k, v in scaler.c.items()]).to_csv(registry.version_dir(args.version) / s / 'dispersion_c.csv', index=False)
    res = pd.DataFrame(all_rows)
    out = Path('reports/props/phase3_dispersion_eval.csv')
    res.to_csv(out, index=False)
    print('\nc table (variance inflation, per season/stat/minutes bucket 0:<15 1:15-28 2:28+):')
    print(pd.DataFrame(c_rows).pivot_table(index=['stat', 'bucket'], columns='season', values='c').round(3).to_string())
    print('\nmean over the three folds, by variant (lower crps/logloss/ece better; varR, PIT tails -> 1, .1, .1):')
    print(res.groupby(['stat', 'variant'])[['crps', 'logloss', 'ece', 'varR', 'PIT<.1', 'PIT>.9']].mean().round(4).to_string())
    print('\nfold-by-fold wins for widen vs raw (CRPS / logloss), out of 3:')
    p = res.pivot_table(index=['season', 'stat'], columns='variant', values=['crps', 'logloss'])
    w = pd.DataFrame({'crps_better': (p['crps']['widen'] < p['crps']['raw']).groupby('stat').sum(), 'logloss_better': (p['logloss']['widen'] < p['logloss']['raw']).groupby('stat').sum(),
                      'iso_logloss_better': (p['logloss']['raw+iso'] < p['logloss']['raw']).groupby('stat').sum()})
    print(w.to_string())


if __name__ == '__main__':
    main()
