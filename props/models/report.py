"""
Smoke report for the development folds: full-distribution CRPS against two simple rolling-average baselines (Poisson and discretized Normal around
rate x expected minutes), bias, PIT tails, the variance check on the combos, and calibration before and after. A guard against building on a broken
model, NOT the formal Phase 4 evaluation (no confidence intervals, no ablations, no market comparison).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from props.models import calibrate, outcome as oc
from props.models.pipeline import observed
from props.quality import md_table

BASE_RATES = {'points': ['r_points_h15'], 'rebounds': ['r_rebounds_h15'], 'assists': ['r_assists_h15'], 'threes': ['r_fg3m_h15'],
              'pra': ['r_points_h15', 'r_rebounds_h15', 'r_assists_h15'], 'pr': ['r_points_h15', 'r_rebounds_h15'],
              'pa': ['r_points_h15', 'r_assists_h15'], 'ra': ['r_rebounds_h15', 'r_assists_h15']}


def baseline_pmfs(rows, stat, y, cap):
    mu = (rows[BASE_RATES[stat]].sum(axis=1) * rows['m_min_h15']).to_numpy(float)
    mu = np.where(np.isnan(mu), np.nanmean(mu), mu)
    ys = np.arange(cap + 1)[None, :]
    pois = poisson.pmf(ys, np.maximum(mu, 1e-3)[:, None])
    pois[:, -1] += np.maximum(1 - pois.sum(1), 0)
    sd = float(np.sqrt(np.nanmean((y - mu) ** 2)))                       # one scalar sd per stat and fold: the baseline gets the best constant sd
    cdf = norm.cdf(ys + 0.5, mu[:, None], sd)
    nrm = np.diff(np.c_[np.zeros(len(mu)), cdf], axis=1)
    nrm[:, -1] += np.maximum(1 - nrm.sum(1), 0)
    return pois.astype(np.float32), nrm.astype(np.float32)


def smoke_table(played, pmfs, calibrator=None):
    rows = []
    for stat, cap in oc.CAPS.items():
        y = observed(played, stat)
        ok = ~np.isnan(y) & played['m_min_h15'].notna().to_numpy()
        pm, yy, sub = pmfs[stat][ok], y[ok].astype(int), played[ok]
        pois, nrm = baseline_pmfs(sub, stat, yy, cap)
        cdf = np.cumsum(pm.astype(float), axis=1)
        hi = cdf[np.arange(len(yy)), np.minimum(yy, cap)]
        lo = np.where(yy > 0, cdf[np.arange(len(yy)), np.clip(yy - 1, 0, cap)], 0.0)
        pit = (hi + lo) / 2
        var = calibrate.pmf_sd(pm) ** 2
        r = {'stat': stat, 'n': len(yy), 'mean pred': oc.pmf_mean(pm).mean(), 'mean obs': yy.mean(),
             'CRPS model': oc.crps(pm, yy).mean(), 'CRPS poisson': oc.crps(pois, yy).mean(), 'CRPS normal': oc.crps(nrm, yy).mean(),
             'PIT<.1': (pit < 0.1).mean(), 'PIT>.9': (pit > 0.9).mean(), 'var ratio (obs/pred)': np.mean((yy - oc.pmf_mean(pm)) ** 2) / np.mean(var)}
        rows.append(r)
    return pd.DataFrame(rows)


def calibration_table(played, pmfs, calibrator):
    out = []
    for stat in oc.CAPS:
        y = observed(played, stat)
        ok = ~np.isnan(y)
        out.append(calibrate.calibration_report(stat, pmfs[stat][ok], y[ok], calibrator))
    return pd.DataFrame(out)


def markdown(title, smoke_by_fold, calib_by_fold, families_table, minutes_info, notes=''):
    md = [f'# {title}\n', notes]
    md += ['## Rate-family selection (mean loss across the development folds; lower is better)\n', md_table(families_table, 5)]
    for season in smoke_by_fold:
        md += [f'## Test season {season}\n', '### Full-distribution CRPS against rolling-average baselines (played rows, all three decision times)\n',
               md_table(smoke_by_fold[season].round(4), 4), '### Calibration of P(X > line), raw vs calibrated (calibration window to test season)\n',
               md_table(calib_by_fold[season].round(4), 4)]
    return '\n'.join(md)
