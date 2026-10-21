"""
Stage F: calibration of P(X > line) on a time-ordered holdout, by stat and by line region.

For each calibration row (the latest 20% of the training window by date, predicted by models that never saw those rows) and a set of half-integer lines around
the predicted mean, the raw model probability of the over is paired with whether the over happened. One isotonic regression per (stat, region) maps the raw
probability to a calibrated one. Regions describe where the line sits relative to the predicted distribution: well below the mean ("low"), around it
("mid"), well above it ("high"), in units of the predicted standard deviation. Only the probability at a line is recalibrated; the stored pmf stays raw.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression

from props.models import outcome as oc

REGIONS = ('low', 'mid', 'high')
LINE_OFFSETS = (-2, -1, 0, 1, 2)          # lines at floor(mean) + offset + 0.5, like a book's ladder around the main line
EPS = 1e-4


def pmf_sd(pmf):
    ys = np.arange(pmf.shape[1])[None, :]
    m = oc.pmf_mean(pmf)
    return np.sqrt(np.maximum((pmf * ys ** 2).sum(axis=1) - m ** 2, 1e-6))


def region_of(line, mean, sd):
    z = (np.asarray(line, dtype=float) - mean) / sd
    return np.where(z < -0.5, 'low', np.where(z > 0.5, 'high', 'mid'))


def over_prob_table(pmf, offsets=LINE_OFFSETS):
    """For every row and offset: (line, raw P(X > line), region). Arrays shaped (n, len(offsets))."""
    mean = oc.pmf_mean(pmf)
    sd = pmf_sd(pmf)
    base = np.floor(mean)
    lines = np.stack([base + o + 0.5 for o in offsets], axis=1)
    lines = np.maximum(lines, 0.5)
    cdf = np.cumsum(pmf.astype(np.float64), axis=1)
    k = np.clip(np.floor(lines).astype(int), 0, pmf.shape[1] - 1)
    raw = 1.0 - np.take_along_axis(cdf, k, axis=1)
    reg = region_of(lines, mean[:, None], sd[:, None])
    return lines, raw, reg


class ThresholdCalibrator:
    def __init__(self):
        self.maps = {}                         # (stat, region) -> IsotonicRegression
        self.n = {}

    def fit(self, stat, pmf, obs):
        lines, raw, reg = over_prob_table(pmf)
        hit = (obs[:, None] > lines).astype(float)
        for r in REGIONS:
            sel = reg == r
            if sel.sum() < 200:
                continue
            iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds='clip', increasing=True).fit(raw[sel], hit[sel])
            self.maps[(stat, r)] = iso
            self.n[(stat, r)] = int(sel.sum())
        return self

    def transform(self, stat, raw, region):
        out = np.array(raw, dtype=float, copy=True)
        region = np.asarray(region)
        for r in REGIONS:
            iso = self.maps.get((stat, r))
            sel = region == r
            if iso is not None and sel.any():
                out[sel] = iso.predict(raw[sel])
        return out

    def over_prob(self, stat, pmf, line):
        """Calibrated P(X > line) for each row at its own line."""
        mean, sd = oc.pmf_mean(pmf), pmf_sd(pmf)
        raw = oc.prob_over(pmf, line)
        return self.transform(stat, raw, region_of(line, mean, sd))


def calibration_report(stat, pmf, obs, calibrator=None):
    """Log loss and mean |predicted - realized| over equal-count bins, for the raw and (if given) the calibrated over-probabilities."""
    lines, raw, reg = over_prob_table(pmf)
    hit = (obs[:, None] > lines).astype(float)
    cal = None
    if calibrator is not None:
        cal = np.stack([calibrator.transform(stat, raw[:, j], reg[:, j]) for j in range(raw.shape[1])], axis=1)

    def ll(p):
        p = np.clip(p, EPS, 1 - EPS)
        return float(-np.mean(hit * np.log(p) + (1 - hit) * np.log(1 - p)))

    def ece(p, bins=10):
        flat_p, flat_h = p.ravel(), hit.ravel()
        q = np.quantile(flat_p, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(q, flat_p, side='right') - 1, 0, bins - 1)
        return float(sum(abs(flat_p[idx == b].mean() - flat_h[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))

    out = {'stat': stat, 'logloss_raw': ll(raw), 'ece_raw': ece(raw)}
    if cal is not None:
        out.update({'logloss_cal': ll(cal), 'ece_cal': ece(cal)})
    return out
