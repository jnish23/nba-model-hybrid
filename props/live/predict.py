"""Predictions for a set of feature rows from a saved fold of the frozen pipeline (the arithmetic of scripts/props_phase5_predict.py, as a function the live job and the parity test share)."""
import numpy as np

from props.models import outcome as oc, pipeline


def predict_rows(bundle, rows, n_sims=20000, seed=21, chunk_rows=30000):
    """
    bundle: a saved fold bundle (refit on fit + calibration when it has a 'refit' entry, with its dispersion scaler and families). rows: model-input rows. Returns
    (minutes_mean, {stat: pmf}) with the dispersion scaling applied and no other recalibration. Same rows in the same order with the same seed give the same draws.
    """
    final = dict(bundle, **bundle['refit']) if 'refit' in bundle else bundle
    if 'refit' in bundle:
        final['rates'].nb_k = bundle['rates'].nb_k
    parts, mins = {s: [] for s in oc.CAPS}, []
    for lo in range(0, len(rows), chunk_rows):
        res = pipeline.distributions(final, rows.iloc[lo:lo + chunk_rows], bundle['families'], n_sims, seed + lo)
        res.pop('sim_means', None)
        mm = res.pop('minutes_mean')
        for s in oc.CAPS:
            parts[s].append(bundle['dispersion'].transform(s, res[s], mm))
        mins.append(mm)
    if not len(rows):
        return np.empty(0), {s: np.empty((0, oc.CAPS[s] + 1), dtype=np.float32) for s in oc.CAPS}
    return np.concatenate(mins), {s: np.concatenate(v) for s, v in parts.items()}
