"""Re-run the tracking / fatigue program (E25 references, E28 screen, E29 forced inclusion) under the post-audit harness defaults
(residual feature selection, capped calibration block, refit on model_train + calibration, Platt ML calibration) and the corrected
`trk_prev_opp_pace` feature, into THIS folder, leaving reports/experiments/ and data/processed/experiments/ untouched.

    python reports/audit_fixes/rerun_new_defaults/run.py variant regen screen explore e29

The registered scripts (e28_screen, e28_explore, e29_run) are not edited: their path constants are re-pointed here. The only change in
behaviour is the one the audit made: e29_run pre-writes each arm's features.json, which the harness now reuses only when told to
(features_from), so the wrapper below passes it.
"""
import functools
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from experiments import lib

OUTN = HERE / 'reports' / 'experiments'
DATAN = HERE / 'data' / 'processed' / 'experiments'
OLD_DATA = ROOT / 'data' / 'processed' / 'experiments'
lib.OUT = OUTN


def stage_variant():
    from experiments import features_tracking as ft
    from team.data_paths import latest_training_set
    DATAN.mkdir(parents=True, exist_ok=True)
    new = DATAN / 'e28_training_set.csv'
    t0 = time.time()
    ft.build_variant(latest_training_set(), new, ROOT / 'data' / 'nba_stats.db')
    shutil.copy(OLD_DATA / 'e28_mechanism.csv', DATAN / 'e28_mechanism.csv')          # outcomes of game t only: unaffected by the pace fix
    print(f'variant built in {time.time() - t0:.0f}s')
    old = pd.read_csv(OLD_DATA / 'e28_training_set.csv', low_memory=False)
    cur = pd.read_csv(new, low_memory=False)
    assert list(old.columns) == list(cur.columns) and len(old) == len(cur)
    diff = []
    for c in old.columns:
        a, b = old[c], cur[c]
        if a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu':
            same = np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
        else:
            same = a.equals(b)
        if not same:
            diff.append(c)
    print('columns that differ from the saved variant:', diff)


def _link_db():
    d = HERE / 'data'
    d.mkdir(exist_ok=True)
    dst = d / 'nba_stats.db'
    if not dst.exists():
        try:
            os.link(ROOT / 'data' / 'nba_stats.db', dst)
        except OSError:
            shutil.copy(ROOT / 'data' / 'nba_stats.db', dst)


def stage_regen():
    from experiments import e25_regen
    e25_regen.main(['with_odds', 'residual'])


def _patch_sc():
    from experiments import e28_screen as sc
    sc.ROOT, sc.OUT = HERE, OUTN / 'E28_tracking'
    sc.VARIANT, sc.MECH = DATAN / 'e28_training_set.csv', DATAN / 'e28_mechanism.csv'
    return sc


def stage_screen():
    sc = _patch_sc()
    sc.main()


def stage_explore():
    sc = _patch_sc()
    _link_db()
    from experiments import e28_explore as ex
    ex.ROOT, ex.OUT = HERE, sc.OUT
    ex.main()


def stage_e29():
    from experiments import e29_run as e29
    e29.OUT, e29.VARIANT, e29.F1_POOL = OUTN, DATAN / 'e28_training_set.csv', DATAN / 'e29_f1pool_training_set.csv'

    def rw(name, **kw):
        pre_written = (OUTN / name / 'features.json').exists()
        return lib.run_walk_forward(name, features_from=name if pre_written else None, **kw)
    e29.run_walk_forward = rw
    for a in list(e29.ARMS):
        e29.run_arm(a)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    for s in sys.argv[1:]:
        print(f'=== {s}', flush=True)
        globals()[f'stage_{s}']()
