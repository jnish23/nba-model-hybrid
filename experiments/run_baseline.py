"""Run and save the baseline: python -m experiments.run_baseline"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.lib import run_walk_forward           # noqa: E402
from experiments.report import evaluate                 # noqa: E402

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    test, calib, info = run_walk_forward('baseline')
    res = evaluate('baseline', test)
    print({k: round(v, 4) for k, v in res['pooled'].items()})
