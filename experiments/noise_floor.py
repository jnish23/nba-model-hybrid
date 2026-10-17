"""Selection-noise floor: the baseline re-run with other feature-selection seeds, compared with the seed-42 baseline using the
same paired machinery. Any feature experiment that re-runs selection carries this noise. python -m experiments.noise_floor"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.lib import OUT, load_run, run_walk_forward           # noqa: E402
from experiments.report import compare, fmt_compare                    # noqa: E402

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    base, _ = load_run('baseline')
    lines = ['# Selection-noise floor (baseline with other selection seeds vs the seed-42 baseline)\n']
    for seed in (43, 44):
        name = f'baseline_seed{seed}'
        test, _, _ = run_walk_forward(name, selection_seed=seed)
        cmp = compare(name, exp_test=test, base_test=base)
        lines += [f'## seed {seed}\n', fmt_compare(cmp).to_markdown(index=False), '']
    (OUT / 'noise_floor.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
