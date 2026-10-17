"""E8 step 2 (only because step 1 supported the hypothesis): add inj_fresh_lost and the decayed chronic value as candidate
features and evaluate on folds 2022-23..2025-26 only (the discovery seasons stay out of the evaluation).
python -m experiments.e08_run"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import decide                                       # noqa: E402
from experiments.e08_hypothesis import FEATS, VARIANT                # noqa: E402
from experiments.feature_exp import MAIN, run_feature_experiment     # noqa: E402

NAME = 'E08_fresh_chronic'
EVAL_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']


def decision(cmp, **_):
    """KEEP iff, for >= 1 of the three primary losses, it improves in >= 3 of 4 folds with a pooled CI excluding zero; no-harm on all."""
    passed = {k: decide.mc(cmp[c], min_seasons=3) for k, c in MAIN.items()}
    nh = {k: decide.nh(cmp[c]) for k, c in MAIN.items()}
    return bool(any(passed.values()) and all(nh.values())), passed, nh


if __name__ == '__main__':
    path = VARIANT.with_name('e08_step2_training_set.csv')
    df = pd.read_csv(VARIANT, low_memory=False)
    drop = [f'{c}_{s}' for c in ('inj_chronic_lost', 'inj_mid_lost') for s in ('home', 'away')] + ['diff_inj_chronic_lost', 'diff_inj_mid_lost']
    df.drop(columns=drop).to_csv(path, index=False)
    run_feature_experiment(NAME + '_step2', path, 'fresh vs chronic absences (folds 2022-23..2025-26)', seasons=EVAL_SEASONS, decision_fn=decision)
