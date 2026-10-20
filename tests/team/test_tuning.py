import json

import numpy as np
import pandas as pd

from synthetic_history import training_csv
from team import tuning
from team.train_baseline import BaselineModel


def test_grid_search_is_deterministic_and_picks_a_config_from_the_grid():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 4)), columns=list('abcd'))
    y = X['a'] * 2 + rng.normal(size=600)
    cfg1, s1 = tuning.pick_config('reg', X, y, 480)
    cfg2, s2 = tuning.pick_config('reg', X, y, 480)
    assert cfg1 == cfg2 and s1 == s2 and cfg1 in tuning.GRID
    yc = (y > 0).astype(int)
    cfg, score = tuning.pick_config('clf', X, yc, 480)
    assert cfg in tuning.GRID and score < np.log(2)                       # better than a coin flip


def test_the_backtest_and_the_trainer_search_the_same_grid():
    from experiments import lib
    assert lib.GRID is tuning.GRID and lib.make_est is tuning.make_est


def test_final_models_can_be_trained_with_the_validated_grid_instead_of_a_wall_clock_search(tmp_path):
    csv = tmp_path / 'train.csv'
    training_csv(csv, per_season=300)
    m = BaselineModel(csv, model_dir=str(tmp_path / 's'))
    m.train_and_save_all(use_odds=True, save_suffix='_g', time_budget=1, calib_holdout_days=80, tuner='grid')
    for target in ('ml', 'spread', 'total'):
        cfg = json.loads((tmp_path / 's' / f'best_config_{target}_g.json').read_text())
        assert cfg in tuning.GRID
