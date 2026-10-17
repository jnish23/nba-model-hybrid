import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')

from models.train_baseline import BaselineModel, NEUTRAL_PARAMS


def _training_csv(path, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season, start in (('2017-18', '2017-10-20'), ('2018-19', '2018-10-20'), ('2019-20', '2019-10-20')):
        dates = pd.Timestamp(start) + pd.to_timedelta(np.arange(400) // 4, unit='D')
        n = len(dates)
        x = rng.normal(size=(n, 6))
        margin = 3 * x[:, 0] + rng.normal(scale=8, size=n)
        total = 215 + 4 * x[:, 1] + rng.normal(scale=12, size=n)
        df = pd.DataFrame({'gameDate': dates, 'seasonYear': season, 'gameId': [f'{season}{i}' for i in range(n)],
                           'teamId_home': 1, 'teamId_away': 2, 'target_point_diff': margin.round(),
                           'target_total_pts': total.round(), 'target_home_win': (margin > 0).astype(int),
                           'points_home': 100, 'points_away': 100})
        for j in range(6):
            df[f'diff_f{j}'] = x[:, j]
            df[f'f{j}_home'] = x[:, j]
            df[f'f{j}_away'] = rng.normal(size=n)
        rows.append(df)
    pd.concat(rows).to_csv(path, index=False)


def test_default_walk_forward_does_not_load_configs_tuned_on_all_seasons(tmp_path, caplog):
    csv = tmp_path / 'train.csv'
    _training_csv(csv)
    model_dir = tmp_path / 'saved'
    model_dir.mkdir()
    # a poisoned "tuned" config: loading it would raise (negative trees) - it must be ignored by default
    for name in ('ml', 'spread', 'total'):
        (model_dir / f'best_config_{name}.json').write_text(json.dumps({'n_estimators': -5}))
    m = BaselineModel(csv, model_dir=str(model_dir))
    metrics, preds = m.backtest_walk_forward(start_season='2019-20', use_odds=False, calibrate=True, save_suffix='_t')
    assert len(metrics) == 1 and len(preds) == 400
    assert not any('NOT out-of-sample' in r.message for r in caplog.records)


def test_opting_into_tuned_configs_warns(tmp_path, caplog):
    csv = tmp_path / 'train.csv'
    _training_csv(csv)
    m = BaselineModel(csv, model_dir=str(tmp_path))
    m.backtest_walk_forward(start_season='2019-20', use_tuned_params=True, save_suffix='_t')   # no config files: falls back
    assert any('NOT out-of-sample' in r.message for r in caplog.records)


def test_neutral_params_are_fixed_constants():
    assert NEUTRAL_PARAMS['n_estimators'] == 1000 and NEUTRAL_PARAMS['learning_rate'] == 0.01


def test_per_fold_tuning_runs_on_training_rows_only(tmp_path, caplog):
    csv = tmp_path / 'train.csv'
    _training_csv(csv)
    m = BaselineModel(csv, model_dir=str(tmp_path))
    seen = []
    orig = m._tune_fold_params

    def spy(X, y, task, budget):
        seen.append(len(X))
        return orig(X, y, task, budget)

    m._tune_fold_params = spy
    with caplog.at_level('INFO'):
        m.backtest_walk_forward(start_season='2019-20', tune_per_fold=True, tune_time_budget=2, save_suffix='_t')
    # 800 earlier-season rows, 80% of them model-train (the rest is the calibration holdout); 3 targets
    assert seen == [640, 640, 640]
