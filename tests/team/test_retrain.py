import datetime as dt
import json

import pandas as pd
import pytest

from synthetic_history import training_csv
from team import model_registry as mr
from team import retrain
from team.train_baseline import BaselineModel


def test_verdict_passes_within_tolerance_and_fails_beyond_it():
    inc = {'ml_logloss': 0.60, 'spread_mae': 10.5, 'total_mae': 14.5}
    assert retrain.verdict({'ml_logloss': 0.603, 'spread_mae': 10.4, 'total_mae': 14.6}, inc)['ok']
    v = retrain.verdict({'ml_logloss': 0.61, 'spread_mae': 10.5, 'total_mae': 14.8}, inc)
    assert not v['ok'] and set(v['worse']) == {'ml_logloss', 'total_mae'}


def test_window_metrics_scores_a_prediction_frame():
    frame = pd.DataFrame({'target_home_win': [1, 0], 'target_point_diff': [5.0, -3.0], 'target_total_pts': [220.0, 230.0]})
    preds = pd.DataFrame({'ml_prob': [0.8, 0.3], 'spread_pred': [3.0, float('nan')], 'total_pred': [222.0, 226.0]})
    m = retrain.window_metrics(preds, frame)
    assert m['spread_mae'] == 2.0 and m['spread_n'] == 1 and m['total_mae'] == 3.0 and 0.2 < m['ml_logloss'] < 0.4


@pytest.fixture(scope='module')
def retrained(tmp_path_factory):
    d = tmp_path_factory.mktemp('retrain')
    csv = d / 'training_set.csv'
    training_csv(csv, per_season=500)
    saved = d / 'saved'
    # the "deployed" set: trained earlier on the first three seasons only
    early = pd.read_csv(csv)
    early[early['seasonYear'] < '2023-24'].to_csv(d / 'early.csv', index=False)
    BaselineModel(d / 'early.csv', model_dir=str(saved)).train_and_save_all(use_odds=True, save_suffix='_inc', time_budget=1,
                                                                            calib_holdout_days=90, tuner='grid')
    report = retrain.run_retrain(csv, model_dir=saved, out_dir=d / 'reports', tuner='grid', holdout_days=90, gate_days=150,
                                 tag='t1', incumbent_suffix='_inc', today=dt.date(2025, 1, 1))
    return d, saved, report


def test_candidate_is_a_new_hashed_suffix_and_the_incumbent_is_untouched(retrained):
    d, saved, report = retrained
    assert report['candidate_suffix'] == '_with_odds_cand_t1'
    m = mr.read_manifest('_with_odds_cand_t1', saved)
    assert m['model_set_hash'] == report['candidate']['model_set_hash'] and m['refit_on_all'] is True
    assert report['candidate']['trained_through'] == str(pd.read_csv(d / 'training_set.csv', parse_dates=['gameDate'])['gameDate'].max())
    mr.load_bundle('_inc', saved)                                           # and the earlier set still loads with its own manifest


def test_gate_compares_candidate_and_incumbent_on_the_same_recent_window(retrained):
    d, saved, report = retrained
    gate = report['gate']
    assert gate['candidate']['games'] == gate['incumbent']['games'] > 150
    assert gate['incumbent_saw_window'] is False
    assert set(gate['verdict']) == {'ok', 'worse'}
    assert (d / 'reports' / 'retrain_t1.json').exists()
    assert json.loads((d / 'reports' / 'retrain_t1.json').read_text())['candidate_suffix'] == '_with_odds_cand_t1'


def test_rerunning_with_the_same_tag_is_refused(retrained):
    d, saved, _ = retrained
    with pytest.raises(FileExistsError):
        retrain.run_retrain(d / 'training_set.csv', model_dir=saved, out_dir=d / 'reports', tuner='grid', holdout_days=90,
                            gate_days=150, tag='t1', incumbent_suffix='_inc')
