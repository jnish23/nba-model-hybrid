import numpy as np
import pandas as pd

from props.live import at_t, parity


def _pair(n=50, **live_changes):
    k = pd.DataFrame({'game_id': [f'g{i}' for i in range(n)], 'player_id': range(n), 'minutes_before': 60})
    train = k.assign(m_min_h15=np.linspace(10, 30, n), tm_pts=np.linspace(100, 120, n), inj_status_code=0, inj_p_play=1.0, av_n_out=0.0, av_miss_pts=0.1)
    live = k.assign(m_min_h15=train['m_min_h15'], tm_pts=train['tm_pts'], inj_status_code=0, inj_p_play=1.0, av_n_out=0.0, av_miss_pts=0.1)
    for col, fn in live_changes.items():
        live[col] = fn(live[col])
    return live, train


def test_a2_demands_exact_history_columns_and_tolerates_1e7_in_context():
    live, train = _pair(m_min_h15=lambda s: s + 1e-8, tm_pts=lambda s: s + 5e-7)
    r = parity.a2_identical(parity.merge_frames(live, train), list(train.columns))
    t = r['table'].set_index('column')
    assert t.loc['m_min_h15', 'disagree'] == 50 and t.loc['tm_pts', 'disagree'] == 0 and not r['pass']


def test_a2_skips_rows_where_the_training_value_is_missing_but_not_a_live_gap():
    live, train = _pair()
    train.loc[0, 'm_min_h15'] = np.nan
    live.loc[1, 'm_min_h15'] = np.nan
    t = parity.a2_identical(parity.merge_frames(live, train), list(train.columns))['table'].set_index('column')
    assert t.loc['m_min_h15', 'compared'] == 49 and t.loc['m_min_h15', 'disagree'] == 1


def test_a3_and_a4_use_their_tolerances():
    live, train = _pair(av_n_out=lambda s: s + 1.0, av_miss_pts=lambda s: s + 0.02)
    m = parity.merge_frames(live, train)
    assert parity.a3_injury(m)['pass']
    assert parity.a4_availability(m)['pass']
    live2, train2 = _pair(av_miss_pts=lambda s: s + 0.04)
    r = parity.a4_availability(parity.merge_frames(live2, train2))
    assert not r['pass'] and r['failing_columns'] == 1


def test_a6_counts_overlap_side_and_ev_correlation():
    key = {'game_id': ['a', 'b', 'c', 'd'], 'player_id': 1, 'stat': 'assists'}
    train = pd.DataFrame({**key, 'best_side': ['over', 'under', 'over', 'over'], 'best_ev': [0.05, 0.04, 0.06, 0.08]})
    live = pd.DataFrame({'game_id': ['a', 'b', 'c', 'e'], 'player_id': 1, 'stat': 'assists', 'best_side': ['over', 'under', 'over', 'over'], 'best_ev': [0.05, 0.04, 0.06, 0.05]})
    r = parity.a6_decisions(live, train)
    assert r['shared'] == 3 and abs(r['jaccard'] - 3 / 5) < 1e-12 and r['side_agreement'] == 1.0 and not r['pass']


def test_asof_reports_drops_reports_inside_the_publish_lag():
    tips = pd.Series({'0022500001': pd.Timestamp('2025-10-21 19:30')})
    rep = pd.DataFrame({'gameId': ['0022500001'] * 3, 'report_ts': pd.to_datetime(['2025-10-21 17:15', '2025-10-21 17:30', '2025-10-21 18:30']), 'personId': [1, 2, 3]})
    kept = at_t.asof_reports(rep, tips, 60)                       # cutoff 19:30 - 60 - 15 = 18:15
    assert list(kept['personId']) == [1, 2]
