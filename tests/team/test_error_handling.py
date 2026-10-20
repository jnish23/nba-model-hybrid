import logging

import numpy as np
import pandas as pd
import pytest

from team import team_features_v2 as tf


def test_searchsorted_count_equals_rolling_count_exactly():
    rng = np.random.default_rng(0)
    rows = []
    for team in range(1, 5):
        d = pd.Timestamp('2025-10-22') + pd.to_timedelta(np.sort(rng.choice(160, 70, replace=False)), unit='D')
        rows += [(team, x) for x in d]
    df = pd.DataFrame(rows, columns=['teamId', 'gameDate']).sort_values(['teamId', 'gameDate']).reset_index(drop=True)
    df['c'] = 1
    ref = df.groupby('teamId').rolling('7D', on='gameDate', closed='left')['c'].count().to_numpy()
    got = tf.games_in_previous_7_days(df['teamId'], df['gameDate'])
    np.testing.assert_allclose(got, np.nan_to_num(ref))
    shuffled = df.sample(frac=1, random_state=1)                    # any row order
    got2 = tf.games_in_previous_7_days(shuffled['teamId'], shuffled['gameDate'])
    np.testing.assert_allclose(pd.Series(got2, index=shuffled.index).sort_index().to_numpy(), np.nan_to_num(ref))


def test_failed_rolling_falls_back_to_the_exact_count_not_a_constant(team_db, monkeypatch, caplog):
    builder = tf.TeamFeatureBuilder(team_db)
    ref = builder.build_entire_history()['games_last_7d']

    def boom(*a, **k):
        raise ValueError("index must be monotonic")

    monkeypatch.setattr(pd.core.groupby.DataFrameGroupBy, 'rolling', boom)
    with caplog.at_level(logging.ERROR):
        got = builder.build_entire_history()['games_last_7d']
    np.testing.assert_allclose(got, ref)
    assert got.nunique() > 1
    assert any("rolling('7D') failed" in r.message for r in caplog.records)


def test_no_bare_except_left_in_the_production_modules():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for rel in ['team/team_features_v2.py', 'team/game_features_v2.py', 'scripts/run_inference_v2.py',
                'core/prediction_logger.py']:
        src = (root / rel).read_text(encoding='utf-8')
        assert not re.search(r'^\s*except\s*:', src, re.M), f'bare except in {rel}'
