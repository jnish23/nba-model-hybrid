import pandas as pd
import pytest

from props.features import guards


def test_dedupe_logs_small_and_refuses_large_duplication(caplog):
    df = pd.DataFrame({'k': list(range(200)) + [0, 1], 'v': range(202)})
    with caplog.at_level('INFO'):
        out = guards.dedupe(df, ['k'], 'toy')
    assert len(out) == 200 and 'dropped 2 duplicate rows' in caplog.text
    bad = pd.DataFrame({'k': [0] * 50 + list(range(1, 51)), 'v': range(100)})
    with pytest.raises(ValueError):
        guards.dedupe(bad, ['k'], 'toy')


def test_cardinality_validation_catches_a_many_to_many_blow_up():
    left = pd.DataFrame({'k': [1, 2, 3], 'a': [1, 2, 3]})
    right_dup = pd.DataFrame({'k': [1, 1, 2], 'b': [1, 2, 3]})
    with pytest.raises(pd.errors.MergeError):
        left.merge(right_dup, on='k', how='left', validate='many_to_one')
