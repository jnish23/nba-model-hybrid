import logging

import numpy as np
import pandas as pd
import pytest

from team.live_features import MissingFeaturesError, align_features


def _row(**kw):
    return pd.DataFrame([kw])


def test_missing_feature_is_nan_not_zero_and_logged(caplog):
    cols = [f'f{i}' for i in range(20)]
    row = _row(**{c: 1.0 for c in cols[:19]})
    with caplog.at_level(logging.ERROR):
        out = align_features(row, cols, 'ML')
    assert np.isnan(out.loc[0, 'f19'])
    assert (out[cols[:19]] == 1.0).all().all()
    assert any('f19' in r.message and r.levelno == logging.ERROR for r in caplog.records)


def test_column_order_follows_the_model():
    out = align_features(_row(b=2.0, a=1.0), ['a', 'b'])
    assert list(out.columns) == ['a', 'b']


def test_mostly_missing_row_is_refused():
    with pytest.raises(MissingFeaturesError):
        align_features(_row(a=1.0), ['a', 'b', 'c', 'd'], 'Spread')


def test_nothing_missing_is_silent(caplog):
    with caplog.at_level(logging.ERROR):
        align_features(_row(a=1.0, b=2.0), ['a', 'b'])
    assert not caplog.records
