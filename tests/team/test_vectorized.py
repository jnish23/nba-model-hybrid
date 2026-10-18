"""The vectorized implementations must reproduce the original (row-by-row / transform(lambda)) ones."""
import numpy as np
import pandas as pd
import pytest

from team import team_features_v2 as tf
from core import injury_report_features as irf
from core.bpm_builder import BPMBuilder


# ------------------------------------------------------------------ win streak / home record

@pytest.mark.parametrize('seed', range(5))
def test_vectorized_win_streak_equals_the_loop(seed):
    rng = np.random.default_rng(seed)
    win = pd.Series((rng.random(120) < 0.55).astype(int), index=np.arange(120) * 3)
    pd.testing.assert_series_equal(tf.win_streak(win), tf._win_streak_loop(win))
    with_nan = pd.concat([win, pd.Series([np.nan], index=[999])])                  # live: the unplayed game
    pd.testing.assert_series_equal(tf.win_streak(with_nan), tf._win_streak_loop(with_nan))
    interior = win.astype(float).copy()
    interior.iloc[[7, 8, 50]] = np.nan                                             # falls back to the loop
    pd.testing.assert_series_equal(tf.win_streak(interior), tf._win_streak_loop(interior))


def test_win_streak_edge_cases():
    assert tf.win_streak(pd.Series([], dtype=float)).tolist() == []
    assert tf.win_streak(pd.Series([1])).tolist() == [0]
    assert tf.win_streak(pd.Series([0, 0, 0, 1, 1])).tolist() == [0, -1, -2, -3, 1]


def test_home_win_pct_by_team_equals_the_per_team_helper():
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({'teamId': np.repeat([1, 2, 3, 4], n // 4)})
    df['is_home'] = rng.integers(0, 2, n)
    df['home_win'] = df['is_home'] * rng.integers(0, 2, n)
    got = tf.home_win_pct_20g_by_team(df['home_win'], df['is_home'], df['teamId'])
    ref = df.groupby('teamId', group_keys=False).apply(
        lambda g: tf.home_win_pct_20g(g['home_win'], g['is_home']), include_groups=False)
    pd.testing.assert_series_equal(got, ref.reindex(df.index), check_names=False)


def test_groupby_ewm_equals_transform_lambda():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({'teamId': np.repeat([1, 2, 3], 80), 'a': rng.normal(size=240), 'b': rng.normal(size=240)})
    df.loc[[5, 100], 'a'] = np.nan
    for span in (5, 20):
        ref = df.groupby('teamId')[['a', 'b']].transform(lambda x: x.ewm(span=span).mean().shift(1))
        new = (df.groupby('teamId')[['a', 'b']].ewm(span=span).mean().reset_index(level=0, drop=True)
               .reindex(df.index).groupby(df['teamId']).shift(1))
        pd.testing.assert_frame_equal(new, ref, check_exact=True)


# ------------------------------------------------------------------ name normalisation

def test_map_unique_equals_map_and_calls_func_once_per_value():
    calls = []

    def f(x):
        calls.append(x)
        return irf.report_name_key(x)

    names = pd.Series(['Williams III, Robert', 'SamuelsJr.,Jermaine', None, 'Williams III, Robert'] * 500
                      + [np.nan], index=np.arange(2001) * 2)
    got = irf.map_unique(names, f)
    ref = names.map(irf.report_name_key)
    assert got.index.equals(names.index)
    assert [None if pd.isna(x) else x for x in got] == [None if pd.isna(x) else x for x in ref]
    assert len(calls) <= 3                              # 2 distinct names (+ nothing for the NaNs)


def test_map_unique_keeps_numeric_dtype_with_missing_ids():
    lookup = {'a': 1, 'b': 2}
    s = pd.Series(['a', 'b', 'zzz', 'a'])
    got, ref = irf.map_unique(s, lookup.get), s.map(lookup.get)
    pd.testing.assert_series_equal(got, ref)


# ------------------------------------------------------------------ BPM

def test_vectorized_position_coefficients_equal_the_per_row_apply():
    b = BPMBuilder('unused')
    labels = pd.Series(['Guard', 'Forward', 'Center', 'Guard-Forward', 'Forward-Center', 'Center-Forward', '', None, np.nan] * 40)
    scalar = labels.map({l: b._map_position(l) for l in labels.unique()}).astype(float)
    ref_scalar = labels.apply(b._map_position)
    pd.testing.assert_series_equal(scalar, ref_scalar)
    for key, tup in b.coeffs_pos.items():
        if isinstance(tup, tuple):
            slope = (tup[1] - tup[0]) / 4.0
            new = tup[0] + slope * (scalar - 1.0)
            old = scalar.apply(lambda x: tup[0] + ((tup[1] - tup[0]) / 4.0) * (x - 1.0))
            np.testing.assert_allclose(new, old, rtol=1e-14)


def test_bpm_ewma_and_cumulative_minutes_equal_the_transform_versions():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({'personId': np.repeat([10, 11, 12], 70), 'minutes': rng.integers(0, 40, 210).astype(float),
                       'points': rng.integers(0, 35, 210).astype(float)}).sample(frac=1, random_state=1)
    df = df.sort_values(['personId']).reset_index(drop=True)
    cols = ['minutes', 'points']
    ref = df.groupby('personId')[cols].transform(lambda x: x.shift(1).ewm(span=40, adjust=False).mean())
    new = (df.groupby('personId')[cols].shift(1).groupby(df['personId']).ewm(span=40, adjust=False).mean()
           .reset_index(level=0, drop=True).reindex(df.index))
    pd.testing.assert_frame_equal(new, ref, check_exact=True)
    ref_cum = df.groupby('personId')['minutes'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    new_cum = df.groupby('personId')['minutes'].shift(1).groupby(df['personId']).cumsum().fillna(0)
    pd.testing.assert_series_equal(new_cum, ref_cum)


@pytest.mark.db
def test_real_history_outputs_unchanged_by_vectorization():
    """Spot-check on the real database against the values stored in the training set builder's
    previous implementation: the roster EWMA of the latest games is reproduced to float precision."""
    import sqlite3
    from conftest import DB_PATH
    from core.player_features import PlayerFeatureBuilder
    with sqlite3.connect(DB_PATH) as conn:
        seasons = sorted(pd.read_sql("SELECT DISTINCT seasonYear FROM team_game_full", conn)['seasonYear'])
    feats = PlayerFeatureBuilder(str(DB_PATH)).build_roster_features(seasons)
    assert feats['roster_bpm'].notna().all() and 0.5 < feats['roster_bpm'].std() < 3
