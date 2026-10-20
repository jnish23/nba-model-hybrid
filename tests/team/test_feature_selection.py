import numpy as np
import pandas as pd

from team.train_baseline import BaselineModel


def _frame(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 12))
    cols = {}
    for j in range(12):
        cols[f'diff_f{j:02d}'] = base[:, j]
    # redundant copies (correlation ~1) with tied importance: the cluster representative must be stable
    for j in range(4):
        cols[f'diff_copy{j}a'] = base[:, j] + rng.normal(scale=1e-3, size=n)
        cols[f'diff_copy{j}b'] = base[:, j] + rng.normal(scale=1e-3, size=n)
    for j in range(30):
        cols[f'diff_noise{j:02d}'] = rng.normal(size=n)
    df = pd.DataFrame(cols)
    margin = 4 * base[:, 0] + 2 * base[:, 1] + rng.normal(scale=6, size=n)
    df['target_point_diff'] = margin
    df['target_home_win'] = (margin > 0).astype(int)
    df['target_total_pts'] = 215 + 3 * base[:, 2] + rng.normal(scale=10, size=n)
    return df


def _selector(df):
    m = BaselineModel('unused.csv')
    m.features_diff = [c for c in df.columns if c.startswith('diff_')]
    m.features_sep = list(m.features_diff)
    return m


def test_selection_is_identical_across_runs():
    df = _frame()
    a = _selector(df)._select_features_from_subset(df, 'target_point_diff')
    b = _selector(df)._select_features_from_subset(df, 'target_point_diff')
    assert a == b and len(a) >= 20


def test_selection_does_not_depend_on_column_order():
    df = _frame()
    ref = _selector(df)._select_features_from_subset(df, 'target_home_win')
    rng = np.random.default_rng(5)
    for _ in range(3):
        cols = list(df.columns)
        rng.shuffle(cols)
        shuffled = df[cols]
        got = _selector(shuffled)._select_features_from_subset(shuffled, 'target_home_win')
        assert got == ref


def test_ties_are_broken_by_name_and_clusters_keep_one_member():
    df = _frame()
    sel = _selector(df)._select_features_from_subset(df, 'target_point_diff')
    for j in range(4):
        members = [c for c in sel if c.startswith(f'diff_copy{j}') or c == f'diff_f{j:02d}']
        assert len(members) <= 1, members              # near-duplicates collapse into a single feature
