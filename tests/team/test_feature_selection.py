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


# ------------------------------------------------------------------ audit: gain importance, stratified null, odds-free proxy selection

def _era_frame(seed=0, n_seasons=8, per=250):
    """Totals drift upward by 2.5 points per season; `diff_era` is a pure proxy for the season, `diff_signal` is real."""
    rng = np.random.default_rng(seed)
    n = n_seasons * per
    season = np.repeat(np.arange(n_seasons), per)
    cols = {'diff_era': season + rng.normal(scale=0.3, size=n), 'diff_signal': rng.normal(size=n)}
    for j in range(25):
        cols[f'diff_noise{j:02d}'] = rng.normal(size=n)
    df = pd.DataFrame(cols)
    df['seasonYear'] = [f'20{13 + s}-{14 + s}' for s in season]
    df['target_total_pts'] = 200 + 2.5 * season + 4 * df['diff_signal'] + rng.normal(scale=8, size=n)
    return df


def test_null_importance_shuffles_within_seasons_so_an_era_proxy_is_not_credited_for_the_drift():
    df = _era_frame()
    sel = _selector(df)
    stratified = sel._select_features_from_subset(df, 'target_total_pts', n_min=1)
    unstratified = sel._select_features_from_subset(df, 'target_total_pts', n_min=1, stratify=False)
    assert 'diff_signal' in stratified
    assert 'diff_era' in unstratified                  # shuffling across seasons destroys the drift, so the proxy "beats" its null
    assert 'diff_era' not in stratified                # within a season the proxy carries no information about the total


def test_explicit_groups_override_the_season_column():
    df = _era_frame()
    sel = _selector(df)
    by_season = sel._select_features_from_subset(df.drop(columns='seasonYear'), 'target_total_pts', n_min=1,
                                                 groups=df['seasonYear'])
    assert 'diff_era' not in by_season and 'diff_signal' in by_season


def test_stage_two_threshold_is_stable_and_rejects_most_noise():
    df = _frame(n=2000)
    sel = _selector(df)
    for seed in (1, 2, 3):
        got = sel._select_features_from_subset(df, 'target_point_diff', n_min=1, seed=seed)
        for j in (0, 1):                                                       # one member of each real cluster (f, copya, copyb)
            assert any(c in (f'diff_f{j:02d}', f'diff_copy{j}a', f'diff_copy{j}b') for c in got), got
        assert sum(c.startswith('diff_noise') for c in got) <= 8, got        # 30 noise columns; ~10% pass a 1.28-sigma bar by chance


def test_legacy_flag_restores_the_split_count_and_percentile_rule():
    df = _frame()
    sel = _selector(df)
    legacy = sel._select_features_from_subset(df, 'target_point_diff', legacy=True)
    explicit = sel._select_features_from_subset(df, 'target_point_diff', importance_type='split', stratify=False, n_permutations=10, legacy=True)
    assert legacy == explicit and len(legacy) >= 20


def test_select_market_features_ranks_what_the_line_misses_and_appends_the_odds():
    rng = np.random.default_rng(3)
    n = 3000
    X = rng.normal(size=(n, 10))
    cols = {f'diff_f{j}': X[:, j] for j in range(10)}
    for j in range(25):
        cols[f'diff_noise{j:02d}'] = rng.normal(size=n)
    df = pd.DataFrame(cols)
    df['odds_home_spread'] = -(4 * X[:, 0] + 2 * X[:, 1])                       # the market already knows f0 and f1 (f0 is also ~0.9 correlated with the line)
    df['odds_total'] = 215 + 3 * X[:, 2]
    df['odds_home_ml'] = -110.0 + 0 * X[:, 0] + rng.normal(size=n)
    df['target_point_diff'] = -df['odds_home_spread'] + 3 * X[:, 5] + rng.normal(scale=6, size=n)   # f5 is what the line misses
    df['target_total_pts'] = df['odds_total'] + 2 * X[:, 6] + rng.normal(scale=8, size=n)
    df['target_home_win'] = (df['target_point_diff'] > 0).astype(int)
    df.loc[:49, 'odds_home_spread'] = np.nan                                      # games without a line have no residual
    sel = _selector(df)
    sel.features_diff = [c for c in df.columns if c.startswith(('diff_', 'odds_'))]
    sel.features_sep = list(sel.features_diff)
    raw = sel._select_features_from_subset(df, 'target_point_diff', n_min=1)
    got = sel.select_market_features(df, n_min=1, min_rows=100)
    assert 'diff_f1' in raw and 'diff_f1' not in got['spread']                    # raw margin credits f1 for what the line already prices
    assert 'diff_f5' in got['spread'] and 'diff_f6' in got['total']
    for k in ('ml', 'spread', 'total'):
        assert {'odds_home_spread', 'odds_total', 'odds_home_ml'} <= set(got[k])
        assert len(got[k]) == len(set(got[k]))
    assert got['ml'] == got['spread']


def test_averaging_the_bagged_importances_makes_the_lists_steadier_across_seeds():
    df = _frame(n=2000)
    sel = _selector(df)

    def mean_jaccard(k):
        lists = [set(sel._select_features_from_subset(df, 'target_point_diff', n_min=1, seed=s, n_importance_fits=k)) for s in (1, 2, 3, 4)]
        return float(np.mean([len(a & b) / len(a | b) for i, a in enumerate(lists) for b in lists[i + 1:]]))

    assert mean_jaccard(5) > mean_jaccard(1)                 # one bagged fit picks a different near-duplicate for each seed
