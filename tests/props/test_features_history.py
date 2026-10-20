import numpy as np
import pandas as pd
import pytest

from conftest import DB_PATH
from props.features import decay, history, rows


def naive_before(values, groups, played, alpha):
    out = np.zeros(len(values))
    for i in range(len(values)):
        total, k = 0.0, 0
        for j in range(i - 1, -1, -1):                      # walk back through earlier rows of the same group
            if groups[j] != groups[i]:
                break
            if played[j]:
                total += alpha ** k * (0.0 if np.isnan(values[j]) else values[j])
                k += 1
        out[i] = total
    return out


def test_decayed_before_hand_example():
    got = decay.decayed_before([10, 99, 20, 30, 5, 7], [1, 1, 1, 1, 2, 2], [1, 0, 1, 1, 1, 1], 0.5)
    # group 1: row0 sees nothing; the DNP row sees 10; row2 sees 10 (the DNP row adds nothing); row3 sees 20 + 0.5*10
    assert got.tolist() == [0.0, 10.0, 10.0, 25.0, 0.0, 5.0]


def test_decayed_before_equals_a_naive_loop_and_never_sees_the_row_itself():
    rng = np.random.default_rng(0)
    n = 300
    groups = np.sort(rng.integers(0, 12, n))
    played = rng.random(n) < 0.8
    values = rng.normal(10, 3, n)
    values[rng.random(n) < 0.05] = np.nan
    a = decay.alpha_for(7)
    assert decay.decayed_before(values, groups, played, a) == pytest.approx(naive_before(values, groups, played, a))
    # perturbing a row's own value and everything after it must not change its state
    t = 150
    v2 = values.copy()
    v2[t:] = rng.normal(500, 50, n - t)
    p2 = played.copy()
    p2[t:] = ~p2[t:]
    assert decay.decayed_before(values, groups, played, a)[:t + 1] == pytest.approx(decay.decayed_before(v2, groups, p2, a)[:t + 1])


def test_half_life_halves_the_weight():
    a = decay.alpha_for(5)
    assert a ** 5 == pytest.approx(0.5)
    assert decay.effective_games(a, 10_000) == pytest.approx(1 / (1 - a))


# ---- on the real database (skipped in CI) --------------------------------------------------------------------------
pytestmark_db = pytest.mark.db

LABEL_COLS = ['minutes', 'points', 'rebounds', 'assists', 'fg3m', 'fg3a', 'fga', 'fgm', 'ftm', 'fta', 'oreb', 'dreb', 'tov', 'pf', 'stl',
              'blk', 'reb_chances', 'reb_chances_off', 'reb_chances_def', 'touches', 'passes', 'secondary_ast', 'ft_ast',
              'contested_fga', 'usage_pct', 'lbl_team_margin', 'y_points', 'y_rebounds', 'y_assists', 'y_threes', 'y_pra']


@pytest.fixture(scope='module')
def hist():
    return rows.load_history(str(DB_PATH), first_season='2017-18')           # a shorter history keeps the test quick


@pytest.fixture(scope='module')
def pos():
    return history.position_groups(str(DB_PATH))


@pytest.fixture(scope='module')
def base_features(hist, pos):
    return history.add_history_features(hist, pos, first_dataset_season='2019-20')


def _feature_cols(f):
    return [c for c in f.columns if c not in ('game_id', 'player_id', 'team_id', 'opp_team_id', 'game_date', 'season', 'game_type',
                                              'is_home', 'pos_group')]


@pytest.mark.db
def test_features_on_a_date_ignore_everything_from_that_date_on(hist, pos, base_features):
    """Mask every outcome dated on or after D (stats, minutes, starter flag, played flag). Features of D's games must not move."""
    D = '2023-01-20'
    masked = hist.copy()
    sel = masked['game_date'] >= D
    for c in LABEL_COLS:
        if c in masked.columns:
            masked.loc[sel, c] = np.nan
    masked.loc[sel, ['lbl_played', 'lbl_started']] = 0
    again = history.add_history_features(masked, pos, first_dataset_season='2019-20')
    key = ['game_id', 'player_id']
    a = base_features[base_features['game_date'] == D].set_index(key)
    b = again[again['game_date'] == D].set_index(key).loc[a.index]
    assert len(a) > 100
    cols = _feature_cols(a)
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9, atol=1e-9)


@pytest.mark.db
def test_shuffling_the_future_does_not_change_the_past(hist, pos, base_features):
    D = '2023-03-01'
    rng = np.random.default_rng(1)
    shuffled = hist.copy()
    fut = np.flatnonzero((shuffled['game_date'] >= D).to_numpy())
    perm = rng.permutation(fut)
    for c in LABEL_COLS + ['lbl_played', 'lbl_started']:
        if c in shuffled.columns:
            shuffled.loc[shuffled.index[fut], c] = shuffled[c].to_numpy()[perm]
    again = history.add_history_features(shuffled, pos, first_dataset_season='2019-20')
    key = ['game_id', 'player_id']
    a = base_features[base_features['game_date'] < D].set_index(key)
    b = again[again['game_date'] < D].set_index(key).loc[a.index]
    cols = _feature_cols(a)
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9, atol=1e-9)


@pytest.mark.db
def test_no_feature_is_a_label_and_the_dictionary_prefixes_hold(base_features):
    cols = _feature_cols(base_features)
    assert not [c for c in cols if c.startswith(('y_', 'lbl_'))]
    assert all(c.startswith(('m_', 'role_', 'rest_', 'r_', 'c_')) for c in cols), [c for c in cols if not c.startswith(('m_', 'role_', 'rest_', 'r_', 'c_'))]


@pytest.mark.db
def test_a_rookies_first_game_uses_only_the_league_prior(base_features):
    first = base_features[base_features['m_games_eff_h15'] == 0]
    assert len(first) > 100
    assert first['m_min_h15'].isna().all()                                           # no history: minutes unknown, not zero
    assert first['r_points_h15'].between(0.2, 0.7).all()                              # but rates fall back to the league prior
