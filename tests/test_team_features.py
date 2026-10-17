import numpy as np
import pandas as pd
import pytest

from models import team_features_v2 as tf
from models.team_features_v2 import TeamFeatureBuilder


def _batch(team_db):
    b = TeamFeatureBuilder(team_db)
    return b, b.build_entire_history()


def test_momentum_features_exist_in_batch_build(team_db):
    """Regression: WL/MATCHUP are upper-case in team_game_full, so the lower-case checks used to skip them."""
    _, full = _batch(team_db)
    for col in ('win_pct_last10', 'win_streak', 'home_win_pct_20g'):
        assert col in full.columns
        assert full[col].notna().any()


def test_batch_and_live_features_agree_for_every_game(team_db):
    """For each historical game, build_features() on the history before that date == the batch row."""
    builder, full = _batch(team_db)
    checked = 0
    for _, row in full.iterrows():
        live = builder.build_features(int(row['teamId']), row['gameDate'].strftime('%Y-%m-%d'),
                                      row['seasonYear'], game_id=row['gameId'])
        if not live:                      # first game of the log: no history
            continue
        for k, v in live.items():
            assert k in full.columns, f"live-only feature {k}"
            b = row[k]
            if pd.isna(v) and pd.isna(b):
                continue
            np.testing.assert_allclose(float(v), float(b), rtol=1e-9, atol=1e-9,
                                       err_msg=f"{k} team {row['teamId']} {row['gameDate']}")
        checked += 1
    assert checked > 40


def test_live_feature_set_covers_batch_feature_set(team_db):
    builder, full = _batch(team_db)
    r = full.iloc[-1]
    live = builder.build_features(int(r['teamId']), r['gameDate'].strftime('%Y-%m-%d'), r['seasonYear'], r['gameId'])
    meta = {'teamId', 'gameDate', 'seasonYear', 'gameId'}
    missing = set(full.columns) - meta - set(live)
    assert not missing, f"batch features missing from the live path: {sorted(missing)[:5]}"


def test_days_rest_is_not_capped_after_offseason(team_db):
    builder, full = _batch(team_db)
    first_of_new_season = full[(full['seasonYear'] == '2025-26')].groupby('teamId').head(1).iloc[0]
    assert first_of_new_season['days_rest'] > 7
    live = builder.build_features(int(first_of_new_season['teamId']),
                                  first_of_new_season['gameDate'].strftime('%Y-%m-%d'), '2025-26')
    assert live['days_rest'] == first_of_new_season['days_rest']


def test_get_latest_team_features_uses_one_history_load_and_matches_per_team(team_db):
    builder = TeamFeatureBuilder(team_db)
    date = '2025-12-20'
    latest = builder.get_latest_team_features(date)
    for tid in (1, 2):
        single = builder.build_features(tid, date, '2025-26')
        assert set(latest[tid]) == set(single)
        for k, v in single.items():
            np.testing.assert_allclose(latest[tid][k], v, equal_nan=True)


def test_momentum_helpers_known_values():
    win = pd.Series([1, 1, 0, 0, 0, 1.0])
    assert tf.win_streak(win).tolist() == [0, 1, 2, -1, -2, -3]
    pct = tf.win_pct_last10(win)
    assert np.isnan(pct.iloc[2]) and pct.iloc[3] == pytest.approx(2 / 3)


def test_missing_exact_opponent_columns_raise_instead_of_summing_positions(tmp_path):
    """The positional C/F/G buckets overlap; summing them inflates opponent totals ~22%."""
    import sqlite3
    from conftest import make_game_log
    df = make_game_log()
    exact = [c for c in df.columns if c.startswith('OPP_')]
    df = df.drop(columns=exact)
    for stat in ('FGA', 'PTS'):                       # an old table: positional columns only
        for pos in 'CFG':
            df[f'OPP_{stat}_{pos}'] = 10.0
    path = tmp_path / 'old.db'
    with sqlite3.connect(path) as conn:
        df.to_sql('team_game_full', conn, index=False)
        df[['teamId', 'teamTricode']].drop_duplicates().assign(teamName=lambda d: 'Team ' + d['teamTricode']) \
            .to_sql('team_basic_stats', conn, index=False)
    builder = TeamFeatureBuilder(str(path))
    with pytest.raises(ValueError, match='create_team_full_table'):
        builder.build_entire_history()
    with pytest.raises(ValueError, match='create_team_full_table'):
        builder.build_features(1, '2025-12-01', '2025-26')


def test_create_team_full_table_delegates_to_the_maintained_rebuild(monkeypatch):
    import create_team_full_table as ctft
    from update_database import NBADatabaseSync
    calls = []
    monkeypatch.setattr(NBADatabaseSync, '__init__', lambda self, db_path, *a, **k: calls.append(db_path))
    monkeypatch.setattr(NBADatabaseSync, 'rebuild_team_game_full', lambda self: calls.append('rebuilt'))
    ctft.create_full_table('some.db')
    assert calls == ['some.db', 'rebuilt']


@pytest.mark.db
def test_real_table_has_exact_opponent_columns_and_they_differ_from_the_position_sum():
    import sqlite3
    import pandas as pd
    from conftest import DB_PATH
    with sqlite3.connect(DB_PATH) as conn:
        d = pd.read_sql("SELECT OPP_FGA, OPP_FGA_C + OPP_FGA_F + OPP_FGA_G AS s FROM team_game_full "
                        "WHERE seasonYear >= '2023-24'", conn).dropna()
    assert 1.15 < d['s'].sum() / d['OPP_FGA'].sum() < 1.30
