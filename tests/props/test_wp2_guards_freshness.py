"""WP2: data-gap guards (pure), their logging in rows.py, and freshness.missing_stat_tables on a toy sqlite database."""
import sqlite3

import numpy as np
import pandas as pd

from props.features import guards, rows
from props.live import freshness


def test_table_coverage_lists_games_missing_from_each_table():
    got = guards.table_coverage(['0022600001', '0022600002', '0022600003'],
                                {'basic': ['0022600001', '0022600002', '0022600003'],
                                 'track': ['0022600001', '0022600003', '0022699999'],        # extra game in the table is not a gap
                                 'usage': []})
    assert got == {'track': ['0022600002'], 'usage': ['0022600001', '0022600002', '0022600003']}
    assert guards.table_coverage([], {'basic': ['x']}) == {}
    assert guards.table_coverage([1, 2], {'t': ['1', '2']}) == {}                           # ids compare as strings


def test_played_null_report_counts_only_played_rows_with_nulls():
    df = pd.DataFrame({'lbl_played': [1, 1, 1, 0, 0], 'touches': [5, np.nan, np.nan, np.nan, np.nan], 'passes': [1, 2, 3, np.nan, 4],
                       'usage_pct': [.1, .2, .3, np.nan, np.nan]})
    assert guards.played_null_report(df, ['touches', 'passes', 'usage_pct', 'not_a_column']) == {'touches': 2}


def _frame(played_by_game):
    recs = []
    for g, flags in played_by_game.items():
        for i, p in enumerate(flags):
            recs.append({'game_id': g, 'player_id': i, 'lbl_played': p, 'touches': np.nan if p else np.nan, 'points': 1.0 if p else np.nan})
    return pd.DataFrame(recs)


def test_rows_warns_but_does_not_raise_for_missing_tables_unplayed_games_and_null_stats(caplog):
    df = _frame({'0022600001': [1, 1, 0], 'G_EMPTY': [0, 0, 0]})
    with caplog.at_level('WARNING'):
        rows._warn_on_incomplete_tables(df, {'player_basic_stats': pd.DataFrame({'game_id': ['0022600001']}),
                                             'player_track_stats_v3': pd.DataFrame({'game_id': ['0022600001', 'G_EMPTY']})})
        rows._warn_on_unplayed_games_and_null_stats(df)
    text = caplog.text
    assert 'missing from player_basic_stats: G_EMPTY' in text
    assert 'player_track_stats_v3' not in text                                                  # the track table is complete: no gap logged
    assert 'zero played rows' in text and 'G_EMPTY' in text
    assert "played rows with null stats {'touches': 2}" in text and '0022600001' in text
    assert all(r.levelname == 'WARNING' for r in caplog.records)


def test_rows_is_silent_when_the_data_is_complete(caplog):
    df = _frame({'0022600001': [1, 0]})
    df['touches'] = np.where(df['lbl_played'] == 1, 3.0, np.nan)
    with caplog.at_level('WARNING'):
        rows._warn_on_incomplete_tables(df, {'player_basic_stats': df[['game_id']], 'player_track_stats_v3': df[['game_id']]})
        rows._warn_on_unplayed_games_and_null_stats(df)
    assert caplog.text == ''


def _toy_db(path, games_by_table):
    conn = sqlite3.connect(path)
    for t, games in games_by_table.items():
        conn.execute(f'CREATE TABLE "{t}" (gameId TEXT, personId INTEGER)')
        conn.executemany(f'INSERT INTO "{t}" VALUES (?, ?)', [(g, p) for g in games for p in (1, 2)])
    conn.commit()
    conn.close()


def test_missing_stat_tables_reports_the_gap_the_roster_based_check_cannot_see(tmp_path):
    db = (tmp_path / 'toy.db').as_posix()
    base = ['0022600001', '0022600002', '0022600003', '0042600001']
    _toy_db(db, {'player_advanced_stats_v3': base + ['0022500999', '0012600001'],        # last season's game and a preseason game: ignored
                 'player_basic_stats': base,
                 'player_track_stats_v3': [g for g in base if g != '0022600002'],
                 'player_usage_stats_v3': [g for g in base if g != '0042600001']})
    assert freshness.missing_stat_tables(db, '2026-27') == [('0022600002', 'player_track_stats_v3'), ('0042600001', 'player_usage_stats_v3')]
    assert freshness.missing_stat_tables(db, '2025-26') == [                                 # the game only the roster table has is a gap for the other three
        ('0022500999', 'player_basic_stats'), ('0022500999', 'player_track_stats_v3'), ('0022500999', 'player_usage_stats_v3')]
    assert freshness.missing_stat_tables(db, '2024-25') == []
    # a game that only the basic table has is also a gap for the other three
    _toy_db((tmp_path / 'toy2.db').as_posix(), {'player_advanced_stats_v3': base, 'player_basic_stats': base + ['0022600004'],
                                        'player_track_stats_v3': base, 'player_usage_stats_v3': base})
    assert freshness.missing_stat_tables((tmp_path / 'toy2.db').as_posix(), '2026-27') == [
        ('0022600004', 'player_advanced_stats_v3'), ('0022600004', 'player_track_stats_v3'), ('0022600004', 'player_usage_stats_v3')]
