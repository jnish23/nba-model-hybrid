import sqlite3

import numpy as np
import pandas as pd
import pytest

from conftest import DB_PATH
from props import odds_store, schedule
from props.features import market_env as me


def build_store(path):
    conn = odds_store.connect(path)
    conn.executescript(schedule.REF_SCHEMA)
    snaps = [('2026-03-17T22:10:00Z', 'hist_T-240', 'e1'), ('2026-03-18T01:05:00Z', 'hist_T-5', 'e1')]
    for ts, mode, eid in snaps:
        for book, hs, tot, hml, aml in (('dk', -5.5, 221.5, -220, 180), ('fd', -5.0, 222.0, -210, 175), ('mgm', -6.0, 221.0, -230, 190),
                                        ('br', -5.5, 221.5, -225, 185)):
            conn.execute("INSERT INTO team_lines (snapshot_ts_utc, event_id_vendor, book, home_ml, away_ml, home_spread, total) VALUES (?,?,?,?,?,?,?)",
                         (ts, eid, book, hml, aml, hs + (0.5 if mode == 'hist_T-5' else 0.0), tot))
        conn.execute("INSERT INTO snapshot_log (snapshot_ts_utc, event_id_vendor, mode, minutes_to_tip, n_prop_rows) VALUES (?,?,?,?,?)", (ts, eid, mode, 0, 1))
        conn.execute("INSERT OR REPLACE INTO snapshot_ref VALUES (?,?,?,?)", (ts, eid, 244.0 if mode == 'hist_T-240' else 9.0, 0))
    conn.execute("INSERT INTO event_ref VALUES ('e1','0022500998','x','y',NULL,NULL,0)")
    conn.commit()
    conn.close()


def test_snapshot_consensus_is_the_median_across_books_at_the_requested_grid_point(tmp_path):
    build_store(tmp_path / 'p.db')
    env = me.snapshot_env(str(tmp_path / 'p.db')).set_index('minutes_before')
    assert env.loc[240, 'home_spread'] == -5.5 and env.loc[240, 'total'] == 221.5 and env.loc[240, 'n_books'] == 4
    assert env.loc[5, 'home_spread'] == -5.0                                    # same books moved half a point toward the dog
    assert env.loc[240, 'spread_dispersion'] == pytest.approx(1.0)
    assert 0.6 < env.loc[240, 'home_win_prob'] < 0.7 and (env['env_source'] == 1).all()


def test_too_few_books_gives_no_consensus(tmp_path):
    build_store(tmp_path / 'p.db')
    c = sqlite3.connect(tmp_path / 'p.db')
    c.execute("DELETE FROM team_lines WHERE book IN ('fd','mgm')")
    c.commit()
    c.close()
    env = me.snapshot_env(str(tmp_path / 'p.db'))
    assert env[['home_spread', 'total', 'home_win_prob']].isna().all().all() and (env['n_books'] == 2).all()


def test_sbr_proxy_covers_open_and_close_only_and_snapshots_win():
    clean = pd.DataFrame({'gameId': ['0021900001'], 'open_home_spread': [-3.0], 'open_total': [220.0], 'open_home_prob': [0.6],
                          'open_n_books_spread': [5], 'close_home_spread': [-4.0], 'close_total': [222.0], 'close_home_prob': [0.65],
                          'close_n_books_spread': [6]})
    p = me.sbr_proxy(clean).set_index('minutes_before')
    assert sorted(p.index) == [5, 240] and p.loc[240, 'env_source'] == 2 and p.loc[5, 'env_source'] == 3
    assert p.loc[240, 'home_spread'] == -3.0 and p.loc[5, 'total'] == 222.0
    snap = pd.DataFrame({'game_id': ['0021900001'], 'minutes_before': [5], 'home_spread': [-9.9], 'total': [200.0], 'home_win_prob': [0.9],
                         'n_books': [8], 'spread_dispersion': [0.5], 'env_source': [1]})
    both = me.combine(snap, me.sbr_proxy(clean))
    five = both[both.minutes_before == 5].iloc[0]
    assert five['home_spread'] == -9.9 and five['env_source'] == 1                # the snapshot, not the proxy
    assert len(both[both.minutes_before == 240]) == 1                              # the proxy fills the gap


def test_team_view_signs_and_implied_totals():
    env = pd.DataFrame({'game_id': ['g'], 'minutes_before': [240], 'home_spread': [-5.5], 'total': [221.5], 'home_win_prob': [0.7],
                        'n_books': [6], 'spread_dispersion': [1.0], 'env_source': [1]})
    rows = pd.DataFrame({'game_id': ['g', 'g'], 'player_id': [1, 2], 'is_home': [1, 0]})
    v = me.team_view(rows, env, 240).set_index('player_id')
    assert v.loc[1, 'env_team_spread'] == -5.5 and v.loc[2, 'env_team_spread'] == 5.5
    assert v.loc[1, 'env_team_total'] == 113.5 and v.loc[1, 'env_opp_total'] == 108.0
    assert v.loc[2, 'env_team_total'] == 108.0 and v.loc[1, 'env_team_total'] + v.loc[1, 'env_opp_total'] == 221.5
    assert v.loc[1, 'env_team_win_prob'] == pytest.approx(0.7) and v.loc[2, 'env_team_win_prob'] == pytest.approx(0.3)
    assert v.loc[1, 'env_team_big_win_prob'] > v.loc[2, 'env_team_big_win_prob']
    assert v.loc[1, 'env_blowout_prob'] == pytest.approx(v.loc[2, 'env_blowout_prob'])          # symmetric
    pk = me.team_view(pd.DataFrame({'game_id': ['g'], 'player_id': [3], 'is_home': [1]}),
                      env.assign(home_spread=0.0), 240)
    assert pk.loc[0, 'env_blowout_prob'] == pytest.approx(2 * (1 - 0.8944), abs=0.002)           # P(|N(0,12)| >= 15) = 0.211


@pytest.mark.db
@pytest.mark.skipif(not (DB_PATH.parent / 'props_odds.db').exists(), reason='props store not available')
def test_no_snapshot_used_for_a_decision_time_is_later_than_that_time():
    """Every snapshot behind env features at T-X must have minutes-to-reference-tip >= X (it was taken at or before T)."""
    with sqlite3.connect(f"file:{DB_PATH.parent / 'props_odds.db'}?mode=ro", uri=True, timeout=120) as conn:
        d = pd.read_sql("""SELECT l.mode, s.minutes_to_ref FROM snapshot_log l JOIN snapshot_ref s
                           ON s.snapshot_ts_utc = l.snapshot_ts_utc AND s.event_id_vendor = l.event_id_vendor
                           WHERE l.mode IN ('hist_T-240', 'hist_T-60', 'hist_T-5') AND s.in_game = 0""", conn)
    d['x'] = d['mode'].str.replace('hist_T-', '', regex=False).astype(int)
    late = d[d['minutes_to_ref'] < d['x'] - 1.0]                       # one minute of tolerance for the 5-minute grid and reference-tip rounding
    assert len(d) > 10000
    assert len(late) / len(d) < 0.001, late.groupby('x').size().to_dict()          # only vendor tip-time revisions may breach, and rarely


def test_a_snapshot_taken_after_T_is_never_used(tmp_path):
    build_store(tmp_path / 'p.db')
    c = sqlite3.connect(tmp_path / 'p.db')
    c.execute("UPDATE snapshot_ref SET minutes_to_ref = 3.0 WHERE snapshot_ts_utc = '2026-03-17T22:10:00Z'")     # the T-240 snapshot is really T-3
    c.commit()
    c.close()
    env = me.snapshot_env(str(tmp_path / 'p.db'))
    assert 240 not in set(env['minutes_before']) and 5 in set(env['minutes_before'])
