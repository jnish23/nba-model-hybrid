import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location('backfill_props_clv', Path(__file__).resolve().parents[2] / 'scripts' / 'backfill_props_clv.py')
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

TIP = '2026-11-02T00:10:00Z'


def _cands(*rows):
    return pd.DataFrame(rows, columns=['game_id', 'event_id_vendor', 'snapshot_ts_utc', 'mode', 'n_prop_rows'])


def test_live_snapshot_closest_to_tip_wins_and_hist_is_the_fallback():
    c = _cands(('g1', 'e1', '2026-11-01T23:10:00Z', 'slow', 50),      # 60 min: too early
               ('g1', 'e1', '2026-11-01T23:40:00Z', 'slow', 50),      # 30 min
               ('g1', 'e1', '2026-11-02T00:00:00Z', 'close', 50),     # 10 min
               ('g1', 'e1', '2026-11-02T00:05:00Z', 'hist_T-5', 50))  # 5 min, hist
    ch = bf.ordered_choices(c, TIP)
    assert [(s, round(m)) for s, _, _, m in ch] == [('live', 10), ('live', 30), ('hist', 5)]


def test_snapshots_at_or_after_tip_or_too_early_are_rejected():
    c = _cands(('g1', 'e1', '2026-11-02T00:10:00Z', 'close', 50), ('g1', 'e1', '2026-11-02T00:20:00Z', 'slow', 50), ('g1', 'e1', '2026-11-01T23:00:00Z', 'hist_T-5', 50))
    assert bf.ordered_choices(c, TIP) == []


def _cons(game_id, player, stat, line, fair):
    return pd.DataFrame({'game_id': game_id, 'player_id': player, 'minutes_before': 5, 'stat': stat, 'consensus_line': line, 'consensus_fair_over': fair}, index=[0])


def _bets():
    base = {'game_id': 'g1', 'game_date': '2026-11-01', 'tip_ref_utc': TIP, 'stat': 'points', 'minutes_before': 240}
    return pd.DataFrame([{**base, 'bet_id': 'a', 'player_id': 1, 'side': 'over', 'line': 20.5, 'p_over_market': 0.50},
                         {**base, 'bet_id': 'b', 'player_id': 2, 'side': 'under', 'line': 10.5, 'p_over_market': 0.50},
                         {**base, 'bet_id': 'c', 'player_id': 3, 'side': 'over', 'line': 5.5, 'p_over_market': 0.50},
                         {**base, 'bet_id': 'd', 'player_id': 4, 'side': 'over', 'line': 5.5, 'p_over_market': 0.50, 'game_id': 'g2'}])


def test_build_computes_clv_move_missing_player_and_no_snapshot():
    cons = pd.concat([_cons('g1', 1, 'points', 20.5, 0.55), _cons('g1', 2, 'points', 11.5, 0.60)], ignore_index=True)
    cands = _cands(('g1', 'e1', '2026-11-02T00:00:00Z', 'close', 10))
    r = bf.build(_bets(), cands, 'unused', slot_fn=lambda *a: (cons, None, None)).set_index('bet_id')
    assert abs(r.loc['a', 'clv_prob'] - 0.05) < 1e-12 and r.loc['a', 'reason'] == 'line_unchanged'
    assert np.isnan(r.loc['b', 'clv_prob']) and r.loc['b', 'line_move_favour'] == -1.0 and r.loc['b', 'close_line'] == 11.5      # under bet, line rose: against
    assert r.loc['b', 'mkt_5'] == 0.60 and r.loc['b', 'line_5'] == 11.5
    assert r.loc['c', 'reason'] == 'player_stat_missing_from_closing_consensus' and r.loc['c', 'source'] == 'live'
    assert r.loc['d', 'source'] == 'none' and r.loc['d', 'reason'] == 'no_closing_snapshot'


def test_empty_live_snapshot_falls_through_to_hist():
    cons = _cons('g1', 1, 'points', 20.5, 0.55)
    cands = _cands(('g1', 'e1', '2026-11-02T00:00:00Z', 'close', 10), ('g1', 'e2', '2026-11-02T00:05:00Z', 'hist_T-5', 10))

    def slot(db, eid, gid, ts, T):
        return (cons, None, None) if eid == 'e2' else (None, None, pd.DataFrame())

    r = bf.build(_bets()[lambda d: d['game_id'] == 'g1'], cands, 'unused', slot_fn=slot)
    assert set(r['source']) == {'hist'}


def _live_db(path):
    c = sqlite3.connect(path)
    c.execute('CREATE TABLE prop_bets (bet_id TEXT, game_id TEXT, game_date TEXT, tip_ref_utc TEXT, minutes_before INT, player_id INT, stat TEXT, side TEXT, line REAL, p_over_market REAL)')
    c.execute('CREATE TABLE prop_results (bet_id TEXT, clv_prob REAL, line_move_favour REAL, close_line REAL, line_5 REAL, mkt_5 REAL, profit_stands_best REAL)')
    for b in _bets().itertuples(index=False):
        c.execute('INSERT INTO prop_bets VALUES (?,?,?,?,?,?,?,?,?,?)', (b.bet_id, b.game_id, b.game_date, b.tip_ref_utc, b.minutes_before, b.player_id, b.stat, b.side, b.line, b.p_over_market))
        c.execute('INSERT INTO prop_results (bet_id, profit_stands_best) VALUES (?, 1.0)', (b.bet_id,))
    c.execute("UPDATE prop_results SET clv_prob = 0.123, close_line = 9.5 WHERE bet_id = 'd'")     # already filled: must be skipped and never overwritten
    c.commit()
    c.close()


def test_write_updates_only_blank_rows_after_a_backup(tmp_path):
    db = tmp_path / 'live.db'
    _live_db(db)
    bets = bf.load_bets(db)
    assert sorted(bets['bet_id']) == ['a', 'b', 'c']
    cons = pd.concat([_cons('g1', 1, 'points', 20.5, 0.55), _cons('g1', 2, 'points', 11.5, 0.60)], ignore_index=True)
    r = bf.build(bets, _cands(('g1', 'e1', '2026-11-02T00:00:00Z', 'close', 10)), 'unused', slot_fn=lambda *a: (cons, None, None))
    backup, n = bf.write_results(db, r, tmp_path)
    assert n == 2 and backup.exists()
    c = sqlite3.connect(db)
    got = {row[0]: row[1:] for row in c.execute('SELECT bet_id, clv_prob, line_move_favour, close_line, line_5, mkt_5, profit_stands_best FROM prop_results')}
    assert abs(got['a'][0] - 0.05) < 1e-12 and got['a'][2] == 20.5 and got['a'][3] == 20.5 and got['a'][4] == 0.55
    assert got['b'][0] is None and got['b'][1] == -1.0 and got['b'][2] == 11.5
    assert got['c'][2] is None                                       # player missing: stays blank
    assert got['d'][0] == 0.123 and got['d'][2] == 9.5               # untouched
    assert got['a'][5] == 1.0                                        # other columns kept
    old = sqlite3.connect(backup).execute('SELECT count(*) FROM prop_results WHERE close_line IS NOT NULL').fetchone()[0]
    assert old == 1                                                  # backup holds the pre-write state
    c.close()
    assert bf.write_results(db, r, tmp_path)[1] == 0                 # a second write changes nothing


def test_odds_db_loader_maps_hist_through_event_ref_and_drops_empty_snapshots(tmp_path):
    p = tmp_path / 'odds.db'
    c = sqlite3.connect(p)
    c.execute('CREATE TABLE snapshot_log (snapshot_ts_utc TEXT, event_id_vendor TEXT, mode TEXT, n_prop_rows INT)')
    c.execute('CREATE TABLE events (event_id_vendor TEXT, game_id TEXT)')
    c.execute('CREATE TABLE event_ref (event_id_vendor TEXT, game_id TEXT)')
    c.executemany('INSERT INTO events VALUES (?,?)', [('e1', None), ('e2', 'g2')])
    c.execute("INSERT INTO event_ref VALUES ('e1', 'g1')")
    c.executemany('INSERT INTO snapshot_log VALUES (?,?,?,?)', [('t1', 'e1', 'hist_T-5', 5), ('t2', 'e1', 'slow', 0), ('t3', 'e2', 'close', 5), ('t4', 'e2', 'hist_T-240', 5)])
    c.commit()
    c.close()
    d = bf.load_candidates(p, ['g1', 'g2'])
    assert sorted(zip(d['game_id'], d['snapshot_ts_utc'])) == [('g1', 't1'), ('g2', 't3')]
