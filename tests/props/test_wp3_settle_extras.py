"""Settlement side tables: detect_corrections, clv_audit_rows, the new store tables and the settle job's helpers (fill-check failure never blocks)."""
import importlib.util
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from props.live import settle, store

_spec = importlib.util.spec_from_file_location('props_settle_job_wp3', Path(__file__).resolve().parents[2] / 'scripts' / 'props_settle_job.py')
job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(job)


def _bets():
    base = {'season': 'S', 'game_id': 'g1', 'stat': 'assists', 'dec_median': 1.9, 'dec_flat': 1.909091, 'expected_minutes': 30.0, 'book': 'fanduel', 'minutes_before': 240, 'p_over_market': 0.5}
    return pd.DataFrame([{**base, 'bet_id': f'S|g1|{p}|assists', 'player_id': p, 'side': s, 'line': ln, 'dec_best': 2.0, 'tip_ref_utc': '2026-11-13T01:10:00Z',
                          'snapshot_ts_utc': '2026-11-13T00:10:00Z'} for p, s, ln in ((1, 'over', 4.5), (2, 'under', 6.5), (3, 'over', 3.5))])


def _labels(assists):
    return pd.DataFrame({'game_id': 'g1', 'player_id': [1, 2, 3], 'lbl_played': 1, 'lbl_minutes': 30.0, 'y_points': 10, 'y_rebounds': 5, 'y_assists': assists, 'y_threes': 1, 'y_pra': 20})


def _settled(assists=(6, 8, 4)):
    res = settle.settle_bets(_bets(), _labels(list(assists)), None, 'old', 'r0')
    return res


def test_no_correction_when_labels_are_unchanged():
    res = _settled()
    out = settle.detect_corrections(_bets(), res, _labels([6, 8, 4]))
    assert out.empty and list(out.columns)[:3] == ['bet_id', 'y_old', 'y_new']


def test_stat_change_that_flips_the_result_is_reported_with_old_and_new_profits():
    res = _settled()                                                       # p1 over 4.5 with 6 = win, p2 under 6.5 with 8 = loss, p3 over 3.5 with 4 = win
    out = settle.detect_corrections(_bets(), res, _labels([4, 6, 4]), detected_utc='now', run_id='r1').set_index('bet_id')
    assert sorted(out.index) == ['S|g1|1|assists', 'S|g1|2|assists']
    a = out.loc['S|g1|1|assists']
    assert (a['y_old'], a['y_new'], a['status_old'], a['status_new']) == (6.0, 4.0, 'win', 'loss') and a['profit_best_old'] == 1.0 and a['profit_best_new'] == -1.0
    assert np.isclose(a['profit_median_new'], -1.0) and np.isclose(a['profit_flat_new'], -1.0) and a['detected_utc'] == 'now' and a['run_id'] == 'r1'
    b = out.loc['S|g1|2|assists']
    assert (b['status_old'], b['status_new'], b['profit_best_new']) == ('loss', 'win', 1.0)


def test_stat_change_that_keeps_the_status_is_still_a_y_change():
    out = settle.detect_corrections(_bets(), _settled(), _labels([7, 8, 4]))
    assert list(out['bet_id']) == ['S|g1|1|assists'] and out['status_old'].iloc[0] == out['status_new'].iloc[0] == 'win' and out['y_new'].iloc[0] == 7.0


def test_only_settled_bets_with_labels_are_compared():
    res = _settled().iloc[:2]                                              # the third bet is unsettled
    out = settle.detect_corrections(_bets(), res, _labels([0, 0, 0]))
    assert 'S|g1|3|assists' not in set(out['bet_id'])
    assert settle.detect_corrections(_bets(), res, _labels([0, 0, 0]).assign(game_id='other')).empty        # game not in the labels: skipped
    assert settle.detect_corrections(_bets().iloc[0:0], res, _labels([0, 0, 0])).empty


def test_missing_player_in_the_labels_matches_the_original_settlement():
    labels = _labels([6, 8, 4])
    res = settle.settle_bets(_bets(), labels.iloc[:2], None, 'old', 'r0')       # p3 had no box-score row: settled void
    assert settle.detect_corrections(_bets(), res, labels.iloc[:2]).empty


# ---- job helpers ---------------------------------------------------------------------------------------------------------------------------------------------------

def _store_with_settled(tmp_path, assists=(6, 8, 4)):
    conn = store.connect(tmp_path / 'live.db')
    b = _bets()
    b = b.assign(created_utc='c', run_id='r0', game_date='2026-11-12', tip_ref_utc='2026-11-13T01:10:00Z')
    with conn:
        store._insert(conn, 'prop_bets', b)
        store._insert(conn, 'prop_results', _settled(assists))
    return conn


def test_record_corrections_appends_once_and_skips_an_unchanged_latest_correction(tmp_path):
    conn = _store_with_settled(tmp_path)
    labels = _labels([4, 8, 4])
    new = job.record_corrections(conn, labels, '2026-11-14T10:00:00Z', 'run2', dry_run=True)
    assert len(new) == 1 and conn.execute('SELECT COUNT(*) FROM prop_result_corrections').fetchone()[0] == 0          # dry run writes nothing
    new = job.record_corrections(conn, labels, '2026-11-14T10:00:00Z', 'run2', dry_run=False)
    assert len(new) == 1
    again = job.record_corrections(conn, labels, '2026-11-15T10:00:00Z', 'run3', dry_run=False)                       # same y_new: nothing new
    assert again.empty and conn.execute('SELECT COUNT(*) FROM prop_result_corrections').fetchone()[0] == 1
    changed = job.record_corrections(conn, _labels([3, 8, 4]), '2026-11-16T10:00:00Z', 'run4', dry_run=False)         # a further revision is a new row
    assert len(changed) == 1 and conn.execute('SELECT COUNT(*) FROM prop_result_corrections').fetchone()[0] == 2
    row = conn.execute("SELECT y_old, y_new, status_old, status_new, profit_best_old, profit_best_new, run_id FROM prop_result_corrections ORDER BY detected_utc LIMIT 1").fetchone()
    assert row == (6.0, 4.0, 'win', 'loss', 1.0, -1.0, 'run2')
    assert conn.execute("SELECT y FROM prop_results WHERE bet_id = 'S|g1|1|assists'").fetchone()[0] == 6.0           # prop_results is untouched
    with pytest.raises(sqlite3.DatabaseError, match='append-only'):
        conn.execute('DELETE FROM prop_result_corrections')
    conn.close()


def test_clv_audit_rows_take_the_snapshot_from_the_slot5_signal():
    sig = pd.DataFrame([{'game_id': 'g1', 'player_id': p, 'stat': 'assists', 'slot_minutes': sl, 'line': 4.5 if p != 2 else 6.5, 'p_market': 0.5 + 0.06 * (sl == 5), 'snapshot_ts_utc': f'ts{sl}-{p}',
                         'snapshot_minutes': 9.5 if sl == 5 else 239.0} for p in (1, 2, 3) for sl in (240, 5)])
    res = settle.settle_bets(_bets(), _labels([6, 8, 4]), sig, 'now', 'r1')
    a = settle.clv_audit_rows(res, sig, 'now', 'r1').set_index('bet_id')
    assert set(a['source']) == {'signals'} and a.loc['S|g1|1|assists', 'close_snapshot_ts_utc'] == 'ts5-1' and a.loc['S|g1|1|assists', 'close_snapshot_minutes'] == 9.5
    assert np.isclose(a.loc['S|g1|1|assists', 'clv_prob'], 0.06) and a.loc['S|g1|1|assists', 'mkt_close'] == res.set_index('bet_id').loc['S|g1|1|assists', 'mkt_5']
    assert a.loc['S|g1|2|assists', 'close_line'] == 6.5 and list(a.columns)[0] == 'written_utc'
    assert settle.clv_audit_rows(settle.settle_bets(_bets(), _labels([6, 8, 4]), None, 'now', 'r1'), None, 'now', 'r1').empty       # no signals: no audit rows


def test_audit_rows_insert_into_the_append_only_table(tmp_path):
    conn = store.connect(tmp_path / 'live.db')
    sig = pd.DataFrame([{'game_id': 'g1', 'player_id': 1, 'stat': 'assists', 'slot_minutes': sl, 'line': 4.5, 'p_market': 0.5, 'snapshot_ts_utc': 't', 'snapshot_minutes': 9.0} for sl in (240, 5)])
    res = settle.settle_bets(_bets().iloc[:1], _labels([6, 8, 4]), sig, 'now', 'r1')
    audit = settle.clv_audit_rows(res, sig, '2026-11-14T00:00:00Z', 'r1')
    with conn:
        assert store._insert(conn, 'prop_clv_audit', audit, ignore=True) == 1
        assert store._insert(conn, 'prop_clv_audit', audit, ignore=True) == 0
    with pytest.raises(sqlite3.DatabaseError, match='append-only'):
        conn.execute("UPDATE prop_clv_audit SET clv_prob = 0")
    conn.close()


def test_fill_check_failure_is_logged_and_never_raises(tmp_path, caplog):
    conn = store.connect(tmp_path / 'live.db')
    assert job.record_fill_checks(conn, _bets(), str(tmp_path / 'missing.db'), 'now', 'r', False) == 0               # odds store absent
    bad = tmp_path / 'bad.db'
    bad.write_bytes(b'not a database')
    with caplog.at_level('ERROR'):
        assert job.record_fill_checks(conn, _bets(), str(bad), 'now', 'r', False) == 0
    assert 'fill checks failed' in caplog.text and conn.execute('SELECT COUNT(*) FROM prop_fill_checks').fetchone()[0] == 0
    conn.close()


def test_new_tables_are_additive_and_prop_results_has_only_the_write_once_lock(tmp_path):
    conn = store.connect(tmp_path / 'live.db')
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {'prop_clv_audit', 'prop_result_corrections', 'prop_fill_checks'} <= names
    assert {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'prop_results'")} == {'prop_results_no_delete', 'prop_results_no_overwrite'}
    store.connect(tmp_path / 'live.db').close()                                                                  # re-running the schema is harmless
    conn.close()


def test_settle_job_end_to_end_settles_audits_corrects_and_survives_a_missing_odds_store(tmp_path, monkeypatch):
    db = tmp_path / 'live.db'
    conn = store.connect(db)
    b = _bets().assign(created_utc='c', run_id='r0', game_date='2026-11-12')
    sig = pd.DataFrame([{'signal_id': f's{p}{sl}', 'game_id': 'g1', 'player_id': p, 'stat': 'assists', 'slot_minutes': sl, 'line': 4.5 if p != 2 else 6.5, 'p_market': 0.5 + 0.06 * (sl == 5),
                         'snapshot_ts_utc': f't{sl}', 'snapshot_minutes': 9.0 if sl == 5 else 239.0, 'qualified': 1, 'already_bet': 0} for p in (1, 2, 3) for sl in (240, 5)])
    with conn:
        store._insert(conn, 'prop_bets', b.iloc[:2])                                  # two bets settled on an earlier night with stale stats
        store._insert(conn, 'prop_results', _settled((6, 8, 4)).iloc[:2])
        store._insert(conn, 'prop_signals', sig)
        store._insert(conn, 'prop_bets', b.iloc[2:])                                  # one new unsettled bet
    conn.close()
    hist = _labels([4, 8, 4]).rename(columns={'lbl_minutes': 'minutes'})              # the box score now says player 1 had 4 assists
    monkeypatch.setattr(job.rows_mod, 'load_history', lambda path: hist.copy())
    monkeypatch.setattr(sys, 'argv', ['job', '--db', str(db), '--props-db', str(tmp_path / 'absent.db')])
    assert job.main() == 0
    c = sqlite3.connect(db)
    assert c.execute('SELECT COUNT(*) FROM prop_results').fetchone()[0] == 3
    assert c.execute('SELECT COUNT(*) FROM prop_clv_audit').fetchone()[0] == 1 and c.execute('SELECT source FROM prop_clv_audit').fetchone()[0] == 'signals'
    assert c.execute('SELECT bet_id, y_old, y_new FROM prop_result_corrections').fetchall() == [('S|g1|1|assists', 6.0, 4.0)]
    assert c.execute('SELECT COUNT(*) FROM prop_fill_checks').fetchone()[0] == 0           # odds store absent: skipped, settlement unaffected
    assert c.execute("SELECT status FROM pipeline_health WHERE job = 'settle'").fetchone()[0] == 'ok'
    c.close()
    assert job.main() == 0                                                               # second run: idempotent
    c = sqlite3.connect(db)
    assert c.execute('SELECT COUNT(*) FROM prop_result_corrections').fetchone()[0] == 1 and c.execute('SELECT COUNT(*) FROM prop_clv_audit').fetchone()[0] == 1
    c.close()
