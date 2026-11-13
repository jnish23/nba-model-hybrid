"""Audit-review additions for scripts/backfill_props_clv.py: live T-5 ordering, audit rows, never overwriting, verified backup, the --import-audit-csv mode."""
import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location('backfill_props_clv_wp3', Path(__file__).resolve().parents[2] / 'scripts' / 'backfill_props_clv.py')
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

TIP = '2026-11-02T00:10:00Z'


def _cands(*rows):
    return pd.DataFrame(rows, columns=['game_id', 'event_id_vendor', 'snapshot_ts_utc', 'mode', 'n_prop_rows'])


def _cons(game_id, player, stat, line, fair):
    return pd.DataFrame({'game_id': game_id, 'player_id': player, 'minutes_before': 5, 'stat': stat, 'consensus_line': line, 'consensus_fair_over': fair}, index=[0])


def _bets():
    base = {'game_id': 'g1', 'game_date': '2026-11-01', 'tip_ref_utc': TIP, 'stat': 'points', 'minutes_before': 240}
    return pd.DataFrame([{**base, 'bet_id': 'a', 'player_id': 1, 'side': 'over', 'line': 20.5, 'p_over_market': 0.50},
                         {**base, 'bet_id': 'b', 'player_id': 2, 'side': 'under', 'line': 10.5, 'p_over_market': 0.50},
                         {**base, 'bet_id': 'c', 'player_id': 3, 'side': 'over', 'line': 5.5, 'p_over_market': 0.50}])


def _live_db(path, mkt5_a=None):
    c = sqlite3.connect(path)
    c.execute('CREATE TABLE prop_bets (bet_id TEXT, game_id TEXT, game_date TEXT, tip_ref_utc TEXT, minutes_before INT, player_id INT, stat TEXT, side TEXT, line REAL, p_over_market REAL)')
    c.execute('CREATE TABLE prop_results (bet_id TEXT, clv_prob REAL, line_move_favour REAL, close_line REAL, line_5 REAL, mkt_5 REAL, profit_stands_best REAL)')
    for b in _bets().itertuples(index=False):
        c.execute('INSERT INTO prop_bets VALUES (?,?,?,?,?,?,?,?,?,?)', (b.bet_id, b.game_id, b.game_date, b.tip_ref_utc, b.minutes_before, b.player_id, b.stat, b.side, b.line, b.p_over_market))
        c.execute('INSERT INTO prop_results (bet_id, profit_stands_best) VALUES (?, 1.0)', (b.bet_id,))
    if mkt5_a is not None:
        c.execute("UPDATE prop_results SET mkt_5 = ? WHERE bet_id = 'a'", (mkt5_a,))
    c.commit()
    c.close()


def _built(db):
    cons = pd.concat([_cons('g1', 1, 'points', 20.5, 0.55), _cons('g1', 2, 'points', 11.5, 0.60)], ignore_index=True)
    return bf.build(bf.load_bets(db), _cands(('g1', 'e1', '2026-11-02T00:00:00Z', 'close', 10)), 'unused', slot_fn=lambda *a: (cons, None, None))


def test_close_snapshot_order_follows_the_live_t5_slot_rule():
    c = _cands(('g1', 'e1', '2026-11-02T00:02:00Z', 'close', 50),      # 8 min before tip (> 5)
               ('g1', 'e1', '2026-11-02T00:06:00Z', 'close', 50),      # 4 min
               ('g1', 'e1', '2026-11-02T00:08:00Z', 'close', 50),      # 2 min
               ('g1', 'e1', '2026-11-01T23:50:00Z', 'slow', 50))       # 20 min
    assert [round(m) for _, _, _, m in bf.ordered_choices(c, TIP)] == [4, 2, 8, 20]      # inside 0.5..5 min: closest to 5 first, then the freshest before the nominal time


def test_bet_placed_at_the_close_slot_gets_its_own_reason():
    cons = _cons('g1', 1, 'points', 20.5, 0.55)
    bets = _bets()[lambda d: d['bet_id'] == 'a'].assign(minutes_before=5)
    r = bf.build(bets, _cands(('g1', 'e1', '2026-11-02T00:00:00Z', 'close', 10)), 'unused', slot_fn=lambda *a: (cons, None, None))
    assert r['reason'].iloc[0] == 'bet_at_close_slot' and np.isnan(r['clv_prob'].iloc[0]) and r['close_line'].iloc[0] == 20.5


def test_write_appends_audit_rows_only_for_filled_rows_and_is_idempotent(tmp_path):
    db = tmp_path / 'live.db'
    _live_db(db)
    r = _built(db)
    backup, n = bf.write_results(db, r, tmp_path, run_id='rid', written_utc='2026-11-12T10:00:00Z')
    assert n == 2 and backup.exists()
    c = sqlite3.connect(db)
    a = c.execute('SELECT bet_id, source, close_snapshot_ts_utc, close_snapshot_minutes, close_line, mkt_close, clv_prob, line_move_favour, run_id, written_utc '
                  'FROM prop_clv_audit ORDER BY bet_id').fetchall()
    assert [x[0] for x in a] == ['a', 'b'] and all(x[1] == 'snapshot_backfill' and x[8] == 'rid' and x[9] == '2026-11-12T10:00:00Z' for x in a)
    assert a[0][2] == '2026-11-02T00:00:00Z' and abs(a[0][3] - 10.0) < 1e-9 and a[0][4] == 20.5 and a[0][5] == 0.55 and abs(a[0][6] - 0.05) < 1e-12 and a[0][7] is None
    assert a[1][6] is None and a[1][7] == -1.0
    with pytest.raises(sqlite3.DatabaseError, match='append-only'):
        c.execute("UPDATE prop_clv_audit SET clv_prob = 1 WHERE bet_id = 'a'")
    c.close()
    assert bf.write_results(db, r, tmp_path)[1] == 0                  # second run: nothing filled
    assert sqlite3.connect(db).execute('SELECT COUNT(*) FROM prop_clv_audit').fetchone()[0] == 2


def test_write_refuses_to_overwrite_any_non_empty_closing_cell(tmp_path):
    db = tmp_path / 'live.db'
    _live_db(db, mkt5_a=0.9)
    assert sorted(bf.load_bets(db)['bet_id']) == ['b', 'c']           # 'a' has mkt_5 filled: not selected
    r = _built(db)
    c = sqlite3.connect(db)
    c.execute("UPDATE prop_results SET line_5 = 7.5 WHERE bet_id = 'b'")        # filled between the dry run and the write: the guard must still hold
    c.commit()
    c.close()
    _, n = bf.write_results(db, r, tmp_path)
    c = sqlite3.connect(db)
    assert n == 0 and c.execute("SELECT line_5, close_line FROM prop_results WHERE bet_id = 'b'").fetchone() == (7.5, None)
    assert c.execute('SELECT COUNT(*) FROM prop_clv_audit').fetchone()[0] == 0          # no audit row for a row that was not filled


def test_backup_is_closed_verified_and_never_overwrites(tmp_path):
    db = tmp_path / 'live.db'
    _live_db(db)
    b1 = bf.backup_db(db, tmp_path)
    b1.unlink()                                                        # would fail on Windows if the backup connection were still open
    assert bf.backup_db(db, tmp_path).exists()


def _csv(path):
    row = {'bet_id': 'a', 'source': 'hist', 'close_snapshot_ts_utc': '2026-11-02T00:00:00Z', 'close_snapshot_minutes_before_tip': 9.4, 'close_line': 20.5, 'mkt_5': 0.55,
           'clv_prob': 0.05, 'line_move_favour': np.nan, 'reason': 'line_unchanged'}
    rows = [row, {**row, 'bet_id': 'b', 'close_line': 11.5, 'mkt_5': 0.60, 'clv_prob': np.nan, 'line_move_favour': -1.0, 'reason': 'line_moved'},
            {**row, 'bet_id': 'c', 'close_line': np.nan, 'mkt_5': np.nan, 'clv_prob': np.nan, 'source': 'none', 'reason': 'no_closing_snapshot'},
            {**row, 'bet_id': 'zz'}]
    pd.DataFrame(rows).to_csv(path, index=False)


def _backfilled(tmp_path):
    """A database as on the real one: filled by the backfill, which predates the audit table (audit rows removed)."""
    db = tmp_path / 'live.db'
    _live_db(db)
    bf.write_results(db, _built(db), tmp_path)
    c = sqlite3.connect(db)
    c.execute('DROP TRIGGER prop_clv_audit_no_delete')
    c.execute('DELETE FROM prop_clv_audit')
    c.commit()
    c.close()
    return db


def test_import_audit_csv_dry_run_default_then_apply_without_touching_prop_results(tmp_path):
    db = _backfilled(tmp_path)
    csv = tmp_path / 'clv_backfill.csv'
    _csv(csv)

    def snap():
        return sqlite3.connect(db).execute('SELECT * FROM prop_results ORDER BY bet_id').fetchall()

    before = snap()
    n, skipped = bf.import_audit(db, csv)                              # dry run
    assert n == 2 and len(skipped['no_prop_results_row']) == 1 and sqlite3.connect(db).execute('SELECT COUNT(*) FROM prop_clv_audit').fetchone()[0] == 0
    n, _ = bf.import_audit(db, csv, apply=True, backup_dir=tmp_path, written_utc='2026-11-12T10:02:49Z')
    assert n == 2 and snap() == before
    got = sqlite3.connect(db).execute('SELECT bet_id, source, close_snapshot_minutes, close_line, mkt_close, clv_prob, line_move_favour, written_utc, run_id FROM prop_clv_audit ORDER BY bet_id').fetchall()
    assert got[0][:3] == ('a', 'snapshot_backfill', 9.4) and got[1][5] is None and got[1][6] == -1.0 and got[0][7] == '2026-11-12T10:02:49Z' and got[0][8].startswith('import_audit_csv:')
    n2, skipped2 = bf.import_audit(db, csv, apply=True, backup_dir=tmp_path)           # idempotent
    assert n2 == 0 and len(skipped2['already_audited']) == 2 and snap() == before


def test_import_skips_rows_where_prop_results_differs_from_the_csv(tmp_path):
    db = _backfilled(tmp_path)
    c = sqlite3.connect(db)
    c.execute("UPDATE prop_results SET close_line = 21.5 WHERE bet_id = 'a'")
    c.commit()
    c.close()
    csv = tmp_path / 'clv_backfill.csv'
    _csv(csv)
    n, skipped = bf.import_audit(db, csv, apply=True, backup_dir=tmp_path)
    assert n == 1 and skipped['prop_results_differs'] == ['a']


def test_import_dry_run_works_on_a_database_that_has_no_audit_table_and_does_not_create_it(tmp_path):
    db = tmp_path / 'old.db'
    _live_db(db)
    c = sqlite3.connect(db)
    c.execute("UPDATE prop_results SET close_line = 20.5, clv_prob = 0.05, mkt_5 = 0.55, line_5 = 20.5 WHERE bet_id = 'a'")
    c.commit()
    c.close()
    csv = tmp_path / 'clv_backfill.csv'
    _csv(csv)
    n, _ = bf.import_audit(db, csv)
    assert n == 1
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'prop_clv_audit'").fetchone()[0] == 0


def test_script_has_no_network_or_paid_api_code():
    src = (Path(__file__).resolve().parents[2] / 'scripts' / 'backfill_props_clv.py').read_text(encoding='utf-8')
    for word in ('import requests', 'urllib', 'http.client', 'the-odds-api', 'odds_api', 'ODDS_API_KEY', 'props.collect'):
        assert word not in src


def test_cli_import_dry_run_prints_counts_and_writes_nothing(tmp_path, capsys):
    db = _backfilled(tmp_path)
    csv = tmp_path / 'clv_backfill.csv'
    _csv(csv)
    bf.main(['--live-db', str(db), '--import-audit-csv', str(csv)])
    assert 'dry run, would import 2' in capsys.readouterr().out
    assert sqlite3.connect(db).execute('SELECT COUNT(*) FROM prop_clv_audit').fetchone()[0] == 0
