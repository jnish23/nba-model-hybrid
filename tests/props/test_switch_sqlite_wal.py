"""scripts/switch_sqlite_wal.py: dry run by default, backup before the change, a clean refusal when the file is busy, and the property the switch is for (a reader no longer blocks the writer)."""
import hashlib
import importlib.util
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('switch_sqlite_wal', ROOT / 'scripts' / 'switch_sqlite_wal.py')
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)


def _store(tmp_path, rows=3000):
    p = tmp_path / 'store.db'
    c = sqlite3.connect(p)
    c.execute('CREATE TABLE snapshot_log (i INTEGER PRIMARY KEY, x TEXT)')
    c.execute('CREATE TABLE prop_snapshots (i INTEGER PRIMARY KEY, x TEXT)')
    c.executemany('INSERT INTO snapshot_log (x) VALUES (?)', [('a' * 50,)] * 40)
    c.executemany('INSERT INTO prop_snapshots (x) VALUES (?)', [('b' * 200,)] * rows)
    c.commit()
    c.close()
    return p


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _mode(p):
    c = sqlite3.connect(f'file:{Path(p).as_posix()}?mode=ro', uri=True)
    try:
        return c.execute('PRAGMA journal_mode').fetchone()[0].lower()
    finally:
        c.close()


def _writer_blocked_by_open_read(p):
    """True when a commit fails while another connection is in the middle of a read."""
    r = sqlite3.connect(f'file:{Path(p).as_posix()}?mode=ro', uri=True)
    cur = r.execute('SELECT * FROM prop_snapshots')
    cur.fetchone()
    w = sqlite3.connect(p, timeout=0.3)
    try:
        w.execute("INSERT INTO snapshot_log (x) VALUES ('w')")
        w.commit()
        return False
    except sqlite3.OperationalError:
        return True
    finally:
        w.close()
        cur.fetchall()
        r.close()


def test_dry_run_is_the_default_and_changes_nothing(tmp_path, capsys):
    p = _store(tmp_path)
    before = _sha(p)
    assert sw.main(['--db', str(p)]) == 0
    assert _sha(p) == before and _mode(p) == 'delete'
    assert sorted(f.name for f in tmp_path.iterdir()) == ['store.db']
    assert 'dry run' in capsys.readouterr().out


def test_apply_needs_a_backup_choice(tmp_path):
    p = _store(tmp_path)
    assert sw.main(['--db', str(p), '--apply']) == 2
    assert _mode(p) == 'delete'


def test_apply_backs_up_then_switches_and_the_reader_no_longer_blocks_the_writer(tmp_path):
    p = _store(tmp_path)
    assert _writer_blocked_by_open_read(p) is True                       # the problem, in rollback-journal mode
    bdir = tmp_path / 'bak'
    assert sw.main(['--db', str(p), '--backup-dir', str(bdir), '--apply']) == 0
    assert _mode(p) == 'wal'
    backups = list(bdir.glob('store_before_wal_*.db'))
    assert len(backups) == 1
    b = sqlite3.connect(backups[0])
    assert b.execute('SELECT COUNT(*) FROM prop_snapshots').fetchone()[0] == 3000 and b.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    b.close()
    assert _writer_blocked_by_open_read(p) is False                      # the point of the switch
    c = sqlite3.connect(p)                                                # the mode persists for a later connection and the data is intact
    assert c.execute('PRAGMA journal_mode').fetchone()[0].lower() == 'wal' and c.execute('SELECT COUNT(*) FROM prop_snapshots').fetchone()[0] == 3000
    c.close()


def test_already_in_wal_is_a_no_op(tmp_path, capsys):
    p = _store(tmp_path)
    assert sw.main(['--db', str(p), '--backup-dir', str(tmp_path / 'bak'), '--apply']) == 0
    assert sw.main(['--db', str(p), '--backup-dir', str(tmp_path / 'bak2'), '--apply']) == 0
    assert 'nothing to do' in capsys.readouterr().out and not (tmp_path / 'bak2').exists()


def test_a_busy_file_is_left_unchanged_and_reported(tmp_path, capsys):
    p = _store(tmp_path)
    busy = sqlite3.connect(p)
    busy.execute('BEGIN IMMEDIATE')                                       # another program is in the middle of a write
    busy.execute("INSERT INTO snapshot_log (x) VALUES ('held')")
    try:
        rc = sw.main(['--db', str(p), '--existing-backup', str(p), '--apply', '--lock-timeout', '0.3'])
    finally:
        busy.rollback()
        busy.close()
    assert rc == 1 and _mode(p) == 'delete'
    assert 'NOT switched' in capsys.readouterr().out


def test_an_open_read_also_stops_the_switch_cleanly(tmp_path):
    p = _store(tmp_path)
    r = sqlite3.connect(f'file:{p.as_posix()}?mode=ro', uri=True)
    cur = r.execute('SELECT * FROM prop_snapshots')
    cur.fetchone()
    try:
        rc = sw.main(['--db', str(p), '--existing-backup', str(p), '--apply', '--lock-timeout', '0.3'])
    finally:
        cur.fetchall()
        r.close()
    assert rc == 1 and _mode(p) == 'delete'
    assert sw.main(['--db', str(p), '--existing-backup', str(p), '--apply']) == 0 and _mode(p) == 'wal'     # idle now: it goes through


def test_revert_goes_back_to_the_rollback_journal(tmp_path):
    p = _store(tmp_path)
    assert sw.main(['--db', str(p), '--existing-backup', str(p), '--apply']) == 0
    assert sw.main(['--db', str(p), '--revert', '--existing-backup', str(p), '--apply']) == 0
    assert _mode(p) == 'delete'
    assert sorted(f.name for f in tmp_path.iterdir() if f.name.startswith('store.db')) == ['store.db']


def test_missing_store_and_missing_named_backup(tmp_path):
    assert sw.main(['--db', str(tmp_path / 'nope.db')]) == 2
    p = _store(tmp_path)
    assert sw.main(['--db', str(p), '--existing-backup', str(tmp_path / 'nope.bak'), '--apply']) == 2
    assert _mode(p) == 'delete'


@pytest.mark.parametrize('rows', [0, 10])
def test_backup_of_a_small_store(tmp_path, rows):
    p = _store(tmp_path, rows=rows)
    dst, a, b = sw.backup(p, tmp_path / 'bak')
    assert dst.exists() and a == b
