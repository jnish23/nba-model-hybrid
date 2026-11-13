"""
Switch a SQLite store to write-ahead logging (WAL), so that a long read no longer blocks the writer.

Why: in the default rollback-journal mode a reader that is in the middle of a query holds a shared lock, and a writer cannot commit until that read ends. The props collector waits
120 seconds for the lock (props/odds_store.connect), so a research read longer than that can cost a snapshot. In WAL mode readers and the writer do not block each other. The mode is
stored in the database file and stays until it is changed back. Every program must be on the same machine and the file on a local disk (true here).

    python scripts/switch_sqlite_wal.py --db data/props_odds.db                                   # dry run: read-only checks, changes nothing
    python scripts/switch_sqlite_wal.py --db data/props_odds.db --backup-dir D:/nba_backups/wal_switch --apply
    python scripts/switch_sqlite_wal.py --db data/props_odds.db --apply --existing-backup <file>     # skip the copy, name a backup you already have
    python scripts/switch_sqlite_wal.py --db data/props_odds.db --revert --apply                     # back to the rollback journal

What --apply does, in order: (1) backs the store up through the SQLite backup API and checks the copy; (2) asks for the new journal mode, which needs a moment with no open read or
write on the file and fails cleanly, changing nothing, if it cannot get it within --lock-timeout seconds; (3) re-opens the file read-only and confirms the mode and that a read works.
The switch itself rewrites only the file header. After it, two side files appear next to the store while it is in use (<name>-wal and <name>-shm): copy the store only through the
SQLite backup API (scripts/backup_data.ps1 already does), never by copying the .db file alone.
"""
import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

TARGET = {'wal': 'wal', 'revert': 'delete'}


def ro(path, timeout=30):
    return sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True, timeout=timeout)


def inspect(path):
    """Read-only facts about the store."""
    p = Path(path)
    conn = ro(p)
    try:
        mode = conn.execute('PRAGMA journal_mode').fetchone()[0].lower()
        pages = conn.execute('PRAGMA page_count').fetchone()[0]
        page_size = conn.execute('PRAGMA page_size').fetchone()[0]
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    finally:
        conn.close()
    return {'path': str(p), 'journal_mode': mode, 'bytes': p.stat().st_size, 'page_count': pages, 'page_size': page_size, 'tables': tables,
            'side_files': sorted(f.name for f in p.parent.glob(p.name + '-*')), 'sqlite': sqlite3.sqlite_version}


def row_counts(conn, tables, limit=4):
    """Row counts of a few small tables; used only to compare a backup with its source."""
    out = {}
    for t in tables[:limit]:
        out[t] = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    return out


def backup(path, backup_dir, pages=20000, sleep=0.02):
    """Copy through the backup API in steps (the source lock is released between steps) and check the copy. Returns the backup path."""
    src_p = Path(path)
    dst_dir = Path(backup_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_p = dst_dir / f"{src_p.stem}_before_wal_{time.strftime('%Y%m%dT%H%M%S')}{src_p.suffix}"
    if dst_p.exists():
        raise SystemExit(f'refusing to overwrite {dst_p}')
    free = shutil.disk_usage(dst_dir).free
    if free < src_p.stat().st_size * 1.1:
        raise SystemExit(f'not enough room in {dst_dir}: {free / 1e9:.1f} GB free, the store is {src_p.stat().st_size / 1e9:.1f} GB')
    src, dst = ro(src_p, timeout=120), sqlite3.connect(str(dst_p))
    try:
        src.backup(dst, pages=pages, sleep=sleep)
        dst.commit()
        ok = dst.execute('PRAGMA quick_check').fetchone()[0]
        if ok != 'ok':
            raise SystemExit(f'backup failed its check: {ok}')
        small = [t for t in ('snapshot_log', 'events', 'event_ref', 'slot_log', 'prop_bets') if t in
                 {r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}]
        b, a = row_counts(dst, small), row_counts(src, small)              # the source is counted after the copy: it can only have gained rows since
        if dst.execute('PRAGMA page_count').fetchone()[0] == 0:
            raise SystemExit('backup is empty')
        bad = [t for t in small if b[t] > a[t] or (a[t] > 0 and b[t] == 0)]
        if bad:
            raise SystemExit(f'backup does not match the source in {bad}: source {a}, backup {b}')
    finally:
        src.close()
        dst.close()
    return dst_p, a, b


def set_mode(path, target, lock_timeout=30.0):
    """Ask for the journal mode. SQLite returns the mode actually in force: when another connection is reading or writing it stays unchanged, and so does the file."""
    conn = sqlite3.connect(str(path), timeout=lock_timeout)
    try:
        try:
            got = conn.execute(f'PRAGMA journal_mode = {target.upper()}').fetchone()[0].lower()
        except sqlite3.OperationalError as e:                                # 'database is locked' after the timeout
            return None, repr(e)
        return got, None
    finally:
        conn.close()


def verify(path, target):
    conn = ro(path)
    try:
        mode = conn.execute('PRAGMA journal_mode').fetchone()[0].lower()
        n = conn.execute('SELECT COUNT(*) FROM sqlite_master').fetchone()[0]
    finally:
        conn.close()
    return mode == target and n > 0, mode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--db', required=True)
    ap.add_argument('--apply', action='store_true', help='make the change (default: read-only dry run)')
    ap.add_argument('--revert', action='store_true', help='switch back to the rollback journal (DELETE) instead of to WAL')
    ap.add_argument('--backup-dir', help='folder for the backup taken before the change')
    ap.add_argument('--existing-backup', help='skip the copy: path of a backup of this store that already exists')
    ap.add_argument('--lock-timeout', type=float, default=30.0, help='seconds to wait for the moment with no other open read or write')
    a = ap.parse_args(argv)
    path = Path(a.db)
    if not path.exists():
        print(f'{path} does not exist')
        return 2
    target = TARGET['revert' if a.revert else 'wal']
    info = inspect(path)
    print(f"store        {info['path']}  ({info['bytes'] / 1e9:.2f} GB, {info['page_count']} pages of {info['page_size']})")
    print(f"journal mode {info['journal_mode']}  ->  {target}")
    print(f"sqlite       {info['sqlite']}  (read-only access to a WAL store needs 3.22 or later)")
    print(f"side files   {', '.join(info['side_files']) or 'none'}")
    if info['journal_mode'] == target:
        print('nothing to do: the store is already in that mode')
        return 0
    if tuple(int(x) for x in info['sqlite'].split('.')[:2]) < (3, 22):
        print('this SQLite is too old for read-only access to a WAL store; not switching')
        return 2
    if not a.apply:
        print('dry run: nothing changed. With --apply this will back the store up, then change the mode (a header change; it needs a moment with no other open read or write).')
        return 0
    if a.existing_backup:
        if not Path(a.existing_backup).exists():
            print(f'--existing-backup {a.existing_backup} does not exist')
            return 2
        print(f'backup       using the existing {a.existing_backup}')
    elif a.backup_dir:
        t = time.time()
        dst, src_n, dst_n = backup(path, a.backup_dir)
        print(f'backup       {dst}  ({dst.stat().st_size / 1e9:.2f} GB in {time.time() - t:.0f} s; rows source/backup: ' + ', '.join(f'{k} {src_n[k]}/{dst_n[k]}' for k in src_n) + ')')
    else:
        print('--apply needs --backup-dir or --existing-backup')
        return 2
    got, err = set_mode(path, target, a.lock_timeout)
    if got != target:
        print(f"NOT switched: the mode is {got or 'unchanged'}" + (f' ({err})' if err else '') + '. Another program had the file open. Nothing was changed; run again when it is idle.')
        return 1
    ok, mode = verify(path, target)
    print(f"switched     journal mode is now {mode}; read-only check {'ok' if ok else 'FAILED'}")
    print(f"side files   {', '.join(inspect(path)['side_files']) or 'none'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
