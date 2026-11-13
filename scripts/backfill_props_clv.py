"""
Backfill closing-line value for settled paper props bets whose T-5 slot never fired (prop_results.clv_prob and close_line are NULL), and record where every filled value came from.

Closing snapshot per game (one per game), in the order the live T-5 slot uses (props.live.quotes.choose_snapshot, T=5):
  1. a live snapshot (modes slow/fast/close, n_prop_rows > 0) taken MIN_MIN..5 minutes before the bets' tip_ref_utc, the one closest to 5 minutes;
  2. else the live snapshot closest to 5 minutes among those 5..MAX_MIN minutes before tip (the freshest one before the nominal time);
  3. else the same two rules over the historical 'hist_T-5' snapshots of the game;
  4. else none: the bets stay blank.
The closing consensus comes from props.live.quotes.slot_quotes(T=5), which calls props.betting.prices.quotes_from_snapshots (the code the live job and the backtest use); CLV, line move and close line
come from props.betting.export.clv_per_bet, called as settle_bets calls it (same sign, same "line unchanged" condition).

No network call and no paid API credit is spent anywhere in this script: it only reads the odds store (opened mode=ro). Default is a dry run that writes a CSV. --write updates prop_results
in one transaction after a verified, timestamped backup of the live database beside the CSV; only rows whose clv_prob, line_move_favour, close_line, line_5 and mkt_5 are all still NULL are touched, so a
second run changes nothing. Every row filled gets an append-only prop_clv_audit row (source 'snapshot_backfill') in the same transaction.

    python scripts/backfill_props_clv.py --live-db <props_live.db> --odds-db <props_odds.db> --out clv_backfill_dryrun.csv [--write]
    python scripts/backfill_props_clv.py --live-db <props_live.db> --import-audit-csv <clv_backfill.csv> [--apply] [--written-utc 2026-11-12T10:02:49Z]

--import-audit-csv loads the per-bet CSV of an earlier --write run into prop_clv_audit without touching prop_results (dry run unless --apply). A CSV row is imported only if prop_results still holds
exactly the CSV's close_line / clv_prob / line_move_favour for the bet and the bet has no 'snapshot_backfill' audit row yet; the rest are listed as skipped.
"""
import argparse
import datetime as dt
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from props.betting import export                                   # noqa: E402
from props.live import quotes, store                               # noqa: E402

MIN_MIN = quotes.MIN_MINUTES_TO_TIP     # a snapshot closer than this to the reference tip (or after it) is in-game
MAX_MIN = 45.0                          # a snapshot further than this before tip is not a closing line (the live rule: nominal time + MAX_STALE_MIN)
CLOSE_T = 5
OUT_COLS = ['bet_id', 'game_id', 'game_date', 'player_id', 'stat', 'side', 'minutes_before', 'line', 'p_over_market', 'source', 'close_snapshot_ts_utc',
            'close_snapshot_minutes_before_tip', 'close_line', 'mkt_5', 'clv_prob', 'line_move_favour', 'reason']
AUDIT_SOURCE = 'snapshot_backfill'
AUDIT_COLS = ['bet_id', 'written_utc', 'source', 'close_snapshot_ts_utc', 'close_snapshot_minutes', 'close_line', 'mkt_close', 'clv_prob', 'line_move_favour', 'run_id']
TOL = 1e-9


def _ro(path):
    return sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True, timeout=120)


def _none(x):
    return None if x is None or pd.isna(x) else float(x)


def _close(a, b):
    a, b = _none(a), _none(b)
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= TOL)


def load_bets(live_db):
    """Settled bets whose closing-quote cells are all still blank (the only rows --write will touch)."""
    q = """SELECT b.bet_id, b.game_id, b.game_date, b.tip_ref_utc, b.minutes_before, b.player_id, b.stat, b.side, b.line, b.p_over_market
           FROM prop_bets b JOIN prop_results r USING (bet_id)
           WHERE r.clv_prob IS NULL AND r.close_line IS NULL AND r.line_move_favour IS NULL AND r.line_5 IS NULL AND r.mkt_5 IS NULL ORDER BY b.bet_id"""
    with _ro(live_db) as conn:
        return pd.read_sql(q, conn)


def load_candidates(odds_db, game_ids):
    """Every snapshot of every vendor event mapped to one of `game_ids` (event_ref.game_id preferred, else events.game_id)."""
    with _ro(odds_db) as conn:
        has_ref = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'event_ref'").fetchone() is not None
        gid = 'COALESCE(r.game_id, e.game_id)' if has_ref else 'e.game_id'
        join = 'LEFT JOIN event_ref r USING (event_id_vendor)' if has_ref else ''
        d = pd.read_sql(f"SELECT {gid} AS game_id, l.event_id_vendor, l.snapshot_ts_utc, l.mode, l.n_prop_rows FROM snapshot_log l JOIN events e USING (event_id_vendor) {join} "
                        f"WHERE l.mode IN ('slow','fast','close','hist_T-5') AND l.n_prop_rows > 0", conn)
    return d[d['game_id'].isin(set(game_ids))].reset_index(drop=True)


def ordered_choices(cands, tip_ref_utc):
    """
    Snapshots of one game usable as the close, in preference order: [(source, event_id, ts, minutes_before_tip)]. Live first, then historical; within a source the live T-5 slot rule:
    snapshots MIN_MIN..5 minutes before tip first (closest to 5 minutes first), then those 5..MAX_MIN minutes before tip (closest to 5 minutes first, i.e. the freshest before the nominal time).
    """
    if cands.empty:
        return []
    c = cands.copy()
    c['minutes'] = [quotes.minutes_before_tip(t, tip_ref_utc) for t in c['snapshot_ts_utc']]
    c = c[(c['minutes'] >= MIN_MIN) & (c['minutes'] <= MAX_MIN)]
    out = []
    for source, modes in (('live', quotes.LIVE_MODES), ('hist', ('hist_T-5',))):
        s = c[c['mode'].isin(modes)]
        inside = s[s['minutes'] <= CLOSE_T].sort_values(['minutes', 'snapshot_ts_utc'], ascending=[False, True])
        before = s[s['minutes'] > CLOSE_T].sort_values(['minutes', 'snapshot_ts_utc'])
        out += [(source, r.event_id_vendor, r.snapshot_ts_utc, float(r.minutes)) for r in pd.concat([inside, before]).itertuples(index=False)]
    return out


def closing_consensus(odds_db, game_id, choices, slot_fn=quotes.slot_quotes):
    """First choice whose snapshot yields a consensus: (source, ts, minutes, cons) or ('none', None, nan, None)."""
    for source, eid, ts, minutes in choices:
        cons, _, _ = slot_fn(odds_db, eid, game_id, ts, CLOSE_T)
        if cons is not None and len(cons):
            return source, ts, minutes, cons
    return 'none', None, np.nan, None


def build(bets, cands, odds_db, slot_fn=quotes.slot_quotes):
    """One row per bet with OUT_COLS (+ line_5)."""
    res = []
    for gid, g in bets.groupby('game_id', sort=True):
        choices = ordered_choices(cands[cands['game_id'] == gid], g['tip_ref_utc'].iloc[0])
        source, ts, minutes, cons = closing_consensus(odds_db, gid, choices, slot_fn)
        out = g[['bet_id', 'game_id', 'game_date', 'player_id', 'stat', 'side', 'minutes_before', 'line', 'p_over_market']].copy()
        out['source'], out['close_snapshot_ts_utc'], out['close_snapshot_minutes_before_tip'] = source, ts, minutes
        out['close_line'] = out['mkt_5'] = out['clv_prob'] = out['line_move_favour'] = np.nan
        if cons is not None:
            sel = g.rename(columns={'side': 'best_side'}).assign(consensus_fair_over=g['p_over_market'].to_numpy())
            clv, move, close_line = export.clv_per_bet(sel, cons)           # the same call settle_bets makes
            out['clv_prob'], out['line_move_favour'], out['close_line'] = clv, move, close_line
            c5 = cons[cons['minutes_before'] == CLOSE_T][['game_id', 'player_id', 'stat', 'consensus_fair_over']].rename(columns={'consensus_fair_over': 'mkt_5'})
            out['mkt_5'] = out[['game_id', 'player_id', 'stat']].merge(c5, on=['game_id', 'player_id', 'stat'], how='left', validate='many_to_one')['mkt_5'].to_numpy()
        res.append(out)
    r = pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=OUT_COLS)
    r['reason'] = np.select([r['source'] == 'none', r['close_line'].isna(), r['minutes_before'].astype(float) <= CLOSE_T, r['clv_prob'].notna(), r['line_move_favour'].notna()],
                            ['no_closing_snapshot', 'player_stat_missing_from_closing_consensus', 'bet_at_close_slot', 'line_unchanged', 'line_moved'], 'other')
    r['line_5'] = r['close_line']
    return r[OUT_COLS + ['line_5']]


def summary(r):
    n = len(r)
    lines = [f'bets: {n}  games: {r["game_id"].nunique()}', 'bets by source: ' + ', '.join(f'{k}={v}' for k, v in r['source'].value_counts().items()),
             f'clv_prob present (line unchanged): {int(r["clv_prob"].notna().sum())}', f'line_move_favour present (line moved): {int(r["line_move_favour"].notna().sum())}',
             f'neither: {int((r["clv_prob"].isna() & r["line_move_favour"].isna()).sum())}  ('
             + ', '.join(f'{k}={v}' for k, v in r.loc[r['clv_prob'].isna() & r['line_move_favour'].isna(), 'reason'].value_counts().items()) + ')',
             f'mean clv_prob overall: {r["clv_prob"].mean():.5f} (n={int(r["clv_prob"].notna().sum())})']
    month = r['game_date'].astype(str).str[:7]
    for k, g in r.groupby(month):
        lines.append(f'  month {k}: bets={len(g)} clv n={int(g["clv_prob"].notna().sum())} mean={g["clv_prob"].mean():.5f}')
    for k, g in r.groupby('minutes_before'):
        lines.append(f'  slot {k}: bets={len(g)} clv n={int(g["clv_prob"].notna().sum())} mean={g["clv_prob"].mean():.5f}')
    return '\n'.join(lines)


def backup_db(live_db, backup_dir):
    """Consistent copy of the live database (sqlite backup API, works while other jobs write), closed properly and verified. Never overwrites an existing file."""
    stamp = dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    backup = Path(backup_dir) / f'{Path(live_db).stem}_backup_{stamp}{Path(live_db).suffix}'
    if backup.exists():
        raise FileExistsError(backup)
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    src, dst = _ro(live_db), sqlite3.connect(backup.as_posix())
    try:
        src.backup(dst)
        n_src = src.execute('SELECT COUNT(*) FROM prop_results').fetchone()[0]
        n_dst = dst.execute('SELECT COUNT(*) FROM prop_results').fetchone()[0]
        ok = dst.execute('PRAGMA integrity_check').fetchone()[0]
    finally:                                                                # sqlite3's `with` does not close a connection
        src.close()
        dst.close()
    if ok != 'ok' or n_src != n_dst:
        raise RuntimeError(f'backup {backup} failed verification (integrity={ok}, prop_results {n_dst} vs {n_src})')
    return backup


def audit_frame(r, written_utc, run_id):
    """prop_clv_audit rows (AUDIT_COLS) for the rows of `r` that have a close_line."""
    r = r[r['close_line'].notna()]
    return pd.DataFrame({'bet_id': r['bet_id'].to_numpy(), 'written_utc': written_utc, 'source': AUDIT_SOURCE, 'close_snapshot_ts_utc': r['close_snapshot_ts_utc'].to_numpy(),
                         'close_snapshot_minutes': r['close_snapshot_minutes_before_tip'].to_numpy(), 'close_line': r['close_line'].to_numpy(), 'mkt_close': r['mkt_5'].to_numpy(),
                         'clv_prob': r['clv_prob'].to_numpy(), 'line_move_favour': r['line_move_favour'].to_numpy(), 'run_id': run_id})[AUDIT_COLS]


def _insert_audit(conn, rows):
    for x in rows:
        conn.execute(f"INSERT OR IGNORE INTO prop_clv_audit ({','.join(AUDIT_COLS)}) VALUES ({','.join('?' * len(AUDIT_COLS))})",
                     [None if (not isinstance(v, str) and pd.isna(v)) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for v in x])


def write_results(live_db, r, backup_dir, run_id=None, written_utc=None):
    """
    Backup the live db, then in one transaction fill prop_results for the rows whose five closing-quote cells are all still blank and append their prop_clv_audit rows (only for rows actually
    filled). Returns (backup_path or None, rows_updated); nothing is touched when there is nothing to write.
    """
    r = r[r['close_line'].notna()]
    if r.empty:
        return None, 0
    now = dt.datetime.now(dt.timezone.utc)
    written_utc = written_utc or now.strftime('%Y-%m-%dT%H:%M:%SZ')
    run_id = run_id or f"clv_backfill_{now.strftime('%Y%m%dT%H%M%SZ')}"
    backup = backup_db(live_db, backup_dir)
    conn = sqlite3.connect(Path(live_db).as_posix(), timeout=120, isolation_level=None)
    try:
        conn.executescript(store.SCHEMA)                                    # additive: creates prop_clv_audit if this database predates it
        conn.execute('BEGIN IMMEDIATE')
        n = 0
        audit = audit_frame(r, written_utc, run_id)
        filled = []
        for x, a in zip(r.itertuples(index=False), audit.itertuples(index=False)):
            k = conn.execute('UPDATE prop_results SET clv_prob = ?, line_move_favour = ?, close_line = ?, line_5 = ?, mkt_5 = ? '
                             'WHERE bet_id = ? AND clv_prob IS NULL AND line_move_favour IS NULL AND close_line IS NULL AND line_5 IS NULL AND mkt_5 IS NULL',
                             (_none(x.clv_prob), _none(x.line_move_favour), float(x.close_line), float(x.line_5), _none(x.mkt_5), x.bet_id)).rowcount
            n += k
            if k:
                filled.append(tuple(a))
        _insert_audit(conn, filled)
        conn.execute('COMMIT')
    except BaseException:
        if conn.in_transaction:
            conn.execute('ROLLBACK')
        raise
    finally:
        conn.close()
    return backup, n


def plan_import(live_db, csv_path):
    """
    (rows to import as a prop_clv_audit-shaped frame without written_utc/run_id, skipped: dict reason -> bet_ids) from a clv_backfill.csv. Read-only. A row is importable if the CSV has a close_line,
    prop_results holds the same close_line / clv_prob / line_move_favour for the bet, and the bet has no 'snapshot_backfill' audit row yet.
    """
    csv = pd.read_csv(csv_path)
    csv = csv[csv['close_line'].notna()]
    skipped = {'no_prop_results_row': [], 'prop_results_differs': [], 'already_audited': []}
    with _ro(live_db) as conn:
        cur = pd.read_sql('SELECT bet_id, close_line, clv_prob, line_move_favour, mkt_5 FROM prop_results', conn).set_index('bet_id')
        has_tbl = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'prop_clv_audit'").fetchone() is not None
        done = set(pd.read_sql(f"SELECT bet_id FROM prop_clv_audit WHERE source = '{AUDIT_SOURCE}'", conn)['bet_id']) if has_tbl else set()
    keep = []
    for x in csv.itertuples(index=False):
        if x.bet_id not in cur.index:
            skipped['no_prop_results_row'].append(x.bet_id)
        elif not (_close(cur.at[x.bet_id, 'close_line'], x.close_line) and _close(cur.at[x.bet_id, 'clv_prob'], x.clv_prob)
                  and _close(cur.at[x.bet_id, 'line_move_favour'], x.line_move_favour) and _close(cur.at[x.bet_id, 'mkt_5'], x.mkt_5)):
            skipped['prop_results_differs'].append(x.bet_id)
        elif x.bet_id in done:
            skipped['already_audited'].append(x.bet_id)
        else:
            keep.append(x.bet_id)
    return csv[csv['bet_id'].isin(keep)].reset_index(drop=True), skipped


def import_audit(live_db, csv_path, apply=False, backup_dir=None, written_utc=None, run_id=None):
    """Load a clv_backfill.csv into prop_clv_audit; prop_results is never written. Returns (rows that would be / were inserted, skipped dict)."""
    rows, skipped = plan_import(live_db, csv_path)
    if not apply or rows.empty:
        return len(rows), skipped
    now = dt.datetime.now(dt.timezone.utc)
    written_utc = written_utc or now.strftime('%Y-%m-%dT%H:%M:%SZ')
    run_id = run_id or f'import_audit_csv:{Path(csv_path).name}:{now.strftime("%Y%m%dT%H%M%SZ")}'
    backup_db(live_db, backup_dir or Path(csv_path).parent)
    audit = audit_frame(rows, written_utc, run_id)
    conn = sqlite3.connect(Path(live_db).as_posix(), timeout=120, isolation_level=None)
    try:
        conn.executescript(store.SCHEMA)
        conn.execute('BEGIN IMMEDIATE')
        before = conn.total_changes
        _insert_audit(conn, [tuple(a) for a in audit.itertuples(index=False)])
        n = conn.total_changes - before
        conn.execute('COMMIT')
    except BaseException:
        if conn.in_transaction:
            conn.execute('ROLLBACK')
        raise
    finally:
        conn.close()
    return n, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--live-db', required=True)
    ap.add_argument('--odds-db', help='odds store (read-only); needed unless --import-audit-csv')
    ap.add_argument('--out', help='CSV path (dry-run output; the --write backup goes beside it); needed unless --import-audit-csv')
    ap.add_argument('--write', action='store_true', help='update prop_results (default is a dry run)')
    ap.add_argument('--import-audit-csv', help='load this clv_backfill.csv into prop_clv_audit; prop_results is not touched')
    ap.add_argument('--apply', action='store_true', help='with --import-audit-csv: actually insert (default is a dry run)')
    ap.add_argument('--written-utc', help='with --import-audit-csv --apply: written_utc to record (default now); use the time of the original backfill if known')
    a = ap.parse_args(argv)
    if a.import_audit_csv:
        n, skipped = import_audit(a.live_db, a.import_audit_csv, apply=a.apply, written_utc=a.written_utc)
        print(f'{"IMPORTED" if a.apply else "dry run, would import"} {n} prop_clv_audit rows (source {AUDIT_SOURCE}); prop_results not touched')
        print('skipped: ' + ', '.join(f'{k}={len(v)}' for k, v in skipped.items()))
        return
    if not (a.odds_db and a.out):
        ap.error('--odds-db and --out are required unless --import-audit-csv is given')
    bets = load_bets(a.live_db)
    cands = load_candidates(a.odds_db, bets['game_id'].unique())
    r = build(bets, cands, a.odds_db)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    r[OUT_COLS].to_csv(a.out, index=False)
    print(summary(r))
    print(f'csv: {a.out}')
    if a.write:
        backup, n = write_results(a.live_db, r, Path(a.out).parent)
        print(f'WROTE {n} prop_results rows; backup {backup}')
    else:
        print('dry run: nothing written to the database')


if __name__ == '__main__':
    main()
