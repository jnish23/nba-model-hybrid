"""
Step S1 of reports/props/audit_2026-11/WP6_WP7_experiment_plan.md (section 7a, item 5): quote caches and price files for 2023-24 and 2024-25.

    PY scripts/a11_cache_quotes.py --props-db data/props_odds.db --seasons 2023-24 2024-25 --out $OUT/quotes --prices-out $OUT/prices

What it does, per season
  1. Event ids come from the `events` table (4,118 rows), grouped by calendar month of the ET game date. Snapshot rows are read month by month with
     `p.event_id_vendor IN (...)`, which SQLite answers through ix_props_event_ts (`--dry-run` prints EXPLAIN QUERY PLAN). The filters are those of
     props.betting.prices.load_season (hist modes, main lines only, game resolved, no team-total ladders, not in game, snapshot at or before T, tip date in range).
     Every step of the consensus is per snapshot and event, so month chunks give exactly the one-pass result.
  2. For each of 3 de-vig methods (core.devig) x 3 consensus sets (all books mean = the current method, bettable books mean, all books median) the consensus table
     is built with props.analysis (add_devig, flag_main_lines, consensus with agg/books). 9 variants.
  3. `prices_<season>.pkl` = pickle of (consensus, bets) exactly as props.betting.prices.load_season returns them, built by prices.quotes_from_snapshots on each month,
     for the current method. Rows are put in a fixed order (sorted by key) and the index is reset; load_season's own row order is arbitrary (it follows the SQL result).

Outputs
  <out>/<season>/cons_<devig>_<set>.pkl   consensus table of one variant (pandas pickle; columns below)
  <out>/<season>/raw_<season>.parquet     the filtered snapshot rows (all books, all markets, one row per quote) the variants were built from (not with --skip-raw)
  <out>/<season>/manifest.json            counts per month, query plan, versions, arguments
  <prices-out>/prices_<season>.pkl        the Phase 5 price file
With --limit-months N the season folder is `<season>.partial` and the price file `prices_<season>.partial.pkl` so that a smoke run can never be mistaken for a season.

Read-only on the odds store (mode=ro, query_only). Writes only under --out and --prices-out. Refuses every season but 2023-24 and 2024-25 (plan A.1: 2025-26 is excluded).
"""
import argparse
import contextlib
import json
import logging
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import devig  # noqa: E402
from props import analysis  # noqa: E402
from props.betting import prices  # noqa: E402

SEASONS = {'2023-24': ('2023-10-01', '2024-06-30'), '2024-25': ('2024-10-01', '2025-06-30')}      # the date ranges of props/betting/export.SEASONS
LAST_ALLOWED_DATE = '2025-06-30'
DEVIG_METHODS = tuple(devig.METHODS)                                                              # multiplicative, power, shin
CONSENSUS_SETS = {'all_mean': {'agg': 'mean', 'books': None},                                     # the current method
                  'bettable_mean': {'agg': 'mean', 'books': prices.BETTABLE},
                  'all_median': {'agg': 'median', 'books': None}}
CURRENT = ('multiplicative', 'all_mean')
# the current method first (it is the reference of every comparison), then the other eight in a fixed order (the tie-break order of the keep rule)
VARIANTS = [CURRENT] + [(m, s) for m in DEVIG_METHODS for s in CONSENSUS_SETS if (m, s) != CURRENT]
KEY = ['game_id', 'player_id', 'minutes_before', 'stat']
SORT_BETS = ['game_id', 'player_id', 'stat', 'minutes_before', 'book', 'line', 'snapshot_ts_utc', 'over_price', 'under_price']
EVENTS_PER_QUERY = 400
RAW_COLUMNS = ['snapshot_ts_utc', 'game_id', 'event_id_vendor', 'book', 'player_id', 'player_name_raw', 'market', 'line', 'over_price', 'under_price',
               'is_alt_line', 'book_last_update_utc', 'minutes_before', 'game_date']
SLIM_COLUMNS = KEY + ['game_date', 'snapshot_ts_utc', 'event_id_vendor', 'consensus_line', 'consensus_fair_over', 'n_at_line', 'n_books', 'book_spread']


class SeasonRefused(ValueError):
    pass


def variant_name(v):
    return f'{v[0]}_{v[1]}'


def check_season(season):
    """Only 2023-24 and 2024-25 are allowed (plan A.1); 2025-26 and any later season are refused with a clear message."""
    if season in SEASONS:
        return SEASONS[season]
    try:
        first = int(str(season)[:4])
    except ValueError:
        first = None
    if first is not None and first >= 2025:
        raise SeasonRefused(f'season {season} is refused: 2025-26 and later seasons are excluded from every a11 experiment (plan A.1); data after {LAST_ALLOWED_DATE} is never read')
    raise SeasonRefused(f'season {season} is not supported: quotes exist from 2023-24 and the plan uses {sorted(SEASONS)}')


def check_range(start, end):
    if end > LAST_ALLOWED_DATE:
        raise SeasonRefused(f'date range {start}..{end} ends after {LAST_ALLOWED_DATE}: refused (plan A.1)')


def connect_ro(props_db):
    """Read-only connection: the collector writes to this file during the day, so wait for locks (timeout) and never write."""
    uri = 'file:' + Path(props_db).resolve().as_posix() + '?mode=ro'
    conn = sqlite3.connect(uri, uri=True, timeout=120)
    conn.execute('PRAGMA query_only = ON')
    return conn


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _modes(grid):
    return [f'hist_T-{x}' for x in grid]


def snapshot_query(n_events, n_modes):
    """prices.load_season's query with an event-id filter (served by ix_props_event_ts) added in front of the date filter."""
    return f"""SELECT p.snapshot_ts_utc, p.game_id, p.event_id_vendor, p.book, p.player_id, p.player_name_raw, p.market, p.line, p.over_price, p.under_price,
                      p.is_alt_line, p.book_last_update_utc, l.mode
               FROM prop_snapshots p
               JOIN snapshot_log l ON l.snapshot_ts_utc = p.snapshot_ts_utc AND l.event_id_vendor = p.event_id_vendor
               LEFT JOIN snapshot_ref s ON s.snapshot_ts_utc = p.snapshot_ts_utc AND s.event_id_vendor = p.event_id_vendor
               WHERE l.mode IN ({','.join('?' * n_modes)}) AND p.is_alt_line = 0 AND p.game_id IS NOT NULL AND (p.player_id IS NULL OR p.player_id <> -1)
                 AND COALESCE(s.in_game, 0) = 0 AND s.minutes_to_ref >= CAST(substr(l.mode, 8) AS REAL)
                 AND p.event_id_vendor IN ({','.join('?' * n_events)})
                 AND substr(p.commence_time_utc, 1, 10) BETWEEN ? AND ?"""


def explain_plan(conn, event_ids, start, end, grid=prices.GRID):
    modes = _modes(grid)
    q = snapshot_query(len(event_ids), len(modes))
    return [r[3] for r in conn.execute('EXPLAIN QUERY PLAN ' + q, modes + list(event_ids) + [start, end]).fetchall()]


def season_events(conn, start, end, grid=prices.GRID, margin_days=3):
    """
    Events that can contribute rows to the season: those with a hist-mode snapshot_log row whose ET game date lies within margin_days of the range, or whose
    tip date lies inside it (the row filter of load_season is on the tip date stored with each quote, which can differ from the ET date by a day).
    Returns event_id_vendor, game_date (ET date, else the tip date), month ('YYYY-MM'), sorted by date.
    """
    modes = _modes(grid)
    hist = pd.read_sql(f"SELECT DISTINCT event_id_vendor FROM snapshot_log WHERE mode IN ({','.join('?' * len(modes))})", conn, params=modes)
    ev = pd.read_sql('SELECT event_id_vendor, et_game_date, commence_time_utc FROM events', conn)
    ev = ev[ev['event_id_vendor'].isin(set(hist['event_id_vendor']))].copy()
    tip = ev['commence_time_utc'].astype(str).str[:10]
    ev['game_date'] = ev['et_game_date'].where(ev['et_game_date'].notna(), tip)
    lo = (pd.Timestamp(start) - pd.Timedelta(days=margin_days)).strftime('%Y-%m-%d')
    hi = (pd.Timestamp(end) + pd.Timedelta(days=margin_days)).strftime('%Y-%m-%d')
    keep = ev['game_date'].between(lo, hi) | tip.between(start, end)
    ev = ev[keep].sort_values(['game_date', 'event_id_vendor'])
    ev['month'] = ev['game_date'].astype(str).str[:7]
    return ev[['event_id_vendor', 'game_date', 'month']].reset_index(drop=True)


def read_chunk(conn, event_ids, start, end, grid=prices.GRID):
    """Snapshot rows of the given events with load_season's filters, plus minutes_before (the slot of the snapshot's mode)."""
    modes = _modes(grid)
    parts = []
    ids = list(event_ids)
    for lo in range(0, len(ids), EVENTS_PER_QUERY):
        batch = ids[lo:lo + EVENTS_PER_QUERY]
        parts.append(pd.read_sql(snapshot_query(len(batch), len(modes)), conn, params=modes + batch + [start, end]))
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['snapshot_ts_utc', 'game_id', 'event_id_vendor', 'book', 'player_id', 'player_name_raw',
                                                                                'market', 'line', 'over_price', 'under_price', 'is_alt_line', 'book_last_update_utc', 'mode'])
    if d.empty:
        d['minutes_before'] = pd.Series(dtype=int)
    else:
        d['minutes_before'] = d['mode'].str.replace('hist_T-', '', regex=False).astype(int)
    return d


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def finish_consensus(cons_raw, d):
    """The post-processing of prices.quotes_from_snapshots applied to analysis.consensus output (adds game, player and slot; one row per game-player-slot-market). Keeps n_books."""
    cons = cons_raw.dropna(subset=['consensus_fair_over'])
    keys = d[['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw', 'game_id', 'player_id', 'minutes_before']].dropna(subset=['player_id']).drop_duplicates(
        ['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw'])
    cons = cons.merge(keys, on=['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw'], how='inner', validate='many_to_one')
    cons['player_id'] = cons['player_id'].astype(int)
    key = ['game_id', 'player_id', 'minutes_before', 'market']
    cons = cons.sort_values(key + ['n_at_line', 'snapshot_ts_utc'], ascending=[True] * len(key) + [False, True]).drop_duplicates(key)
    return cons.rename(columns={'market': 'stat'})[['game_id', 'player_id', 'minutes_before', 'stat', 'snapshot_ts_utc', 'event_id_vendor', 'player_name_raw',
                                                     'consensus_line', 'consensus_fair_over', 'n_at_line', 'n_books', 'book_spread']]


def variant_tables(d, game_dates, min_books=prices.MIN_BOOKS, stale_minutes=prices.STALE_MINUTES):
    """{(devig, set): slim consensus table} for the 9 variants of snapshot rows d. game_dates: event_id_vendor -> ET game date."""
    out = {}
    if d.empty:
        return {v: pd.DataFrame(columns=SLIM_COLUMNS) for v in VARIANTS}
    for method in DEVIG_METHODS:
        flagged = analysis.flag_main_lines(analysis.add_devig(d, method=method, stale_after_minutes=stale_minutes))
        for sname, spec in CONSENSUS_SETS.items():
            cons = finish_consensus(analysis.consensus(flagged, min_books=min_books, agg=spec['agg'], books=spec['books']), d)
            cons['game_date'] = cons['event_id_vendor'].map(game_dates)
            out[(method, sname)] = cons[SLIM_COLUMNS].reset_index(drop=True)
        del flagged
    return {v: out[v] for v in VARIANTS}


def sort_prices(cons, bets):
    cons = cons.sort_values(['game_id', 'player_id', 'minutes_before', 'stat'], kind='mergesort').reset_index(drop=True)
    bets = bets.sort_values(SORT_BETS, kind='mergesort').reset_index(drop=True)
    return cons, bets


def _same_cons(a, b):
    cols = ['game_id', 'player_id', 'minutes_before', 'stat', 'snapshot_ts_utc', 'event_id_vendor', 'consensus_line', 'consensus_fair_over', 'n_at_line', 'book_spread']
    a = a[cols].sort_values(KEY).reset_index(drop=True)
    b = b[cols].sort_values(KEY).reset_index(drop=True)
    return a.equals(b)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Raw cache
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _raw_schema():
    import pyarrow as pa
    s, f = pa.string(), pa.float64()
    return pa.schema([('snapshot_ts_utc', s), ('game_id', s), ('event_id_vendor', s), ('book', s), ('player_id', f), ('player_name_raw', s), ('market', s), ('line', f),
                      ('over_price', f), ('under_price', f), ('is_alt_line', pa.int64()), ('book_last_update_utc', s), ('minutes_before', pa.int64()), ('game_date', s)])


class RawWriter:
    """Streams month chunks of the filtered snapshot rows to one parquet file (the cache the variants were built from)."""

    def __init__(self, path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.pa, self.path, self.schema = pa, Path(path), _raw_schema()
        self.tmp = self.path.with_suffix('.parquet.tmp')
        self.writer = pq.ParquetWriter(str(self.tmp), self.schema, compression='zstd')
        self.rows = 0

    def write(self, d, game_dates):
        if d.empty:
            return
        x = d.assign(game_date=d['event_id_vendor'].map(game_dates))[RAW_COLUMNS].copy()
        x['player_id'] = x['player_id'].astype(float)
        self.writer.write_table(self.pa.Table.from_pandas(x, schema=self.schema, preserve_index=False))
        self.rows += len(x)

    def close(self):
        self.writer.close()
        os.replace(self.tmp, self.path)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# One season
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def collect_season(props_db, start, end, chunk='month', limit_months=None, raw_writer=None, selfcheck=True, grid=prices.GRID, log=None):
    """
    Reads and processes one season. Returns dict(cons, bets, variants, months, plan). chunk='month' (default) or 'all' (a single pass; for tests and small stores).
    With selfcheck the generic current-method table of every chunk is compared with the one prices.quotes_from_snapshots builds (they must agree).
    """
    check_range(start, end)
    log = log or logging.getLogger('a11_cache')
    with contextlib.closing(connect_ro(props_db)) as conn:
        ev = season_events(conn, start, end, grid)
        if ev.empty:
            return {'cons': pd.DataFrame(), 'bets': pd.DataFrame(), 'variants': {v: pd.DataFrame(columns=SLIM_COLUMNS) for v in VARIANTS}, 'months': [], 'plan': []}
        plan = explain_plan(conn, ev['event_id_vendor'].head(2).tolist(), start, end, grid)
        groups = list(ev.groupby('month', sort=True)) if chunk == 'month' else [('all', ev)]
        if limit_months:
            groups = groups[:limit_months]
        game_dates = dict(zip(ev['event_id_vendor'], ev['game_date']))
        cons_parts, bets_parts, var_parts, months = [], [], {v: [] for v in VARIANTS}, []
        for month, g in groups:
            t0 = time.time()
            d = read_chunk(conn, g['event_id_vendor'].tolist(), start, end, grid)
            t_read = time.time() - t0
            if raw_writer is not None:
                raw_writer.write(d, game_dates)
            if d.empty:
                months.append({'month': month, 'events': int(len(g)), 'rows': 0, 'seconds_read': round(t_read, 1), 'seconds_total': round(time.time() - t0, 1)})
                continue
            cons, bets = prices.quotes_from_snapshots(d)
            tabs = variant_tables(d, game_dates)
            if selfcheck and not _same_cons(tabs[CURRENT], cons):
                raise RuntimeError(f'self-check failed in {month}: the generic current-method table differs from prices.quotes_from_snapshots')
            cons_parts.append(cons)
            bets_parts.append(bets)
            for v in VARIANTS:
                var_parts[v].append(tabs[v])
            months.append({'month': month, 'events': int(len(g)), 'rows': int(len(d)), 'consensus_rows_current': int(len(cons)), 'bettable_quotes': int(len(bets)),
                           'seconds_read': round(t_read, 1), 'seconds_total': round(time.time() - t0, 1)})
            log.info(f'{month}: {len(g)} events, {len(d)} rows, read {t_read:.0f}s, total {time.time() - t0:.0f}s')
            del d, tabs
    cons = pd.concat(cons_parts, ignore_index=True) if cons_parts else pd.DataFrame()
    bets = pd.concat(bets_parts, ignore_index=True) if bets_parts else pd.DataFrame()
    if len(cons):
        cons, bets = sort_prices(cons, bets)
    variants = {v: (pd.concat(p, ignore_index=True).sort_values(KEY, kind='mergesort').reset_index(drop=True) if p else pd.DataFrame(columns=SLIM_COLUMNS)) for v, p in var_parts.items()}
    return {'cons': cons, 'bets': bets, 'variants': variants, 'months': months, 'plan': plan}


def _atomic_pickle(obj, path):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def targets(season, out, prices_out, partial):
    tag = f'{season}.partial' if partial else season
    sdir = Path(out) / tag
    ppath = Path(prices_out) / f'prices_{tag}.pkl'
    return sdir, ppath


def run_season(props_db, season, out, prices_out, limit_months=None, skip_raw=False, overwrite=False, chunk='month', selfcheck=True, log=None):
    start, end = check_season(season)
    sdir, ppath = targets(season, out, prices_out, bool(limit_months))
    if (ppath.exists() or (sdir / 'manifest.json').exists()) and not overwrite:
        raise FileExistsError(f'{ppath} or {sdir / "manifest.json"} exists; pass --overwrite to replace it')
    sdir.mkdir(parents=True, exist_ok=True)
    Path(prices_out).mkdir(parents=True, exist_ok=True)
    writer = None if skip_raw else RawWriter(sdir / f'raw_{season}.parquet')
    try:
        res = collect_season(props_db, start, end, chunk=chunk, limit_months=limit_months, raw_writer=writer, selfcheck=selfcheck, log=log)
    finally:
        if writer is not None:
            writer.close()
    cons_out = res['cons'].drop(columns=[c for c in ('n_books',) if c in res['cons'].columns])
    _atomic_pickle((cons_out, res['bets']), ppath)
    for v, t in res['variants'].items():
        _atomic_pickle(t, sdir / f'cons_{variant_name(v)}.pkl')
    p = Path(props_db)
    manifest = {'season': season, 'start': start, 'end': end, 'partial': bool(limit_months), 'chunk': chunk, 'months': res['months'], 'query_plan': res['plan'],
                'rows_total': int(sum(m['rows'] for m in res['months'])), 'consensus_rows': {variant_name(v): int(len(t)) for v, t in res['variants'].items()},
                'prices_file': str(ppath), 'prices_consensus_rows': int(len(res['cons'])), 'prices_bettable_quotes': int(len(res['bets'])),
                'raw_cache_rows': (writer.rows if writer is not None else None), 'props_db': str(p), 'props_db_bytes': p.stat().st_size,
                'props_db_mtime': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(p.stat().st_mtime)), 'pandas': pd.__version__, 'numpy': np.__version__,
                'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}
    (sdir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return res, manifest


def dry_run(props_db, seasons, limit_months=None):
    """Monthly event counts, hist-mode row counts from snapshot_log and the query plan. Reads the two small tables (events, snapshot_log) only."""
    with contextlib.closing(connect_ro(props_db)) as conn:
        modes = _modes(prices.GRID)
        for season in seasons:
            start, end = check_season(season)
            ev = season_events(conn, start, end)
            print(f'\n== {season}: {start} .. {end}: {len(ev)} events with hist-mode snapshots')
            if ev.empty:
                continue
            log = pd.read_sql(f"SELECT event_id_vendor, SUM(n_prop_rows) AS n FROM snapshot_log WHERE mode IN ({','.join('?' * len(modes))}) GROUP BY event_id_vendor", conn, params=modes)
            ev = ev.merge(log, on='event_id_vendor', how='left')
            m = ev.groupby('month').agg(events=('event_id_vendor', 'size'), prop_rows_all_lines=('n', 'sum'))
            if limit_months:
                m = m.head(limit_months)
            print(m.to_string())
            print(f'total prop rows (all lines, upper bound before the main-line and time filters): {int(m["prop_rows_all_lines"].sum())}')
            print('EXPLAIN QUERY PLAN of the chunk query (two event ids):')
            for line in explain_plan(conn, ev['event_id_vendor'].head(2).tolist(), start, end):
                print('   ', line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--props-db', required=True)
    ap.add_argument('--seasons', nargs='+', required=True, help='2023-24 and/or 2024-25 (2025-26 is refused)')
    ap.add_argument('--out', required=True, help='quote cache folder (one sub-folder per season)')
    ap.add_argument('--prices-out', required=True, help='folder for prices_<season>.pkl')
    ap.add_argument('--limit-months', type=int, default=None, help='smoke run: only the first N months of each season (outputs are marked .partial)')
    ap.add_argument('--dry-run', action='store_true', help='print monthly event counts and the query plan; read nothing else, write nothing')
    ap.add_argument('--skip-raw', action='store_true', help='do not write the raw parquet cache')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no-selfcheck', action='store_true')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        for s in a.seasons:
            check_season(s)
        for d in (a.out, a.prices_out):
            if Path(d).resolve() == Path(a.props_db).resolve().parent:
                raise SeasonRefused(f'{d} is the folder of the odds store; pass a sub-folder')
        if a.dry_run:
            dry_run(a.props_db, a.seasons, a.limit_months)
            return 0
        for s in a.seasons:
            t0 = time.time()
            res, man = run_season(a.props_db, s, a.out, a.prices_out, a.limit_months, a.skip_raw, a.overwrite, selfcheck=not a.no_selfcheck)
            print(f'{s}: {man["rows_total"]} rows, {man["prices_consensus_rows"]} consensus rows, {man["prices_bettable_quotes"]} bettable quotes, {time.time() - t0:.0f}s')
            print('query plan:', *man['query_plan'], sep='\n   ')
    except SeasonRefused as e:
        sys.exit(f'REFUSED: {e}')
    return 0


if __name__ == '__main__':
    main()
