"""
Fill realism: did the price a paper bet was recorded at survive after the quote time?

The paper record assumes an instant fill at the best of the bettable books at snapshot_ts_utc. A limit order on an exchange placed after the signal (the signal is written about 4 minutes after the
snapshot, about 9 minutes for T-5 bets) is only worth the paper price if the sportsbook price it is pegged to is still there. For one bet this module follows the same player-market through the
LATER live snapshots of the same event (before the reference tip) and reports, per snapshot, the bet book's price on the bet side at the bet line, the best and the median bettable-book price
at that line and the consensus line. The quote builder is the backtest's: props.analysis add_devig / flag_main_lines / consensus and props.betting.prices.BETTABLE, so a "price" here is a main-line price
of a bettable book, exactly as at bet time. Read-only: the odds store is only ever opened with mode=ro.

Pure parts: price_path (per-snapshot path) and summarise_path (one prop_fill_checks row). I/O parts: load_later_quotes / fill_checks.
Snapshot cadence limits the resolution: a price seen at one snapshot and gone at the next left at some unknown time in between, so the availability time is given as a lower and an upper bound.
"""
import sqlite3

import numpy as np
import pandas as pd

from props import analysis
from props.betting import prices
from props.live import quotes

PRICE_EPS = 1e-6                   # decimal-price tolerance when comparing a later price with the recorded dec_best
LINE_EPS = 1e-9
CHECK_COLS = ['bet_id', 'checked_utc', 'n_later_snapshots', 'next_snapshot_ts_utc', 'next_minutes_after', 'book_price_next', 'best_price_next', 'median_price_next', 'line_next',
              'book_still_at_or_better_next', 'best_still_at_or_better_next', 'minutes_price_available', 'max_best_price_before_tip', 'run_id', 'minutes_price_available_lo', 'price_ended']
PATH_COLS = ['snapshot_ts_utc', 'minutes_before', 'minutes_after', 'book_line', 'book_price', 'best_price', 'median_price', 'n_books_at_line', 'consensus_line']
QUOTE_COLS = ['snapshot_ts_utc', 'minutes_before', 'event_id_vendor', 'book', 'player_id', 'player_name_raw', 'market', 'line', 'over_price', 'under_price', 'is_alt_line',
              'book_last_update_utc']


def _minutes_between(a_utc, b_utc):
    return (pd.to_datetime(b_utc, utc=True) - pd.to_datetime(a_utc, utc=True)).total_seconds() / 60.0


def _empty_path():
    return pd.DataFrame({c: pd.Series(dtype=float if c not in ('snapshot_ts_utc',) else object) for c in PATH_COLS})


def price_path(bet, later_quotes):
    """
    bet: mapping with side, line, book, stat, player_id, snapshot_ts_utc. later_quotes: raw prop_snapshots rows of the bet's player-market for the later snapshots, with a minutes_before column
    (minutes before the reference tip), PLUS one placeholder row per later snapshot that has no row for this player-market (book NaN), so a market pulled by every book still shows up as a snapshot.
    Returns one row per later snapshot in time order (PATH_COLS):
      book_price / book_line: the bet book's MAIN-line price on the bet side and that main line; NaN when the book's main line is no longer the bet line (line moved) or it has no row (pulled);
      best_price / median_price / n_books_at_line: over bettable books whose main line is the bet line, passing the same quality flags as at bet time; NaN when none;
      consensus_line: the consensus line at that snapshot (NaN when fewer than MIN_BOOKS books agree).
    """
    q = later_quotes
    if q is None or len(q) == 0:
        return _empty_path()
    snaps = q[['snapshot_ts_utc', 'minutes_before']].drop_duplicates('snapshot_ts_utc').sort_values('snapshot_ts_utc').reset_index(drop=True)
    d = q[q['book'].notna()].copy()
    if 'market' in d.columns:
        d = d[d['market'] == bet['stat']]
    if 'player_id' in d.columns and d['player_id'].notna().any():
        d = d[d['player_id'] == bet['player_id']]
    for c, v in (('is_alt_line', 0), ('book_last_update_utc', None), ('player_name_raw', ''), ('event_id_vendor', '')):
        if c not in d.columns:
            d[c] = v
    d = d[d['is_alt_line'] == 0]
    over = str(bet['side']) == 'over'
    line = float(bet['line'])
    out = pd.DataFrame({'snapshot_ts_utc': snaps['snapshot_ts_utc'], 'minutes_before': snaps['minutes_before'].astype(float)})
    out['minutes_after'] = [_minutes_between(bet['snapshot_ts_utc'], t) for t in out['snapshot_ts_utc']]
    for c in ('book_line', 'book_price', 'best_price', 'median_price', 'n_books_at_line', 'consensus_line'):
        out[c] = np.nan
    if len(d):
        flagged = analysis.flag_main_lines(analysis.add_devig(d.reset_index(drop=True), stale_after_minutes=prices.STALE_MINUTES))
        flagged['dec'] = prices.american_to_decimal(flagged['over_price'] if over else flagged['under_price'])
        cons = analysis.consensus(flagged, min_books=prices.MIN_BOOKS).dropna(subset=['consensus_fair_over'])
        cons_line = cons.groupby('snapshot_ts_utc')['consensus_line'].first()
        main = flagged[flagged['is_main_line']]
        for i, ts in enumerate(out['snapshot_ts_utc']):
            m = main[main['snapshot_ts_utc'] == ts]
            mine = m[m['book'] == bet['book']]
            if len(mine):
                out.loc[i, 'book_line'] = float(mine['line'].iloc[0])
                if abs(float(mine['line'].iloc[0]) - line) <= LINE_EPS and np.isfinite(mine['dec'].iloc[0]):
                    out.loc[i, 'book_price'] = float(mine['dec'].iloc[0])
            at = m[(np.abs(m['line'] - line) <= LINE_EPS) & m['book'].isin(prices.BETTABLE) & ~m['one_sided'] & ~m['no_margin'] & ~m['wide'] & ~m['stale'] & np.isfinite(m['dec'])]
            if len(at):
                out.loc[i, 'best_price'], out.loc[i, 'median_price'], out.loc[i, 'n_books_at_line'] = float(at['dec'].max()), float(at['dec'].median()), float(at['book'].nunique())
            if ts in cons_line.index:
                out.loc[i, 'consensus_line'] = float(cons_line.loc[ts])
    return out[PATH_COLS]


def _ok(price, ref):
    return bool(np.isfinite(price) and price >= ref - PRICE_EPS)


def summarise_path(bet, path, checked_utc=None, run_id=None):
    """
    One prop_fill_checks row (dict, CHECK_COLS) from price_path(bet, ...). The reference price is the paper price bet['dec_best'].
    book_still_at_or_better_next: the bet book's main-line price at the bet line at the next snapshot is >= dec_best. best_still_at_or_better_next: the same for the best bettable price.
    minutes_price_available: minutes from the bet's snapshot until the first later snapshot at which the bet book's price at the bet line was worse than dec_best, the line had moved or the market
    was pulled (an upper bound: it went away at some point between the previous snapshot and that one); minutes_price_available_lo is the time to the last snapshot that still showed it (lower
    bound). If it never went away, both are the minutes from the bet's snapshot to the reference tip and price_ended is 0 (right-censored at the tip).
    A bet with no later snapshot has n_later_snapshots 0 and NULLs; a bet with no snapshot time (restored rows) has n_later_snapshots NULL.
    """
    row = {c: None for c in CHECK_COLS}
    row.update(bet_id=bet['bet_id'], checked_utc=checked_utc, run_id=run_id)
    if pd.isna(bet.get('snapshot_ts_utc')):
        return row
    row['n_later_snapshots'] = int(len(path))
    if len(path) == 0:
        return row
    ref = float(bet['dec_best'])
    first = path.iloc[0]
    row.update(next_snapshot_ts_utc=first['snapshot_ts_utc'], next_minutes_after=float(first['minutes_after']),
               book_price_next=_nan_none(first['book_price']), best_price_next=_nan_none(first['best_price']), median_price_next=_nan_none(first['median_price']),
               line_next=_nan_none(first['consensus_line']), book_still_at_or_better_next=int(_ok(first['book_price'], ref)), best_still_at_or_better_next=int(_ok(first['best_price'], ref)),
               max_best_price_before_tip=_nan_none(path['best_price'].max()))
    good = [_ok(p, ref) for p in path['book_price']]
    after = path['minutes_after'].to_numpy(float)
    if all(good):
        end = _minutes_between(bet['snapshot_ts_utc'], bet['tip_ref_utc']) if bet.get('tip_ref_utc') else float(after[-1])
        row.update(minutes_price_available=end, minutes_price_available_lo=end, price_ended=0)
    else:
        j = good.index(False)
        row.update(minutes_price_available=float(after[j]), minutes_price_available_lo=float(after[j - 1]) if j > 0 else 0.0, price_ended=1)
    return row


def _nan_none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else float(x)


def _ro(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=120)


def event_for_bet(conn, bet):
    """The vendor event whose snapshot the bet was recorded from: the one mapped to the bet's game that has a snapshot_log row at the bet's snapshot time."""
    q = """SELECT l.event_id_vendor FROM snapshot_log l LEFT JOIN events e USING (event_id_vendor) LEFT JOIN event_ref r USING (event_id_vendor)
           WHERE l.snapshot_ts_utc = ? AND COALESCE(r.game_id, e.game_id) = ?"""
    got = conn.execute(q, (bet['snapshot_ts_utc'], str(bet['game_id']))).fetchall()
    return got[0][0] if got else None


def load_later_quotes(conn, bet, event_id=None, modes=quotes.LIVE_MODES):
    """
    Read-only: later live snapshots of the bet's event (strictly after the bet's snapshot, at least MIN_MINUTES_TO_TIP before the reference tip, with prop rows) and the bet player-market's raw rows in
    them, as price_path expects (placeholder rows for snapshots without a row of this player-market).
    """
    event_id = event_id or event_for_bet(conn, bet)
    if event_id is None:
        return pd.DataFrame(columns=QUOTE_COLS)
    ph = ','.join('?' * len(modes))
    snaps = pd.read_sql(f'SELECT snapshot_ts_utc FROM snapshot_log WHERE event_id_vendor = ? AND snapshot_ts_utc > ? AND n_prop_rows > 0 AND mode IN ({ph}) ORDER BY snapshot_ts_utc',
                        conn, params=[event_id, bet['snapshot_ts_utc'], *modes])
    snaps['minutes_before'] = [quotes.minutes_before_tip(t, bet['tip_ref_utc']) for t in snaps['snapshot_ts_utc']]
    snaps = snaps[snaps['minutes_before'] >= quotes.MIN_MINUTES_TO_TIP].reset_index(drop=True)
    if snaps.empty:
        return pd.DataFrame(columns=QUOTE_COLS)
    rows = pd.read_sql('SELECT snapshot_ts_utc, event_id_vendor, book, player_id, player_name_raw, market, line, over_price, under_price, is_alt_line, book_last_update_utc FROM prop_snapshots '
                       'WHERE event_id_vendor = ? AND snapshot_ts_utc >= ? AND snapshot_ts_utc <= ? AND player_id = ? AND market = ? AND is_alt_line = 0',
                       conn, params=[event_id, snaps['snapshot_ts_utc'].iloc[0], snaps['snapshot_ts_utc'].iloc[-1], int(bet['player_id']), bet['stat']])
    rows = rows[rows['snapshot_ts_utc'].isin(set(snaps['snapshot_ts_utc']))].merge(snaps, on='snapshot_ts_utc', how='right')     # right join keeps snapshots with no row of this player-market
    return rows[QUOTE_COLS]


def fill_checks(bets, props_db, checked_utc, run_id):
    """
    prop_fill_checks rows (DataFrame, CHECK_COLS) for `bets` (prop_bets rows). Opens props_db read-only. Exceptions propagate: the settle job wraps the call in try/except so that a fill-check
    failure never blocks settlement.
    """
    rows = []
    conn = _ro(props_db)
    try:
        for bet in bets.to_dict('records'):
            if pd.isna(bet.get('snapshot_ts_utc')):
                rows.append(summarise_path(bet, _empty_path(), checked_utc, run_id))
                continue
            later = load_later_quotes(conn, bet)
            rows.append(summarise_path(bet, price_path(bet, later), checked_utc, run_id))
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=CHECK_COLS)
