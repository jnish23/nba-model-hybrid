"""
From stored quotes to analysis-ready rows: de-vig, quality flags, main-line selection, consensus probability.

Why main-line selection exists: some books (Bovada 92% of player-markets, BetRivers 38% in the pilot) post a ladder of lines for
one player and market under the main market key, not marked alternate. Every quote is kept in the store; analysis uses one
line per (snapshot, book, player, market): the one whose de-vigged over probability is closest to 50%. Ladders stay available
(is_ladder, n_lines) for calibration work at specific lines.

Consensus: among books whose main line equals the consensus line (the most common main line; ties broken toward the median),
the mean de-vigged over probability, computed only when at least `min_books` books agree on that line.
"""
import numpy as np
import pandas as pd

from core import devig

KEY = ['snapshot_ts_utc', 'event_id_vendor', 'book', 'player_name_raw', 'market', 'is_alt_line']
GROUP = ['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw', 'market']


def add_devig(df, method='multiplicative', stale_after_minutes=60):
    """Adds fair_over, fair_under, booksum and the boolean flags one_sided, no_margin, wide, stale."""
    fo, fu, bs = devig.devig_prices(df['over_price'], df['under_price'], method)
    out = df.copy()
    out['fair_over'], out['fair_under'], out['booksum'] = fo, fu, bs
    flags = devig.quality_flags(df['over_price'], df['under_price'], df.get('book_last_update_utc'),
                                df['snapshot_ts_utc'], stale_after_minutes)
    flags.index = df.index
    return out.join(flags)


def flag_main_lines(df):
    """df must have fair_over (see add_devig). Adds n_lines, is_ladder and is_main_line (never set on alt-market rows)."""
    out = df.copy()
    out['n_lines'] = out.groupby(KEY)['line'].transform('size')
    out['is_ladder'] = out['n_lines'] > 1
    score = (out['fair_over'] - 0.5).abs().fillna(np.inf)                 # one-sided quotes only win by default
    order = out.assign(_score=score).sort_values(KEY + ['_score', 'line'], kind='mergesort').index
    first = pd.Series(~out.loc[order].duplicated(subset=KEY).to_numpy(), index=order)
    out['is_main_line'] = first.reindex(out.index) & (out['is_alt_line'] == 0)
    return out


def load_flagged(conn, where='1=1', params=(), method='multiplicative'):
    """Stored quotes with de-vig, quality flags and main-line flags. Rows tagged player_id = -1 (team-total ladders posted under a
    player market) are never returned; unresolved player_id NULL rows are returned (they are real players we could not identify)."""
    df = pd.read_sql(f"SELECT * FROM prop_snapshots WHERE ({where}) AND (player_id IS NULL OR player_id <> -1)", conn,
                     params=params)
    if df.empty:
        return df
    return flag_main_lines(add_devig(df, method))


def consensus(flagged, min_books=3, exclude_stale=True, exclude_flagged=True):
    """
    One row per (snapshot, event, player, market): consensus_line, n_books (quoting any main line), n_at_line,
    book_spread (max - min of main lines), consensus_fair_over (NaN unless n_at_line >= min_books).
    """
    m = flagged[flagged['is_main_line']].copy()
    ok = ~m['one_sided']
    if exclude_flagged:
        ok &= ~m['no_margin'] & ~m['wide']
    if exclude_stale:
        ok &= ~m['stale']
    m = m[ok]
    if m.empty:
        return pd.DataFrame(columns=GROUP + ['consensus_line', 'n_books', 'n_at_line', 'book_spread', 'consensus_fair_over'])
    med = m.groupby(GROUP)['line'].transform('median')
    cnt = m.groupby(GROUP + ['line'])['book'].transform('nunique')
    ranked = m.assign(_cnt=cnt, _dist=(m['line'] - med).abs()).sort_values(
        GROUP + ['_cnt', '_dist', 'line'], ascending=[True] * len(GROUP) + [False, True, True], kind='mergesort')
    pick = ranked.drop_duplicates(subset=GROUP)[GROUP + ['line']].rename(columns={'line': 'consensus_line'})
    m = m.merge(pick, on=GROUP, validate='many_to_one')
    at = m[m['line'] == m['consensus_line']]
    g = m.groupby(GROUP)
    res = pick.set_index(GROUP)
    res['n_books'] = g['book'].nunique()
    res['book_spread'] = g['line'].max() - g['line'].min()
    ag = at.groupby(GROUP)
    res['n_at_line'] = ag['book'].nunique()
    res['consensus_fair_over'] = ag['fair_over'].mean()
    res.loc[res['n_at_line'] < min_books, 'consensus_fair_over'] = np.nan
    return res.reset_index()
