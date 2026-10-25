"""
Bettable prices and the consensus, point in time (Phase 5 plan, section 1).

A quote is usable for a bet at decision time T only if its snapshot was stamped at or before T (minutes_to_ref >= T), the game was not under way, the book updated
it within STALE_MINUTES, both sides are priced, it is the book's MAIN line for that player-market and it sits on the consensus line (at least MIN_BOOKS agreeing
books, counting every book we hold, bettable or not). The opening line is never a price here.
"""
import sqlite3

import numpy as np
import pandas as pd

from props import analysis

BETTABLE = ('draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'betrivers', 'unibet_us', 'fanatics')      # the user's US-regulated set (williamhill_us = Caesars)
GRID = (240, 60, 5)
MIN_BOOKS = 3
STALE_MINUTES = 60


def american_to_decimal(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))


def load_season(props_db, start, end, grid=GRID, min_books=MIN_BOOKS):
    """Returns (consensus, bets). consensus: one row per (game_id, player_id, minutes_before, market). bets: one row per bettable-book main-line quote on the consensus line."""
    modes = [f'hist_T-{x}' for x in grid]
    q = f"""SELECT p.snapshot_ts_utc, p.game_id, p.event_id_vendor, p.book, p.player_id, p.player_name_raw, p.market, p.line, p.over_price, p.under_price,
                   p.is_alt_line, p.book_last_update_utc, l.mode
            FROM prop_snapshots p
            JOIN snapshot_log l ON l.snapshot_ts_utc = p.snapshot_ts_utc AND l.event_id_vendor = p.event_id_vendor
            LEFT JOIN snapshot_ref s ON s.snapshot_ts_utc = p.snapshot_ts_utc AND s.event_id_vendor = p.event_id_vendor
            WHERE l.mode IN ({','.join('?' * len(modes))}) AND p.is_alt_line = 0 AND p.game_id IS NOT NULL AND (p.player_id IS NULL OR p.player_id <> -1)
              AND COALESCE(s.in_game, 0) = 0 AND s.minutes_to_ref >= CAST(substr(l.mode, 8) AS REAL)
              AND substr(p.commence_time_utc, 1, 10) BETWEEN ? AND ?"""
    with sqlite3.connect(f'file:{props_db}?mode=ro', uri=True, timeout=120) as conn:
        d = pd.read_sql(q, conn, params=modes + [start, end])
    d['minutes_before'] = d['mode'].str.replace('hist_T-', '', regex=False).astype(int)
    return quotes_from_snapshots(d, min_books)


def quotes_from_snapshots(d, min_books=MIN_BOOKS):
    """
    The consensus and the bettable quotes from snapshot rows `d` (the columns of load_season's query plus minutes_before, the decision slot each row serves). Shared by the backtest and the
    live job so both build quotes with the same code. One slot per call when a snapshot can serve several slots.
    """
    flagged = analysis.flag_main_lines(analysis.add_devig(d, stale_after_minutes=STALE_MINUTES))
    cons = analysis.consensus(flagged, min_books=min_books).dropna(subset=['consensus_fair_over'])
    keys = d[['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw', 'game_id', 'player_id', 'minutes_before']].dropna(subset=['player_id']).drop_duplicates(
        ['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw'])
    cons = cons.merge(keys, on=['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw'], how='inner', validate='many_to_one')
    cons['player_id'] = cons['player_id'].astype(int)
    key = ['game_id', 'player_id', 'minutes_before', 'market']
    cons = cons.sort_values(key + ['n_at_line', 'snapshot_ts_utc'], ascending=[True] * len(key) + [False, True]).drop_duplicates(key)
    cons = cons.rename(columns={'market': 'stat'})[['game_id', 'player_id', 'minutes_before', 'stat', 'snapshot_ts_utc', 'event_id_vendor', 'player_name_raw',
                                                     'consensus_line', 'consensus_fair_over', 'n_at_line', 'book_spread']]
    b = flagged[flagged['is_main_line'] & flagged['book'].isin(BETTABLE) & ~flagged['one_sided'] & ~flagged['no_margin'] & ~flagged['wide'] & ~flagged['stale']].copy()
    b = b.rename(columns={'market': 'stat'})
    b = b.merge(cons[['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw', 'stat', 'consensus_line', 'consensus_fair_over', 'n_at_line', 'player_id', 'game_id']]
                .rename(columns={'player_id': 'cons_player_id', 'game_id': 'cons_game_id'}),
                on=['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw', 'stat'], how='inner')
    b = b[b['line'] == b['consensus_line']]
    b['player_id'] = b['cons_player_id']
    b['game_id'] = b['cons_game_id']
    b['quote_age_min'] = (pd.to_datetime(b['snapshot_ts_utc'], utc=True) - pd.to_datetime(b['book_last_update_utc'], utc=True)).dt.total_seconds() / 60.0
    b['over_dec'] = american_to_decimal(b['over_price'])
    b['under_dec'] = american_to_decimal(b['under_price'])
    bets = b[['game_id', 'player_id', 'minutes_before', 'stat', 'book', 'line', 'over_price', 'under_price', 'over_dec', 'under_dec', 'fair_over', 'consensus_fair_over',
              'n_at_line', 'snapshot_ts_utc', 'quote_age_min']].reset_index(drop=True)
    return cons, bets
