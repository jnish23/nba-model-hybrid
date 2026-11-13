"""props.live.fills: price path of a recorded bet through the later snapshots (synthetic sequences) and the read-only loader."""
import sqlite3

import numpy as np
import pandas as pd

from props.live import fills, store

T0 = '2026-11-13T00:10:00Z'           # the bet's snapshot
TIP = '2026-11-13T01:10:00Z'          # 60 minutes later
BET = {'bet_id': 'S|g1|1|points', 'game_id': 'g1', 'player_id': 1, 'stat': 'points', 'side': 'over', 'line': 5.5, 'book': 'draftkings', 'dec_best': 2.05, 'snapshot_ts_utc': T0, 'tip_ref_utc': TIP}
# american prices (over, under) of the four books at the bet time: draftkings +105 is the best over price (decimal 2.05)
BOOKS = {'draftkings': (105, -125), 'fanduel': (100, -120), 'betmgm': (-105, -115), 'williamhill_us': (-110, -110)}


def _snap(ts, minutes, books=BOOKS, line=5.5, **moves):
    rows = []
    for book, (o, u) in books.items():
        mv, ln = moves.get(book), line
        if mv == 'pull':
            continue
        if mv is not None:
            ln, o, u = mv
        rows.append({'snapshot_ts_utc': ts, 'minutes_before': minutes, 'event_id_vendor': 'e1', 'book': book, 'player_id': 1, 'player_name_raw': 'P', 'market': 'points', 'line': ln,
                     'over_price': o, 'under_price': u, 'is_alt_line': 0, 'book_last_update_utc': ts})
    return pd.DataFrame(rows)


def _empty_snap(ts, minutes):
    """A later snapshot in which no book quotes this player-market at all (placeholder row)."""
    return pd.DataFrame([{'snapshot_ts_utc': ts, 'minutes_before': minutes, 'event_id_vendor': 'e1', 'book': None, 'player_id': None, 'player_name_raw': None, 'market': None, 'line': np.nan,
                          'over_price': np.nan, 'under_price': np.nan, 'is_alt_line': None, 'book_last_update_utc': None}])


def _run(*snaps, bet=BET):
    q = pd.concat(snaps, ignore_index=True) if snaps else pd.DataFrame()
    path = fills.price_path(bet, q)
    return path, fills.summarise_path(bet, path, '2026-11-13T09:00:00Z', 'run')


def test_price_holds_at_every_later_snapshot():
    path, row = _run(_snap('2026-11-13T00:40:00Z', 30.0), _snap('2026-11-13T01:00:00Z', 10.0))
    assert list(path['minutes_after']) == [30.0, 50.0] and list(path['minutes_before']) == [30.0, 10.0]
    assert np.allclose(path['book_price'], 2.05) and np.allclose(path['best_price'], 2.05) and np.allclose(path['consensus_line'], 5.5)
    assert row['n_later_snapshots'] == 2 and row['next_snapshot_ts_utc'] == '2026-11-13T00:40:00Z' and row['next_minutes_after'] == 30.0
    assert row['book_still_at_or_better_next'] == 1 and row['best_still_at_or_better_next'] == 1 and abs(row['book_price_next'] - 2.05) < 1e-9
    assert row['price_ended'] == 0 and row['minutes_price_available'] == 60.0 and row['minutes_price_available_lo'] == 60.0      # right-censored at the tip
    assert abs(row['max_best_price_before_tip'] - 2.05) < 1e-9 and row['line_next'] == 5.5 and row['median_price_next'] is not None


def test_price_worsens_at_the_next_snapshot_but_another_book_still_has_it():
    # draftkings drops to +100 (2.00); fanduel is raised to +105: the best bettable price is unchanged, the bet book's is not
    path, row = _run(_snap('2026-11-13T00:40:00Z', 30.0, draftkings=(5.5, 100, -120), fanduel=(5.5, 105, -125)), _snap('2026-11-13T01:00:00Z', 10.0))
    assert row['book_still_at_or_better_next'] == 0 and row['best_still_at_or_better_next'] == 1
    assert abs(row['book_price_next'] - 2.0) < 1e-9 and abs(row['best_price_next'] - 2.05) < 1e-9
    assert row['price_ended'] == 1 and row['minutes_price_available'] == 30.0 and row['minutes_price_available_lo'] == 0.0       # gone somewhere in the first 30 minutes


def test_price_worsens_at_the_second_snapshot_gives_bounds_between_the_snapshots():
    _, row = _run(_snap('2026-11-13T00:40:00Z', 30.0), _snap('2026-11-13T01:00:00Z', 10.0, draftkings=(5.5, -105, -115)))
    assert row['book_still_at_or_better_next'] == 1 and row['minutes_price_available_lo'] == 30.0 and row['minutes_price_available'] == 50.0 and row['price_ended'] == 1


def test_book_moves_the_line_counts_as_price_gone_and_best_is_read_at_the_bet_line():
    path, row = _run(_snap('2026-11-13T00:40:00Z', 30.0, draftkings=(6.5, 105, -125)))
    assert path['book_line'].iloc[0] == 6.5 and np.isnan(path['book_price'].iloc[0])
    assert row['book_still_at_or_better_next'] == 0 and row['book_price_next'] is None and row['price_ended'] == 1 and row['minutes_price_available'] == 30.0
    # the best price at the bet line now comes from fanduel (+100 = 2.00): worse than the paper price
    assert abs(row['best_price_next'] - 2.0) < 1e-9 and row['best_still_at_or_better_next'] == 0 and row['line_next'] == 5.5


def test_book_pulls_the_market():
    path, row = _run(_snap('2026-11-13T00:40:00Z', 30.0, draftkings='pull'))
    assert np.isnan(path['book_line'].iloc[0]) and np.isnan(path['book_price'].iloc[0])
    assert row['book_still_at_or_better_next'] == 0 and row['price_ended'] == 1 and row['book_price_next'] is None


def test_every_book_pulls_the_market_still_counts_the_snapshot():
    path, row = _run(_empty_snap('2026-11-13T00:40:00Z', 30.0))
    assert len(path) == 1 and np.isnan(path['best_price'].iloc[0]) and np.isnan(path['consensus_line'].iloc[0])
    assert row['n_later_snapshots'] == 1 and row['book_still_at_or_better_next'] == 0 and row['best_still_at_or_better_next'] == 0 and row['best_price_next'] is None


def test_no_later_snapshot_gives_a_zero_count_row_with_nulls():
    path, row = _run()
    assert len(path) == 0 and row['n_later_snapshots'] == 0
    assert all(row[c] is None for c in ('next_snapshot_ts_utc', 'book_price_next', 'book_still_at_or_better_next', 'minutes_price_available', 'price_ended'))


def test_bet_without_a_snapshot_time_gets_null_count():
    bet = {**BET, 'snapshot_ts_utc': None}
    row = fills.summarise_path(bet, fills.price_path(bet, pd.DataFrame()), 'now', 'run')
    assert row['n_later_snapshots'] is None and row['bet_id'] == BET['bet_id'] and list(row) == fills.CHECK_COLS
    nan_bet = {**BET, 'snapshot_ts_utc': float('nan')}
    assert fills.summarise_path(nan_bet, fills.price_path(nan_bet, pd.DataFrame()), 'now', 'run')['n_later_snapshots'] is None


def test_under_bet_reads_the_under_price():
    bet = {**BET, 'side': 'under', 'book': 'draftkings', 'dec_best': 1.8}          # -125 = 1.80
    _, row = _run(_snap('2026-11-13T00:40:00Z', 30.0), bet=bet)
    assert abs(row['book_price_next'] - 1.8) < 1e-9 and row['book_still_at_or_better_next'] == 1


# ---- loader --------------------------------------------------------------------------------------------------------------------------------------------------------------

def _odds_db(path):
    c = sqlite3.connect(path)
    c.executescript("""CREATE TABLE snapshot_log (snapshot_ts_utc TEXT, event_id_vendor TEXT, mode TEXT, minutes_to_tip REAL, n_prop_rows INT, PRIMARY KEY (snapshot_ts_utc, event_id_vendor));
        CREATE TABLE events (event_id_vendor TEXT PRIMARY KEY, game_id TEXT);
        CREATE TABLE event_ref (event_id_vendor TEXT PRIMARY KEY, game_id TEXT);
        CREATE TABLE prop_snapshots (snapshot_ts_utc TEXT, game_id TEXT, event_id_vendor TEXT, book TEXT, player_id INT, player_name_raw TEXT, market TEXT, line REAL, over_price REAL,
                                     under_price REAL, is_alt_line INT, book_last_update_utc TEXT);""")
    c.execute("INSERT INTO events VALUES ('e1', NULL)")
    c.execute("INSERT INTO event_ref VALUES ('e1', 'g1')")
    logs = [(T0, 'slow'), ('2026-11-13T00:40:00Z', 'slow'), ('2026-11-13T01:00:00Z', 'close'), ('2026-11-13T00:50:00Z', 'hist_T-60'), ('2026-11-13T01:12:00Z', 'slow'), ('2026-11-13T00:00:00Z', 'slow')]
    c.executemany("INSERT INTO snapshot_log VALUES (?, 'e1', ?, NULL, 100)", logs)
    # a live snapshot with prop rows for another player only (the bet player-market vanished) and one whose rows exist for everybody
    for ts in (T0, '2026-11-13T00:40:00Z', '2026-11-13T01:00:00Z', '2026-11-13T00:00:00Z'):
        s = _snap(ts, 0)
        for r in s.itertuples():
            pid = 2 if (ts == '2026-11-13T01:00:00Z') else r.player_id
            c.execute("INSERT INTO prop_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (ts, 'g1', 'e1', r.book, pid, r.player_name_raw, r.market, r.line, r.over_price, r.under_price, 0, ts))
    c.commit()
    c.close()


def test_loader_picks_later_live_snapshots_before_tip_and_keeps_empty_ones(tmp_path):
    db = tmp_path / 'odds.db'
    _odds_db(db)
    conn = fills._ro(db.as_posix())
    q = fills.load_later_quotes(conn, BET)
    conn.close()
    assert sorted(q['snapshot_ts_utc'].unique()) == ['2026-11-13T00:40:00Z', '2026-11-13T01:00:00Z']      # not the bet snapshot, the earlier one, the hist one or the in-game one
    path = fills.price_path(BET, q)
    assert len(path) == 2 and np.isclose(path['book_price'].iloc[0], 2.05) and np.isnan(path['book_price'].iloc[1])        # at 01:00 only another player is quoted


def test_fill_checks_runs_read_only_and_writes_a_row_per_bet(tmp_path):
    db = tmp_path / 'odds.db'
    _odds_db(db)
    restored = {**BET, 'bet_id': 'S|g1|9|points', 'snapshot_ts_utc': None}
    out = fills.fill_checks(pd.DataFrame([BET, restored]), db.as_posix(), '2026-11-13T09:00:00Z', 'run')
    assert list(out.columns) == fills.CHECK_COLS and out['n_later_snapshots'].iloc[0] == 2 and pd.isna(out['n_later_snapshots'].iloc[1])
    assert out['book_still_at_or_better_next'].iloc[0] == 1 and out['price_ended'].iloc[0] == 1 and out['minutes_price_available'].iloc[0] == 50.0
    # the rows load into the real table
    lc = store.connect(tmp_path / 'live.db')
    with lc:
        assert store._insert(lc, 'prop_fill_checks', out, ignore=True) == 2
        assert store._insert(lc, 'prop_fill_checks', out, ignore=True) == 0                 # re-insert of the same bets is ignored (bet_id is the key)
    import pytest
    with pytest.raises(sqlite3.DatabaseError, match='append-only'):
        lc.execute("UPDATE prop_fill_checks SET n_later_snapshots = 0")
    lc.close()
    # a write attempt on the odds store through the read-only connection must fail
    ro = fills._ro(db.as_posix())
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("DELETE FROM prop_snapshots")
    ro.close()


def test_unknown_event_gives_no_later_snapshots(tmp_path):
    db = tmp_path / 'odds.db'
    _odds_db(db)
    out = fills.fill_checks(pd.DataFrame([{**BET, 'game_id': 'gX'}]), db.as_posix(), 'now', 'run')
    assert out['n_later_snapshots'].iloc[0] == 0
