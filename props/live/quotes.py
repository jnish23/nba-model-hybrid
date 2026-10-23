"""
Live quotes for one decision slot. The quote builder is the backtest's (props.betting.prices.quotes_from_snapshots); only the choice of the snapshot differs, because the live
collector snapshots every 30 minutes (plus a closing snapshot) instead of at exactly T-240, T-60 and T-5.

Slot rule (fixed in reports/props/live_pipeline_plan.md, section 10): the slot with nominal time T (minutes before the reference tip) uses
  1. the snapshot taken between T and T-SLOT_AFTER_MIN minutes before tip (at most SLOT_AFTER_MIN minutes AFTER the nominal time), the one closest to the nominal time; if none exists yet
     the job waits for one until the nominal time plus SLOT_AFTER_MIN has passed;
  2. then the freshest snapshot taken before the nominal time, if it is at most MAX_STALE_MIN minutes older than the nominal time;
  3. otherwise the slot is missed.
Whatever snapshot is used, every other input (injury reports, quotes) is limited to what was stamped at or before that snapshot.
"""
import sqlite3

import pandas as pd

from props.betting import prices

LIVE_MODES = ('slow', 'fast', 'close')
SLOT_AFTER_MIN = 10.0
MAX_STALE_MIN = 40.0
MIN_MINUTES_TO_TIP = 0.5            # a snapshot at or after the reference tip is in-game and never used
MAX_LATE_MIN = 25.0                 # a slot is not worked more than this long after its nominal time (the bet would not be a bet at T)


def choose_snapshot(snapshot_minutes, T, now_minutes):
    """
    snapshot_minutes: Series indexed by snapshot_ts_utc with the minutes before the reference tip at which each snapshot was taken. T: nominal slot minutes. now_minutes: minutes
    before the reference tip now. Returns (status, snapshot_ts_utc or None, minutes) with status 'wait' (a fresher snapshot may still come), 'use' or 'missed'.
    """
    if now_minutes > T:
        return 'wait', None, None                                                       # the nominal time has not arrived
    if now_minutes <= MIN_MINUTES_TO_TIP or T - now_minutes > MAX_LATE_MIN:
        return 'missed', None, None                                                     # the game is under way, or the job was down too long: no bet at T
    m = snapshot_minutes[snapshot_minutes >= now_minutes]                               # snapshots already taken
    after = m[(m <= T) & (m >= T - SLOT_AFTER_MIN) & (m >= MIN_MINUTES_TO_TIP)]
    if len(after):
        ts = after.idxmax()
        return 'use', ts, float(after[ts])
    wait_until = T - SLOT_AFTER_MIN if T >= 3 * SLOT_AFTER_MIN else T                   # the last slot (T-5) does not wait: the closing snapshot is the freshest there will be
    if now_minutes > wait_until:
        return 'wait', None, None
    before = m[(m > T) & (m <= T + MAX_STALE_MIN)]
    if len(before):
        ts = before.idxmin()
        return 'use', ts, float(before[ts])
    return 'missed', None, None


def event_snapshots(props_db, event_id, since_utc=None):
    """Snapshot timestamps of one vendor event taken by the live collector (not the historical backfill)."""
    q = f"SELECT snapshot_ts_utc, mode FROM snapshot_log WHERE event_id_vendor = ? AND mode IN ({','.join('?' * len(LIVE_MODES))})"
    params = [event_id, *LIVE_MODES]
    if since_utc:
        q += ' AND snapshot_ts_utc >= ?'
        params.append(since_utc)
    with sqlite3.connect(f'file:{props_db}?mode=ro', uri=True, timeout=120) as conn:
        return pd.read_sql(q, conn, params=params)


def minutes_before_tip(snapshot_ts_utc, tip_ref_utc):
    return (pd.to_datetime(tip_ref_utc, utc=True) - pd.to_datetime(snapshot_ts_utc, utc=True)).total_seconds() / 60.0


def slot_quotes(props_db, event_id, game_id, snapshot_ts_utc, T, min_books=prices.MIN_BOOKS):
    """(consensus, bets, raw snapshot rows) of one snapshot served as the slot with nominal minutes T. game_id is the NBA game matched to the vendor event."""
    q = """SELECT snapshot_ts_utc, event_id_vendor, book, player_id, player_name_raw, market, line, over_price, under_price, is_alt_line, book_last_update_utc, home_team, away_team
           FROM prop_snapshots WHERE event_id_vendor = ? AND snapshot_ts_utc = ? AND is_alt_line = 0 AND player_id IS NOT NULL AND player_id <> -1"""
    with sqlite3.connect(f'file:{props_db}?mode=ro', uri=True, timeout=120) as conn:
        d = pd.read_sql(q, conn, params=[event_id, snapshot_ts_utc])
    d['game_id'] = game_id
    d['minutes_before'] = int(T)
    if d.empty:
        return None, None, d
    cons, bets = prices.quotes_from_snapshots(d, min_books)
    return cons, bets, d
