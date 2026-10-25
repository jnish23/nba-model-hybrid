"""Vendor events of the live collector, matched to NBA games (shared by the signal job and the current-odds export)."""
import sqlite3

import pandas as pd


def iso(ts):
    return pd.Timestamp(ts).tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')


def events_in_window(odds_db, now, hours_back=1.0, hours_ahead=7.0):
    """Events whose vendor tip lies in [now - hours_back, now + hours_ahead]; the tip is the vendor's EARLIEST estimate (the reference clock uses it, as in the backtest)."""
    lo, hi = iso(now - pd.Timedelta(hours=hours_back)), iso(now + pd.Timedelta(hours=hours_ahead))
    with sqlite3.connect(f'file:{odds_db}?mode=ro', uri=True, timeout=120) as conn:
        ev = pd.read_sql('SELECT event_id_vendor, home_team, away_team, commence_time_utc FROM events WHERE commence_time_utc BETWEEN ? AND ?', conn, params=[lo, hi])
        if ev.empty:
            return ev
        first = pd.read_sql(f"SELECT event_id_vendor, MIN(commence_time_utc) AS first_tip FROM prop_snapshots WHERE snapshot_ts_utc >= ? AND event_id_vendor IN ({','.join('?' * len(ev))}) "
                            'GROUP BY event_id_vendor', conn, params=[iso(now - pd.Timedelta(days=3)), *ev['event_id_vendor']])
    ev = ev.merge(first, on='event_id_vendor', how='left')
    ev['commence_time_utc'] = ev['first_tip'].fillna(ev['commence_time_utc'])
    return ev.drop(columns='first_tip')
