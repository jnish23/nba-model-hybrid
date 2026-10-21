"""
Health check for the live props collector. Run after a game day:
    python scripts/check_props_collection.py                 # today's ET game date
    python scripts/check_props_collection.py --date 2026-10-20

Per game: tip time, the snapshots taken (minutes before tip), rows, players, books, and flags for the usual failures:
no snapshot at all, first snapshot later than T-4h30, a gap longer than 45 minutes, no closing snapshot (last one more
than 12 minutes before tip), and credit/ledger/unmatched-name totals. Prints no secrets.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.timeutil import ET, now_utc_iso, parse_utc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/props_odds.db')
    ap.add_argument('--date', default=None, help='ET game date (default: today ET)')
    args = ap.parse_args()
    date = args.date or parse_utc(now_utc_iso()).astimezone(ET).strftime('%Y-%m-%d')
    c = sqlite3.connect(args.db)
    ev = pd.read_sql("SELECT * FROM events WHERE et_game_date = ? ORDER BY commence_time_utc", c, params=(date,))
    print(f"ET game date {date}: {len(ev)} events in the store")
    problems = 0
    for e in ev.itertuples(index=False):
        s = pd.read_sql("SELECT mode, minutes_to_tip, n_prop_rows, n_players, books, credit_cost FROM snapshot_log "
                        "WHERE event_id_vendor = ? AND mode NOT LIKE 'hist_%' ORDER BY snapshot_ts_utc", c,
                        params=(e.event_id_vendor,))
        tip_et = parse_utc(e.commence_time_utc).astimezone(ET).strftime('%H:%M ET')
        flags = []
        if s.empty:
            flags.append('NO SNAPSHOTS')
        else:
            m = s.minutes_to_tip.round().astype(int).tolist()
            if m[0] > 270:
                flags.append('first snapshot earlier than expected?')
            if m[0] < 225:
                flags.append(f'first snapshot late (T-{m[0]})')
            if any(a - b > 45 for a, b in zip(m, m[1:])):
                flags.append('gap > 45 min')
            if m[-1] > 12:
                flags.append(f'no closing snapshot (last T-{m[-1]})')
            if s.n_prop_rows.iloc[-1] == 0:
                flags.append('last snapshot has no props')
        problems += bool(flags)
        last = s.iloc[-1] if len(s) else None
        print(f"  {e.away_team} @ {e.home_team} {tip_et}: {len(s)} snapshots "
              f"{[int(x) for x in s.minutes_to_tip.round()] if len(s) else ''}"
              + (f" | last: {int(last.n_prop_rows)} rows, {int(last.n_players)} players, "
                 f"{len(str(last.books).split(',')) if last.books else 0} books" if last is not None else '')
              + (f"  <-- {'; '.join(flags)}" if flags else ''))
    try:
        led = pd.read_sql("SELECT ts_utc, cost, estimate, remaining, status FROM api_ledger", c)
        today = led[led.ts_utc >= pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT00:00:00Z')]
        print(f"\ncredits: used today {today.cost.fillna(today.estimate).sum():g} (UTC day), "
              f"remaining {led.remaining.dropna().iloc[-1]:g}; failed calls today: {(today.status >= 400).sum()}")
    except Exception as ex:
        print('ledger unavailable:', ex)
    un = pd.read_sql("SELECT player_name_raw, team_context, reason, n_rows FROM unmatched_names", c)
    nulls = c.execute("SELECT COUNT(*) FROM prop_snapshots WHERE player_id IS NULL").fetchone()[0]
    print(f"unmatched names: {len(un)} ({nulls} prop rows without player_id)")
    if len(un):
        print(un.to_string(index=False))
    print(f"\n{'ALL GAMES LOOK OK' if not problems and len(ev) else str(problems) + ' game(s) flagged'}")


if __name__ == '__main__':
    main()
