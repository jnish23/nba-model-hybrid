"""
Early prop snapshots for the "bet before T-240" test (stage 1): one snapshot per game at each early offset
(default T-480 and T-360 minutes), for a fixed number of game days, into a store of its OWN (data/props_odds_early.db).

Why a separate store: everything that reads data/props_odds.db (the signal job, settlement, the dashboards and other readers) assumes its snapshots start at T-240.
A separate file means none of them can see an early snapshot, and the credit ledger inside it counts only this test, so its total cap IS the approved upper bound.

Pure planning plus one tick; the script (scripts/snapshot_props_early.py) owns the vendor, the caps and the clock.
"""
import logging
from dataclasses import dataclass
from datetime import timedelta

from core.timeutil import et_game_date, minutes_to_tip, parse_utc, to_utc_iso
from props import collect

OFFSETS_MIN = (480, 360)
GRACE_MIN = 25.0            # a snapshot for offset T is taken on the first tick with T - GRACE_MIN < minutes to tip <= T (the scheduler fires every 10 minutes)
MAX_GAME_DAYS = 14
CREDITS_PER_SNAPSHOT = 11   # 11 markets x 1 region, as the live collector
MAX_GAMES_PER_DAY = 15
TOTAL_CAP = len(OFFSETS_MIN) * CREDITS_PER_SNAPSHOT * MAX_GAMES_PER_DAY * MAX_GAME_DAYS          # 4,620: the upper bound the owner approved on 2026-11-13
DAILY_CAP = len(OFFSETS_MIN) * CREDITS_PER_SNAPSHOT * MAX_GAMES_PER_DAY                          # 330
MODE_PREFIX = 'early_T-'


def mode_of(offset):
    return f'{MODE_PREFIX}{int(offset)}'


@dataclass
class EarlyTarget:
    event_id: str
    commence_time_utc: str
    minutes_to_tip: float
    offset: int
    game_date: str

    @property
    def mode(self):
        return mode_of(self.offset)


def done_and_days(conn):
    """From the early store: {(event_id, mode)} already snapshotted, and the set of ET game dates that have any early snapshot."""
    rows = conn.execute("SELECT l.event_id_vendor, l.mode, e.et_game_date FROM snapshot_log l LEFT JOIN events e USING (event_id_vendor) WHERE l.mode LIKE ?", (MODE_PREFIX + '%',)).fetchall()
    return {(r[0], r[1]) for r in rows}, {r[2] for r in rows if r[2]}


def plan_early(events, now_utc, done, days, offsets=OFFSETS_MIN, grace_min=GRACE_MIN, max_game_days=MAX_GAME_DAYS):
    """
    events: vendor event dicts (id, commence_time). done: {(event_id, mode)} already taken. days: ET game dates that already have an early snapshot.
    Returns the EarlyTargets due now, soonest tip first. An offset is due once per game, in the window (offset - grace_min, offset] minutes before tip; a tick that
    arrives later than that skips the offset for the game (the snapshot would no longer be "early" in the sense of the test). A game on a date that would be the
    (max_game_days + 1)-th date is never snapshotted.
    """
    out, days = [], set(days)
    for e in sorted(events, key=lambda x: x['commence_time']):
        m = minutes_to_tip(now_utc, e['commence_time'])
        date = et_game_date(e['commence_time'])
        for off in offsets:
            if not (off - grace_min < m <= off) or (e['id'], mode_of(off)) in done:
                continue
            if date not in days:
                if len(days) >= max_game_days:
                    continue
                days.add(date)
            out.append(EarlyTarget(e['id'], to_utc_iso(e['commence_time']), m, int(off), date))
    return sorted(out, key=lambda t: t.minutes_to_tip)


def run_early(vendor, conn, now_utc, markets=None, regions='us', bookmakers=None, offsets=OFFSETS_MIN, grace_min=GRACE_MIN, max_game_days=MAX_GAME_DAYS,
              dry_run=False, resolver=None):
    """One scheduler tick. Lists upcoming events (free), snapshots the ones due, resolves player ids, then writes the snapshot_log rows. Returns per-event result dicts."""
    markets = markets or collect.ALL_MARKET_KEYS
    horizon = to_utc_iso(parse_utc(now_utc) + timedelta(minutes=max(offsets) + 60))
    events = vendor.events(commence_from=to_utc_iso(now_utc), commence_to=horizon).payload
    done, days = done_and_days(conn)
    targets = plan_early(events, now_utc, done, days, offsets, grace_min, max_game_days)
    logging.info(f"{len(events)} upcoming events, {len(targets)} early snapshots due; game days used so far: {len(days)} of {max_game_days}")
    results, pending = [], []
    try:
        for t in targets:
            if dry_run:
                results.append({'event_id': t.event_id, 'mode': t.mode, 'minutes_to_tip': round(t.minutes_to_tip), 'game_date': t.game_date, 'dry_run': True})
                continue
            resp = vendor.event_odds(t.event_id, markets, regions=regions, bookmakers=bookmakers)
            out = collect.store_event_odds(conn, resp, t.mode, pending_logs=pending)
            results.append({'event_id': t.event_id, 'mode': t.mode, 'cost': resp.cost, 'game_date': t.game_date, **{k: out[k] for k in ('prop_rows', 'team_rows')}})
    finally:
        if resolver is not None and not dry_run and pending:
            try:
                resolver(conn)
            except Exception:
                logging.exception('id resolution failed; snapshot_log rows are written with unresolved ids')        # paid for: log them so they are not fetched again
        if pending:
            collect.write_pending_logs(conn, pending)
    return results
