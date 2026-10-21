"""
Collection logic for the props odds store, kept free of I/O decisions so it can be tested with a fake vendor.

Live: the scheduler fires every 10 minutes; plan_live() decides per event whether this run snapshots it:
  tip in the last FAST_WINDOW_MIN minutes  -> every FAST_CADENCE_MIN
  tip within WINDOW_HOURS                  -> every SLOW_CADENCE_MIN
  tip already passed                       -> never (prop lines go in-game; excluded by design)
Historical: plan_historical() turns a game's tip time into the snapshot instants to fetch (minutes before tip).
"""
import logging
from datetime import timedelta
from dataclasses import dataclass

from core.timeutil import et_game_date, minutes_to_tip, parse_utc, to_utc_iso
from props import odds_store
from props.parse import parse_event_odds

PROP_MARKET_KEYS = ['player_points', 'player_rebounds', 'player_assists', 'player_threes',
                    'player_points_rebounds_assists', 'player_points_rebounds', 'player_points_assists',
                    'player_rebounds_assists']
TEAM_MARKET_KEYS = ['h2h', 'spreads', 'totals']
ALL_MARKET_KEYS = PROP_MARKET_KEYS + TEAM_MARKET_KEYS

WINDOW_HOURS = 6.0
FAST_WINDOW_MIN = 120.0
SLOW_CADENCE_MIN = 30.0
FAST_CADENCE_MIN = 10.0
CLOSE_MIN_GAP = 8.0         # minutes since the previous snapshot before a closing snapshot is taken
JITTER_MIN = 3.0            # a scheduler tick may arrive a little early; do not skip a due snapshot for that


@dataclass
class LiveTarget:
    event_id: str
    commence_time_utc: str
    minutes_to_tip: float
    mode: str               # 'fast' | 'slow'


def plan_live(events, now_utc, last_snapshot_by_event, window_hours=WINDOW_HOURS, fast_window_min=FAST_WINDOW_MIN,
              slow_cadence_min=SLOW_CADENCE_MIN, fast_cadence_min=FAST_CADENCE_MIN, close_snapshot_min=0.0):
    """events: vendor event dicts (id, commence_time). Returns the LiveTargets to snapshot now, soonest tip first."""
    out = []
    for e in events:
        m = minutes_to_tip(now_utc, e['commence_time'])
        if m <= 0 or m > window_hours * 60:
            continue
        mode, cadence = ('fast', fast_cadence_min) if m <= fast_window_min else ('slow', slow_cadence_min)
        last = last_snapshot_by_event.get(e['id'])
        if close_snapshot_min and m <= close_snapshot_min:
            # one closing snapshot just before tip (needs a scheduler tick at least every close_snapshot_min minutes);
            # skipped only if a snapshot was taken in the last CLOSE_MIN_GAP minutes
            gap = None if last is None else (parse_utc(now_utc) - parse_utc(last)).total_seconds() / 60.0
            if gap is None or gap >= CLOSE_MIN_GAP:
                out.append(LiveTarget(e['id'], to_utc_iso(e['commence_time']), m, 'close'))
            continue
        if last is not None:
            gap = (parse_utc(now_utc) - parse_utc(last)).total_seconds() / 60.0
            if gap < cadence - JITTER_MIN:
                continue
        out.append(LiveTarget(e['id'], to_utc_iso(e['commence_time']), m, mode))
    return sorted(out, key=lambda t: t.minutes_to_tip)


def plan_historical(commence_time_utc, offsets_min):
    """Snapshot instants (UTC ISO) for a game: tip minus each offset in minutes (e.g. 180, 90, 5)."""
    tip = parse_utc(commence_time_utc)
    return [(off, to_utc_iso(tip - timedelta(minutes=off))) for off in offsets_min]


def store_event_odds(conn, resp, mode, markets_requested=None):
    """Parse one event-odds VendorResponse into the store (props, team lines, event row, snapshot log)."""
    ev = resp.payload
    props, teams, stats = parse_event_odds(ev, resp.snapshot_ts_utc)
    odds_store.upsert_event(conn, ev['id'], to_utc_iso(ev['commence_time']), ev['home_team'], ev['away_team'],
                            et_game_date(ev['commence_time']))
    n_new = odds_store.insert_props(conn, props)
    odds_store.insert_team_lines(conn, teams)
    odds_store.log_snapshot(conn, resp.snapshot_ts_utc, ev['id'], mode,
                            minutes_to_tip(resp.snapshot_ts_utc, ev['commence_time']), props, resp.cost, resp.raw_path)
    conn.commit()
    if stats['unknown_markets']:
        logging.info(f"unknown markets ignored: {stats['unknown_markets']}")
    return {'prop_rows': len(props), 'new_prop_rows': n_new, 'team_rows': len(teams), 'stats': stats}


def run_live(vendor, conn, now_utc, markets=None, regions='us', bookmakers=None, window_hours=WINDOW_HOURS,
             dry_run=False, fast_window_min=FAST_WINDOW_MIN, slow_cadence_min=SLOW_CADENCE_MIN,
             fast_cadence_min=FAST_CADENCE_MIN, close_snapshot_min=0.0):
    """One scheduler tick. Returns a list of per-event result dicts."""
    markets = markets or ALL_MARKET_KEYS
    horizon = to_utc_iso(parse_utc(now_utc) + timedelta(hours=window_hours + 1))
    events = vendor.events(commence_from=to_utc_iso(now_utc), commence_to=horizon).payload
    last = {e['id']: odds_store.last_snapshot_ts(conn, e['id']) for e in events}
    targets = plan_live(events, now_utc, last, window_hours=window_hours, fast_window_min=fast_window_min,
                        slow_cadence_min=slow_cadence_min, fast_cadence_min=fast_cadence_min,
                        close_snapshot_min=close_snapshot_min)
    logging.info(f"{len(events)} upcoming events, {len(targets)} due for a snapshot")
    results = []
    for t in targets:
        if dry_run:
            results.append({'event_id': t.event_id, 'mode': t.mode, 'minutes_to_tip': round(t.minutes_to_tip),
                            'dry_run': True})
            continue
        resp = vendor.event_odds(t.event_id, markets, regions=regions, bookmakers=bookmakers)
        out = store_event_odds(conn, resp, t.mode)
        results.append({'event_id': t.event_id, 'mode': t.mode, 'cost': resp.cost, **{k: out[k] for k in ('prop_rows', 'team_rows')}})
    return results
