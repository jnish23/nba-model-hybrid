"""
Collect player-prop and team-line odds snapshots into data/props_odds.db (separate from nba_stats.db).

Live (scheduled every 10 minutes on game days; per-event cadence is decided inside, see props/collect.py):
    python scripts/snapshot_props.py
    python scripts/snapshot_props.py --dry-run          # what would be requested; spends no credits

Historical (costs 10x credits; always set --max-events while piloting):
    python scripts/snapshot_props.py --historical --date 2025-01-15 --offsets 180,90,5 --max-events 2

Credit safety: hard caps are enforced before every request and persisted in the ledger (api_ledger table):
--max-credits-per-run, --daily-cap, --total-cap. The API key is read from the ODDS_API_KEY environment variable.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.odds_vendor.base import BudgetExceeded, CreditBudget  # noqa: E402
from core.odds_vendor.theoddsapi import OddsApiError, TheOddsApi  # noqa: E402
from core.timeutil import et_game_date, now_utc_iso, parse_utc, to_utc_iso  # noqa: E402
from props import collect, odds_store  # noqa: E402
from props.player_ids import PlayerMatcher, resolve_store  # noqa: E402

STORE_PATH = Path('data/props_odds.db')
NBA_DB = Path('data/nba_stats.db')


def resolve_ids(conn):
    if not NBA_DB.exists():
        logging.warning('nba_stats.db not found; player ids left unresolved')
        return
    summary = resolve_store(conn, PlayerMatcher(NBA_DB), NBA_DB)
    logging.info(f"id resolution: {summary}")


def spread_sample(games, n):
    """n games spread evenly across the slate (first and last tip included), not just the earliest tips."""
    if n >= len(games):
        return games
    if n == 1:
        return games[:1]
    idx = sorted({round(i * (len(games) - 1) / (n - 1)) for i in range(n)})
    return [games[i] for i in idx]


def run_historical(vendor, conn, date, offsets, max_events, markets, regions, bookmakers):
    """Fetch snapshots at tip - offset for each game on an ET game date. Event ids come from a historical events
    listing taken at 16:00 UTC of that date (1 credit), which lists that day's games."""
    listing = vendor.historical_events(f'{date}T16:00:00Z')
    games = [e for e in listing.payload if et_game_date(e['commence_time']) == date]
    games = spread_sample(sorted(games, key=lambda e: e['commence_time']), max_events)
    logging.info(f"{len(games)} games on {date} (max_events={max_events})")
    for e in games:
        for off, ts in collect.plan_historical(e['commence_time'], offsets):
            resp = vendor.historical_event_odds(e['id'], ts, markets, regions=regions, bookmakers=bookmakers)
            if not resp.payload:
                logging.info(f"no data for {e['id']} at {ts}")
                continue
            out = collect.store_event_odds(conn, resp, mode=f'hist_T-{off}')
            logging.info(f"{e['away_team']} @ {e['home_team']} T-{off}m (snapshot {resp.snapshot_ts_utc}): "
                         f"{out['prop_rows']} prop rows, {out['team_rows']} team rows, cost {resp.cost}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--historical', action='store_true')
    ap.add_argument('--date', help='ET game date for --historical')
    ap.add_argument('--offsets', default='180,90,5', help='minutes before tip for --historical')
    ap.add_argument('--max-events', type=int, default=2, help='games per date for --historical, spread evenly across the slate (pilot guard)')
    ap.add_argument('--markets', default=','.join(collect.ALL_MARKET_KEYS))
    ap.add_argument('--regions', default='us')
    ap.add_argument('--bookmakers', default=None)
    ap.add_argument('--window-hours', type=float, default=collect.WINDOW_HOURS)
    ap.add_argument('--fast-window-min', type=float, default=collect.FAST_WINDOW_MIN,
                    help='minutes before tip where the fast cadence applies (0 = never)')
    ap.add_argument('--slow-cadence-min', type=float, default=collect.SLOW_CADENCE_MIN)
    ap.add_argument('--fast-cadence-min', type=float, default=collect.FAST_CADENCE_MIN)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--max-credits-per-run', type=float, default=600)
    ap.add_argument('--daily-cap', type=float, default=2000)
    ap.add_argument('--total-cap', type=float, default=18000, help='all credits ever recorded; plan allowance minus reserve')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    conn = odds_store.connect(STORE_PATH)
    budget = CreditBudget(STORE_PATH, per_run=args.max_credits_per_run, per_day=args.daily_cap, total=args.total_cap)
    vendor = TheOddsApi(budget)
    markets = [m for m in args.markets.split(',') if m]
    try:
        if args.historical:
            if not args.date:
                ap.error('--historical needs --date')
            run_historical(vendor, conn, args.date, [int(x) for x in args.offsets.split(',')], args.max_events,
                           markets, args.regions, args.bookmakers)
        else:
            res = collect.run_live(vendor, conn, now_utc_iso(), markets, args.regions, args.bookmakers,
                                   args.window_hours, args.dry_run, args.fast_window_min, args.slow_cadence_min,
                                   args.fast_cadence_min)
            for r in res:
                logging.info(r)
    except BudgetExceeded as e:
        logging.error(f'STOPPED by credit cap: {e}')
        return 2
    except OddsApiError as e:
        logging.error(f'API error: {e}')
        return 1
    finally:
        if not args.dry_run:
            resolve_ids(conn)
        conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
