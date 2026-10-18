"""
Historical props backfill. Two steps, on purpose:

1. PLAN (default, spends nothing; reads the NBA schedule, cached after the first call):
    python scripts/backfill_props.py --start 2023-10-24 --end 2024-06-30 --offsets 240,120,60,30,5
   prints games, snapshots still to fetch (already-stored ones are skipped) and the upper-bound credit cost, and the exact
   command to run it.

2. RUN (spends credits; refuses unless you pass back the planned figure):
    python scripts/backfill_props.py ... --run --confirm-credits <planned credits> --total-cap <cap>
   Safe to interrupt and rerun; each rerun skips what is stored. --limit-games N does a staged run.

Credit caps are enforced before every request. --total-cap counts every credit ever recorded in the ledger
(data/props_backfill_ledger.db by default, separate from the live collector).
"""
import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.odds_vendor.base import BudgetExceeded, CreditBudget  # noqa: E402
from core.odds_vendor.theoddsapi import TheOddsApi  # noqa: E402
from core.season_utils import season_for_date  # noqa: E402
from props import backfill, collect, odds_store, schedule  # noqa: E402
from props.player_ids import PlayerMatcher, resolve_store  # noqa: E402

STORE = Path('data/props_odds.db')
NBA_DB = Path('data/nba_stats.db')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', required=True, help='first ET game date, YYYY-MM-DD')
    ap.add_argument('--end', required=True, help='last ET game date, YYYY-MM-DD')
    ap.add_argument('--offsets', default=','.join(map(str, backfill.DEFAULT_OFFSETS)), help='minutes before the reference tip')
    ap.add_argument('--markets', default=','.join(collect.ALL_MARKET_KEYS))
    ap.add_argument('--regions', default='us')
    ap.add_argument('--bookmakers', default=None)
    ap.add_argument('--limit-games', type=int, default=None)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--confirm-credits', type=float, default=None, help='must equal the planned credits printed by the plan step')
    ap.add_argument('--ledger', default='data/props_backfill_ledger.db',
                    help='credit ledger for backfill spending; separate from the live collector so its daily and total caps '
                         'are not eaten by the backfill')
    ap.add_argument('--total-cap', type=float, default=None, help='caps credits recorded in --ledger across all backfill runs')
    ap.add_argument('--daily-cap', type=float, default=None)
    ap.add_argument('--sleep', type=float, default=0.1, help='seconds between calls')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    offsets = tuple(int(x) for x in args.offsets.split(','))
    markets = [m for m in args.markets.split(',') if m]
    n_regions = len([r for r in args.regions.split(',') if r]) if not args.bookmakers else -(-len(args.bookmakers.split(',')) // 10)
    seasons = sorted({season_for_date(d) for d in pd.date_range(args.start, args.end, freq='7D').strftime('%Y-%m-%d')}
                     | {season_for_date(args.end)})
    sched = pd.concat([schedule.load_schedule(s) for s in seasons], ignore_index=True)

    conn = odds_store.connect(STORE)
    conn.executescript(schedule.REF_SCHEMA)
    done = backfill.done_snapshots(conn)
    p = backfill.plan(sched, args.start, args.end, offsets, len(markets), n_regions, done, args.limit_games)
    summ = backfill.summarize(p)
    print(f"\nWindow {args.start} to {args.end}; snapshots at T-{list(offsets)} minutes before the reference tip "
          f"(NBA scheduled tip + {schedule.TIP_OFFSET_MIN}); {len(markets)} markets x {n_regions} region(s)")
    print(summ.to_string())
    print(f"\nalready stored (skipped): {sum(1 for g, o in done if g in set(p.games['game_id']) and o in offsets)} snapshots")
    print(f"TO FETCH: {p.snapshot_count:,} snapshots over {len(p.dates_todo)} game dates")
    print(f"PLANNED CREDITS (upper bound): {p.credits:,.0f}  = 10 x {len(markets)} x {n_regions} x {p.snapshot_count:,}"
          f" + {len(p.dates_todo)} event listings")
    try:                                    # a free call (the events listing costs nothing) reports the live balance
        remaining = TheOddsApi(CreditBudget(args.ledger, per_run=0)).events().remaining
        print(f"credits remaining right now (vendor header): {remaining:,.0f}"
              + ("  <-- NOT ENOUGH for this plan" if remaining is not None and remaining < p.credits else ""))
    except Exception as e:
        print(f"could not read the credit balance ({type(e).__name__}: {e})")
    if not args.run:
        print(f"\nNothing spent. To run it: add  --run --confirm-credits {p.credits:.0f} --total-cap <cap>")
        return 0
    if args.confirm_credits != p.credits:
        print(f"\nREFUSED: --confirm-credits must equal the planned {p.credits:.0f} (got {args.confirm_credits}).")
        return 2
    budget = CreditBudget(args.ledger, per_run=p.credits + 5, per_day=args.daily_cap, total=args.total_cap)
    vendor = TheOddsApi(budget)
    matcher = PlayerMatcher(NBA_DB)
    try:
        stats = backfill.run(vendor, conn, p, matcher.team_id, markets, args.regions, args.bookmakers, args.sleep)
        print('finished:', stats)
        rc = 0
    except BudgetExceeded as e:
        print(f'STOPPED by credit cap: {e}\nRerun the same command to resume (stored snapshots are skipped).')
        rc = 2
    finally:
        conn.commit()
        print('id resolution:', resolve_store(conn, matcher, NBA_DB))
        print('annotation:', schedule.annotate(conn, sched, matcher.team_id))
    return rc


if __name__ == '__main__':
    sys.exit(main())
