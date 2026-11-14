"""
Early prop snapshots (T-480 and T-360) for 14 game days, into data/props_odds_early.db. Stage 1 of the "bet before T-240" test.
The production collector (scripts/snapshot_props.py) and its store (data/props_odds.db) are not touched and never see these snapshots.

    python scripts/snapshot_props_early.py --dry-run     # lists what is due now; calls only the free events listing; spends no credits, writes no snapshot
    python scripts/snapshot_props_early.py --status      # credits used and game days covered so far; no network call at all
    python scripts/snapshot_props_early.py               # one tick (scheduled every 10 minutes)

Credit safety: the ledger lives in the early store, so it counts only this test. Hard caps, checked before every request: 4,620 credits in total
(2 snapshots x 11 credits x at most 15 games x 14 game days: the bound approved on 2026-11-13) and 330 per UTC day. The caps cannot be raised from the command line.
The collection also stops by itself after 14 distinct game dates. The API key is read from ODDS_API_KEY (the gitignored .env), never printed.
"""
import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.odds_vendor.base import BudgetExceeded, CreditBudget  # noqa: E402
from core.odds_vendor.theoddsapi import OddsApiError, TheOddsApi  # noqa: E402
from core.timeutil import now_utc_iso  # noqa: E402
from props import collect, collect_early, odds_store  # noqa: E402
from props.player_ids import PlayerMatcher, has_pending, resolve_store  # noqa: E402

STORE_PATH = Path('data/props_odds_early.db')
RAW_DIR = Path('data/raw/props_api_early')
NBA_DB = Path('data/nba_stats.db')


def resolve_ids(conn):
    if not NBA_DB.exists():
        logging.warning('nba_stats.db not found; player ids left unresolved')
        return
    if not has_pending(conn):
        return
    logging.info(f"id resolution: {resolve_store(conn, PlayerMatcher(NBA_DB), NBA_DB)}")


def status(path=STORE_PATH):
    """Credits used and game days covered, read-only; no network."""
    if not Path(path).exists():
        return {'store': str(path), 'exists': False, 'credits_used': 0.0, 'total_cap': collect_early.TOTAL_CAP, 'game_days': 0, 'max_game_days': collect_early.MAX_GAME_DAYS, 'snapshots': {}}
    with sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        used = float(c.execute('SELECT COALESCE(SUM(COALESCE(cost, estimate)), 0) FROM api_ledger').fetchone()[0]) if 'api_ledger' in tables else 0.0
        snaps = dict(c.execute("SELECT mode, COUNT(*) FROM snapshot_log WHERE mode LIKE 'early_T-%' GROUP BY 1").fetchall()) if 'snapshot_log' in tables else {}
        days = c.execute("SELECT COUNT(DISTINCT e.et_game_date) FROM snapshot_log l JOIN events e USING (event_id_vendor) WHERE l.mode LIKE 'early_T-%'").fetchone()[0] if 'snapshot_log' in tables else 0
    return {'store': str(path), 'exists': True, 'credits_used': used, 'total_cap': collect_early.TOTAL_CAP, 'game_days': int(days), 'max_game_days': collect_early.MAX_GAME_DAYS, 'snapshots': snaps}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--regions', default='us')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    if a.status:
        s = status()
        print(f"early store {s['store']}: {'present' if s['exists'] else 'not created yet'}; credits used {s['credits_used']:g} of {s['total_cap']}; "
              f"game days {s['game_days']} of {s['max_game_days']}; snapshots {s['snapshots']}")
        return 0
    conn = odds_store.connect(STORE_PATH)
    budget = CreditBudget(STORE_PATH, per_run=collect_early.DAILY_CAP, per_day=collect_early.DAILY_CAP, total=collect_early.TOTAL_CAP)
    vendor = TheOddsApi(budget, raw_dir=RAW_DIR)
    try:
        for r in collect_early.run_early(vendor, conn, now_utc_iso(), collect.ALL_MARKET_KEYS, a.regions, dry_run=a.dry_run, resolver=resolve_ids):
            logging.info(r)
    except BudgetExceeded as e:
        logging.error(f'STOPPED by credit cap: {e}')
        return 2
    except OddsApiError as e:
        logging.error(f'API error: {e}')
        return 1
    finally:
        conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
