"""
The settlement job (D1.4): settles every recorded bet whose game is in the box-score history. Run nightly after the box-score sync; idempotent (a settled bet is never rewritten).

    python scripts/props_settle_job.py [--dry-run]
"""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.features import rows as rows_mod  # noqa: E402
from props.live import settle, store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=store.DEFAULT_PATH)
    ap.add_argument('--nba-db', default='data/nba_stats.db')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    now = pd.Timestamp.now(tz='UTC')
    run_id, started, job = now.strftime('%Y%m%dT%H%M%SZ'), now.strftime('%Y-%m-%dT%H:%M:%SZ'), 'settle'
    conn = store.connect(a.db)
    try:
        bets = pd.read_sql('SELECT * FROM prop_bets WHERE bet_id NOT IN (SELECT bet_id FROM prop_results)', conn)
        if bets.empty:
            store.health(conn, run_id, job, started, started, 'idle', 0, {})
            logging.info('nothing to settle')
            return 0
        hist = rows_mod.load_history(a.nba_db)
        hist['game_id'] = hist['game_id'].astype(str)
        done = bets[bets['game_id'].isin(set(hist['game_id']))]
        if done.empty:
            store.health(conn, run_id, job, started, started, 'idle', 0, {'waiting_for_box_scores': int(len(bets))})
            logging.info(f'{len(bets)} bets are waiting for their box scores')
            return 0
        signals = pd.read_sql(f"SELECT * FROM prop_signals WHERE game_id IN ({','.join('?' * done['game_id'].nunique())})", conn, params=list(done['game_id'].unique()))
        res = settle.settle_bets(done, settle.labels_from_history(hist), signals, started, run_id)
        print(res[['bet_id', 'y', 'status_stands', 'profit_stands_median', 'clv_prob']].round(3).to_string(index=False))
        if not a.dry_run:
            with conn:
                n = store._insert(conn, 'prop_results', res, ignore=True)
            store.health(conn, run_id, job, started, pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ'), 'ok', n, {'settled': n, 'waiting': int(len(bets) - len(done))})
        return 0
    except Exception as e:                       # noqa: BLE001
        logging.error(traceback.format_exc())
        store.health(conn, run_id, job, started, started, 'error', 0, {'error': repr(e)[:500]})
        return 1


if __name__ == '__main__':
    sys.exit(main())
