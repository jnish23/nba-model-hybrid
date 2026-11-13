"""
The settlement job (D1.4): settles every recorded bet whose game is in the box-score history. Run nightly after the box-score sync; idempotent (a settled bet is never rewritten).
Besides prop_results it appends, never rewriting anything:
  prop_clv_audit          where each settled bet's closing quote came from (the recorded slot-5 signal);
  prop_result_corrections settled bets whose y or status changed after the box score was re-read (settle.detect_corrections);
  prop_fill_checks        whether the recorded price survived to the later snapshots (props.live.fills; reads the odds store read-only; a failure is logged and never blocks settlement).

    python scripts/props_settle_job.py [--dry-run] [--props-db data/props_odds.db]
"""
import argparse
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.features import rows as rows_mod  # noqa: E402
from props.live import fills, settle, store  # noqa: E402

PROPS_DB = 'data/props_odds.db'


def _same(a, b):
    a_nan, b_nan = a is None or pd.isna(a), b is None or pd.isna(b)
    return (a_nan and b_nan) or (not a_nan and not b_nan and abs(float(a) - float(b)) < 1e-9)


def new_corrections(conn, found):
    """Drop the corrections whose bet already has a latest correction with the same y_new (nothing new to record)."""
    if found.empty:
        return found
    last = pd.read_sql('SELECT bet_id, y_new FROM prop_result_corrections WHERE (bet_id, detected_utc) IN (SELECT bet_id, MAX(detected_utc) FROM prop_result_corrections GROUP BY bet_id)', conn)
    prev = dict(zip(last['bet_id'], last['y_new']))
    keep = [not (r.bet_id in prev and _same(prev[r.bet_id], r.y_new)) for r in found.itertuples(index=False)]
    return found[keep].reset_index(drop=True)


def record_corrections(conn, labels, started, run_id, dry_run):
    """Re-settle every settled bet with the current labels and append the changes not already recorded to prop_result_corrections. Returns the new rows."""
    try:
        settled = pd.read_sql('SELECT b.* FROM prop_bets b JOIN prop_results r USING (bet_id)', conn)
        results = pd.read_sql('SELECT bet_id, y, status_stands, profit_stands_best FROM prop_results', conn)
        found = new_corrections(conn, settle.detect_corrections(settled, results, labels, detected_utc=started, run_id=run_id))
        if len(found):
            logging.warning(f'{len(found)} settled bets changed after re-reading the box scores')
            print(found.round(3).to_string(index=False))
            if not dry_run:
                with conn:
                    store._insert(conn, 'prop_result_corrections', found)
        return found
    except Exception:                                    # noqa: BLE001 - a side table must never block settlement
        logging.error('corrections check failed (settlement unaffected):\n' + traceback.format_exc())
        return pd.DataFrame()


def record_clv_audit(conn, res, signals, started, run_id):
    """prop_clv_audit rows for the bets just settled. Never raises: a failure is logged and settlement stands."""
    try:
        with conn:
            return store._insert(conn, 'prop_clv_audit', settle.clv_audit_rows(res, signals, started, run_id), ignore=True)
    except Exception:                                    # noqa: BLE001
        logging.error('CLV audit rows failed (settlement unaffected):\n' + traceback.format_exc())
        return 0


def record_fill_checks(conn, done, props_db, started, run_id, dry_run):
    """prop_fill_checks for the bets just settled. Never raises: a failure is logged and settlement stands."""
    try:
        if not Path(props_db).exists():
            logging.warning(f'fill checks skipped: {props_db} not found')
            return 0
        chk = fills.fill_checks(done, props_db, started, run_id)
        if dry_run:
            return len(chk)
        with conn:
            return store._insert(conn, 'prop_fill_checks', chk, ignore=True)
    except Exception:                                    # noqa: BLE001
        logging.error('fill checks failed (settlement unaffected):\n' + traceback.format_exc())
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=store.DEFAULT_PATH)
    ap.add_argument('--nba-db', default='data/nba_stats.db')
    ap.add_argument('--props-db', default=PROPS_DB, help='odds store, opened read-only for the fill checks')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    now = pd.Timestamp.now(tz='UTC')
    run_id, started, job = now.strftime('%Y%m%dT%H%M%SZ'), now.strftime('%Y-%m-%dT%H:%M:%SZ'), 'settle'
    conn = store.connect(a.db)
    try:
        bets = pd.read_sql('SELECT * FROM prop_bets WHERE bet_id NOT IN (SELECT bet_id FROM prop_results)', conn)
        n_settled = conn.execute('SELECT COUNT(*) FROM prop_results').fetchone()[0]
        if bets.empty and not n_settled:
            store.health(conn, run_id, job, started, started, 'idle', 0, {})
            logging.info('nothing to settle')
            return 0
        hist = rows_mod.load_history(a.nba_db)
        hist['game_id'] = hist['game_id'].astype(str)
        labels = settle.labels_from_history(hist)
        n_corr = len(record_corrections(conn, labels, started, run_id, a.dry_run)) if n_settled else 0
        if bets.empty:
            store.health(conn, run_id, job, started, started, 'idle', 0, {'corrections': n_corr})
            logging.info('nothing to settle')
            return 0
        done = bets[bets['game_id'].isin(set(hist['game_id']))]
        if done.empty:
            store.health(conn, run_id, job, started, started, 'idle', 0, {'waiting_for_box_scores': int(len(bets)), 'corrections': n_corr})
            logging.info(f'{len(bets)} bets are waiting for their box scores')
            return 0
        signals = pd.read_sql(f"SELECT * FROM prop_signals WHERE game_id IN ({','.join('?' * done['game_id'].nunique())})", conn, params=list(done['game_id'].unique()))
        res = settle.settle_bets(done, labels, signals, started, run_id)
        print(res[['bet_id', 'y', 'status_stands', 'profit_stands_median', 'clv_prob']].round(3).to_string(index=False))
        if not a.dry_run:
            with conn:
                n = store._insert(conn, 'prop_results', res, ignore=True)
            record_clv_audit(conn, res, signals, started, run_id)
            n_fill = record_fill_checks(conn, done, a.props_db, started, run_id, False)
            store.health(conn, run_id, job, started, pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ'), 'ok', n,
                         {'settled': n, 'waiting': int(len(bets) - len(done)), 'corrections': n_corr, 'fill_checks': n_fill})
        else:
            record_fill_checks(conn, done, a.props_db, started, run_id, True)
        return 0
    except Exception as e:                       # noqa: BLE001
        logging.error(traceback.format_exc())
        store.health(conn, run_id, job, started, started, 'error', 0, {'error': repr(e)[:500]})
        return 1


if __name__ == '__main__':
    sys.exit(main())
