"""
One-off: rebuild data/props_live.db from the surviving live dashboard export (data/props_dashboard_live), after the 2026-10-31 data loss.

    .venv/Scripts/python.exe reports/props/restore_2026-11/import_live_record.py [--db data/props_live.db]

prop_bets: one row per bets.csv line (pmf_json / books_json from pmf.json / books.json, hashes of the old pinned config, roster rule v2).
prop_results: one row per settled bet (y or status_stands present). Profits come from props.betting.settle.settle, as in props/live/settle.py.
prop_signals, slot_log and pipeline_health are not written: they were lost and the jobs create new rows.
Refuses to run if the target database exists. Everything is inserted in one transaction.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from props.betting import settle as settle_mod  # noqa: E402
from props.live import store  # noqa: E402

EXPORT = ROOT / 'data/props_dashboard_live'
OLD_CONFIG = ROOT / 'reports/props/restore_2026-11/live_config_v3_live_2026-27.old.json'
RUN_ID = 'restored_from_dashboard_export_2026-11'
RESULT_COPY = ['y', 'played', 'minutes_actual', 'status_stands', 'status_flagged', 'status_all', 'clv_prob', 'line_move_favour', 'close_line',
               'line_240', 'line_60', 'line_5', 'mkt_240', 'mkt_60', 'mkt_5']


def table_cols(conn, table):
    return [r[1] for r in conn.execute(f'PRAGMA table_info({table})')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=str(ROOT / 'data/props_live.db'))
    a = ap.parse_args()
    if Path(a.db).exists():
        raise SystemExit(f'{a.db} already exists; refusing to run')

    bets = pd.read_csv(EXPORT / 'bets.csv', dtype={'game_id': str, 'bet_id': str})
    pmf = json.loads((EXPORT / 'pmf.json').read_text(encoding='utf-8'))
    books = json.loads((EXPORT / 'books.json').read_text(encoding='utf-8'))
    old = json.loads(OLD_CONFIG.read_text(encoding='utf-8'))

    # game_id: the 10-character id with leading zeros, as signal.py writes it; bet_id = season|game_id|player_id|stat holds the original
    parts = bets['bet_id'].str.split('|', expand=True)
    gid = bets['game_id'].astype(str).str.zfill(10)
    assert (gid == parts[1]).all(), 'game_id does not match bet_id'
    assert (bets['player_id'].astype(int).astype(str) == parts[2]).all() and (bets['stat'] == parts[3]).all() and (bets['season'] == parts[0]).all()
    assert gid.str.len().eq(10).all() and bets['bet_id'].is_unique
    bets['game_id'] = gid
    assert set(bets['bet_id']) == set(pmf) == set(books), 'pmf.json / books.json keys differ from bets.csv'

    conn = store.connect(a.db)
    bet_cols = table_cols(conn, 'prop_bets')
    pb = bets[[c for c in bets.columns if c in bet_cols]].copy()
    for c in ('player_id', 'minutes_before', 'n_at_line', 'n_quotes'):
        pb[c] = pb[c].astype('Int64')
    pb['pmf_json'] = [json.dumps(pmf[k]) for k in pb['bet_id']]
    pb['books_json'] = [json.dumps(books[k]) for k in pb['bet_id']]
    pb['model_sha256'] = old['model_bundle_sha256']
    pb['features_db_sha256'] = old['features_db_sha256']
    pb['rule_sha256'] = old['rule_sha256']
    pb['roster_rule'] = 'v2'
    pb['run_id'] = RUN_ID
    for c in ('snapshot_ts_utc', 'quote_age_min', 'code_commit'):
        pb[c] = None
    assert pb['model_version'].eq('v3_live_2026-27').all()
    pb = pb.reindex(columns=bet_cols)

    settled = bets[bets['y'].notna() | bets['status_stands'].notna()].reset_index(drop=True)
    settled_utc = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    pr = settled[['bet_id'] + RESULT_COPY].copy()
    pr['played'] = pr['played'].astype(int)
    for name in ('best', 'median', 'flat'):
        prof, status = settle_mod.settle(settled['side'].to_numpy(), settled['line'], settled[f'dec_{name}'], settled['y'], settled['played'].astype(bool),
                                         settled['minutes_actual'], settled['expected_minutes'], settled['book'].to_numpy(), 'stands')
        pr[f'profit_stands_{name}'] = prof
        mismatch = int((status != settled['status_stands'].to_numpy()).sum())
        assert mismatch == 0, f'{mismatch} recomputed stands statuses differ from the export ({name})'
    pr['settled_utc'] = settled_utc
    pr['run_id'] = RUN_ID
    pr = pr.reindex(columns=table_cols(conn, 'prop_results'))

    with conn:
        n_bets = store._insert(conn, 'prop_bets', pb)
        n_res = store._insert(conn, 'prop_results', pr)
        if n_bets != len(bets) or n_res != len(settled):
            raise RuntimeError(f'wrote {n_bets}/{len(bets)} bets and {n_res}/{len(settled)} results')
    counts = {t: conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ('prop_bets', 'prop_results', 'prop_signals', 'slot_log', 'pipeline_health')}
    conn.close()
    print(json.dumps({'db': a.db, 'settled_utc': settled_utc, 'unsettled': int(len(bets) - len(settled)),
                      'settled_by_status': settled['status_stands'].value_counts().to_dict(), **counts}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
