"""One refresh of the live dashboard export: pipeline checks, the odds now for the pending bets, and the export tables. Used by the export script and the private publisher."""
import json
from pathlib import Path

import pandas as pd

from props import schedule as S
from props.live import config, current, export, health, scores, store

ODDS_DB = 'data/props_odds.db'
REPORT_DIR = 'data/raw/injury_reports'


ACTIVE_BEFORE_H, ACTIVE_AFTER_H = 8.0, 7.0


def active_now(now=None, db=store.DEFAULT_PATH):
    """True while there is something to keep fresh: an unsettled bet whose game tips within the next 8 hours or tipped within the last 7 (prices before, provisional results after)."""
    now = now or pd.Timestamp.now(tz='UTC')
    conn = store.connect(db)
    tips = pd.read_sql('SELECT tip_ref_utc FROM prop_bets WHERE bet_id NOT IN (SELECT bet_id FROM prop_results)', conn)['tip_ref_utc']
    tips = pd.to_datetime(tips, utc=True)
    return bool(((tips > now - pd.Timedelta(hours=ACTIVE_AFTER_H)) & (tips < now + pd.Timedelta(hours=ACTIVE_BEFORE_H))).any())


def refresh_live_export(out='data/props_dashboard_live', odds_db=ODDS_DB, n_boot=1000, now=None, verbose=True):
    cfg = config.load()
    now = now or pd.Timestamp.now(tz='UTC')
    sched = S.load_schedule(cfg['model_season'])
    tips = pd.to_datetime(sched['nba_tip_utc'], utc=True)
    soon = bool(((tips > now - pd.Timedelta(hours=1)) & (tips < now + pd.Timedelta(hours=6))).any())
    conn = store.connect(store.DEFAULT_PATH)
    checks = health.checks(conn, odds_db, REPORT_DIR, now, soon)
    unsettled = pd.read_sql('SELECT * FROM prop_bets WHERE bet_id NOT IN (SELECT bet_id FROM prop_results)', conn)
    pending = unsettled[pd.to_datetime(unsettled['tip_ref_utc'], utc=True) > now]
    rule = json.loads(Path(cfg['rule_file']).read_text(encoding='utf-8'))
    cur = current.current_odds(pending, sched, odds_db, now, rule)
    res_now = scores.provisional_results(unsettled, now)
    export.export_live(store.DEFAULT_PATH, out, cfg, 1000 if n_boot is None else n_boot, checks, cur, res_now)
    if verbose:
        print(checks.to_string(index=False))
        print(f'odds now for {len(cur)} of {len(pending)} upcoming pending bets; provisional results for {len(res_now)} of {len(unsettled)} unsettled; wrote {out}')
    return {'pending': int(len(pending)), 'with_odds_now': int(len(cur)), 'provisional': int(len(res_now)), 'checks': checks}
