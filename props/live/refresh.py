"""One refresh of the live dashboard export: pipeline checks, the odds now for the pending bets, and the export tables. Used by the export script and the private publisher."""
import json
from pathlib import Path

import pandas as pd

from props import schedule as S
from props.live import config, current, export, health, store

ODDS_DB = 'data/props_odds.db'
REPORT_DIR = 'data/raw/injury_reports'


def refresh_live_export(out='data/props_dashboard_live', odds_db=ODDS_DB, n_boot=1000, now=None, verbose=True):
    cfg = config.load()
    now = now or pd.Timestamp.now(tz='UTC')
    sched = S.load_schedule(cfg['model_season'])
    tips = pd.to_datetime(sched['nba_tip_utc'], utc=True)
    soon = bool(((tips > now - pd.Timedelta(hours=1)) & (tips < now + pd.Timedelta(hours=6))).any())
    conn = store.connect(store.DEFAULT_PATH)
    checks = health.checks(conn, odds_db, REPORT_DIR, now, soon)
    pending = pd.read_sql('SELECT * FROM prop_bets WHERE bet_id NOT IN (SELECT bet_id FROM prop_results) AND tip_ref_utc > ?', conn, params=[now.strftime('%Y-%m-%dT%H:%M:%SZ')])
    rule = json.loads(Path(cfg['rule_file']).read_text(encoding='utf-8'))
    cur = current.current_odds(pending, sched, odds_db, now, rule)
    export.export_live(store.DEFAULT_PATH, out, cfg, 1000 if n_boot is None else n_boot, checks, cur)
    if verbose:
        print(checks.to_string(index=False))
        print(f'odds now for {len(cur)} of {len(pending)} pending bets; wrote {out}')
    return {'pending': int(len(pending)), 'with_odds_now': int(len(cur)), 'checks': checks}
