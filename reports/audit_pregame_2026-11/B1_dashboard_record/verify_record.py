"""B1 verification: the dashboard's season record at its default settings, in three data situations, against A1's reconstruction of the live run's bets.

Run from the worktree root (one scenario per process; the app module reads its data locations at import time):
  PY=C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe
  $PY reports/audit_pregame_2026-11/B1_dashboard_record/verify_record.py fallback  <db>            # <db> without bet_recommendations_log (the worktree snapshot DB, read only)
  $PY reports/audit_pregame_2026-11/B1_dashboard_record/verify_record.py local     <db_backfilled> # local database that has the table (a COPY with the backfill applied)
  $PY reports/audit_pregame_2026-11/B1_dashboard_record/verify_record.py hosted    <export_dir>    # no database: the exported files (scripts/export_dashboard_data.py write_files)
It never writes to the database and makes no network call (the app's hard-coded GitHub URLs are pointed at local files).
"""
import json
import os
import sys
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

scenario, target = sys.argv[1], Path(sys.argv[2])
if scenario == 'hosted':
    os.environ['NBA_DASHBOARD_DB'] = str(target / 'no_such.db')
    os.environ['NBA_DASHBOARD_DATA_BASE'] = target.as_uri() + '/'
else:
    os.environ['NBA_DASHBOARD_DB'] = str(target)

import numpy as np
import pandas as pd
import logging
logging.disable(logging.WARNING)
import dashboard.app as app                                 # streamlit imports in bare mode (warnings only)
if scenario == 'hosted':
    app.REMOTE_DATA_URL = (target / 'predictions_log.csv').as_uri()      # the app's hard-coded GitHub URL -> the local export
    app.SNAPSHOT_PATH = target / 'no_snapshot.csv'
    app.RECS_SNAPSHOT_PATH = target / 'no_snapshot_recs.csv'
    app.RECORD_META_SNAPSHOT_PATH = target / 'no_snapshot_meta.json'

df, source = app.load_data()
ctx = app.load_record_context()
print(f"scenario {scenario}: predictions from {source}; log {ctx['log_source']} ({0 if ctx['log'] is None else len(ctx['log'])} rows); sigmas {ctx['sigmas']['spread_std']:.3f} / "
      f"{ctx['sigmas']['total_std']:.3f} from {ctx['sigma_source']}")
season = df[df['game_date'] >= datetime.date(2026, 10, 20)]

A1 = pd.read_csv(ROOT / 'reports/audit_pregame_2026-11/A1_season_bets/bets.csv', dtype={'game_id': str})
A1 = A1[A1.status == 'settled']
out = {'scenario': scenario, 'source': source, 'log_source': ctx['log_source'], 'markets': {}}
rows = []
for bt, mk in (('Moneyline', 'ML'), ('Spread', 'Spread'), ('Totals', 'Total')):
    sim = app.simulate_strategies(season, 0.05, bt)
    b = sim[sim.bet_outcome != 'No Bet']
    rec = app.record_of(sim)
    a = A1[A1.market == mk]
    mine = set(zip(b.game_id.str.zfill(10), b.bet_side))
    ref = set(zip(a.game_id.str.zfill(10), a.side))
    clv_a1 = a.clv.mean()
    out['markets'][bt] = dict(bets=rec['bets'], record=f"{rec['wins']}-{rec['losses']}-{rec['pushes']}", units=round(rec['units'], 2), roi=round(rec['roi'], 1),
                              mean_clv=None if np.isnan(rec['mean_clv']) else round(rec['mean_clv'], 3), n_clv=rec['n_clv'], by_src=rec['by_src'],
                              a1_bets=len(a), a1_record=f"{(a.result == 'W').sum()}-{(a.result == 'L').sum()}-{(a.result == 'P').sum()}", a1_units=round(a.profit.sum(), 2),
                              a1_clv=round(float(clv_a1), 3) if mk != 'ML' else round(float(a.clv_fair_to_fair.mean() * 100), 3),
                              common=len(mine & ref), dashboard_only=sorted(mine - ref), a1_only=sorted(ref - mine))
    # per bet differences on the common bets (price / line / result / profit)
    bm = b.assign(gid=b.game_id.str.zfill(10)).set_index('gid')
    for r in a.itertuples():
        if r.game_id in bm.index and (r.game_id, r.side) in mine:
            x = bm.loc[r.game_id]
            rows.append(dict(market=mk, game_id=r.game_id, side=r.side, a1_line=r.line, line=x.bet_line, a1_price=r.price, price=x.bet_price, a1_result=r.result,
                             result=x.bet_outcome, a1_profit=round(r.profit, 3), profit=round(x.bet_result / 100, 3), src=x.bet_src))
diff = pd.DataFrame(rows)
diff['same_result'] = diff.a1_result.map({'W': 'Win', 'L': 'Loss', 'P': 'Push'}) == diff.result
diff['profit_diff'] = (diff.profit - diff.a1_profit).round(3)
out['common_bets_with_different_result'] = int((~diff.same_result).sum())
out['common_bets_with_profit_diff'] = diff[diff.profit_diff.abs() > 0.0015][['market', 'game_id', 'side', 'a1_price', 'price', 'profit_diff']].to_dict('records')
(HERE / f'verify_{scenario}.json').write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
diff.to_csv(HERE / f'verify_{scenario}_common_bets.csv', index=False)
for bt, m in out['markets'].items():
    print(f"{bt:9s} dashboard {m['bets']:2d} bets {m['record']:7s} {m['units']:+6.2f}u ROI {m['roi']:+6.1f}% CLV {m['mean_clv']} (n={m['n_clv']}) src {m['by_src']}\n"
          f"          A1        {m['a1_bets']:2d} bets {m['a1_record']:7s} {m['a1_units']:+6.2f}u CLV {m['a1_clv']} | common {m['common']}, dashboard-only {m['dashboard_only']}, A1-only {m['a1_only']}")
print('common bets with a different result:', out['common_bets_with_different_result'], '| with a profit difference > 0.0015u:', len(out['common_bets_with_profit_diff']))
