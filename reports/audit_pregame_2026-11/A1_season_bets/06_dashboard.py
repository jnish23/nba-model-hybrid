"""Step 6: what would the dashboard (dashboard/app.py) show for the 2026-27 season, versus the reconstruction?
streamlit is not installed, so the dashboard's own functions are extracted with ast and executed unchanged
(compute_model_accuracy_and_roi, simulate_strategies) on the same table the dashboard reads (predictions_log_v2).
Also compares the CSV mirror (predictions/predictions_2026-27.csv in the production checkout, read-only) with the table (finding P5)."""
import ast
import datetime
import subprocess
import json
from common import *
from team import betting as bt
from team.betting import *

src = (ROOT / 'dashboard' / 'app.py').read_text(encoding='utf-8')
tree = ast.parse(src)                                   # also the requested ast.parse check
want = {'compute_model_accuracy_and_roi', 'simulate_strategies'}
ns = dict(pd=pd, np=np)
ns.update({k: getattr(bt, k) for k in dir(bt) if not k.startswith('_')})
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(ast.get_source_segment(src, node), ns)
md = ["## T6. Dashboard replication (dashboard/app.py functions run unchanged on predictions_log_v2)\n"]

# same preprocessing as dashboard load_data()
df = pd.read_sql("SELECT * FROM predictions_log_v2", con())
df = df.sort_values(by=['game_date', 'prediction_date'], ascending=[False, False]).drop_duplicates(subset=['game_id'], keep='first')
df['game_date'] = pd.to_datetime(df['game_date']).dt.date
df['prediction_date'] = pd.to_datetime(df['prediction_date'])
for col in ['home_score', 'away_score', 'ml_prob', 'odds_home_ml', 'odds_away_ml', 'spread_pred', 'total_pred']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
season = df[df.game_date >= datetime.date(2026, 10, 20)]

B = pd.read_csv(OUT / 'bets.csv', dtype={'game_id': str})
S = B[B.status == 'settled']

# ---- tab 3 "Simulation": what-if with the default sliders
std_sp, std_tt = DEFAULT_SPREAD_STD, DEFAULT_TOTAL_STD
pts_sp = round(points_threshold(std_sp, DEFAULT_PROB_EDGE), 1)
pts_tt = round(points_threshold(std_tt, DEFAULT_PROB_EDGE), 1)
thr = {'Moneyline': DEFAULT_PROB_EDGE, 'Spread': implied_prob_edge(pts_sp, std_sp), 'Totals': implied_prob_edge(pts_tt, std_tt)}
md.append(f"Default sliders: ML {thr['Moneyline']:.3f}; Spread {pts_sp} pts -> prob threshold {thr['Spread']:.4f} (std {std_sp}); "
          f"Totals {pts_tt} pts -> {thr['Totals']:.4f} (std {std_tt}). The live run uses spread std {14.287:.2f}, a per-game total sigma (about 18-20) and the side's own price.\n")
md.append("| bet type | dashboard bets | W-L | profit (u) | ROI | my reconstruction: bets | W-L | units | ROI | identical bet sets? |\n|---|---|---|---|---|---|---|---|---|---|")
sims = {}
for bt_name, mk in [('Moneyline', 'ML'), ('Spread', 'Spread'), ('Totals', 'Total')]:
    sim = ns['simulate_strategies'](season, thr[bt_name], bt_name)
    b = sim[sim.bet_outcome != 'No Bet']
    sims[bt_name] = b
    w, l = (b.bet_outcome == 'Win').sum(), (b.bet_outcome == 'Loss').sum()
    m = S[S.market == mk]
    mw, ml_ = (m.result == 'W').sum(), (m.result == 'L').sum()
    mine = set(zip(m.game_id, m.side))
    col_ = 'bet_subgroup' if bt_name == 'Totals' else 'bet_home_away'
    theirs = set(zip(b.game_id, b[col_].str.lower()))
    both, only_d, only_m = len(mine & theirs), len(theirs - mine), len(mine - theirs)
    md.append(f"| {bt_name} | {len(b)} | {w}-{l} | {b.bet_result.sum()/100:+.2f} | {100*b.bet_result.sum()/(100*len(b)):+.1f}% | {len(m)} | {mw}-{ml_} | {m.profit.sum():+.2f} | "
              f"{100*m.profit.sum()/len(m):+.1f}% | common {both}, dashboard-only {only_d}, mine-only {only_m} |")

# ---- tab 2 "Recent Performance": a pick on EVERY game for spread and total, ML at 5%
metrics, mdf = ns['compute_model_accuracy_and_roi'](season, 0.05)
md.append("\n'Recent Performance' logic (compute_model_accuracy_and_roi) over the whole season (the tab itself uses the last 14 days): a spread and a total PICK on every game, no edge rule, -110, pushes counted as losses:\n")
md.append("| market | picks | wins | win rate | ROI |\n|---|---|---|---|---|")
for k, v in metrics.items():
    md.append(f"| {k} | {v['bets']} | {v['wins']} | {v['win_rate']:.1f}% | {v['roi']:+.1f}% |")
today = datetime.date(2026, 11, 11)
wk = mdf[(mdf.game_date >= today - datetime.timedelta(days=14)) & (mdf.game_date < today)]
wm, _ = ns['compute_model_accuracy_and_roi'](wk, 0.05)
md.append(f"\nLast 14 days as the tab would show them on 2026-11-11 ({wk.game_date.min()} to {wk.game_date.max()}): " +
          '; '.join(f"{k} {v['wins']}-{v['bets']-v['wins']} ROI {v['roi']:+.1f}%" for k, v in wm.items()))

# ---- pushes / whole-number lines and missing prices
whole_sp = (season.odds_home_spread.dropna() == season.odds_home_spread.dropna().round()).sum()
whole_tt = (season.odds_total.dropna() == season.odds_total.dropna().round()).sum()
md.append(f"\nWhole-number lines this season (a push is possible): spread {whole_sp}, total {whole_tt} of {len(season)} games. In the dashboard a push is a 'loss' "
          f"(tab 3: `won` stays False; tab 2: `spread_correct`/`total_correct` are False) and costs -100; betting.py cover_probability refunds it. Pushes among my reconstructed bets: "
          f"{int((S.result == 'P').sum())}.")

# ---- P5: CSV mirror vs table
mirror = pd.read_csv('C:/Users/jorda/nba-model-hybrid/predictions/predictions_2026-27.csv', dtype={'game_id': str})
tab = pd.read_sql("SELECT * FROM predictions_log_v2 WHERE game_date >= '2026-10-20'", con())
mm = mirror.merge(tab, on='game_id', suffixes=('_csv', '_db'))
md.append("\n## P5. CSV mirror versus predictions_log_v2\n")
md.append(f"rows: csv {len(mirror)}, db {len(tab)}, joined {len(mm)}. Status differences: {int((mm.status_csv != mm.status_db).sum())} "
          f"({mm[mm.status_csv != mm.status_db].groupby(['game_date_db', 'status_csv', 'status_db']).size().to_dict()}).")
for col in ['ml_prob', 'spread_pred', 'total_pred', 'odds_home_ml', 'odds_away_ml', 'odds_home_spread', 'odds_total']:
    a, b2 = pd.to_numeric(mm[col + '_csv'], errors='coerce'), pd.to_numeric(mm[col + '_db'], errors='coerce')
    md.append(f"- {col}: max abs diff {np.nanmax(np.abs(a - b2)):.2e}")
diff_scores = (pd.to_numeric(mm.home_score_csv, errors='coerce').fillna(-1) != pd.to_numeric(mm.home_score_db, errors='coerce').fillna(-1)).sum()
md.append(f"- home_score differs on {int(diff_scores)} rows (the PENDING csv rows have no score)")
# published branch
try:
    pub = subprocess.run(['git', 'show', 'dashboard-data:predictions_log.csv'], capture_output=True, text=True, cwd=ROOT).stdout
    from io import StringIO
    pdf = pd.read_csv(StringIO(pub), dtype={'game_id': str})
    pdf = pdf[pdf.game_date >= '2026-10-20']
    md.append(f"- published dashboard-data branch (git show dashboard-data:predictions_log.csv): {len(pdf)} rows from 2026-10-20, COMPLETED {int((pdf.status == 'COMPLETED').sum())}, PENDING {int((pdf.status == 'PENDING').sum())}")
except Exception as e:
    md.append(f"- could not read the dashboard-data branch: {e}")
(OUT / 'tables_dashboard.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md))
