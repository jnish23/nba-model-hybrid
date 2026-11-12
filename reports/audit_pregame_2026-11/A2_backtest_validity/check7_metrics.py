"""Check 7: reproduce the E25_with_odds metrics and the 5%-rule bets from predictions.csv alone, with own code.
Also settle-logic unit tests, and a cross-check against the project's experiments.metrics.make_bets."""
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
sys.path.insert(0, 'C:/Users/jorda/nba-pregame-audit')
from common import *   # noqa
from mymetrics import *   # noqa

p = load_e25()
print('rows', len(p), 'seasons', p.seasonYear.nunique(), ' dup gameIds:', int(p.gameId.duplicated().sum()))
y = p.target_home_win
M = {}


def line_metrics(df, tag):
    r = {}
    # ML
    r['ml_ll_model'] = logloss(df.target_home_win, df.ml_prob); r['ml_brier_model'] = brier(df.target_home_win, df.ml_prob)
    r['ml_ll_raw'] = logloss(df.target_home_win, df.ml_prob_raw)
    r['spread_mae_model'] = float((df.target_point_diff - df.spread_pred).abs().mean())
    r['spread_rmse_model'] = float(np.sqrt(((df.target_point_diff - df.spread_pred) ** 2).mean()))
    r['spread_bias_model(pred-act)'] = float((df.spread_pred - df.target_point_diff).mean())
    r['total_mae_model'] = float((df.target_total_pts - df.total_pred).abs().mean())
    r['total_rmse_model'] = float(np.sqrt(((df.target_total_pts - df.total_pred) ** 2).mean()))
    r['total_bias_model(pred-act)'] = float((df.total_pred - df.target_total_pts).mean())
    return r


pooled = line_metrics(p, 'all')
pooled['n'] = len(p)
rows = [{'season': 'pooled', **pooled}]
for s, g in p.groupby('seasonYear'):
    rows.append({'season': s, 'n': len(g), **line_metrics(g, s)})
tab = pd.DataFrame(rows)
tab.to_csv(OUT / 'check7_model_metrics.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(tab.round(4).to_string())

# ECE for reference to the report (10 equal-width bins, as in project metrics)
def ece(y, p, bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))
print('ECE pooled', round(ece(y.to_numpy(float), p.ml_prob.to_numpy(float)), 4))

# ---- model vs open vs close on common games
res = {}
# spread
for mk, act, mod, opn, cls in (('spread', 'target_point_diff', 'spread_pred', 'odds_home_spread', 'close_home_spread'),
                               ('total', 'target_total_pts', 'total_pred', 'odds_total', 'close_total')):
    sgn = -1.0 if mk == 'spread' else 1.0
    c = p.dropna(subset=[opn, cls]).copy()
    c['pred_model'] = c[mod]; c['pred_open'] = sgn * c[opn]; c['pred_close'] = sgn * c[cls]
    for k in ('model', 'open', 'close'):
        c['ae_' + k] = (c[act] - c['pred_' + k]).abs(); c['se_' + k] = (c[act] - c['pred_' + k]) ** 2
    print(f'\n[{mk}] common games with open and close: {len(c)}')
    out = {}
    for k in ('model', 'open', 'close'):
        out[k] = {'mae': float(c['ae_' + k].mean()), 'rmse': float(np.sqrt(c['se_' + k].mean())), 'bias(pred-act)': float((c['pred_' + k] - c[act]).mean())}
    print(pd.DataFrame(out).round(3).to_string())
    for a, b in (('model', 'open'), ('model', 'close'), ('open', 'close')):
        pt, lo, hi = paired_day_boot(c, c['ae_' + a], c['ae_' + b])
        print(f'   MAE {a} - {b}: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]')
    for k in ('model', 'open', 'close'):
        res[(mk, k)] = out[k]
    # by season MAE table
    print((c.groupby('seasonYear')[['ae_model', 'ae_open', 'ae_close']].mean()).round(3).to_string())
# ML
c = p.dropna(subset=['open_home_prob', 'close_home_prob']).copy()
print(f'\n[ml] common games with open and close prob: {len(c)}')
for nm, col in (('model', 'ml_prob'), ('open(consensus devig)', 'open_home_prob'), ('close(consensus devig)', 'close_home_prob')):
    print(f'   {nm:24s} logloss {logloss(c.target_home_win, c[col]):.4f} brier {brier(c.target_home_win, c[col]):.4f}')
ll = lambda col: -(c.target_home_win * np.log(c[col].clip(1e-9, 1 - 1e-9)) + (1 - c.target_home_win) * np.log((1 - c[col]).clip(1e-9, 1 - 1e-9)))
for a, b in (('ml_prob', 'open_home_prob'), ('ml_prob', 'close_home_prob'), ('open_home_prob', 'close_home_prob')):
    pt, lo, hi = paired_day_boot(c, ll(a), ll(b))
    print(f'   logloss {a} - {b}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]')
# single-book open de-vig
mo = p.dropna(subset=['odds_home_ml', 'odds_away_ml']).copy()
ph, pa = implied(mo.odds_home_ml), implied(mo.odds_away_ml)
mo['devig_book_open'] = ph / (ph + pa)
print('   single-book (training-set) open de-vig logloss on its own %d games: %.4f ; model on same: %.4f' %
      (len(mo), logloss(mo.target_home_win, mo.devig_book_open), logloss(mo.target_home_win, mo.ml_prob)))

# ---- 5% rule bets priced at the open
bets, pushes = make_bets(p)
print('\npushes refunded at the open:', pushes)
rows = []
for mk in ('ml', 'spread', 'total'):
    b = bets[bets.market == mk]
    r = day_boot_stat(b)
    r.update({'market': mk, 'pushes': pushes.get(mk, 0)})
    rows.append(r)
    print(f"{mk:7s} bets {r['bets']:5d}  W-L-P {r['wins']}-{r['losses']}-{r['pushes']}  hit {r['hit']:.2f}% [{r['hit_ci'][0]:.1f}, {r['hit_ci'][1]:.1f}]  "
          f"ROI {r['roi']:+.2f}% [{r['roi_ci'][0]:+.1f}, {r['roi_ci'][1]:+.1f}]")
    print('        by season ROI:', {s: round(g.profit.sum() / (100 * len(g)) * 100, 1) for s, g in b.groupby('seasonYear')},
          ' n:', {s: len(g) for s, g in b.groupby('seasonYear')})
pd.DataFrame(rows).to_csv(OUT / 'check7_bets_at_open.csv', index=False)
bets.to_csv(OUT / 'check7_bets_at_open_detail.csv', index=False)
allb = day_boot_stat(bets)
print('all markets: bets', allb['bets'], 'ROI', round(allb['roi'], 2), allb['roi_ci'])

# ---- cross-check with the project's own make_bets
try:
    from experiments import metrics as pm
    pb = pm.make_bets(p)
    pb['gameId'] = pb['gameId'].astype(str).str.zfill(10)
    for mk in ('ml', 'spread', 'total'):
        a = bets[bets.market == mk].set_index('gameId')
        b = pb[pb.market == mk].set_index('gameId')
        j = a.join(b[['profit', 'side']], rsuffix='_proj', how='outer')
        print(f'cross-check {mk}: mine {len(a)} project {len(b)} ; same game set: {set(a.index) == set(b.index)} ; max |profit diff| '
              f'{(j.profit - j.profit_proj).abs().max():.6f}; sides differ in {(j.side != j.side_proj).sum()}')
except Exception as e:                      # keep going if the project module cannot be imported in this env
    print('project make_bets cross-check failed:', repr(e))

# ---- settle unit tests (hand-built cases)
cases = [
    ('home -3.5, margin +4 -> home covers', 4, -3.5, +1), ('home -3.5, margin +3 -> away covers', 3, -3.5, -1),
    ('home -3, margin +3 -> push', 3, -3.0, 0), ('home +3.5, margin -3 -> home covers', -3, 3.5, +1),
    ('home +3.5, margin -4 -> away covers', -4, 3.5, -1), ('home -7, margin -2 -> away covers', -2, -7.0, -1), ('home +6, margin -6 -> push', -6, 6.0, 0)]
ok = True
for nm, m, ln, exp in cases:
    got = int(settle_spread_home(m, ln))
    ok &= (got == exp)
    print(f'  settle {nm}: got {got} expected {exp} {"OK" if got == exp else "FAIL"}')
print('  total: 230 vs 225.5 ->', int(settle_total_over(230, 225.5)), '; 220 vs 220 ->', int(settle_total_over(220, 220)), '; 210 vs 220.5 ->', int(settle_total_over(210, 220.5)))
# away ML priced at its own odds: home -300 / away +250, model P(home)=0.60: home edge .60-.75=-.15 ; away edge .40-.2857=+.114
t = pd.DataFrame({'gameId': ['x'], 'gameDate': ['2020-01-01'], 'seasonYear': ['2019-20'], 'target_point_diff': [-3], 'target_total_pts': [200],
                  'target_home_win': [0], 'ml_prob': [0.60], 'odds_home_ml': [-300.0], 'odds_away_ml': [250.0], 'odds_home_spread': [np.nan],
                  'odds_total': [np.nan], 'spread_pred': [0.0], 'total_pred': [0.0], 'sigma_spread': [13.0], 'sigma_total': [18.0]})
tb, _ = make_bets(t, markets=('ml',))
print('  away-ML test: side', tb.side.iloc[0], 'price', tb.price.iloc[0], 'profit (win at +250, per $100)', tb.profit.iloc[0], '(expect 250)')
tb2, ps = make_bets(t.assign(odds_home_spread=np.nan, odds_total=np.nan), markets=('spread', 'total'))
print('  missing lines -> bets:', len(tb2), '(expect 0)')
# random side, -110, every game -> expect -4.545%
rng = np.random.default_rng(SEED)
for mk, act, ln in (('spread', 'target_point_diff', 'odds_home_spread'), ('total', 'target_total_pts', 'odds_total')):
    d = p.dropna(subset=[ln])
    r = (d[act] + d[ln]) if mk == 'spread' else (d[act] - d[ln])
    push = r == 0
    side = rng.integers(0, 2, len(d)) == 0
    won = np.where(side, r > 0, r < 0)
    sel = ~push
    prof = np.where(won[sel], 100 / 1.1, -100.0)
    print(f'  random side every {mk} game at -110: n={int(sel.sum())} (pushes {int(push.sum())}) hit {won[sel].mean():.4f} ROI {prof.mean():.3f}%')
