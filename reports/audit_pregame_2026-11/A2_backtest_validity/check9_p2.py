"""Check 9: verify lead finding P2 on E25_with_odds with day-bootstrap CIs (resample dates, 1,000 draws, seed 42).
 (a) 5% rule priced and settled at the CLOSE (spread, total at -110; moneyline at the consensus close prices and at zero vig)
 (b) share of those bets that fade the open-to-close move
 (c) regression of (result - close) on (model - open) [single regressor] and on both (model - open) and (open - close), coefficients with
     day-bootstrap CIs. Everything restricted to games that have BOTH an opening and a closing line.
 (d) the same rule priced at the OPEN on the same game set (for the 629 / 1,924 comparison)."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
sys.path.insert(0, 'C:/Users/jorda/nba-pregame-audit')
from common import *   # noqa
from mymetrics import *   # noqa
import warnings; warnings.filterwarnings('ignore')
from core.odds_clean import load_clean_odds

p = load_e25()
o = load_clean_odds(DB)
o['gameId'] = o['gameId'].astype(str).str.zfill(10)
p = p.merge(o[['gameId', 'close_home_ml', 'close_away_ml', 'open_home_ml', 'open_away_ml', 'open_home_spread', 'open_total']], on='gameId', how='left')
print('E25 rows', len(p))

# ---------------------------------------------------------------- (d) open-priced, on games with both lines
sp = p.dropna(subset=['odds_home_spread', 'close_home_spread']).copy()
to = p.dropna(subset=['odds_total', 'close_total']).copy()
ml = p.dropna(subset=['odds_home_ml', 'odds_away_ml', 'close_home_prob', 'open_home_prob']).copy()
print(f'games with both lines: spread {len(sp)}, total {len(to)}, ML (open prices and close prob) {len(ml)}')
rows = []


def show(tag, b):
    st = day_boot_stat(b)
    rows.append({'rule': tag, **{k: st[k] for k in ('bets', 'wins', 'losses', 'hit', 'roi')}, 'hit_lo': st['hit_ci'][0], 'hit_hi': st['hit_ci'][1],
                 'roi_lo': st['roi_ci'][0], 'roi_hi': st['roi_ci'][1]})
    print(f"  {tag:46s} bets {st['bets']:5d} W-L {st['wins']}-{st['losses']}  hit {st['hit']:.2f}% [{st['hit_ci'][0]:.1f},{st['hit_ci'][1]:.1f}]  "
          f"ROI {st['roi']:+.2f}% [{st['roi_ci'][0]:+.1f},{st['roi_ci'][1]:+.1f}]")
    return st


b_open_sp, push_o_sp = make_bets(sp, markets=('spread',))
b_open_to, push_o_to = make_bets(to, markets=('total',))
print('\n(d) 5% rule priced at the OPEN (games that also have a close):')
show('spread @open', b_open_sp); show('total @open', b_open_to)

# ---------------------------------------------------------------- (a) priced at the close
b_cl_sp, push_c_sp = make_bets(sp, markets=('spread',), line_sp='close_home_spread')
b_cl_to, push_c_to = make_bets(to, markets=('total',), line_tot='close_total')
print('\n(a) 5% rule priced and settled at the CLOSE (-110):')
s1 = show('spread @close', b_cl_sp); s2 = show('total @close', b_cl_to)
print('   pushes refunded:', push_c_sp, push_c_to)
# at zero vig (even money) for reference
for nm, b in (('spread', b_cl_sp), ('total', b_cl_to)):
    pr = np.where(b.won, 100.0, -100.0)
    z = b.assign(profit=pr)
    st = day_boot_stat(z)
    print(f'   {nm} @close at ZERO vig (even money): ROI {st["roi"]:+.2f}% [{st["roi_ci"][0]:+.1f},{st["roi_ci"][1]:+.1f}] hit {st["hit"]:.2f}%')
    rows.append({'rule': f'{nm} @close zero vig', 'bets': st['bets'], 'hit': st['hit'], 'roi': st['roi'], 'roi_lo': st['roi_ci'][0], 'roi_hi': st['roi_ci'][1],
                 'hit_lo': st['hit_ci'][0], 'hit_hi': st['hit_ci'][1], 'wins': st['wins'], 'losses': st['losses']})
# ML at close prices (consensus median prices, vig included) : bets chosen against the close prices
mlc = p.dropna(subset=['close_home_ml', 'close_away_ml']).copy()
b_cl_ml, _ = make_bets(mlc, markets=('ml',), ml_cols=('close_home_ml', 'close_away_ml'))
b_open_ml, _ = make_bets(mlc, markets=('ml',))
print('\nMoneyline, games with close prices (%d):' % len(mlc))
show('ML @open prices (same games)', b_open_ml)
show('ML @close median prices', b_cl_ml)
# ML at the close, zero vig: bet when model prob - de-vigged close prob > 5%, paid at fair odds of the close
q = mlc.close_home_prob.to_numpy(float)
pm = mlc.ml_prob.to_numpy(float)
eh, ea = pm - q, (1 - pm) - (1 - q)
home = eh >= ea
edge = np.where(home, eh, ea)
ok = edge > 0.05
won = np.where(home, mlc.target_home_win == 1, mlc.target_home_win == 0)
fairp = np.where(home, q, 1 - q)
prof = np.where(won, 100 * (1 / fairp - 1), -100.0)
zb = mlc[ok][['gameId', 'gameDate', 'seasonYear']].assign(won=won[ok], profit=prof[ok])
zb['gameDate'] = pd.to_datetime(zb.gameDate)
show('ML @close de-vigged prob, zero vig', zb)

# ---------------------------------------------------------------- (b) fade share
print('\n(b) share of the close-priced bets that FADE the open-to-close move:')
fade_rows = []
for nm, b, d_, opn, cls in (('spread', b_cl_sp, sp, 'odds_home_spread', 'close_home_spread'), ('total', b_cl_to, to, 'odds_total', 'close_total')):
    j = b.merge(d_[['gameId', opn, cls]], on='gameId')
    if nm == 'spread':
        toward_home = (j[opn] - j[cls])           # >0: line moved toward the home team (home became bigger favourite)
        s_bet = np.where(j.side == 'home', 1.0, -1.0)
        mv = toward_home * s_bet                   # >0: bet side is the side the market moved toward (WITH the move)
    else:
        toward_over = (j[cls] - j[opn])
        s_bet = np.where(j.side == 'over', 1.0, -1.0)
        mv = toward_over * s_bet
    fade, withm, flat = int((mv < 0).sum()), int((mv > 0).sum()), int((mv == 0).sum())
    # day bootstrap of the fade share among all bets
    jj = j.assign(won=(mv < 0).astype(float), profit=0.0)
    st = day_boot_stat(jj)
    print(f'  {nm}: {len(j)} bets: fade {fade} ({fade / len(j):.1%}; 95% day-boot [{st["hit_ci"][0]:.1f}, {st["hit_ci"][1]:.1f}]), with the move {withm} ({withm / len(j):.1%}), no move {flat} ({flat / len(j):.1%})')
    fade_rows.append({'market': nm, 'bets': len(j), 'fade': fade, 'with_move': withm, 'flat': flat, 'fade_share': fade / len(j), 'lo': st['hit_ci'][0], 'hi': st['hit_ci'][1]})
    # hit rate by fade / with
    print(f'     hit rate fading: {j[mv < 0].won.mean():.3f} (n={int((mv<0).sum())}); with move: {j[mv > 0].won.mean() if (mv > 0).any() else float("nan"):.3f} (n={int((mv>0).sum())}); no move: {j[mv == 0].won.mean() if (mv == 0).any() else float("nan"):.3f}')
# ML
j = b_cl_ml.merge(mlc[['gameId', 'close_home_prob', 'open_home_prob']], on='gameId')
move_home = j.close_home_prob - j.open_home_prob       # >0: market moved toward home
s_bet = np.where(j.side == 'home', 1.0, -1.0)
mv = move_home * s_bet
print(f'  ml: {len(j)} bets @close prices: fade {(mv < 0).sum()} ({(mv < 0).mean():.1%}), with move {(mv > 0).sum()}, none {(mv == 0).sum()}')
pd.DataFrame(fade_rows).to_csv(OUT / 'check9_fade_share.csv', index=False)

# ---------------------------------------------------------------- (c) regressions with day bootstrap
def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def reg_boot(d, y, x1, x2, n=1000, seed=SEED):
    """OLS of y on [x1] and on [x1, x2]; day-bootstrap CIs. d has gameDate."""
    days, inv = np.unique(d.gameDate.to_numpy(), return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(days))]
    rng = np.random.default_rng(seed)
    Y, X1, X2 = y.to_numpy(float), x1.to_numpy(float), x2.to_numpy(float)
    pt1 = ols(X1[:, None], Y); pt2 = ols(np.column_stack([X1, X2]), Y)
    b1, b2 = np.empty((n, 2)), np.empty((n, 3))
    for k in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(days), len(days))])
        b1[k] = ols(X1[idx][:, None], Y[idx]); b2[k] = ols(np.column_stack([X1[idx], X2[idx]]), Y[idx])
    lo1, hi1 = np.percentile(b1, [2.5, 97.5], axis=0); lo2, hi2 = np.percentile(b2, [2.5, 97.5], axis=0)
    # simple (iid) OLS standard errors for comparison with the lead's numbers
    X = np.column_stack([np.ones(len(Y)), X1]); res = Y - X @ pt1; se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var(ddof=2))
    return pt1, lo1, hi1, pt2, lo2, hi2, se, np.corrcoef(X1, X2)[0, 1]


reg_rows = []
print('\n(c) regressions (day-bootstrap 95% CI):')
for nm, d_, y, x1, x2 in (
        ('spread', sp, sp.target_point_diff + sp.close_home_spread, sp.spread_pred + sp.odds_home_spread, sp.close_home_spread - sp.odds_home_spread),
        ('total', to, to.target_total_pts - to.close_total, to.total_pred - to.odds_total, to.odds_total - to.close_total),
        ('ml (prob)', ml, ml.target_home_win - ml.close_home_prob, ml.ml_prob - ml.open_home_prob, ml.open_home_prob - ml.close_home_prob)):
    # orientation: x2 is defined so that a positive value means the market moved AWAY from the home/over side (open-minus-close in model units);
    # spread: model margin = -spread, so open-close in margin units = (-open)-(-close) = close - open as coded.
    pt1, lo1, hi1, pt2, lo2, hi2, se, rho = reg_boot(d_, y, x1, x2)
    print(f'  {nm:10s} n={len(d_)}  y = result - close, x1 = model - open, x2 = open - close (margin/total/prob units; corr(x1,x2)={rho:+.2f})')
    print(f'     single regressor: b1 = {pt1[1]:+.3f} [{lo1[1]:+.3f}, {hi1[1]:+.3f}] (iid se {se[1]:.3f})')
    print(f'     both regressors : b1 = {pt2[1]:+.3f} [{lo2[1]:+.3f}, {hi2[1]:+.3f}] ; b2 = {pt2[2]:+.3f} [{lo2[2]:+.3f}, {hi2[2]:+.3f}]')
    reg_rows.append({'market': nm, 'n': len(d_), 'b1_single': pt1[1], 'b1_single_lo': lo1[1], 'b1_single_hi': hi1[1], 'b1_single_iid_se': se[1],
                     'b1_both': pt2[1], 'b1_both_lo': lo2[1], 'b1_both_hi': hi2[1], 'b2_both': pt2[2], 'b2_both_lo': lo2[2], 'b2_both_hi': hi2[2], 'corr_x1_x2': rho})
pd.DataFrame(reg_rows).to_csv(OUT / 'check9_regressions.csv', index=False)
pd.DataFrame(rows).to_csv(OUT / 'check9_bets.csv', index=False)
# the same regression restricted to the games where the rule would bet at the close (selection check) and by season for b1_single
for nm, d_, y, x1 in (('spread', sp, sp.target_point_diff + sp.close_home_spread, sp.spread_pred + sp.odds_home_spread),
                      ('total', to, to.target_total_pts - to.close_total, to.total_pred - to.odds_total)):
    per = {s: round(float(np.polyfit(x1[d_.seasonYear == s], y[d_.seasonYear == s], 1)[0]), 2) for s in sorted(d_.seasonYear.unique())}
    print(f'  {nm} single-regressor slope by season:', per)

# ---------------------------------------------------------------- robustness: consensus open instead of the single-book open the model read
print('\nRobustness: x1 = model - CONSENSUS open (core.odds_clean median of books) and fade share / move defined with the consensus open')
spc = sp.dropna(subset=['open_home_spread']); toc = to.dropna(subset=['open_total'])
for nm, d_, y, x1, x2 in (
        ('spread', spc, spc.target_point_diff + spc.close_home_spread, spc.spread_pred + spc.open_home_spread, spc.close_home_spread - spc.open_home_spread),
        ('total', toc, toc.target_total_pts - toc.close_total, toc.total_pred - toc.open_total, toc.open_total - toc.close_total)):
    pt1, lo1, hi1, pt2, lo2, hi2, se, rho = reg_boot(d_, y, x1, x2)
    print(f'  {nm:7s} n={len(d_)} single b1 = {pt1[1]:+.3f} [{lo1[1]:+.3f}, {hi1[1]:+.3f}] ; both: b1 = {pt2[1]:+.3f} [{lo2[1]:+.3f}, {hi2[1]:+.3f}], b2 = {pt2[2]:+.3f} [{lo2[2]:+.3f}, {hi2[2]:+.3f}]')
    reg_rows.append({'market': nm + ' (consensus open)', 'n': len(d_), 'b1_single': pt1[1], 'b1_single_lo': lo1[1], 'b1_single_hi': hi1[1], 'b1_single_iid_se': se[1],
                     'b1_both': pt2[1], 'b1_both_lo': lo2[1], 'b1_both_hi': hi2[1], 'b2_both': pt2[2], 'b2_both_lo': lo2[2], 'b2_both_hi': hi2[2], 'corr_x1_x2': rho})
for nm, b, d_, opn, cls in (('spread', b_cl_sp, spc, 'open_home_spread', 'close_home_spread'), ('total', b_cl_to, toc, 'open_total', 'close_total')):
    j = b.merge(d_[['gameId', opn, cls]], on='gameId')
    if nm == 'spread':
        mv = (j[opn] - j[cls]) * np.where(j.side == 'home', 1.0, -1.0)
    else:
        mv = (j[cls] - j[opn]) * np.where(j.side == 'over', 1.0, -1.0)
    print(f'  {nm}: fade share with consensus open: {(mv < 0).mean():.1%} (with {(mv > 0).mean():.1%}, no move {(mv == 0).mean():.1%}) on {len(j)} bets')
pd.DataFrame(reg_rows).to_csv(OUT / 'check9_regressions.csv', index=False)
