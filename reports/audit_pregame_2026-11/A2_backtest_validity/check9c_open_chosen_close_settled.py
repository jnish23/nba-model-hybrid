"""Check 9c: how much of the open-priced ROI is line movement the bettor could not have bought at the noon information cut?
Bets are CHOSEN with the 5% rule against the open (what the backtest does) and then SETTLED (a) at the open line (backtest ROI) and
(b) at the close line (still -110). Also mean closing-line value in points for the bets. Day-bootstrap CIs."""
import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from common import *
from mymetrics import *
import warnings; warnings.filterwarnings('ignore')
p = load_e25()
for mk, ln, cl in (('spread', 'odds_home_spread', 'close_home_spread'), ('total', 'odds_total', 'close_total')):
    d = p.dropna(subset=[ln, cl]).reset_index(drop=True)
    kw = dict(settle_sp=d[cl]) if mk == 'spread' else dict(settle_tot=d[cl])
    bo, _ = make_bets(d, markets=(mk,))
    bc, pc = make_bets(d, markets=(mk,), **kw)
    so, sc = day_boot_stat(bo), day_boot_stat(bc)
    # CLV in points for the chosen bets
    j = bo.merge(d[['gameId', ln, cl]], on='gameId')
    if mk == 'spread':
        clv = np.where(j.side == 'home', j[ln] - j[cl], -(j[ln] - j[cl]))    # home bet at s is better if close is lower
    else:
        clv = np.where(j.side == 'over', j[cl] - j[ln], j[ln] - j[cl])
    # day bootstrap of mean CLV
    jj = j.assign(won=clv, profit=0.0)
    days, inv = np.unique(jj.gameDate.to_numpy(), return_inverse=True); nd = len(days)
    s = np.bincount(inv, weights=clv, minlength=nd); c = np.bincount(inv, minlength=nd).astype(float)
    rng = np.random.default_rng(SEED); dr = rng.integers(0, nd, (1000, nd)); bs = s[dr].sum(1) / c[dr].sum(1)
    print(f'{mk:6s} chosen at open: {so["bets"]} bets; settled at OPEN: hit {so["hit"]:.1f}% ROI {so["roi"]:+.2f}% [{so["roi_ci"][0]:+.1f},{so["roi_ci"][1]:+.1f}] | '
          f'settled at CLOSE: {sc["bets"]} bets hit {sc["hit"]:.1f}% ROI {sc["roi"]:+.2f}% [{sc["roi_ci"][0]:+.1f},{sc["roi_ci"][1]:+.1f}] | '
          f'mean CLV {clv.mean():+.3f} pts [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]; share with CLV>0 {np.mean(clv > 0):.2f}, <0 {np.mean(clv < 0):.2f}')
