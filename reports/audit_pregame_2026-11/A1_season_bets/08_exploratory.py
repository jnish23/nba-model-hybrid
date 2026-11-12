"""Step 8 (EXPLORATORY, outside the declared split family): the model's side on EVERY completed game (not only the 5% bets), by size of the
edge versus the run-time line. More games (about 150) than the 66 bets, so a monotone relation between edge size and hit rate / CLV would be easier to see.
Settled at the logged line and -110 (spread/total). Intervals: day bootstrap."""
from common import *

G = pd.read_csv(OUT / 'games.csv', dtype={'game_id': str})
G = G[G.home_score.notna()].copy()
G['margin'] = G.home_score - G.away_score
G['tot'] = G.home_score + G.away_score
G['close_ok'] = G.snap_close_mins_to_tip <= 90
md = ["## T7 (exploratory). Model side on every completed game, by edge size versus the run-time line\n"]


def block(mk, edge, win, clv, bins, labels):
    d = pd.DataFrame({'day': G.game_date, 'edge': edge, 'win': win, 'clv': clv}).dropna(subset=['edge', 'win'])
    d['bucket'] = pd.cut(d.edge, bins, labels=labels, right=False)
    md.append(f"\n### {mk}\n\n| |edge| bucket (pts) | games | hit rate [95% CI] | ROI at -110 [CI] | games with close | mean CLV pts [CI] |\n|---|---|---|---|---|---|")
    for lb in labels + ['ALL']:
        x = d if lb == 'ALL' else d[d.bucket == lb]
        if not len(x):
            continue
        hr, lo, hi, _ = day_boot(x.day, x.win)
        profit = np.where(x.win == 1, 100 / 110, -1.0)
        roi, rlo, rhi, _ = day_boot(x.day, profit)
        c = x.dropna(subset=['clv'])
        cm = day_boot(c.day, c.clv) if len(c) else (np.nan,) * 4
        md.append(f"| {lb} | {len(x)} | {100*hr:.1f}% [{100*lo:.1f}, {100*hi:.1f}] | {100*roi:+.1f}% [{100*rlo:+.1f}, {100*rhi:+.1f}] | {len(c)} | {cm[0]:+.2f} [{cm[1]:+.2f}, {cm[2]:+.2f}] |")
    # slope of hit on edge and clv on edge (day bootstrap on cov/var is awkward; use correlation via bootstrap of days)
    ok = d.dropna(subset=['edge'])
    r_, lo_, hi_ = np.nan, np.nan, np.nan
    days = ok.day.values
    ud, inv = np.unique(days, return_inverse=True)
    rng = np.random.default_rng(SEED)
    rs = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(ud), len(ud))
        idx = np.concatenate([np.where(inv == k)[0] for k in pick])
        sub = ok.iloc[idx]
        if sub.win.std() > 0:
            rs.append(np.corrcoef(sub.edge, sub.win)[0, 1])
    r0 = np.corrcoef(ok.edge, ok.win)[0, 1]
    md.append(f"\ncorr(|edge|, win) = {r0:+.3f}, day-bootstrap 95% CI [{np.percentile(rs, 2.5):+.3f}, {np.percentile(rs, 97.5):+.3f}] (n={len(ok)}).")


# spread: model side home if spread_pred + home_line > 0
pe = G.spread_pred + G.odds_home_spread
s = np.where(pe > 0, 1, -1)
win = np.where(G.odds_home_spread.isna(), np.nan, np.where(s == 1, (G.margin + G.odds_home_spread > 0), (G.margin + G.odds_home_spread < 0))).astype(float)
clv = np.where(G.close_ok & G.snap_close_spread.notna(), s * (G.odds_home_spread - G.snap_close_spread), np.nan)
block('Spread', pe.abs(), win, clv, [0, 1, 2, 2.66, 4, 99], ['0-1', '1-2', '2-2.66', '2.66-4 (bet zone)', '4+'])
pt = G.total_pred - G.odds_total
s = np.where(pt > 0, 1, -1)
win = np.where(G.odds_total.isna(), np.nan, np.where(s == 1, G.tot > G.odds_total, G.tot < G.odds_total)).astype(float)
clv = np.where(G.close_ok & G.snap_close_total.notna(), s * (G.snap_close_total - G.odds_total), np.nan)
block('Total', pt.abs(), win, clv, [0, 2, 3.46, 5, 7, 99], ['0-2', '2-3.46', '3.46-5 (bet zone)', '5-7', '7+'])
# unconditional drift of the lines after the run
drift = (G.snap_close_total - G.odds_total)[G.close_ok]
m, lo, hi, _ = day_boot(G.game_date[G.close_ok & G.snap_close_total.notna()], drift.dropna())
md.append(f"\nUnconditional drift of the total from the run-time line to the close, all games with a close: mean {m:+.2f} pts [{lo:+.2f}, {hi:+.2f}] (n={int(drift.notna().sum())}); "
          f"a bettor taking Under at the run-time line on every game would have CLV {-m:+.2f}.")
drift = (G.snap_close_spread - G.odds_home_spread)[G.close_ok]
m, lo, hi, _ = day_boot(G.game_date[G.close_ok & G.snap_close_spread.notna()], drift.dropna())
md.append(f"Home spread drift (close minus run-time home line; positive = home gets more points at the close): mean {m:+.2f} pts [{lo:+.2f}, {hi:+.2f}] (n={int(drift.notna().sum())}).")
(OUT / 'tables_exploratory.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md))
