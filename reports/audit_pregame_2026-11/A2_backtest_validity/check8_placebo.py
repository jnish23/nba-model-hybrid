"""Check 8: placebo. Shuffle the model's predictions across games within each season (seed 42) and re-run the open-priced 5% rule and the
edge-slope regression. Also: bet every game on a random side at -110 (and both sides) to check the settle code returns about -4.5%."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *   # noqa
from mymetrics import *   # noqa

p = load_e25()
PRED = ['ml_prob', 'ml_prob_raw', 'spread_pred', 'total_pred', 'sigma_spread', 'sigma_total']


def shuffled(df, seed):
    rng = np.random.default_rng(seed)
    out = df.copy()
    for s, idx in df.groupby('seasonYear').indices.items():
        perm = rng.permutation(idx)
        out.iloc[idx, [out.columns.get_loc(c) for c in PRED]] = df.iloc[perm][PRED].to_numpy()
    return out


def slopes(df):
    """OLS slope of (actual - line) on (model - line) at the open, no intercept-free: with intercept. Returns {market: slope}."""
    out = {}
    d = df.dropna(subset=['odds_home_spread'])
    x = d.spread_pred + d.odds_home_spread; y = d.target_point_diff + d.odds_home_spread
    out['spread'] = float(np.polyfit(x, y, 1)[0])
    d = df.dropna(subset=['odds_total'])
    x = d.total_pred - d.odds_total; y = d.target_total_pts - d.odds_total
    out['total'] = float(np.polyfit(x, y, 1)[0])
    d = df.dropna(subset=['open_home_prob'])
    x = d.ml_prob - d.open_home_prob; y = d.target_home_win - d.open_home_prob
    out['ml'] = float(np.polyfit(x, y, 1)[0])
    return out


def summarize(df):
    b, push = make_bets(df)
    r = {}
    for mk in ('ml', 'spread', 'total'):
        bb = b[b.market == mk]
        r[mk] = (len(bb), float(bb.profit.sum() / (100 * len(bb)) * 100) if len(bb) else np.nan, float(bb.won.mean() * 100) if len(bb) else np.nan)
    return r, b


real, rb = summarize(p)
print('REAL    : ', {k: (v[0], round(v[1], 2), round(v[2], 1)) for k, v in real.items()}, ' slopes', {k: round(v, 3) for k, v in slopes(p).items()})
ps42 = shuffled(p, SEED)
sh, sb = summarize(ps42)
print('SHUFFLED (seed 42): ', {k: (v[0], round(v[1], 2), round(v[2], 1)) for k, v in sh.items()}, ' slopes', {k: round(v, 3) for k, v in slopes(ps42).items()})
for mk in ('ml', 'spread', 'total'):
    st = day_boot_stat(sb[sb.market == mk])
    print(f'   shuffled {mk:6s} bets {st["bets"]} hit {st["hit"]:.2f}% [{st["hit_ci"][0]:.1f},{st["hit_ci"][1]:.1f}] ROI {st["roi"]:+.2f}% [{st["roi_ci"][0]:+.1f},{st["roi_ci"][1]:+.1f}]')
# distribution over 200 shuffles
rows = []
for k in range(200):
    q = shuffled(p, 1000 + k)
    r, _ = summarize(q)
    sl = slopes(q)
    rows.append({**{f'{m}_roi': r[m][1] for m in r}, **{f'{m}_n': r[m][0] for m in r}, **{f'{m}_slope': sl[m] for m in sl}})
d = pd.DataFrame(rows)
d.to_csv(OUT / 'check8_placebo_200_shuffles.csv', index=False)
print('\n200 shuffles (seeds 1000..1199): mean [2.5%, 97.5%] of ROI % / n bets / slope')
for m in ('ml', 'spread', 'total'):
    print(f'  {m:6s} ROI {d[m+"_roi"].mean():+.2f} [{d[m+"_roi"].quantile(.025):+.2f}, {d[m+"_roi"].quantile(.975):+.2f}]   '
          f'bets {d[m+"_n"].mean():.0f}   slope {d[m+"_slope"].mean():+.3f} [{d[m+"_slope"].quantile(.025):+.3f}, {d[m+"_slope"].quantile(.975):+.3f}]')
print('real ROI percentile within the placebo distribution:',
      {m: round(float((d[m + '_roi'] < real[m][1]).mean()), 3) for m in ('ml', 'spread', 'total')})

# --- random side every game at -110 (settle-code check), and both sides
rng = np.random.default_rng(SEED)
for mk, act, ln in (('spread', 'target_point_diff', 'odds_home_spread'), ('total', 'target_total_pts', 'odds_total')):
    dd = p.dropna(subset=[ln])
    r = (dd[act] + dd[ln]) if mk == 'spread' else (dd[act] - dd[ln])
    sel = (r != 0).to_numpy()
    rois = []
    for k in range(500):
        side = rng.integers(0, 2, len(dd)) == 0
        won = np.where(side, r > 0, r < 0)[sel]
        rois.append(np.where(won, 100 / 1.1, -100.0).mean())
    both = np.where(r[sel] > 0, 100 / 1.1, -100.0) + np.where(r[sel] < 0, 100 / 1.1, -100.0)
    print(f'random side, every {mk} game at -110 ({int(sel.sum())} bets, {int((~sel).sum())} pushes): mean ROI over 500 draws {np.mean(rois):.3f}% '
          f'(sd of one draw {np.std(rois):.2f}); both sides of every game {both.mean() / 2:.3f}% (exact: -4.545)')
# ML: random side at the open prices
mo = p.dropna(subset=['odds_home_ml', 'odds_away_ml'])
rois = []
for k in range(500):
    side = rng.integers(0, 2, len(mo)) == 0
    won = np.where(side, mo.target_home_win == 1, mo.target_home_win == 0)
    price = np.where(side, mo.odds_home_ml, mo.odds_away_ml)
    rois.append(np.where(won, payout(price) * 100, -100.0).mean())
ph, pa = implied(mo.odds_home_ml), implied(mo.odds_away_ml)
print(f'random side, every ML game at open prices ({len(mo)} games): mean ROI {np.mean(rois):.3f}% ; average overround {np.mean(ph + pa) - 1:.4f}')
