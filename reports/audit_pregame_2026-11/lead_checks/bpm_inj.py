"""Lead's check (exploratory): how much do roster BPM and the injury features carry, and is it used well?
(1) alone, against the margin; (2) against what the opening line misses; (3) against the open-to-close line move; (4) against what the close misses."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from core.odds_clean import load_clean_odds

pd.set_option('display.width', 220)
cols = ['gameId', 'gameDate', 'seasonYear', 'target_point_diff', 'target_total_pts', 'odds_home_spread', 'odds_total',
        'diff_roster_bpm', 'diff_roster_bpm_ewma', 'roster_bpm_home', 'roster_bpm_away', 'diff_off_plusMinus_20g', 'diff_off_plusMinus_5g',
        'diff_inj_lost', 'diff_inj_lost_new', 'diff_inj_n_out', 'inj_lost_home', 'inj_lost_away', 'inj_lost_new_home', 'inj_lost_new_away',
        'inj_missing_home', 'inj_missing_away']
d = pd.read_csv('data/processed/training_set_v1_reduced_2026-11-01.csv', usecols=lambda c: c in cols, dtype={'gameId': str})
d['gameId'] = d.gameId.str.zfill(10)
o = load_clean_odds('data/nba_stats.db')
o['gameId'] = o.gameId.astype(str).str.zfill(10)
d = d.merge(o[['gameId', 'close_home_spread', 'close_total']], on='gameId', how='left')
d['gameDate'] = pd.to_datetime(d.gameDate)
d = d[(d.seasonYear >= '2019-20') & (d.seasonYear <= '2025-26')].copy()
d['margin'] = d.target_point_diff
d['open_resid'] = d.margin + d.odds_home_spread                 # what the opening line misses (home margin scale)
d['close_resid'] = d.margin + d.close_home_spread
d['move'] = -(d.close_home_spread - d.odds_home_spread)          # line move toward the home team, in points of home margin
d['tot_open_resid'] = d.target_total_pts - d.odds_total
d['tot_move'] = d.close_total - d.odds_total
d['tot_close_resid'] = d.target_total_pts - d.close_total
d['sum_inj_lost'] = d.inj_lost_home + d.inj_lost_away
d['sum_inj_lost_new'] = d.inj_lost_new_home + d.inj_lost_new_away


def ols(y, X, data, label):
    k = data.dropna(subset=[y] + X)
    A = np.column_stack([k[X].to_numpy(float), np.ones(len(k))])
    yv = k[y].to_numpy(float)
    co = np.linalg.lstsq(A, yv, rcond=None)[0]
    r2 = 1 - ((yv - A @ co) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    codes, uniq = pd.factorize(k.gameDate)
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    boots = []
    for _ in range(400):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(uniq), len(uniq))])
        boots.append(np.linalg.lstsq(A[idx], yv[idx], rcond=None)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print(f'{label}  (n={len(k)}, R2={r2:.4f})')
    for name, c, l, h in zip(X, co, lo, hi):
        print(f'     {name:<26} {c:+.3f}  [{l:+.3f}, {h:+.3f}]   sd of feature {k[name].std():.2f}')


print('feature spread: sd diff_roster_bpm %.2f, diff_roster_bpm_ewma %.2f, diff_inj_lost %.2f, diff_inj_lost_new %.2f; share of games with any fresh loss %.2f'
      % (d.diff_roster_bpm.std(), d.diff_roster_bpm_ewma.std(), d.diff_inj_lost.std(), d.diff_inj_lost_new.std(), ((d.inj_lost_new_home > 0) | (d.inj_lost_new_away > 0)).mean()))
print('\n--- (1) margin, no market')
ols('margin', ['diff_off_plusMinus_20g'], d, 'team point-margin EWMA(20) alone')
ols('margin', ['diff_roster_bpm_ewma'], d, 'roster BPM EWMA alone')
ols('margin', ['diff_roster_bpm'], d, 'previous-game roster BPM alone')
ols('margin', ['diff_off_plusMinus_20g', 'diff_roster_bpm_ewma', 'diff_inj_lost', 'diff_inj_lost_new'], d, 'team EWMA + roster BPM + injuries')
print('\n--- (2) what the OPENING line misses (margin + opening spread)')
ols('open_resid', ['diff_roster_bpm_ewma', 'diff_inj_lost', 'diff_inj_lost_new'], d, 'roster BPM + injuries')
print('\n--- (3) the open-to-close line MOVE (points toward the home team)')
ols('move', ['diff_inj_lost', 'diff_inj_lost_new'], d, 'injury features')
ols('move', ['diff_inj_lost_new', 'diff_roster_bpm_ewma'], d, 'fresh injuries + roster BPM')
print('\n--- (4) what the CLOSING line misses')
ols('close_resid', ['diff_roster_bpm_ewma', 'diff_inj_lost', 'diff_inj_lost_new'], d, 'roster BPM + injuries')
print('\n--- totals: injuries (sum of both teams)')
ols('tot_open_resid', ['sum_inj_lost', 'sum_inj_lost_new'], d, 'total minus opening total')
ols('tot_move', ['sum_inj_lost', 'sum_inj_lost_new'], d, 'open-to-close total move')
ols('tot_close_resid', ['sum_inj_lost', 'sum_inj_lost_new'], d, 'total minus closing total')
# how big are the moves when there is fresh news
k = d.dropna(subset=['move', 'diff_inj_lost_new'])
big = k[k.diff_inj_lost_new.abs() >= k.diff_inj_lost_new.abs().quantile(0.9)]
print('\ntop 10%% of games by |fresh injury difference|: n=%d, mean |open-to-close spread move| %.2f (all games %.2f); share where the move goes against the team with the fresh loss %.2f'
      % (len(big), big.move.abs().mean(), k.move.abs().mean(), (np.sign(big.move) == -np.sign(big.diff_inj_lost_new))[big.move != 0].mean()))
