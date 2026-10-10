"""
Point-in-time backtest of injury-report information against the betting market.

For several snapshot times (what the official NBA injury report said at that moment), build
reported-absence features and ask:
  A. Does the reported information explain the line's move (open -> close), the opening line's
     error, the closing line's error? (same regressions as scripts/evaluate_availability.py)
  B. How much of the realized-availability ("oracle") signal does each snapshot capture?
  C. Walk-forward betting simulation: predict the opening line's error from reported absences
     (coefficients fit on prior seasons only), bet when the predicted error exceeds a threshold.
     Results are reported at the OPENING price (upper bound: assumes you bet before the line
     moved) and the CLOSING price (lower bound). The opening line has no timestamp in our data,
     so the truth lies in between.

P(play | status) used to weight reported players is estimated from prior seasons only
(2019-20, the first season, uses the later seasons).

Usage: python scripts/evaluate_injury_reports.py [--skip-parse]
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.odds_clean import load_clean_odds
from models.availability_features import build_player_game_candidates, build_availability_features, to_game_level
from models import injury_report_features as irf

DB = Path('data/nba_stats.db')
SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
SNAPSHOTS = irf.SNAPSHOT_CUTOFFS  # clock cutoffs capped at tip - 30 min (see module)
REP = ['diff_rep_lost_new', 'diff_rep_lost_old']
STANDARD_PAYOUT = 100.0 / 110.0
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0]


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(resid @ resid / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return beta, se, 1 - resid.var() / y.var()


def build_snapshot_features(linked, cand, schedule, name, cutoff):
    rows, cov = irf.snapshot(linked, cutoff, schedule)
    out = []
    for i, season in enumerate(SEASONS):
        # P(play | status) from prior seasons only; the first season has none, so it uses the others
        prior = rows[rows['seasonYear'].isin(SEASONS[:i])] if i else rows[rows['seasonYear'] != season]
        rates = irf.status_play_rates(prior, cand)
        r_s, c_s = rows[rows['seasonYear'] == season], cov[cov['gameId'].isin(rows.loc[rows['seasonYear'] == season, 'gameId'])]
        out.append(irf.reported_features(r_s, c_s, cand[cand['seasonYear'] == season], rates))
    feats = pd.concat(out, ignore_index=True)
    feats['rep_lost_old'] = feats['rep_lost'] - feats['rep_lost_new']
    logging.info(f"[{name}] team-games with a report: {(~feats['rep_missing']).mean():.1%}")
    return feats


def game_level(feats, games):
    cols = ['rep_lost', 'rep_lost_new', 'rep_lost_old', 'rep_n_out', 'rep_missing']
    f = feats[['gameId', 'teamId'] + cols]
    g = games.merge(f.rename(columns={'teamId': 'teamId_home', **{c: f'{c}_home' for c in cols}}), on=['gameId', 'teamId_home'], how='left')
    g = g.merge(f.rename(columns={'teamId': 'teamId_away', **{c: f'{c}_away' for c in cols}}), on=['gameId', 'teamId_away'], how='left')
    for c in ('rep_lost', 'rep_lost_new', 'rep_lost_old', 'rep_n_out'):
        g[f'diff_{c}'] = g[f'{c}_home'] - g[f'{c}_away']
    g['rep_ok'] = ~(g['rep_missing_home'].fillna(True).astype(bool) | g['rep_missing_away'].fillna(True).astype(bool))
    return g


def regressions(g, label):
    rows = []
    for target in ('move', 'open_error', 'close_error'):
        beta, se, r2 = ols(g[REP].values, g[target].values)
        rows.append({'snapshot': label, 'target': target, 'n': len(g), 'R2': r2,
                     'b_new': beta[1], 't_new': beta[1] / se[1], 'b_old': beta[2], 't_old': beta[2] / se[2]})
    return rows


def bet_sim(g, label, fit_target):
    """
    fit_target='open_error': fit the predicted opening-line error directly (noisy: sd ~13.6 pts).
    fit_target='move': fit the open->close line move instead (sd ~2 pts). Since the closing line's
    error is unpredictable, E[open_error | x] = E[move | x], so this estimates the same quantity
    with far less noise.
    """
    seasons = sorted(g['seasonYear'].unique())
    rows = []
    preds = []
    for i, s in enumerate(seasons[1:], start=1):
        tr, te = g[g.seasonYear.isin(seasons[:i])], g[g.seasonYear == s].copy()
        beta, _, _ = ols(tr[REP].values, tr[fit_target].values)
        te['pred_open_error'] = beta[1] * te[REP[0]] + beta[2] * te[REP[1]]  # no intercept: no home/away bias bets
        preds.append(te)
    p = pd.concat(preds)
    bet_sim.last_preds = p
    for thr in THRESHOLDS:
        b = p[p['pred_open_error'].abs() > thr]
        side = np.sign(b['pred_open_error'])  # +1 = home beats the line
        row = {'snapshot': label, 'fit': fit_target, 'threshold': thr, 'bets': len(b)}
        for price in ('open', 'close'):
            cover = side * (b['margin'] + b[f'{price}_home_spread'])
            n_dec = (cover != 0).sum()
            row[f'win% @{price}'] = (cover > 0).sum() / n_dec * 100 if n_dec else np.nan
            row[f'ROI% @{price}'] = (np.where(cover > 0, STANDARD_PAYOUT, np.where(cover < 0, -1.0, 0.0)).sum() / len(b) * 100) if len(b) else np.nan
        clv = side * (b['open_home_spread'] - b['close_home_spread'])
        row['CLV pts'] = clv.mean()
        row['CLV t'] = clv.mean() / (clv.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else np.nan
        rows.append(row)
    return rows


PRIMARY_FIT, PRIMARY_THRESHOLD = 'move', 1.0
N_BOOT = 2000


def primary_test(p, label, game_ids=None):
    """
    The single headline test (chosen after the exploratory grid above, so not pre-registered;
    the 2026-27 season is the real out-of-sample test): move-fit, |predicted| > 1.0 pt.
    Win % at the opening and closing lines, CLV, per season, and a bootstrap 95% CI that resamples
    whole dates (games on the same date share news and market conditions).
    """
    if game_ids is not None:
        p = p[p['gameId'].isin(game_ids)]
    b = p[p['pred_open_error'].abs() > PRIMARY_THRESHOLD].copy()
    if b.empty:
        return [], {}
    side = np.sign(b['pred_open_error'])
    for price in ('open', 'close'):
        cover = side * (b['margin'] + b[f'{price}_home_spread'])
        b[f'win_{price}'] = np.where(cover > 0, 1.0, np.where(cover < 0, 0.0, np.nan))
    b['clv'] = side * (b['open_home_spread'] - b['close_home_spread'])
    rows = []
    for season, g in list(b.groupby('seasonYear')) + [('ALL', b)]:
        rows.append({'snapshot': label, 'season': season, 'bets': len(g),
                     'win% @open': g['win_open'].mean() * 100, 'win% @close': g['win_close'].mean() * 100,
                     'CLV pts': g['clv'].mean()})
    rng = np.random.default_rng(0)
    dates = b['gameDate'].unique() if 'gameDate' in b else b['game_date'].unique()
    by_date = {d: g for d, g in b.groupby('gameDate' if 'gameDate' in b else 'game_date')}
    boot = []
    for _ in range(N_BOOT):
        sample = pd.concat([by_date[d] for d in rng.choice(dates, size=len(dates), replace=True)])
        boot.append((sample['win_open'].mean() * 100, sample['win_close'].mean() * 100, sample['clv'].mean()))
    boot = np.array(boot)
    ci = {'snapshot': label, 'bets': len(b),
          'win% @open': b['win_open'].mean() * 100, 'CI95 @open': tuple(np.percentile(boot[:, 0], [2.5, 97.5]).round(1)),
          'P(@open <= 52.38%)': float((boot[:, 0] <= 52.38).mean()),
          'win% @close': b['win_close'].mean() * 100, 'CI95 @close': tuple(np.percentile(boot[:, 1], [2.5, 97.5]).round(1)),
          'CLV pts': b['clv'].mean(), 'CI95 CLV': tuple(np.percentile(boot[:, 2], [2.5, 97.5]).round(2))}
    return rows, ci


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-parse', action='store_true', help='Use the parsed-report cache as is')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    pd.set_option('display.width', 250)

    reports = irf.parse_all_reports() if not args.skip_parse else irf.parse_all_reports(report_dir=Path('nonexistent'))
    linked = irf.link_reports(reports, DB)
    cand, _ = build_player_game_candidates(DB, SEASONS)
    schedule = irf.load_schedule(DB)

    odds = load_clean_odds(DB)
    odds['gameId'] = odds['gameId'].astype(str)
    odds['margin'] = odds['home_score'] - odds['away_score']
    odds['move'] = odds['open_home_spread'] - odds['close_home_spread']  # = close_mkt - open_mkt (home margin)
    odds['open_error'] = odds['margin'] + odds['open_home_spread']
    odds['close_error'] = odds['margin'] + odds['close_home_spread']
    odds = odds.dropna(subset=['open_home_spread', 'close_home_spread'])

    realized = to_game_level(build_availability_features(DB, SEASONS), odds)

    reg_rows, sim_rows, cover_rows = [], [], []
    move_preds, covered = {}, {}
    for name, cutoff in SNAPSHOTS.items():
        feats = build_snapshot_features(linked, cand, schedule, name, cutoff)
        g = game_level(feats, odds)
        cover_rows.append({'snapshot': name, 'games': len(g), 'with both reports': g['rep_ok'].sum(),
                           'share': g['rep_ok'].mean()})
        g = g[g['rep_ok']]
        reg_rows += regressions(g, name)
        sim_rows += bet_sim(g, name, 'open_error')
        sim_rows += bet_sim(g, name, 'move')
        move_preds[name] = bet_sim.last_preds
        covered[name] = set(g['gameId'])

        # B. Oracle comparison on the same games: does reported info absorb the realized signal?
        both = g.merge(realized[['gameId', 'diff_avail_lost_new', 'diff_avail_lost_old']], on='gameId')
        X = both[REP + ['diff_avail_lost_new', 'diff_avail_lost_old']].values
        beta, se, r2 = ols(X, both['open_error'].values)
        beta_o, se_o, r2_o = ols(both[['diff_avail_lost_new', 'diff_avail_lost_old']].values, both['open_error'].values)
        reg_rows.append({'snapshot': name, 'target': 'open_error | +oracle', 'n': len(both), 'R2': r2,
                         'b_new': beta[1], 't_new': beta[1] / se[1], 'b_old': beta[2], 't_old': beta[2] / se[2],
                         'oracle_t_new': beta[3] / se[3], 'oracle_only_R2': r2_o})

    print("\n=== Snapshot coverage (games with open+close odds)")
    print(pd.DataFrame(cover_rows).round(3).to_string(index=False))
    print("\n=== A/B. Regressions on home-minus-away reported lost value (b = points per unit)")
    print("    'open_error | +oracle': reported + realized features together; oracle_t_new = realized fresh-absence t after controlling for the report")
    print(pd.DataFrame(reg_rows).round(4).to_string(index=False))
    print("\n=== C. Walk-forward betting simulation (fit on prior seasons; bets from 2020-21 on)")
    print("    Break-even at -110 = 52.38%. @open assumes you got the opening number; @close is the lower bound.")
    print(pd.DataFrame(sim_rows).round(2).to_string(index=False))

    print(f"\n=== D. PRIMARY TEST: fit={PRIMARY_FIT}, |predicted move| > {PRIMARY_THRESHOLD} pt")
    print("    Chosen after seeing the grid above (not pre-registered). Bootstrap resamples dates.")
    common = set.intersection(*covered.values())
    for scope, ids in (("each snapshot's own games", None), (f'common games covered by all snapshots (n={len(common)})', common)):
        season_rows, cis = [], []
        for name, p in move_preds.items():
            r, ci = primary_test(p, name, ids)
            season_rows += r
            if ci:
                cis.append(ci)
        print(f"\n--- {scope}: per season")
        print(pd.DataFrame(season_rows).pivot(index='season', columns='snapshot', values='win% @open').round(1).to_string())
        print(f"--- {scope}: overall with bootstrap 95% CI")
        print(pd.DataFrame(cis).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
