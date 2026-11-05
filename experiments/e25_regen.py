"""E25 step 1 - regenerate the three walk-forward runs on the current training set (see PREREGISTRATION.md, E25).
    E25_with_odds  opening odds in the feature pools (the baseline)
    E25_residual   E1's hooks (market as an offset); features selected against the residual targets, odds left out of the lists
    E25_no_odds    no market information in the pools; features selected afresh per fold
python -m experiments.e25_regen [with_odds|residual|no_odds ...]"""
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import e01_residual as e1                          # noqa: E402
from experiments.lib import ROOT, load_games, run_walk_forward                 # noqa: E402
from team.data_paths import latest_training_set                     # noqa: E402
from team.train_baseline import BaselineModel                       # noqa: E402


def load_games_no_odds(csv):
    """load_games() with the odds-free feature pools; the odds columns stay in the frame for evaluation only."""
    m = BaselineModel(csv)
    m.load_data(use_odds=False)
    leaked = [f for f in m.features_diff + m.features_sep if 'odds' in f.lower() or 'close' in f.lower()]
    assert not leaked, leaked
    df = m.df
    df['gameId'] = df['gameId'].astype(str).str.zfill(10)
    from core.odds_clean import load_clean_odds
    o = load_clean_odds(ROOT / 'data' / 'nba_stats.db')
    o['gameId'] = o['gameId'].astype(str).str.zfill(10)
    keep = ['gameId', 'close_home_spread', 'close_total', 'close_home_prob', 'open_home_prob', 'close_source']
    return m, df.merge(o[keep], on='gameId', how='left')


def init_scores(kind, frames, ctx):
    """E1's ML init score, but its fallback is fitted on rows with an opening spread (the 10-19 set has 44 games without one);
    with neither a moneyline nor a spread the offset is the training home-win rate."""
    if kind != 'ml':
        return None
    mt = frames['mt']
    has = mt['odds_home_spread'].notna()
    lr = e1.LogisticRegression(C=100.0).fit(mt.loc[has, ['odds_home_spread']], mt.loc[has, 'target_home_win'])
    base = float(mt['target_home_win'].mean())
    out = []
    for fr in (mt, frames['calib'], frames['test']):
        p = e1.devig_open_prob(fr)
        sp = fr['odds_home_spread']
        fb = np.full(len(fr), base)
        fb[sp.notna().to_numpy()] = lr.predict_proba(fr.loc[sp.notna(), ['odds_home_spread']])[:, 1]
        p = np.where(np.isfinite(p), p, fb)
        out.append(e1.logit(np.clip(p, 1e-4, 1 - 1e-4)))
    return tuple(out)


def main(which):
    csv = latest_training_set()
    logging.info(f"E25 training set: {csv}")
    if 'with_odds' in which:
        run_walk_forward('E25_with_odds', csv=csv)
    if 'residual' in which:
        m, df = load_games(csv)                      # the residual targets need a line: the games without one are dropped inside each fold
        run_walk_forward('E25_residual', df_override=(m, df), selection='residual_no_odds',
                         hooks={'init_scores': init_scores, 'target': e1.target, 'pred_offset': e1.pred_offset,
                                'require_columns': e1.LINE_COLUMNS})
    if 'no_odds' in which:
        run_walk_forward('E25_no_odds', df_override=load_games_no_odds(csv), selection='raw')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    main(sys.argv[1:] or ['with_odds', 'residual', 'no_odds'])
