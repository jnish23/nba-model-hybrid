"""
Phase 0 gate: dump the outputs of the team pipeline so two code trees (before / after the core/ team/ props/
refactor) can be compared byte for byte.

Run from the root of a code tree whose data/ points at the SAME frozen data for both runs:
    python scripts/phase0_equivalence_check.py --out <dir>
It writes, into <dir>:
    training_set.csv        what scripts/generate_training_set.py produces (the real entry point is executed)
    live_frame_<date>.csv   build_live_game_frame() for historical dates (the inference feature path)
    predictions.csv         deployed model set scored on those frames: raw ML prob, calibrated prob, spread, total
Compare two output folders with:  python scripts/phase0_equivalence_check.py --compare <dirA> <dirB>
Works on both layouts (models.* before the refactor, core.* / team.* after).
"""
import argparse
import datetime as dt
import hashlib
import importlib
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
DATES = ['2026-01-15', '2025-03-12', '2022-11-23', '2020-01-17']          # + the last game date in the database


def imp(name, *candidates):
    """Import the first of `candidates` (module paths) that exists and return it."""
    for c in candidates:
        try:
            return importlib.import_module(c)
        except ModuleNotFoundError as e:
            if e.name and c.startswith(e.name):
                continue
            raise
    raise ModuleNotFoundError(f'{name}: none of {candidates}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    db = str(ROOT / 'data' / 'nba_stats.db')

    # 1. the real training-set entry point
    produced = ROOT / 'data' / 'processed' / f'training_set_v1_reduced_{dt.date.today()}.csv'
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'generate_training_set.py')], cwd=ROOT, check=True)
    shutil.move(str(produced), out / 'training_set.csv')

    # 2. inference feature path + deployed model scoring, replayed on historical dates
    gfb = imp('game_features', 'team.game_features_v2', 'models.game_features_v2')
    lf = imp('live_features', 'team.live_features', 'models.live_features')
    irf = imp('injury_report_features', 'core.injury_report_features', 'models.injury_report_features')
    mr = imp('model_registry', 'team.model_registry', 'models.model_registry')
    mm = imp('market_models', 'team.market_models', 'models.market_models')
    ts = pd.read_csv(out / 'training_set.csv', low_memory=False)
    ts['gameDate'] = pd.to_datetime(ts['gameDate'])
    with sqlite3.connect(db) as conn:
        last = conn.execute("SELECT MAX(gameDate) FROM team_game_full WHERE gameId LIKE '002%'").fetchone()[0]
    bundle = mr.load_bundle()
    reports = irf.parse_all_reports()
    rows = []
    for date in DATES + [last]:
        day = ts[ts['gameDate'] == date]
        games = pd.DataFrame({'gameId': day['gameId'].astype(str).str.zfill(10).values, 'gameDate': date,
                              'teamId_home': day['teamId_home'].astype(int).values,
                              'teamId_away': day['teamId_away'].astype(int).values})
        for c in ('odds_home_ml', 'odds_away_ml', 'odds_home_spread', 'odds_away_spread', 'odds_total'):
            games[c] = day[c].values
        season = day['seasonYear'].iloc[0]
        inj = None
        if season >= gfb.GameFeatureBuilder.INJURY_FIRST_SEASON:
            up = pd.concat([pd.DataFrame({'gameId': games['gameId'], 'teamId': games[f'teamId_{s}']})
                            for s in ('home', 'away')], ignore_index=True)
            up['gameDate'], up['seasonYear'] = date, season
            inj = irf.build_injury_features(db, [season], snapshot_name=gfb.GameFeatureBuilder.INJURY_SNAPSHOT,
                                            as_of=pd.Timestamp(date) + pd.Timedelta(hours=12), upcoming=up,
                                            reports=reports)
            inj = inj[inj['gameId'].astype(str).str.zfill(10).isin(games['gameId'])]
        frame = lf.build_live_game_frame(db, games, injury_features=inj)
        frame.to_csv(out / f'live_frame_{date}.csv', index=False)
        # the real inference path: predict_markets handles the market-aware set (residual spread, margin-stack calibrator, per-game total sigma)
        out_df = mm.predict_markets(bundle, frame, align=lambda fr, name, cols: lf.align_features(fr, cols, name))
        rows.append(pd.DataFrame({'gameDate': date, 'gameId': frame['gameId'].values, 'ml_raw': out_df['ml_prob_raw'].values, 'ml_cal': out_df['ml_prob'].values,
                                  'spread': out_df['spread_pred'].values, 'total': out_df['total_pred'].values,
                                  'spread_std': out_df['spread_std'].values, 'total_std': out_df['total_std'].values}))
    pd.concat(rows).to_csv(out / 'predictions.csv', index=False, float_format='%.17g')
    print(f'wrote {out}')


def compare(a, b):
    a, b = Path(a), Path(b)
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir()), (names, sorted(p.name for p in b.iterdir()))
    bad = 0
    for n in names:
        same = sha(a / n) == sha(b / n)
        rows = len(pd.read_csv(a / n, low_memory=False))
        print(f'{"IDENTICAL" if same else "DIFFERENT"}  {n}  ({rows} rows, sha256 {sha(a / n)[:12]})')
        bad += not same
    print('ALL IDENTICAL' if not bad else f'{bad} FILE(S) DIFFER')
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--compare', nargs=2)
    args = ap.parse_args()
    if args.compare:
        sys.exit(1 if compare(*args.compare) else 0)
    dump(args.out)
