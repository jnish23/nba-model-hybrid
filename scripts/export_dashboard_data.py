"""
Export the dashboard's data (the predictions_log_v2 table) so the dashboard works without the
1 GB database: on Streamlit Community Cloud, or in a fresh clone of the repo.

  python scripts/export_dashboard_data.py             # write the bundled snapshot in the repo
  python scripts/export_dashboard_data.py --publish   # update the live data branch and push it

--publish writes the file to the `dashboard-data` branch (which holds only the exported files) through a
separate git worktree in data/dashboard_publish/ (gitignored), commits if it changed, and pushes.
It never touches the branch you have checked out. The deployed app reads the file from
https://raw.githubusercontent.com/<repo>/dashboard-data/predictions_log.csv (and explanations.csv next to it), so pushes to this
branch refresh the dashboard without redeploying it. Run daily by scripts/run_daily_inference.bat.
"""
import argparse
import json
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.odds_rerun import STATUS_FILE_NAME, read_status, write_status
DB_PATH = ROOT / 'data' / 'nba_stats.db'
SNAPSHOT_DIR = ROOT / 'dashboard' / 'snapshot'
SNAPSHOT_PATH = SNAPSHOT_DIR / 'predictions_log.csv'
PUBLISH_DIR = ROOT / 'data' / 'dashboard_publish'
RERUN_STATUS_PATH = ROOT / 'data' / 'rerun_status.json'     # written by scripts/schedule_odds_rerun.py
BRANCH = 'dashboard-data'
FILE_NAME = 'predictions_log.csv'
EXPLANATIONS_FILE_NAME = 'explanations.csv'      # why the model likes each game (team/explain.py), written next to the predictions


def load_predictions(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM predictions_log_v2", conn)
    return df.sort_values(['game_date', 'game_id', 'prediction_date']).reset_index(drop=True)


HISTORY_DIR = 'explanations_history'             # one gzip per month of games: past months never change, so git history stays small
HISTORY_MANIFEST = 'explanations_manifest.json'
HISTORY_TOP_FEATURES = 15                         # feature rows kept per game and market in the history files (the breakdown pieces are kept whole)
EXPLANATION_DAYS = 7      # the dashboard panel is for today's games; older games' rows stay in the database only


def load_explanations_table(df, db_path=DB_PATH, days=EXPLANATION_DAYS):
    """prediction_explanations rows of the games dated within `days` of the newest game in `df` (empty if none are stored yet)."""
    from team.explain import load_explanations
    expl = load_explanations(db_path)
    if days is None or expl.empty or df.empty:
        return expl
    dates = pd.to_datetime(df['game_date'])
    recent = df.loc[dates >= dates.max() - pd.Timedelta(days=days), 'game_id'].astype(str).str.zfill(10)
    return expl[expl['game_id'].isin(set(recent))].reset_index(drop=True)


def history_shards(df, expl, top=HISTORY_TOP_FEATURES):
    """{'YYYY-MM.csv.gz': rows}: every explained game's pieces and flags plus its `top` biggest feature rows, grouped by the month of the game."""
    if expl is None or expl.empty or df.empty:
        return {}
    feat = expl[expl['kind'] == 'feature']
    rank = feat['contribution'].abs().groupby([feat['game_id'], feat['market']]).rank(method='first', ascending=False)
    keep = pd.concat([expl[expl['kind'] != 'feature'], feat[rank <= top]]).sort_values(['game_id', 'market', 'kind', 'feature'], kind='stable')
    months = pd.to_datetime(df.assign(game_id=df['game_id'].astype(str).str.zfill(10)).drop_duplicates('game_id').set_index('game_id')['game_date']).dt.strftime('%Y-%m')
    keep = keep.assign(month=keep['game_id'].map(months)).dropna(subset=['month'])
    return {f'{m}.csv.gz': part.drop(columns='month').reset_index(drop=True) for m, part in keep.groupby('month')}


def write_files(directory, df, explanations, history=None, status=None):
    """
    Write predictions_log.csv and, when there are explanations, explanations.csv (the recent games, in full) into `directory`.
    `history` (all explanation rows) also writes explanations_history/<month>.csv.gz and a manifest listing them; the gzip header carries no
    timestamp, so a month that did not change produces identical bytes and no git change.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / FILE_NAME, index=False)
    if status is not None:                       # which of today's games still wait for lines, and when the next re-run is (core/odds_rerun.py)
        write_status(directory / STATUS_FILE_NAME, status)
    else:
        (directory / STATUS_FILE_NAME).unlink(missing_ok=True)
    if explanations is not None and not explanations.empty:
        explanations.to_csv(directory / EXPLANATIONS_FILE_NAME, index=False)
    shards = history_shards(df, history)
    if shards:
        (directory / HISTORY_DIR).mkdir(exist_ok=True)
        for name, part in shards.items():
            part.to_csv(directory / HISTORY_DIR / name, index=False, compression={'method': 'gzip', 'mtime': 0})
        (directory / HISTORY_MANIFEST).write_text(json.dumps({'shards': [{'file': n, 'rows': len(p)} for n, p in sorted(shards.items())]}, indent=1))


def git(*args, cwd=ROOT, check=True):
    r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def ensure_worktree():
    """Create (once) the worktree checked out on the data branch."""
    if (PUBLISH_DIR / '.git').exists():
        return
    git('worktree', 'prune')
    git('fetch', 'origin', check=False)
    remote = git('ls-remote', '--heads', 'origin', BRANCH).stdout.strip()
    if remote:
        git('worktree', 'add', '-B', BRANCH, str(PUBLISH_DIR), f'origin/{BRANCH}')
    else:
        git('worktree', 'add', '--orphan', '-b', BRANCH, str(PUBLISH_DIR))
        (PUBLISH_DIR / 'README.md').write_text(
            "Data for the NBA model dashboard, published daily by "
            "`scripts/export_dashboard_data.py --publish`. Do not edit by hand.\n", encoding='utf-8')
    logging.info(f"Worktree for {BRANCH} at {PUBLISH_DIR}")


def publish(df, explanations=None, history=None, status=None):
    ensure_worktree()
    git('fetch', 'origin', BRANCH, cwd=PUBLISH_DIR, check=False)
    if git('rev-parse', '--verify', f'origin/{BRANCH}', cwd=PUBLISH_DIR, check=False).returncode == 0:
        git('reset', '--hard', f'origin/{BRANCH}', cwd=PUBLISH_DIR)  # the branch only ever holds exported data
    write_files(PUBLISH_DIR, df, explanations, history, status)
    git('add', '-A', cwd=PUBLISH_DIR)
    if git('diff', '--cached', '--quiet', cwd=PUBLISH_DIR, check=False).returncode == 0:
        logging.info("Dashboard data unchanged; nothing to publish")
        return False
    last = df['game_date'].max()
    git('commit', '-m', f'Update dashboard data through {last} ({len(df)} rows)', cwd=PUBLISH_DIR)
    git('push', '-u', 'origin', BRANCH, cwd=PUBLISH_DIR)
    logging.info(f"Published {len(df)} rows through {last} to {BRANCH}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--publish', action='store_true', help='Update and push the dashboard-data branch')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    df = load_predictions()
    explanations = load_explanations_table(df)
    history = load_explanations_table(df, days=None)
    status = read_status(RERUN_STATUS_PATH)
    if args.publish:
        publish(df, explanations, history, status)
    else:
        write_files(SNAPSHOT_DIR, df, explanations)
        logging.info(f"Wrote {len(df)} rows to {SNAPSHOT_PATH} and {len(explanations)} explanation rows to {SNAPSHOT_DIR / EXPLANATIONS_FILE_NAME}")


if __name__ == "__main__":
    main()
