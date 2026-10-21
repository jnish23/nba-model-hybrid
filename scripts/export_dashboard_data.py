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
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB_PATH = ROOT / 'data' / 'nba_stats.db'
SNAPSHOT_DIR = ROOT / 'dashboard' / 'snapshot'
SNAPSHOT_PATH = SNAPSHOT_DIR / 'predictions_log.csv'
PUBLISH_DIR = ROOT / 'data' / 'dashboard_publish'
BRANCH = 'dashboard-data'
FILE_NAME = 'predictions_log.csv'
EXPLANATIONS_FILE_NAME = 'explanations.csv'      # why the model likes each game (team/explain.py), written next to the predictions


def load_predictions(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM predictions_log_v2", conn)
    return df.sort_values(['game_date', 'game_id', 'prediction_date']).reset_index(drop=True)


EXPLANATION_DAYS = 7      # the dashboard panel is for today's games; older games' rows stay in the database only


def load_explanations_table(df, db_path=DB_PATH, days=EXPLANATION_DAYS):
    """prediction_explanations rows of the games dated within `days` of the newest game in `df` (empty if none are stored yet)."""
    from team.explain import load_explanations
    expl = load_explanations(db_path)
    if expl.empty or df.empty:
        return expl
    dates = pd.to_datetime(df['game_date'])
    recent = df.loc[dates >= dates.max() - pd.Timedelta(days=days), 'game_id'].astype(str).str.zfill(10)
    return expl[expl['game_id'].isin(set(recent))].reset_index(drop=True)


def write_files(directory, df, explanations):
    """Write predictions_log.csv and, when there are explanations, explanations.csv into `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / FILE_NAME, index=False)
    if explanations is not None and not explanations.empty:
        explanations.to_csv(directory / EXPLANATIONS_FILE_NAME, index=False)


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


def publish(df, explanations=None):
    ensure_worktree()
    git('fetch', 'origin', BRANCH, cwd=PUBLISH_DIR, check=False)
    if git('rev-parse', '--verify', f'origin/{BRANCH}', cwd=PUBLISH_DIR, check=False).returncode == 0:
        git('reset', '--hard', f'origin/{BRANCH}', cwd=PUBLISH_DIR)  # the branch only ever holds exported data
    write_files(PUBLISH_DIR, df, explanations)
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
    if args.publish:
        publish(df, explanations)
    else:
        write_files(SNAPSHOT_DIR, df, explanations)
        logging.info(f"Wrote {len(df)} rows to {SNAPSHOT_PATH} and {len(explanations)} explanation rows to {SNAPSHOT_DIR / EXPLANATIONS_FILE_NAME}")


if __name__ == "__main__":
    main()
