"""
After a daily inference run: find today's games that still lack betting lines, schedule ONE re-run an hour later (a one-shot
Windows task) while that can still help, and write data/rerun_status.json for the dashboard.

  python scripts/schedule_odds_rerun.py              # after the morning run (resets today's re-run count)
  python scripts/schedule_odds_rerun.py --after-rerun # after a re-run (counts it)

The re-run (scripts/run_odds_rerun.bat) predicts only the games that have not started and still lack a line, then calls this
script again, so the chain stops by itself when every line is in, the first unpriced game is about to tip off, or MAX_RERUNS is reached.
"""
import argparse
import datetime as dt
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.odds_rerun import (MAX_RERUNS, build_status, next_rerun_time, read_status, unpriced_games, write_status)

DB_PATH = ROOT / 'data' / 'nba_stats.db'
STATUS_PATH = ROOT / 'data' / 'rerun_status.json'
TASK_NAME = 'NBA Odds Rerun'
RERUN_BAT = ROOT / 'scripts' / 'run_odds_rerun.bat'


def tipoffs_of(game_ids, game_date):
    """{game_id: tip-off (UTC)} of the games that have not started, from the NBA scoreboard."""
    from nba_api.stats.endpoints import scoreboardv3
    games = scoreboardv3.ScoreboardV3(game_date=game_date).get_data_frames()[1]
    games = games[games['gameId'].astype(str).isin(set(game_ids)) & (games['gameStatus'] == 1)]
    return {str(r.gameId): dt.datetime.fromisoformat(r.gameTimeUTC.replace('Z', '+00:00')) for r in games.itertuples()}


def schedule_task(when):
    """Create (or replace) the one-shot task at the local time `when`."""
    local = when.astimezone()
    subprocess.run(['schtasks', '/create', '/tn', TASK_NAME, '/tr', f'"{RERUN_BAT}"', '/sc', 'once',
                    '/st', local.strftime('%H:%M'), '/sd', local.strftime('%m/%d/%Y'), '/f'], check=True, capture_output=True, text=True)


def remove_task():
    subprocess.run(['schtasks', '/delete', '/tn', TASK_NAME, '/f'], capture_output=True, text=True)   # fine if it is not there


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--after-rerun', action='store_true', help='this call follows a re-run: count it')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    from core.scrape_injury_reports import now_et
    now = dt.datetime.now(dt.timezone.utc)
    game_date = now_et().strftime('%Y-%m-%d')

    previous = read_status(STATUS_PATH) or {}
    attempts = previous.get('attempts', 0) if previous.get('game_date') == game_date else 0
    attempts = attempts + 1 if args.after_rerun else 0

    missing = unpriced_games(DB_PATH, game_date)
    next_run, reason = None, 'every game has its lines'
    if len(missing):
        try:
            tips = tipoffs_of(missing['game_id'], game_date)
            missing = missing[missing['game_id'].map(lambda g: tips.get(g, now) > now)].reset_index(drop=True)   # started games can no longer be helped
        except Exception as e:                      # without tip-off times the safe choice is not to schedule
            tips, reason = {}, f'tip-off times unavailable ({e})'
            logging.error(reason)
        next_run = next_rerun_time(now, list(tips.values()), attempts)
        if next_run is not None:
            reason = 'waiting for lines'
        elif tips and attempts >= MAX_RERUNS:
            reason = f'gave up after {MAX_RERUNS} re-runs'
        elif tips:
            reason = 'the next unpriced game tips off before another re-run'
        elif not reason.startswith('tip-off'):
            reason = 'the unpriced games have started'
    write_status(STATUS_PATH, build_status(game_date, missing, next_run, attempts, now, reason))
    if next_run is not None:
        schedule_task(next_run)
        logging.info(f"{len(missing)} game(s) lack lines; re-run {attempts + 1} of {MAX_RERUNS} scheduled for {next_run.astimezone():%H:%M}")
    else:
        remove_task()
        logging.info(f"{len(missing)} game(s) lack lines; no re-run scheduled ({reason})")


if __name__ == '__main__':
    main()
