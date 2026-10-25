"""Linked injury reports for the live job: the training parser and linker, with the upcoming games passed in so reports for games not yet in the box-score tables link to their game id."""
import pandas as pd

from core import injury_report_features as irf
from props.features import build


def upcoming_games(schedule, season):
    """(gameId, teamId, gameDate, seasonYear) rows, two per scheduled game, for the linker."""
    rows = []
    for r in schedule.itertuples():
        for team in (r.home_team_id, r.away_team_id):
            rows.append((str(r.game_id).zfill(10), int(team), r.et_game_date, season))
    return pd.DataFrame(rows, columns=['gameId', 'teamId', 'gameDate', 'seasonYear'])


def live_reports(nba_db, schedule, season, since_date):
    """Parsed reports (incremental cache shared with the training build) for games from `since_date` on, linked to teams, games and players."""
    frames = [irf.parse_all_reports(build.MAIN_REPORT_DIR, build.MAIN_CACHE)]
    if build.HOURLY_DIR.exists():
        frames.append(irf.parse_all_reports(build.HOURLY_DIR, build.HOURLY_CACHE))
    r = pd.concat(frames, ignore_index=True).drop_duplicates(['source_file', 'team', 'player', 'status', 'matchup'])
    r = r[r['game_date'] >= since_date].reset_index(drop=True)
    window = schedule[(schedule['et_game_date'] >= since_date)]
    return irf.link_reports(r, str(nba_db), upcoming_games(window, season))
