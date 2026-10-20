"""Season helpers shared by training and live inference."""
import datetime as dt

import pandas as pd


def season_for_date(d=None):
    """NBA season string for a date (default today): Oct-Dec -> 'YYYY-YY+1', Jan-Sep -> previous year's."""
    d = pd.Timestamp(d if d is not None else dt.date.today())
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


# First regular-season game dates where the data's earliest game is not the opener (e.g. a season that is still beginning).
SEASON_OPENERS = {'2026-27': dt.date(2026, 10, 20)}


def season_options(game_dates, today=None):
    """Seasons present in `game_dates` plus the current season, newest first."""
    today = pd.Timestamp(today if today is not None else dt.date.today()).date()
    seasons = {season_for_date(d) for d in game_dates} | {season_for_date(today)}
    return sorted(seasons, reverse=True)


def season_date_range(game_dates, season, today=None):
    """
    Date-filter defaults for one season: (start, end, min_bound, max_bound), all datetime.date.
    start = the season opener (SEASON_OPENERS, else the first game in the data); end = today for the current season, else the last
    game in the data. The bounds are what a user may move the dates to: the whole span that has games (the current season up to today).
    """
    today = pd.Timestamp(today if today is not None else dt.date.today()).date()
    in_season = sorted(d for d in game_dates if season_for_date(d) == season)
    first_year = int(season[:4])
    opener = SEASON_OPENERS.get(season) or (in_season[0] if in_season else dt.date(first_year, 10, 1))
    if in_season:
        opener = min(opener, in_season[0])
    last = in_season[-1] if in_season else opener
    end = max(today, last) if season == season_for_date(today) else last
    end = max(end, opener)
    start = min(SEASON_OPENERS.get(season, opener), end)
    return start, end, opener, max(end, last)
