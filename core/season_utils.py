"""Season helpers shared by training and live inference."""
import datetime as dt

import pandas as pd


def season_for_date(d=None):
    """NBA season string for a date (default today): Oct-Dec -> 'YYYY-YY+1', Jan-Sep -> previous year's."""
    d = pd.Timestamp(d if d is not None else dt.date.today())
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"
