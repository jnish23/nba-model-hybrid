import sqlite3

import pandas as pd

import snapshot_odds
from models import odds_clean


def test_closing_from_snapshots_pivots_and_fills_missing_books():
    """Regression: the pivot used DataFrame.setdefault (a dict method), so any non-empty odds_snapshots table crashed load_clean_odds."""
    conn = sqlite3.connect(':memory:')
    conn.execute(snapshot_odds.SCHEMA)
    rows = [
        ('2026-10-20 17:00:00', '2026-10-20', '2026-10-20T23:00:00Z', 'Scheduled', 0, 'Detroit Pistons', 'San Antonio Spurs', 'DET', 'SAS', 'current', 'fanduel',
         -3.5, -110, 3.5, -110, -160, 140, 224.5, -108, -112),
        ('2026-10-20 18:30:00', '2026-10-20', '2026-10-20T23:00:00Z', 'Scheduled', 0, 'Detroit Pistons', 'San Antonio Spurs', 'DET', 'SAS', 'current', 'fanduel',
         -4.0, -110, 4.0, -110, -170, 145, 225.0, -110, -110),
        ('2026-10-20 19:00:00', '2026-10-20', '2026-10-20T23:00:00Z', 'In Progress', 1, 'Detroit Pistons', 'San Antonio Spurs', 'DET', 'SAS', 'current', 'fanduel',
         -9.5, -110, 9.5, -110, -900, 600, 231.5, -110, -110),
    ]
    conn.executemany('INSERT INTO odds_snapshots VALUES (' + ','.join('?' * 20) + ')', rows)
    wide = odds_clean._closing_from_snapshots(conn)
    assert len(wide) == 1
    r = wide.iloc[0]
    assert r['home_spread_fanduel'] == -4.0 and r['total_fanduel'] == 225.0                 # the last pre-game line; the in-game one is excluded
    for b in odds_clean.BOOKS:                                                                  # every book column exists, NaN where the book was not scraped
        assert f'total_{b}' in wide.columns
    assert pd.isna(r['total_betmgm'])
    assert r['gameDate'] == '2026-10-20'
