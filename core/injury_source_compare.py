"""
Official NBA injury-report PDFs (the source of every inj_* feature) vs ESPN's public injury list (the old live source).

ESPN is kept here ONLY as a comparator: it never feeds a feature. `log_side_by_side` stores both sources for the games of a day
in table `injury_source_compare`; `historical_overlap` measures how often they disagree on the 2026-01..04 days for which
`nba_injury_logs` holds ESPN scrapes.

Harmonised categories: OUT (PDF Out / ESPN Out), UNCERTAIN (PDF Doubtful, Questionable / ESPN Day-To-Day),
PLAYING (PDF Probable, Available / not listed).
"""
import logging
import sqlite3
import time

import numpy as np
import pandas as pd

from core import injury_report_features as irf

ESPN_URL = "https://www.espn.com/nba/injuries"
SCHEMA = """CREATE TABLE IF NOT EXISTS injury_source_compare (
    logged_at TEXT, game_date TEXT, player_key TEXT, team TEXT, pdf_status TEXT, espn_status TEXT,
    pdf_category TEXT, espn_category TEXT, agree INTEGER)"""


def pdf_category(status):
    if not isinstance(status, str):
        return 'PLAYING'
    return {'Out': 'OUT', 'Doubtful': 'UNCERTAIN', 'Questionable': 'UNCERTAIN'}.get(status, 'PLAYING')


def espn_category(status):
    if not isinstance(status, str):
        return 'PLAYING'
    s = status.strip().lower()
    return 'OUT' if s == 'out' else ('UNCERTAIN' if s in ('day-to-day', 'questionable', 'doubtful') else 'PLAYING')


def fetch_espn(max_retries=3):
    """{player name: status} from ESPN's injuries page (comparator only). Empty dict on failure."""
    import requests
    from bs4 import BeautifulSoup
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    for attempt in range(max_retries):
        try:
            r = requests.get(ESPN_URL, headers=headers, timeout=10)
            if r.status_code != 200:
                time.sleep(2)
                continue
            out = {}
            for table in BeautifulSoup(r.content, 'html.parser').find_all('table'):
                for row in table.find_all('tr'):
                    cols = row.find_all('td')
                    if len(cols) >= 4 and cols[0].get_text().strip():
                        out[cols[0].get_text().strip()] = cols[3].get_text().strip()
            if out:
                return out
        except requests.RequestException as e:
            logging.warning(f"ESPN attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return {}


def latest_pdf_rows(reports, game_date, as_of):
    """Rows of the newest report published at/before `as_of` that covers `game_date`."""
    day = reports[(pd.to_datetime(reports['game_date']).dt.normalize() == pd.Timestamp(game_date).normalize())
                  & (pd.to_datetime(reports['report_ts']) <= pd.Timestamp(as_of))]
    if day.empty:
        return day
    return day[day['report_ts'] == day['report_ts'].max()].dropna(subset=['player'])


def compare_frames(pdf_rows, espn_map, team_of=None):
    """Side-by-side frame, one row per player listed in either source. team_of(key) -> team label (optional)."""
    p = pd.DataFrame({'player_key': pdf_rows['player'].map(irf.report_name_key), 'team': pdf_rows['team'], 'pdf_status': pdf_rows['status']}) \
        if len(pdf_rows) else pd.DataFrame(columns=['player_key', 'team', 'pdf_status'])
    p = p.drop_duplicates('player_key')
    e = pd.DataFrame({'player_key': [irf.db_name_key(n) for n in espn_map], 'espn_status': list(espn_map.values())}).drop_duplicates('player_key')
    m = p.merge(e, on='player_key', how='outer')
    if team_of is not None:
        m['team'] = m['team'].fillna(m['player_key'].map(team_of))
    m['pdf_category'] = m['pdf_status'].map(pdf_category)
    m['espn_category'] = m['espn_status'].map(espn_category)
    m['agree'] = (m['pdf_category'] == m['espn_category']).astype(int)
    return m


def log_side_by_side(db_path, game_date, reports=None, as_of=None, espn_map=None):
    """Fetch ESPN, take the newest official report, and store both. Returns the comparison frame. Never raises."""
    try:
        reports = irf.parse_all_reports() if reports is None else reports
        as_of = pd.Timestamp.now() if as_of is None else as_of
        cmp_ = compare_frames(latest_pdf_rows(reports, game_date, as_of), fetch_espn() if espn_map is None else espn_map)
        if len(cmp_):
            cmp_ = cmp_.assign(logged_at=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), game_date=str(game_date)[:10])
            with sqlite3.connect(db_path) as conn:
                conn.execute(SCHEMA)
                cmp_[['logged_at', 'game_date', 'player_key', 'team', 'pdf_status', 'espn_status', 'pdf_category', 'espn_category', 'agree']] \
                    .to_sql('injury_source_compare', conn, if_exists='append', index=False)
            logging.info(f"Injury sources for {game_date}: {len(cmp_)} listed players, disagreement {1 - cmp_['agree'].mean():.1%}")
        return cmp_
    except Exception as e:                                   # logging must never break inference
        logging.warning(f"injury source comparison failed: {e}")
        return pd.DataFrame()


def historical_overlap(db_path, reports=None):
    """Disagreement between the official report and the stored ESPN scrapes on the days where both exist."""
    reports = irf.parse_all_reports() if reports is None else reports
    with sqlite3.connect(db_path) as conn:
        espn = pd.read_sql("SELECT scrape_date, player_name, status FROM nba_injury_logs WHERE source = 'ESPN'", conn)
        team_lookup = irf._team_lookup(conn)
        last_team = pd.read_sql("SELECT personId, playerName, teamId, MAX(gameDate) AS d FROM player_basic_stats "
                                "WHERE seasonYear >= '2025-26' GROUP BY personId", conn)
    espn['ts'] = pd.to_datetime(espn['scrape_date'])
    key_team = dict(zip(last_team['playerName'].map(irf.db_name_key), last_team['teamId']))
    rows = []
    for ts, g in espn.groupby('ts'):
        pdf = latest_pdf_rows(reports, ts.normalize(), ts)
        if pdf.empty:
            continue
        pdf_teams = {team_lookup.get(irf._norm(t)) for t in pdf['team'].dropna().unique()} - {None}
        emap = dict(zip(g['player_name'], g['status']))
        c = compare_frames(pdf, emap)
        c['teamId'] = c['team'].map(lambda t: team_lookup.get(irf._norm(t)) if isinstance(t, str) else np.nan)
        c['teamId'] = c['teamId'].fillna(c['player_key'].map(key_team))
        c = c[c['teamId'].isin(pdf_teams)]                       # only teams that play that day (ESPN lists everyone)
        c['scrape'] = ts
        rows.append(c)
    allc = pd.concat(rows, ignore_index=True)
    summary = {
        'scrapes compared': allc['scrape'].nunique(), 'listed player-days': len(allc),
        'agreement': float(allc['agree'].mean()),
        'ESPN OUT but PDF not OUT': float(((allc['espn_category'] == 'OUT') & (allc['pdf_category'] != 'OUT')).sum() / max((allc['espn_category'] == 'OUT').sum(), 1)),
        'PDF OUT but ESPN not OUT': float(((allc['pdf_category'] == 'OUT') & (allc['espn_category'] != 'OUT')).sum() / max((allc['pdf_category'] == 'OUT').sum(), 1)),
    }
    matrix = pd.crosstab(allc['pdf_category'], allc['espn_category'])
    return summary, matrix, allc
