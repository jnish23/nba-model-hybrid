"""
Live in-game scores and player box scores for the dashboards (display only: nothing here records or settles a bet).

ESPN's public site API is the only provider. NBA's own live feed (cdn.nba.com liveData) refuses non-browser clients from the owner's machine and we do not work around that.
ESPN knows nothing of NBA ids, so games are matched to NBA game ids by (Eastern date, home tricode, away tricode) from a schedule the caller passes in (bets.csv / predictions_log
both carry it), and players are matched to NBA personIds by normalized name within that game, with core/live_feed_overrides.csv (ESPN athlete id -> NBA personId) for misses.
A player that matches nothing, or matches ambiguously, is dropped and counted, never guessed.

Every fetch has a timeout and every failure returns an empty frame with the normal columns plus a logged warning, so a feed outage can never break a page.
No caching here: the dashboards wrap these calls in st.cache_data so all viewers share one request. Each request is logged at INFO ('live_feed fetch ...') for request counting.
Set LIVE_FEED_ESPN_BASE to point the provider elsewhere (e.g. a bad URL to simulate an outage).
"""
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)
ET = ZoneInfo('America/New_York')
ESPN_BASE = os.environ.get('LIVE_FEED_ESPN_BASE', 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba')
TIMEOUT = 10
HEADERS = {'User-Agent': 'nba-model-hybrid dashboard (live scores)', 'Accept': 'application/json'}
OVERRIDES = Path(__file__).with_name('live_feed_overrides.csv')

SCOREBOARD_COLS = ['nba_game_id', 'espn_event_id', 'game_date', 'status', 'period', 'clock', 'home_tricode', 'away_tricode', 'home_score', 'away_score', 'updated_utc']
BOX_COLS = ['player_id', 'player_name', 'team_tricode', 'minutes', 'points', 'rebounds', 'assists', 'threes', 'on_court']

ESPN_TO_NBA = {'GS': 'GSW', 'NO': 'NOP', 'NY': 'NYK', 'SA': 'SAS', 'UTAH': 'UTA', 'WSH': 'WAS'}   # every other ESPN abbreviation equals NBA's (checked against ESPN's team list)
TEAM_TRICODE = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'}
NBA_TRICODES = set(TEAM_TRICODE.values())
SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
ESPN_STAT = {'minutes': 'minutes', 'points': 'points', 'rebounds': 'rebounds', 'assists': 'assists', 'threes': 'threePointFieldGoalsMade-threePointFieldGoalsAttempted'}


def tricode(team):
    """NBA tricode from an NBA tricode, an ESPN abbreviation or a full team name ('GS', 'GSW', 'Golden State Warriors' -> 'GSW'); None when unknown."""
    if not isinstance(team, str):
        return None
    t = team.strip()
    if t in TEAM_TRICODE:
        return TEAM_TRICODE[t]
    t = ESPN_TO_NBA.get(t.upper(), t.upper())
    return t if t in NBA_TRICODES else None


def norm_name(name):
    """Name key for matching: accents, punctuation and generational suffixes removed ("De'Andre Hunter" -> 'deandre hunter', 'Nikola Jokić' -> 'nikola jokic', 'Darius Acuff Jr.' -> 'darius acuff')."""
    if not isinstance(name, str):
        return ''
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r"['’.]", '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(w for w in s.split() if w not in SUFFIXES)


def _now_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _get(path, params=None):
    """GET one ESPN resource -> parsed JSON, or None (logged) on any failure."""
    url = f'{ESPN_BASE}/{path}'
    t0 = time.perf_counter()
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        log.info('live_feed fetch %s %s -> %s in %.0f ms', path, params or '', r.status_code, (time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        return r.json()
    except Exception as e:                       # timeouts, HTTP errors, bad JSON: the page shows "unavailable", never a traceback
        log.warning('live_feed %s %s failed: %s: %s', path, params or '', type(e).__name__, str(e)[:160])
        return None


def _status(status_type):
    """ESPN status.type -> 'scheduled' / 'live' / 'final' (halftime is live; postponed and cancelled count as scheduled)."""
    state = (status_type or {}).get('state')
    if state == 'in':
        return 'live'
    if state == 'post' and (status_type or {}).get('completed') and 'POSTPONED' not in (status_type or {}).get('name', '') and 'CANCELED' not in (status_type or {}).get('name', ''):
        return 'final'
    return 'scheduled'


def _int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def parse_scoreboard(data, updated_utc=None):
    """ESPN scoreboard JSON -> one row per game (SCOREBOARD_COLS, nba_game_id left empty). Malformed events are skipped."""
    rows = []
    for e in (data or {}).get('events') or []:
        try:
            comp = e['competitions'][0]
            side = {c['homeAway']: c for c in comp['competitors']}
            st = e.get('status') or comp.get('status') or {}
            status = _status(st.get('type'))
            rows.append({'nba_game_id': None, 'espn_event_id': str(e['id']),
                         'game_date': datetime.fromisoformat(e['date'].replace('Z', '+00:00')).astimezone(ET).strftime('%Y-%m-%d'),
                         'status': status, 'period': _int(st.get('period')) if status != 'scheduled' else None,
                         'clock': st.get('displayClock') if status == 'live' else None,
                         'home_tricode': tricode(side['home']['team']['abbreviation']), 'away_tricode': tricode(side['away']['team']['abbreviation']),
                         'home_score': _int(side['home'].get('score')) if status != 'scheduled' else None,
                         'away_score': _int(side['away'].get('score')) if status != 'scheduled' else None,
                         'updated_utc': updated_utc or _now_utc()})
        except (KeyError, IndexError, TypeError, ValueError) as err:
            log.warning('live_feed: skipped a scoreboard event %s: %s', e.get('id') if isinstance(e, dict) else '?', err)
    return pd.DataFrame(rows, columns=SCOREBOARD_COLS)


def attach_nba_ids(board, schedule):
    """Fill nba_game_id by (game_date, home_tricode, away_tricode) from `schedule` (columns nba_game_id or game_id, game_date, home/away as tricodes or full names). Unmatched stay empty."""
    if board.empty or schedule is None or len(schedule) == 0:
        return board
    s = schedule.rename(columns={'game_id': 'nba_game_id'})
    home = s['home_tricode'] if 'home_tricode' in s else s['home_team']
    away = s['away_tricode'] if 'away_tricode' in s else s['away_team']
    key = pd.DataFrame({'game_date': s['game_date'].astype(str).str[:10], 'home_tricode': home.map(tricode), 'away_tricode': away.map(tricode),
                        'nba_game_id': s['nba_game_id'].astype(str)}).drop_duplicates(['game_date', 'home_tricode', 'away_tricode'])
    out = board.drop(columns='nba_game_id').merge(key, on=['game_date', 'home_tricode', 'away_tricode'], how='left')
    return out[SCOREBOARD_COLS]


def scoreboard(game_date, schedule=None):
    """Games of one Eastern date ('YYYY-MM-DD') -> SCOREBOARD_COLS, with NBA game ids from `schedule` when given. Empty frame on any failure."""
    data = _get('scoreboard', {'dates': str(game_date).replace('-', '')[:8]})
    board = parse_scoreboard(data)
    board = board[board['game_date'] == str(game_date)[:10]].reset_index(drop=True)      # ESPN's date window can spill into the next ET day
    return attach_nba_ids(board, schedule)


def load_overrides(path=OVERRIDES):
    """ESPN athlete id -> NBA personId from the manual override CSV (espn_athlete_id, player_id, note); empty when the file is missing or unreadable."""
    try:
        o = pd.read_csv(path, dtype=str, comment='#').dropna(subset=['espn_athlete_id', 'player_id'])
        return dict(zip(o['espn_athlete_id'].str.strip(), o['player_id'].str.strip().astype(int)))
    except Exception:
        return {}


def parse_summary(data):
    """ESPN summary JSON -> one row per ESPN player with a stat line (espn_id, espn_name, team_tricode, minutes, points, rebounds, assists, threes, on_court). DNPs are left out."""
    rows = []
    for team in ((data or {}).get('boxscore') or {}).get('players') or []:
        tri = tricode((team.get('team') or {}).get('abbreviation'))
        for block in team.get('statistics') or []:
            idx = {k: i for i, k in enumerate(block.get('keys') or [])}
            if not all(v in idx for v in ESPN_STAT.values()):
                continue
            for a in block.get('athletes') or []:
                stats = a.get('stats') or []
                if a.get('didNotPlay') or len(stats) < len(idx):
                    continue
                v = {k: stats[idx[key]] for k, key in ESPN_STAT.items()}
                rows.append({'espn_id': str((a.get('athlete') or {}).get('id')), 'espn_name': (a.get('athlete') or {}).get('displayName'), 'team_tricode': tri,
                             'minutes': float(_int(v['minutes']) or 0), 'points': _int(v['points']) or 0, 'rebounds': _int(v['rebounds']) or 0,
                             'assists': _int(v['assists']) or 0, 'threes': _int(str(v['threes']).split('-')[0]) or 0, 'on_court': bool(a.get('active'))})
    return pd.DataFrame(rows, columns=['espn_id', 'espn_name', 'team_tricode', 'minutes', 'points', 'rebounds', 'assists', 'threes', 'on_court'])


def match_players(espn, roster, overrides=None):
    """
    ESPN rows -> BOX_COLS with NBA personIds. `roster` holds the NBA players to look for (player_id, player_name); overrides (ESPN id -> personId) win over names.
    Returns (frame, unmatched) where unmatched lists roster names not found once, ambiguous ones included; ESPN rows outside the roster are dropped.
    """
    overrides = load_overrides() if overrides is None else overrides
    r = roster[['player_id', 'player_name']].drop_duplicates('player_id').copy()
    r['player_id'] = r['player_id'].astype(int)
    r['key'] = r['player_name'].map(norm_name)
    by_key = r.groupby('key')['player_id'].apply(list).to_dict()
    names = dict(zip(r['player_id'], r['player_name']))
    e = espn.copy()
    e['key'] = e['espn_name'].map(norm_name)
    espn_per_key = e['key'].value_counts()
    ids = []
    for _, row in e.iterrows():
        pid = overrides.get(row['espn_id'])
        if pid is None and espn_per_key.get(row['key'], 0) == 1 and len(by_key.get(row['key'], [])) == 1:
            pid = by_key[row['key']][0]
        ids.append(pid if pid in names else None)
    e['player_id'] = ids
    e = e.dropna(subset=['player_id']).drop_duplicates('player_id', keep=False)
    e['player_id'] = e['player_id'].astype(int)
    e['player_name'] = e['player_id'].map(names)
    unmatched = sorted(set(names.values()) - set(e['player_name']))
    return e[BOX_COLS].reset_index(drop=True), unmatched


def box(nba_game_id, board, roster, overrides=None):
    """
    Player lines of one game -> BOX_COLS for the players in `roster` (player_id, player_name). `board` is a scoreboard() frame holding this game (it supplies the ESPN event id).
    Empty frame when the game is unknown, the fetch fails or nobody matches; misses are logged with their names.
    """
    g = board[board['nba_game_id'].astype(str) == str(nba_game_id)] if len(board) else board
    if g.empty or roster is None or len(roster) == 0:
        return pd.DataFrame(columns=BOX_COLS)
    data = _get('summary', {'event': g['espn_event_id'].iloc[0]})
    if data is None:
        return pd.DataFrame(columns=BOX_COLS)
    lines, unmatched = match_players(parse_summary(data), roster, overrides)
    if g['status'].iloc[0] != 'live':
        lines['on_court'] = None                 # ESPN's flag only means "on the floor now" during play
    if unmatched:
        log.warning('live_feed %s: %d of %d players not matched (no line yet, or add them to %s): %s', nba_game_id, len(unmatched),
                    roster['player_id'].nunique(), OVERRIDES.name, ', '.join(unmatched))
    return lines
