"""
One-off feed check for live in-game tracking: can THIS machine reach NBA's live feed (cdn.nba.com liveData) and ESPN's public API? From the owner's home machine the NBA CDN answered 403,
so before building a live panel we need the same answer from Streamlit Community Cloud, where the dashboards run. Self-contained (requests + pandas + streamlit only) so it can be deployed
on its own; scripts/probe_live_feeds.py runs the same checks from a terminal. Scores and box scores only, no odds.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

NBA_SCOREBOARD = 'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json'
NBA_BOX = 'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json'
ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
ESPN_SUMMARY = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}'
FALLBACK_NBA_GAME = '0022600001'                 # first regular-season game of 2026-27, used when the scoreboard gives no game id
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36',
           'Accept': 'application/json, text/plain, */*', 'Accept-Language': 'en-US,en;q=0.9', 'Referer': 'https://www.nba.com/', 'Origin': 'https://www.nba.com'}
TIMEOUT = 10


def fetch(name, url, save_dir=None):
    """One GET -> (result row, parsed JSON or None). Never raises: a failure is a row with the error text."""
    t0 = time.perf_counter()
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        ms = round((time.perf_counter() - t0) * 1000)
        body = None
        try:
            body = r.json() if r.ok else None
        except ValueError:
            pass
        if save_dir and body is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            (Path(save_dir) / f'{name}.json').write_text(json.dumps(body), encoding='utf-8')
        return {'check': name, 'url': url, 'status': r.status_code, 'bytes': len(r.content), 'ms': ms, 'json': body is not None, 'note': '' if r.ok else r.text[:120].replace('\n', ' ')}, body
    except Exception as e:                       # timeouts, DNS, TLS: reported, not raised
        return {'check': name, 'url': url, 'status': None, 'bytes': 0, 'ms': round((time.perf_counter() - t0) * 1000), 'json': False, 'note': f'{type(e).__name__}: {str(e)[:120]}'}, None


def _nba_game_id(board):
    """A game id from the NBA scoreboard, preferring one in progress (gameStatus 2), then final (3)."""
    games = ((board or {}).get('scoreboard') or {}).get('games') or []
    for status in (2, 3, 1):
        for g in games:
            if g.get('gameStatus') == status:
                return g['gameId']
    return None


def _espn_event_id(board):
    """An ESPN event id from its scoreboard, preferring one in progress, then final."""
    events = (board or {}).get('events') or []
    for state in ('in', 'post', 'pre'):
        for e in events:
            if (((e.get('status') or {}).get('type') or {}).get('state')) == state:
                return e['id']
    return None


def run_checks(save_dir=None):
    """The four checks (NBA scoreboard, NBA box, ESPN scoreboard, ESPN summary) as a DataFrame, plus a short context column (which game was probed, how many games were listed)."""
    rows = []
    row, board = fetch('nba_scoreboard', NBA_SCOREBOARD, save_dir)
    games = ((board or {}).get('scoreboard') or {}).get('games') or []
    rows.append({**row, 'context': f'{len(games)} games' if board else ''})
    gid = _nba_game_id(board) or FALLBACK_NBA_GAME
    row, box = fetch('nba_box', NBA_BOX.format(game_id=gid), save_dir)
    rows.append({**row, 'context': f'game {gid}, status {((box or {}).get("game") or {}).get("gameStatusText", "?")}' if box else f'game {gid}'})
    row, eboard = fetch('espn_scoreboard', ESPN_SCOREBOARD, save_dir)
    rows.append({**row, 'context': f'{len((eboard or {}).get("events") or [])} events' if eboard else ''})
    eid = _espn_event_id(eboard)
    if eid:
        row, summ = fetch('espn_summary', ESPN_SUMMARY.format(event_id=eid), save_dir)
        n = sum(len(a.get('athletes') or []) for t in ((summ or {}).get('boxscore') or {}).get('players') or [] for a in t.get('statistics') or [])
        rows.append({**row, 'context': f'event {eid}, {n} player rows'})
    else:
        rows.append({'check': 'espn_summary', 'url': ESPN_SUMMARY.format(event_id='?'), 'status': None, 'bytes': 0, 'ms': 0, 'json': False, 'note': 'no ESPN event id to probe', 'context': ''})
    out = pd.DataFrame(rows)
    out.insert(0, 'checked_utc', pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%SZ'))
    return out


def page():
    import streamlit as st
    st.set_page_config(page_title='Live feed probe', layout='wide')
    st.title('Live feed probe')
    st.caption('Fetches the NBA live feed and the ESPN API once from wherever this app runs. Scores and box scores only. Press the button to run again.')
    st.button('Run again')
    t = run_checks()
    st.dataframe(t, hide_index=True, use_container_width=True)
    st.code(t.drop(columns=['url']).to_csv(index=False), language='text')
    st.caption('Copy the block above and paste it back into the chat.')


if __name__ == '__main__':                       # `streamlit run` executes the file as __main__; importing it (scripts/probe_live_feeds.py) only gets the checks
    page()
