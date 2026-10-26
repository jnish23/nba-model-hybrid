"""
Provisional results for bets whose game has just finished, before the daily box-score sync settles them officially. A game counts as finished when the stats API returns its box score with
both teams at 240 minutes or more (the API returns nothing while a game is in progress). Display only: the recorded bet and its official settlement (settle_bets) are untouched.
Grading follows the 'bet stands once the player takes the court' treatment; a player with no minutes is 'void'.
"""
import logging

import numpy as np
import pandas as pd

TEAM_MINUTES_FINAL = 240.0
COLS = ['bet_id', 'game_final', 'actual', 'minutes', 'provisional', 'checked_utc']
STAT_VALUE = {'points': lambda s: s['points'], 'rebounds': lambda s: s['reboundsTotal'], 'assists': lambda s: s['assists'], 'threes': lambda s: s['threePointersMade'],
              'ra': lambda s: s['reboundsTotal'] + s['assists'], 'pra': lambda s: s['points'] + s['reboundsTotal'] + s['assists']}
MIN_MINUTES_AFTER_TIP = 100                      # no game is over earlier; saves requests during play


def minutes_float(text):
    """'29:09' -> 29.15; blank or missing -> 0."""
    if not isinstance(text, str) or ':' not in text:
        return 0.0
    m, s = text.split(':')[:2]
    return float(m) + float(s) / 60.0


def fetch_box(game_id, timeout=30):
    """Raw box score dict of the stats API, or None when it has none yet (game in progress / not yet published)."""
    from nba_api.stats.library.http import NBAStatsHTTP
    params = {'GameID': game_id, 'StartPeriod': 0, 'EndPeriod': 0, 'StartRange': 0, 'EndRange': 0, 'RangeType': 0}
    try:
        return NBAStatsHTTP().send_api_request(endpoint='boxscoretraditionalv3', parameters=params, timeout=timeout).get_dict().get('boxScoreTraditional')
    except Exception as e:                       # network or API trouble must never break an export
        logging.warning(f'box score {game_id}: {type(e).__name__}: {str(e)[:120]}')
        return None


def player_lines(box):
    """(is_final, DataFrame player_id, minutes, points, reboundsTotal, assists, threePointersMade) from a box score dict; (False, empty) when it has no players yet."""
    rows, totals = [], []
    for side in ('homeTeam', 'awayTeam'):
        team = (box or {}).get(side) or {}
        totals.append(minutes_float((team.get('statistics') or {}).get('minutes')))
        for p in team.get('players') or []:
            s = p.get('statistics') or {}
            rows.append({'player_id': int(p['personId']), 'minutes': minutes_float(s.get('minutes')), 'points': s.get('points') or 0, 'reboundsTotal': s.get('reboundsTotal') or 0,
                         'assists': s.get('assists') or 0, 'threePointersMade': s.get('threePointersMade') or 0})
    final = bool(rows) and min(totals) >= TEAM_MINUTES_FINAL - 1e-6
    return final, pd.DataFrame(rows, columns=['player_id', 'minutes', 'points', 'reboundsTotal', 'assists', 'threePointersMade'])


def grade(side, line, actual, minutes):
    """'win' / 'loss' / 'push' / 'void' for one bet on a finished game."""
    if not np.isfinite(minutes) or minutes <= 0:
        return 'void'
    if actual == line:
        return 'push'
    return 'win' if (actual > line) == (side == 'over') else 'loss'


def provisional_results(pending, now, fetch=fetch_box):
    """
    pending: unsettled recorded bets (bet_id, game_id, player_id, stat, side, line, tip_ref_utc). Returns a frame with COLS for the bets whose game box score is final.
    `fetch(game_id)` returns the raw box dict; injectable for tests.
    """
    if pending.empty:
        return pd.DataFrame(columns=COLS)
    started = pd.to_datetime(pending['tip_ref_utc'], utc=True) + pd.Timedelta(minutes=MIN_MINUTES_AFTER_TIP) <= now
    rows = []
    stamp = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    for gid, g in pending[started].groupby('game_id'):
        final, lines = player_lines(fetch(gid))
        if not final:
            continue
        by_player = lines.set_index('player_id')
        for b in g.itertuples(index=False):
            if int(b.player_id) in by_player.index and b.stat in STAT_VALUE:
                s = by_player.loc[int(b.player_id)]
                actual, minutes = float(STAT_VALUE[b.stat](s)), float(s['minutes'])
            else:
                actual, minutes = np.nan, 0.0                        # not on the box score at all: did not play
            rows.append({'bet_id': b.bet_id, 'game_final': 1, 'actual': actual, 'minutes': minutes, 'provisional': grade(b.side, float(b.line), actual, minutes), 'checked_utc': stamp})
    return pd.DataFrame(rows, columns=COLS)
