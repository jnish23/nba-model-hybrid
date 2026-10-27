"""
Live scores next to the team model's picks on Today's Games (display only: the prediction log and its grading are untouched).
Pure pandas, no Streamlit, so the app stays thin and this is unit-tested (tests/team/test_dashboard_live_scores.py). The scoreboard comes from core/live_feed.py.
The pick sides follow the Today's Games table: moneyline only when get_ml_bet passes the 5% edge, spread and total always lean to the side of the model's number.
"""
import numpy as np
import pandas as pd

from team.betting import get_ml_bet

ML_THRESHOLD = 0.05
LIVE_COLS = ['Game', 'Score', 'Clock', 'ML', 'Spread', 'Total']
RESULT_COLS = {'ML': 'ml_result', 'Spread': 'spread_result', 'Total': 'total_result'}   # 'won' / 'lost' / '' per pick column, for the app's cell colours


def pick_sides(games):
    """Today's Games rows -> the same rows with ml_side ('home'/'away'/None), spread_side ('home'/'away'/None) and total_side ('over'/'under'/None)."""
    g = games.copy()
    g['ml_side'] = [get_ml_bet(r, threshold=ML_THRESHOLD)[0] for _, r in g.iterrows()]
    edge = g['spread_pred'] + g['odds_home_spread']
    g['spread_side'] = np.where(edge.isna(), None, np.where(edge > 0, 'home', 'away'))
    diff = g['total_pred'] - g['odds_total']
    g['total_side'] = np.where(diff.isna(), None, np.where(diff > 0, 'over', 'under'))
    return g


def _pts(x):
    return f'{x:g}'


def ml_state(side, home, away, hs, as_, final=False):
    """'SAC leads by 8' / 'tied'; with a pick: 'SAC (pick) leads by 8' or 'SAC (pick) trails by 8'. Final games: 'won by' / 'lost by'."""
    margin = hs - as_
    if side is None:
        return 'tied' if margin == 0 else f"{home if margin > 0 else away} {'won' if final else 'leads'} by {abs(margin)}"
    team, m = (home, margin) if side == 'home' else (away, -margin)
    if m == 0:
        return f'{team} (pick) tied'
    verb = ('won' if m > 0 else 'lost') if final else ('leads' if m > 0 else 'trails')
    return f'{team} (pick) {verb} by {abs(m)}'


def spread_state(side, home_line, home, away, hs, as_, final=False):
    """Picked side's cover margin now: 'SAC -3.5: covering by 1.5' / 'not covering by 2.5' / 'push now'; final games: 'covered by' / 'did not cover by' / 'push'."""
    if side is None or pd.isna(home_line):
        return ''
    line = home_line if side == 'home' else -home_line
    cover = (hs - as_ if side == 'home' else as_ - hs) + line
    label = f"{home if side == 'home' else away} {line:+g}"
    if cover == 0:
        return f"{label}: push{'' if final else ' now'}"
    words = ('covered', 'did not cover') if final else ('covering', 'not covering')
    return f'{label}: {words[0] if cover > 0 else words[1]} by {_pts(abs(cover))}'


def total_state(side, line, hs, as_, final=False):
    """'Over 224.5: 176 now, needs 49 more' / 'Over 224.5: over already'; 'Under 224.5: 176 now, 48.5 below' / 'Under 224.5: busted'. Final games: 'won' / 'lost' / 'push'."""
    if side is None or pd.isna(line):
        return ''
    total = hs + as_
    label = f"{side.capitalize()} {line:g}"
    if final:
        return f'{label}: push ({total})' if total == line else f"{label}: {'won' if (total > line) == (side == 'over') else 'lost'} ({total})"
    if side == 'over':
        return f'{label}: over already ({total})' if total > line else f'{label}: {total} now, needs {int(np.floor(line - total)) + 1} more'
    return f'{label}: busted ({total})' if total > line else f'{label}: {total} now, {_pts(line - total)} below'


def pick_result(kind, side, line, home_line, hs, as_, final):
    """'won' / 'lost' once the pick's outcome is settled by the score, else ''. Final games settle every pick (a push is ''); while live only the total can
    already be decided (an over already through the line won, an under already through it lost)."""
    if side is None:
        return ''
    if kind == 'total':
        if pd.isna(line):
            return ''
        total = hs + as_
        if not final:
            return ('won' if side == 'over' else 'lost') if total > line else ''
        margin = (total - line) if side == 'over' else (line - total)
    else:
        if not final:
            return ''
        margin = hs - as_ if side == 'home' else as_ - hs
        if kind == 'spread':
            if pd.isna(home_line):
                return ''
            margin += home_line if side == 'home' else -home_line
    return 'won' if margin > 0 else ('lost' if margin < 0 else '')


def period_label(period, clock, status):
    if status == 'final':
        return 'Final' + ('' if pd.isna(period) or int(period) <= 4 else (' (OT)' if int(period) == 5 else f' ({int(period) - 4}OT)'))
    p = int(period)
    name = f'Q{p}' if p <= 4 else ('OT' if p == 5 else f'{p - 4}OT')
    return f'{name} {clock}' if isinstance(clock, str) and clock else name


def live_view(games, board):
    """
    One row per game of `games` (Today's Games rows with pick sides; see pick_sides) that has started, joined to a core/live_feed scoreboard with nba_game_id filled.
    Scheduled games and games the board does not know are left out. The RESULT_COLS columns say which pick cells are settled won or lost.
    """
    cols = LIVE_COLS + list(RESULT_COLS.values())
    if games is None or games.empty or board is None or board.empty:
        return pd.DataFrame(columns=cols)
    started = board[board['status'].isin(['live', 'final'])].dropna(subset=['nba_game_id'])
    picks = games[['game_id', 'ml_side', 'spread_side', 'odds_home_spread', 'total_side', 'odds_total']]   # the log's own score/status columns must not collide with the board's
    d = picks.merge(started, left_on='game_id', right_on='nba_game_id')
    rows = []
    for _, r in d.iterrows():
        h, a, hs, as_, fin = r['home_tricode'], r['away_tricode'], int(r['home_score']), int(r['away_score']), r['status'] == 'final'
        rows.append({'Game': f'{a} @ {h}', 'Score': f'{a} {as_} - {hs} {h}', 'Clock': period_label(r['period'], r['clock'], r['status']),
                     'ML': ml_state(r['ml_side'], h, a, hs, as_, fin), 'Spread': spread_state(r['spread_side'], r['odds_home_spread'], h, a, hs, as_, fin),
                     'Total': total_state(r['total_side'], r['odds_total'], hs, as_, fin),
                     'ml_result': pick_result('ml', r['ml_side'], None, None, hs, as_, fin),
                     'spread_result': pick_result('spread', r['spread_side'], None, r['odds_home_spread'], hs, as_, fin),
                     'total_result': pick_result('total', r['total_side'], r['odds_total'], None, hs, as_, fin)})
    return pd.DataFrame(rows, columns=cols)
