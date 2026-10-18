"""
Vendor JSON -> tidy rows. One prop row per (snapshot, event, book, market, player, line) with both prices on the row.

Input is one event-odds payload in The Odds API v4 shape:
  {id, commence_time, home_team, away_team, bookmakers: [{key, markets: [{key, last_update, outcomes: [
      {name: 'Over'|'Under', description: <player>, point, price}]}]}]}
Team markets (h2h, spreads, totals) go to a separate table with the same snapshot timestamp, so a prop line and
the game total/spread can be compared at exactly the same moment (H2).

Nothing is dropped silently: unknown market keys and outcomes that are neither Over nor Under are counted in the
returned stats.
"""
import pandas as pd

from core.timeutil import to_utc_iso

PROP_MARKETS = {
    'player_points': 'points', 'player_rebounds': 'rebounds', 'player_assists': 'assists',
    'player_threes': 'threes', 'player_points_rebounds_assists': 'pra', 'player_points_rebounds': 'pr',
    'player_points_assists': 'pa', 'player_rebounds_assists': 'ra',
}
ALT_SUFFIX = '_alternate'
TEAM_MARKETS = ('h2h', 'spreads', 'totals')
PROP_COLS = ['snapshot_ts_utc', 'event_id_vendor', 'commence_time_utc', 'home_team', 'away_team', 'book', 'market',
             'player_name_raw', 'line', 'over_price', 'under_price', 'is_alt_line', 'book_last_update_utc', 'source']
TEAM_COLS = ['snapshot_ts_utc', 'event_id_vendor', 'book', 'home_ml', 'away_ml', 'home_spread',
             'home_spread_price', 'away_spread_price', 'total', 'over_price', 'under_price', 'book_last_update_utc',
             'source']


def prop_market(vendor_key):
    """('player_points_alternate') -> ('points', True); unknown keys -> (None, None)."""
    alt = vendor_key.endswith(ALT_SUFFIX)
    base = vendor_key[:-len(ALT_SUFFIX)] if alt else vendor_key
    market = PROP_MARKETS.get(base)
    return (market, alt) if market else (None, None)


def parse_event_odds(event, snapshot_ts_utc, source='theoddsapi'):
    """Returns (props_df, team_df, stats)."""
    snap = to_utc_iso(snapshot_ts_utc)
    ev_id, commence = event['id'], to_utc_iso(event['commence_time'])
    home, away = event['home_team'], event['away_team']
    props, teams = {}, []
    stats = {'unknown_markets': {}, 'odd_outcomes': 0}
    for bk in event.get('bookmakers', []) or []:
        book = bk['key']
        team_row = {'snapshot_ts_utc': snap, 'event_id_vendor': ev_id, 'book': book, 'source': source}
        team_ts = []
        for mk in bk.get('markets', []) or []:
            key, last = mk['key'], mk.get('last_update')
            market, alt = prop_market(key)
            if market is not None:
                for o in mk.get('outcomes', []):
                    side = str(o.get('name', '')).lower()
                    if side not in ('over', 'under') or o.get('description') is None or o.get('point') is None:
                        stats['odd_outcomes'] += 1
                        continue
                    k = (book, market, o['description'], float(o['point']), alt)
                    row = props.setdefault(k, {'snapshot_ts_utc': snap, 'event_id_vendor': ev_id,
                                               'commence_time_utc': commence, 'home_team': home, 'away_team': away,
                                               'book': book, 'market': market, 'player_name_raw': o['description'],
                                               'line': float(o['point']), 'over_price': None, 'under_price': None,
                                               'is_alt_line': int(alt),
                                               'book_last_update_utc': to_utc_iso(last) if last else None,
                                               'source': source})
                    row[f'{side}_price'] = o.get('price')
            elif key in TEAM_MARKETS:
                if last:
                    team_ts.append(to_utc_iso(last))
                _fill_team_row(team_row, key, mk.get('outcomes', []), home, away)
            else:
                stats['unknown_markets'][key] = stats['unknown_markets'].get(key, 0) + 1
        if len(team_row) > 4:
            team_row['book_last_update_utc'] = max(team_ts) if team_ts else None
            teams.append(team_row)
    props_df = pd.DataFrame(list(props.values()), columns=PROP_COLS)
    team_df = pd.DataFrame(teams).reindex(columns=TEAM_COLS)
    return props_df, team_df, stats


def _fill_team_row(row, key, outcomes, home, away):
    for o in outcomes:
        name, price, point = o.get('name'), o.get('price'), o.get('point')
        if key == 'h2h':
            if name == home:
                row['home_ml'] = price
            elif name == away:
                row['away_ml'] = price
        elif key == 'spreads':
            if name == home:
                row['home_spread'], row['home_spread_price'] = point, price
            elif name == away:
                row['away_spread_price'] = price
        elif key == 'totals':
            row['total'] = point
            if str(name).lower() == 'over':
                row['over_price'] = price
            elif str(name).lower() == 'under':
                row['under_price'] = price
