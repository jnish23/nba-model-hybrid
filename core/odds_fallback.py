"""
The Odds API as a backup for SportsbookReview (SBR) team lines.

When SBR lists a game but has no lines for it (all five markets empty), the daily inference fills the game from The Odds API's bulk
endpoint instead. This module is the vendor-neutral part: it turns the API's events into the SBR game shape ({market: {book: value}}),
so core.odds_pick / team.live_odds work on them unchanged, and matches API events to SBR games by team.

One approximation to know about: the model is trained on OPENING lines, and the API only offers the CURRENT line. For a game SBR has
no lines for at all, the API's line at fetch time stands in for both (the morning pull is usually close to the open).
"""
import datetime as dt
import logging

# The Odds API bookmaker key -> the book names core.odds_pick.BOOKS_PRIORITY / SBR use
BOOK_KEYS = {'draftkings': 'draftkings', 'fanduel': 'fanduel', 'betmgm': 'betmgm', 'williamhill_us': 'caesars',
             'betrivers': 'bet_rivers_ny'}
API_BOOKMAKERS = tuple(BOOK_KEYS)                       # asked for by key: up to 10 bookmakers cost one region
SBR_FIELDS = ('home_ml', 'away_ml', 'home_spread', 'away_spread', 'total',
              'home_spread_odds', 'away_spread_odds', 'over_odds', 'under_odds')
SOURCE = 'theoddsapi'


def _nickname(team):
    """'Los Angeles Clippers' and SBR's 'LA Clippers' both end in 'clippers'; every NBA nickname's last word is unique."""
    return str(team).strip().lower().split()[-1] if str(team).strip() else ''


def has_lines(game):
    """True when an SBR game dict has a price for any market."""
    return any(isinstance(game.get(f), dict) and game[f] for f in SBR_FIELDS)


def event_to_sbr_game(event):
    """One Odds API event -> {field: {sbr_book: value}} (the lines of the SBR game dict), plus its teams."""
    out = {f: {} for f in SBR_FIELDS}
    home, away = event.get('home_team'), event.get('away_team')
    for bm in event.get('bookmakers') or []:
        book = BOOK_KEYS.get(bm.get('key'))
        if book is None:
            continue
        for market in bm.get('markets') or []:
            outcomes = market.get('outcomes') or []
            by_name = {o.get('name'): o for o in outcomes}
            if market.get('key') == 'h2h':
                if home in by_name and away in by_name:
                    out['home_ml'][book], out['away_ml'][book] = by_name[home].get('price'), by_name[away].get('price')
            elif market.get('key') == 'spreads':
                if home in by_name and away in by_name and by_name[home].get('point') is not None:
                    out['home_spread'][book], out['away_spread'][book] = by_name[home]['point'], by_name[away].get('point')
                    out['home_spread_odds'][book], out['away_spread_odds'][book] = by_name[home].get('price'), by_name[away].get('price')
            elif market.get('key') == 'totals':
                if 'Over' in by_name and by_name['Over'].get('point') is not None:
                    out['total'][book] = by_name['Over']['point']
                    out['over_odds'][book] = by_name['Over'].get('price')
                    out['under_odds'][book] = (by_name.get('Under') or {}).get('price')
    return {**out, 'home_team': home, 'away_team': away, 'commence_time': event.get('commence_time')}


UPCOMING_HOURS = 30


def index_by_matchup(events, now=None, hours=UPCOMING_HOURS):
    """
    {(away nickname, home nickname): SBR-shaped game} for the API events that carry any line and tip off within the next `hours`.
    A game that has started carries in-game lines (not the pre-game ones the model needs), and the API also lists far-off games (a
    Christmas rematch of tonight's matchup): both are left out. If a matchup still appears twice, the earlier game wins.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    games = {}
    for event in sorted(events or [], key=lambda e: str(e.get('commence_time'))):
        try:
            tip = dt.datetime.fromisoformat(str(event.get('commence_time')).replace('Z', '+00:00'))
        except ValueError:
            continue
        if not now < tip <= now + dt.timedelta(hours=hours):
            continue
        game = event_to_sbr_game(event)
        key = (_nickname(game['away_team']), _nickname(game['home_team']))
        if has_lines(game) and key not in games:
            games[key] = game
    return games


def find_game(games, away_team, home_team):
    """The API game for an SBR matchup (names as SBR spells them), or None."""
    return games.get((_nickname(away_team), _nickname(home_team)))


def fetch_team_odds(budget_db='data/team_odds_ledger.db', raw_dir='data/raw/team_odds_api', per_run=10, per_day=40, api=None, now=None):
    """
    The API's team lines as {(away, home): game}; {} (with a logged reason) if the key is missing, the budget refuses, or the call fails.
    A fallback must never stop the daily run.
    """
    try:
        from core.odds_vendor.base import CreditBudget
        from core.odds_vendor.theoddsapi import TheOddsApi
        api = api or TheOddsApi(CreditBudget(budget_db, per_run=per_run, per_day=per_day), raw_dir=raw_dir)
        response = api.team_odds(bookmakers=API_BOOKMAKERS)
    except Exception as e:                       # no key, over budget, network, HTTP error
        logging.warning(f"Odds API backup unavailable: {e}")
        return {}
    games = index_by_matchup(response.payload, now=now)
    logging.info(f"Odds API backup: lines for {len(games)} games (cost {response.cost}, {response.remaining} credits remaining)")
    return games
