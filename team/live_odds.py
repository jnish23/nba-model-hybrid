"""
Odds handling for live inference.

The models are trained on OPENING lines (nba_odds_opening / the archive's open_* columns), so the
`odds_*` model inputs at inference must be opening lines too. The CURRENT line is what a bet is
actually placed at, so it is kept separately as `curr_odds_*` and used only for edge / EV /
settlement. If the opening line cannot be fetched for a market the model input stays NaN (the current
line is NOT substituted: it already contains the news since the open, which the spread model, trained
to find what the opening line misses, would count twice) and `odds_open_missing` counts those markets;
the caller skips such games.
"""
import numpy as np
import pandas as pd

from core.odds_pick import pick_book_odds

MARKETS = ('home_ml', 'away_ml', 'home_spread', 'away_spread', 'total')
# Prices (juice) of the spread / total bets at the current line; the edge on a bet is judged against its own price
PRICE_MARKETS = ('home_spread_odds', 'away_spread_odds', 'over_odds', 'under_odds')
# Same book priority as the training build (core/odds_pick.BOOKS_PRIORITY)
BOOKS = ('draftkings', 'fanduel', 'betmgm', 'bet365', 'caesars', 'bet_rivers_ny')


def first_book_value(item):
    """First available book's number from an SBR {book: value} dict, else NaN.

    0 is a legitimate value (a pick'em spread), so only missing/blank values are skipped.
    """
    if not isinstance(item, dict):
        return np.nan
    for book in BOOKS:
        val = item.get(book)
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isnan(num):
            return num
    return np.nan


def _per_book(game, fields):
    """SBR {field: {book: value}} -> {book: {field: value}}."""
    return {b: {f: (game.get(f) or {}).get(b) if isinstance(game.get(f), dict) else None for f in fields} for b in BOOKS}


def split_open_and_current(curr_game, open_game):
    """
    curr_game / open_game: SBR game dicts holding {book: value} dicts under MARKETS keys
    (open_game may be None). Every field of a side comes from ONE book (core.odds_pick.pick_book_odds, the rule the
    training build uses), so moneylines, spread and total are mutually consistent. Returns a flat dict:
      odds_<market>       model input: the opening line (NaN when it could not be fetched; never the current line)
      curr_odds_<market>  current line (for bets / EV)
      curr_price_<name>   current spread / total price from the same book as the current line
      odds_open_missing   number of markets on the board whose opening line is missing
    """
    open_game = open_game or {}
    opened = pick_book_odds(_per_book(open_game, MARKETS), BOOKS)
    curr = pick_book_odds(_per_book(curr_game, MARKETS), BOOKS)
    out, missing = {}, 0
    for market in MARKETS:
        if pd.isna(opened[market]) and pd.notna(curr[market]):
            missing += 1
        out[f'odds_{market}'] = opened[market]
        out[f'curr_odds_{market}'] = curr[market]
    book = curr['book']
    for name in PRICE_MARKETS:
        out[f'curr_price_{name}'] = first_book_value({book: curr_game[name].get(book)}) if book and isinstance(curr_game.get(name), dict) else np.nan
    out['odds_open_missing'] = missing
    return out

