"""
Odds handling for live inference.

The models are trained on OPENING lines (nba_odds_opening / the archive's open_* columns), so the
`odds_*` model inputs at inference must be opening lines too. The CURRENT line is what a bet is
actually placed at, so it is kept separately as `curr_odds_*` and used only for edge / EV /
settlement. If the opening line cannot be fetched for a market the current line stands in for the
model input (and `odds_open_missing` counts how many inputs were substituted), because the model
cannot take a missing input from a team that is on the board.
"""
import numpy as np
import pandas as pd

MARKETS = ('home_ml', 'away_ml', 'home_spread', 'away_spread', 'total')
# Same book priority as the training build (models/game_features_v2._load_db_odds)
BOOKS = ('draftkings', 'fanduel', 'betmgm', 'bet365', 'caesars')


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


def split_open_and_current(curr_game, open_game):
    """
    curr_game / open_game: SBR game dicts holding {book: value} dicts under MARKETS keys
    (open_game may be None). Returns a flat dict:
      odds_<market>       model input: opening line, else the current line
      curr_odds_<market>  current line (for bets / EV)
      odds_open_missing   number of model inputs that fell back to the current line
    """
    open_game = open_game or {}
    out, fallbacks = {}, 0
    for market in MARKETS:
        curr = first_book_value(curr_game.get(market))
        opened = first_book_value(open_game.get(market))
        if pd.isna(opened) and pd.notna(curr):
            fallbacks += 1
        out[f'odds_{market}'] = opened if pd.notna(opened) else curr
        out[f'curr_odds_{market}'] = curr
    out['odds_open_missing'] = fallbacks
    return out
