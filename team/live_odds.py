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

from core.devig import devig_prices
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
    # Which book supplied each side (None when no book had a line). Added for the bet log; every output above is unchanged.
    out['open_book'] = opened['book']
    out['curr_book'] = book
    return out


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return x


def _fair_home(home_ml, away_ml):
    """De-vigged (multiplicative) home win probability from a moneyline pair, NaN if either price is missing."""
    fh, _, _ = devig_prices(home_ml, away_ml, 'multiplicative')
    return float(fh)


def bet_log_row(game, pred, bets, run_ts_utc, model_set_hash=None, injury_cutoff=None, injury_report=None, bet_suppressed_reason=None):
    """
    One bet_recommendations_log row (core/prediction_logger.BET_LOG_COLUMNS) for one game of one run.

    game  : mapping with gameId, gameDate, odds_source, the opening lines odds_* and the current lines / prices / book
            curr_odds_*, curr_price_*, curr_book, open_book (split_open_and_current output)
    pred  : mapping with ml_prob, ml_prob_raw, spread_pred, total_pred, spread_std, total_std
    bets  : {'ml': (side, edge), 'spread': (side, edge), 'total': (side, edge)} as returned by team.betting.get_*_bet

    own_view = the model against the OPENING line it read (points; ML: probability against the de-vigged opening price).
    line_move = opening minus current line (spread and total: the raw line numbers; ML: de-vigged home probability).
    Identities, with the model's point edge at the current line:  spread edge_pts = own_view - line_move;
    total edge_pts = own_view + line_move;  ML (model - current fair prob) = own_view + line_move.
    Anything that cannot be computed because a line is missing is None (NULL), never 0.
    """
    g = lambda k: _num(game.get(k))
    p = lambda k: _num(pred.get(k))
    ml_prob, spread_pred, total_pred = p('ml_prob'), p('spread_pred'), p('total_pred')

    open_fair, curr_fair = _fair_home(g('odds_home_ml'), g('odds_away_ml')), _fair_home(g('curr_odds_home_ml'), g('curr_odds_away_ml'))
    own = {'ml': ml_prob - open_fair,
           'spread': spread_pred + g('odds_home_spread'),
           'total': total_pred - g('odds_total')}
    move = {'ml': open_fair - curr_fair,
            'spread': g('odds_home_spread') - g('curr_odds_home_spread'),
            'total': g('odds_total') - g('curr_odds_total')}
    have_curr = {'ml': not (np.isnan(g('curr_odds_home_ml')) or np.isnan(g('curr_odds_away_ml'))),
                 'spread': not np.isnan(g('curr_odds_home_spread')),
                 'total': not np.isnan(g('curr_odds_total'))}

    row = {'game_id': str(game.get('gameId')), 'game_date': str(game.get('gameDate'))[:10], 'run_ts_utc': run_ts_utc,
           'model_set_hash': model_set_hash, 'odds_source': game.get('odds_source'),
           'open_book': game.get('open_book'), 'curr_book': game.get('curr_book')}
    for k in ('odds_home_ml', 'odds_away_ml', 'odds_home_spread', 'odds_total',
              'curr_odds_home_ml', 'curr_odds_away_ml', 'curr_odds_home_spread', 'curr_odds_total',
              'curr_price_home_spread_odds', 'curr_price_away_spread_odds', 'curr_price_over_odds', 'curr_price_under_odds'):
        row[k] = g(k)
    row.update(ml_prob=ml_prob, ml_prob_raw=p('ml_prob_raw'), spread_pred=spread_pred, total_pred=total_pred,
               spread_std=p('spread_std'), total_std=p('total_std'))
    for m in ('ml', 'spread', 'total'):
        side, edge = bets.get(m, (None, np.nan))
        row[f'{m}_bet'] = side
        row[f'{m}_edge'] = edge if have_curr[m] else np.nan
        row[f'{m}_own_view'] = own[m]
        row[f'{m}_line_move'] = move[m]
    row['injury_cutoff_et'] = None if injury_cutoff is None else str(injury_cutoff)
    row['injury_report_file'] = None if injury_report is None else str(injury_report)
    row['bet_suppressed_reason'] = None if bet_suppressed_reason is None else str(bet_suppressed_reason)   # the *_bet sides are then None
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}

