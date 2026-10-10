"""
Clean opening/closing odds into one row per game.

Sources: `nba_odds_opening` (opening) and `nba_odds` (closing) in nba_stats.db, both wide
tables with one column per book. Known problems this module handles:
  - teamId_home/away missing on ~4-8% of rows -> filled from full team name (names map 1:1 to IDs;
    abbreviations are inconsistent, e.g. BK/BKN, GS/GSW, so they are not used)
  - sentinel / junk prices (e.g. 2147483647, -100000) and impossible lines -> NaN
  - individual books in the closing table sometimes hold an in-game line (e.g. -35.5 when other
    books are -2.5); these leak the outcome, so books far from the cross-book median are dropped
  - in two windows (see CLOSING_BAD_WINDOWS) the closing table holds in-game lines for nearly all
    books at once, so the cross-book filter cannot catch them; closing odds there are set to NaN.
    Found by daily closing-vs-opening error: close error was 30-80% below open (normal: ~0-5%).
  - closing-table scores are 0-0 for many rows -> scores come from team_game_full instead

Output columns (prefix `open_` / `close_`):
  home_spread       consensus home spread (median of kept books; negative = home favored)
  total             consensus total
  home_prob         consensus no-vig home win probability (median of per-book de-vigged probs)
  home_ml/away_ml   median price across kept books
  home_ml_best/away_ml_best  best price across kept books (line shopping)
  n_books_spread / n_books_total / n_books_ml
"""
import sqlite3
import logging
from pathlib import Path

import numpy as np
import pandas as pd

BOOKS = ['betmgm', 'fanduel', 'caesars', 'bet365', 'draftkings', 'bet_rivers_ny']

# Plausibility bounds
MAX_ABS_SPREAD = 30.0
TOTAL_RANGE = (170.0, 290.0)
MAX_ABS_PRICE = 20000.0

# Max distance from the cross-book median before a book is treated as stale/in-game
SPREAD_TOL = 3.0
TOTAL_TOL = 6.0
# Opening quotes are posted at different times; a wider tolerance lets stale prices through and
# makes 'best price' look like free edge (betting both sides of every game at best price ~breaks even)
PROB_TOL = 0.03

# Date ranges (inclusive) where nba_odds "closing" lines are in-game snapshots for most books
CLOSING_BAD_WINDOWS = [('2019-10-22', '2019-11-19'), ('2021-10-19', '2021-11-23')]

TEAM_NAME_ALIASES = {'Charlotte Bobcats': 'Charlotte Hornets', 'LA Clippers': 'Los Angeles Clippers'}


def american_to_payout(odds):
    """Profit per 1 unit staked (vectorized)."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(odds < 0, 100.0 / -odds, odds / 100.0)


def payout_to_american(payout):
    """Inverse of american_to_payout (vectorized)."""
    payout = np.asarray(payout, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(payout >= 1.0, payout * 100.0, -100.0 / payout)


def american_to_prob(odds):
    """Vig-inclusive implied probability from American odds (vectorized)."""
    return 1.0 / (1.0 + american_to_payout(odds))


def _clean_price(s):
    s = pd.to_numeric(s, errors='coerce')
    valid = (s.abs() >= 100) & (s.abs() <= MAX_ABS_PRICE)
    return s.where(valid)


def _filter_to_consensus(values: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Mask book values further than `tol` from the per-game cross-book median."""
    med = values.median(axis=1)
    return values.where(values.sub(med, axis=0).abs() <= tol)


def _load_table(conn, table):
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df['gameDate'] = pd.to_datetime(df['gameDate']).dt.strftime('%Y-%m-%d')
    for col in df.columns:
        if any(col.endswith('_' + b) for b in BOOKS):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _fill_team_ids(frames):
    """Fill missing teamId_home/away from full team names (consistent across both tables)."""
    name_to_id = {}
    for df in frames:
        for side in ('home', 'away'):
            known = df[[f'{side}_team', f'teamId_{side}']].dropna()
            names = known[f'{side}_team'].replace(TEAM_NAME_ALIASES)
            name_to_id.update(dict(zip(names, known[f'teamId_{side}'].astype(int))))
    for df in frames:
        for side in ('home', 'away'):
            names = df[f'{side}_team'].replace(TEAM_NAME_ALIASES)
            df[f'teamId_{side}'] = df[f'teamId_{side}'].fillna(names.map(name_to_id))
            unmapped = df[f'teamId_{side}'].isna().sum()
            if unmapped:
                logging.warning(f"{unmapped} rows with unmapped {side} team names: "
                                f"{df.loc[df[f'teamId_{side}'].isna(), f'{side}_team'].unique()[:5]}")
            df[f'teamId_{side}'] = df[f'teamId_{side}'].astype('Int64')


def _consensus(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Collapse per-book columns into consensus lines/prices for one snapshot."""
    def books(col):
        return df[[f'{col}_{b}' for b in BOOKS if f'{col}_{b}' in df.columns]].copy()

    out = pd.DataFrame(index=df.index)

    # Spread (home perspective); require the away side to mirror it
    home_sp = books('home_spread')
    away_sp = books('away_spread')
    away_sp.columns = home_sp.columns
    home_sp = home_sp.where((home_sp.abs() <= MAX_ABS_SPREAD) & ((home_sp + away_sp).abs() < 0.01))
    home_sp = _filter_to_consensus(home_sp, SPREAD_TOL)
    out[f'{prefix}home_spread'] = home_sp.median(axis=1)
    out[f'{prefix}n_books_spread'] = home_sp.notna().sum(axis=1)

    # Total
    tot = books('total')
    tot = tot.where((tot >= TOTAL_RANGE[0]) & (tot <= TOTAL_RANGE[1]))
    tot = _filter_to_consensus(tot, TOTAL_TOL)
    out[f'{prefix}total'] = tot.median(axis=1)
    out[f'{prefix}n_books_total'] = tot.notna().sum(axis=1)

    # Moneyline: de-vig each book (proportional), drop outlier books, take medians
    h = books('home_ml').apply(_clean_price)
    a = books('away_ml').apply(_clean_price)
    a.columns = h.columns
    both = h.notna() & a.notna()
    h, a = h.where(both), a.where(both)
    ph = pd.DataFrame(american_to_prob(h), index=h.index, columns=h.columns)
    pa = pd.DataFrame(american_to_prob(a), index=a.index, columns=a.columns)
    overround = ph + pa
    # A two-way market should have a 0-15% overround; anything else is a bad/stale quote
    fair = (ph / overround).where((overround >= 1.0) & (overround <= 1.15))
    fair = _filter_to_consensus(fair, PROB_TOL)
    keep = fair.notna()
    h, a = h.where(keep), a.where(keep)
    out[f'{prefix}home_prob'] = fair.median(axis=1)
    # Median in payout space: American odds are discontinuous at +/-100, so their median can be invalid
    out[f'{prefix}home_ml'] = payout_to_american(pd.DataFrame(american_to_payout(h), index=h.index).median(axis=1))
    out[f'{prefix}away_ml'] = payout_to_american(pd.DataFrame(american_to_payout(a), index=a.index).median(axis=1))
    out[f'{prefix}home_ml_best'] = h.max(axis=1)  # higher American odds = better payout
    out[f'{prefix}away_ml_best'] = a.max(axis=1)
    out[f'{prefix}n_books_ml'] = keep.sum(axis=1)

    return out


def _game_results(conn):
    """One row per game with gameId and final scores, from team_game_full."""
    tg = pd.read_sql("SELECT gameId, gameDate, teamId, MATCHUP, points, seasonYear FROM team_game_full", conn)
    tg['gameDate'] = pd.to_datetime(tg['gameDate']).dt.strftime('%Y-%m-%d')
    is_home = tg['MATCHUP'].str.contains(' vs. ', regex=False)
    home = tg[is_home].rename(columns={'teamId': 'teamId_home', 'points': 'home_score'})
    away = tg[~is_home][['gameId', 'teamId', 'points']].rename(columns={'teamId': 'teamId_away', 'points': 'away_score'})
    games = home.drop(columns='MATCHUP').merge(away, on='gameId')
    games['teamId_home'] = games['teamId_home'].astype('Int64')
    games['teamId_away'] = games['teamId_away'].astype('Int64')
    return games


def load_clean_odds(db_path, min_date='2019-10-01'):
    """
    Return one row per game with cleaned opening + closing consensus odds, gameId and scores.
    Book-level odds only exist from the 2019-20 season on, hence the default `min_date`.
    """
    with sqlite3.connect(db_path) as conn:
        opening = _load_table(conn, 'nba_odds_opening')
        closing = _load_table(conn, 'nba_odds')
        games = _game_results(conn)

    _fill_team_ids([opening, closing])
    key = ['gameDate', 'teamId_home', 'teamId_away']

    snapshots = []
    for df, prefix in ((opening, 'open_'), (closing, 'close_')):
        df = df[(df['gameDate'] >= min_date) & ~df['status'].str.contains('Postponed', na=False)]
        df = df.drop_duplicates(subset=key, keep='last')
        snapshots.append(pd.concat([df[key], _consensus(df, prefix)], axis=1))

    odds = snapshots[0].merge(snapshots[1], on=key, how='outer')
    close_cols = [c for c in odds.columns if c.startswith('close_')]
    for lo, hi in CLOSING_BAD_WINDOWS:
        bad = odds['gameDate'].between(lo, hi)
        odds.loc[bad, close_cols] = np.nan
        logging.info(f"Dropped closing odds for {bad.sum()} games in contaminated window {lo}..{hi}")
    out = games.merge(odds, on=key, how='inner')
    logging.info(f"Clean odds: {len(out)} games matched to results "
                 f"({out['open_home_spread'].notna().sum()} with opening spread, "
                 f"{out['close_home_spread'].notna().sum()} with closing spread)")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_clean_odds(Path('data/nba_stats.db'))
    out_path = Path('data/processed/odds_clean.csv')
    df.to_csv(out_path, index=False)
    print(df.describe().T.round(2).to_string())
    print(f"Saved {len(df)} rows to {out_path}")
