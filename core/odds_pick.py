"""
One rule for choosing the opening odds that go into the model, shared by the training-set build and live inference.

Before: the training build took DraftKings and filled each of the five fields from the next book that had it, with no
cross-book check. Home moneyline, away moneyline, spread and total could come from different books (12 games since 2019 where
the spread and the moneyline disagree on who is the favourite), and one stale or in-game quote went straight into a feature.

Now: books that are far from the cross-book consensus are dropped (the tolerances of core/odds_clean.py, which the evaluation
uses), then ONE book supplies every field: the first, in priority order, with the spread, the total and both moneylines.
If no book has all of them, the first with a spread and a total (moneyline missing); else the first with anything.
"""
import numpy as np

from core.odds_clean import MAX_ABS_PRICE, MAX_ABS_SPREAD, SPREAD_TOL, TOTAL_RANGE, TOTAL_TOL

BOOKS_PRIORITY = ('draftkings', 'fanduel', 'betmgm', 'bet365', 'caesars', 'bet_rivers_ny')
FIELDS = ('home_spread', 'away_spread', 'home_ml', 'away_ml', 'total')
REQUIRED = ('home_spread', 'home_ml', 'away_ml', 'total')


def _num(v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _clean(values):
    """Plausibility bounds (same as odds_clean): junk prices / impossible lines become NaN."""
    out = {f: _num(values.get(f)) for f in FIELDS}
    if abs(out['home_spread']) > MAX_ABS_SPREAD:
        out['home_spread'] = out['away_spread'] = np.nan
    if not (TOTAL_RANGE[0] <= out['total'] <= TOTAL_RANGE[1]):
        out['total'] = np.nan
    for f in ('home_ml', 'away_ml'):
        if not (100 <= abs(out[f]) <= MAX_ABS_PRICE):
            out[f] = np.nan
    if np.isnan(out['away_spread']) and not np.isnan(out['home_spread']):
        out['away_spread'] = -out['home_spread']
    return out


def pick_book_odds(per_book, books=BOOKS_PRIORITY):
    """
    per_book: {book: {field: value}} with fields among FIELDS (missing books / fields / blanks are fine).
    Returns {'book': name or None, 'home_spread', 'away_spread', 'home_ml', 'away_ml', 'total'} (NaN where unavailable).
    """
    clean = {b: _clean(per_book[b]) for b in books if b in per_book and per_book[b]}
    spreads = [c['home_spread'] for c in clean.values() if not np.isnan(c['home_spread'])]
    totals = [c['total'] for c in clean.values() if not np.isnan(c['total'])]
    med_spread = np.median(spreads) if spreads else np.nan
    med_total = np.median(totals) if totals else np.nan

    def consistent(c):          # a quote far from the other books is stale or in-game: the whole book is out for this game
        if not np.isnan(c['home_spread']) and not np.isnan(med_spread) and abs(c['home_spread'] - med_spread) > SPREAD_TOL:
            return False
        if not np.isnan(c['total']) and not np.isnan(med_total) and abs(c['total'] - med_total) > TOTAL_TOL:
            return False
        return True

    ok = [(b, c) for b, c in clean.items() if consistent(c)]
    tiers = (lambda c: all(not np.isnan(c[f]) for f in REQUIRED),
             lambda c: not np.isnan(c['home_spread']) and not np.isnan(c['total']),
             lambda c: any(not np.isnan(c[f]) for f in FIELDS))
    for has in tiers:
        for b, c in ok:
            if has(c):
                return {'book': b, **c}
    return {'book': None, **{f: np.nan for f in FIELDS}}
