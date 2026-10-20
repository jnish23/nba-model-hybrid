"""
Model versus market at the line the book actually posts (not at a ladder the model chose).

For every (game, player, decision time T, stat) with a consensus main line (>= MIN_BOOKS agreeing books, stale/one-sided/wide quotes excluded; props/analysis.py)
the market gives a de-vigged P(over) at that line. The model's comparable quantity is P(over | no push) from its pmf at the SAME line (pushes refund the stake
and drop out, as they do from the de-vigged price). Realized outcome: over = 1, under = 0, push = dropped.

Descriptive tools for Phase 4/5: they do not select a betting rule. Prices are those stamped at the decision time (snapshots at or before T), never the opening line.
"""
import sqlite3

import numpy as np
import pandas as pd

from props import analysis
from props.models import evaluate as ev, outcome as oc
from props.models.pipeline import observed

GRID = (240, 60, 5)
MIN_BOOKS = 3


def load_consensus(props_db, season_start, season_end, grid=GRID, min_books=MIN_BOOKS):
    """Consensus line and de-vigged P(over) per (game_id, player_id, minutes_before, market) from snapshots stamped at or before T, for games between the two ET dates."""
    modes = [f'hist_T-{x}' for x in grid]
    q = f"""SELECT p.snapshot_ts_utc, p.game_id, p.event_id_vendor, p.book, p.player_id, p.player_name_raw, p.market, p.line, p.over_price, p.under_price,
                   p.is_alt_line, p.book_last_update_utc, l.mode
            FROM prop_snapshots p
            JOIN snapshot_log l ON l.snapshot_ts_utc = p.snapshot_ts_utc AND l.event_id_vendor = p.event_id_vendor
            LEFT JOIN snapshot_ref s ON s.snapshot_ts_utc = p.snapshot_ts_utc AND s.event_id_vendor = p.event_id_vendor
            WHERE l.mode IN ({','.join('?' * len(modes))}) AND p.is_alt_line = 0 AND p.game_id IS NOT NULL AND (p.player_id IS NULL OR p.player_id <> -1)
              AND COALESCE(s.in_game, 0) = 0 AND s.minutes_to_ref >= CAST(substr(l.mode, 8) AS REAL)
              AND substr(p.commence_time_utc, 1, 10) BETWEEN ? AND ?"""
    with sqlite3.connect(f'file:{props_db}?mode=ro', uri=True, timeout=120) as conn:
        d = pd.read_sql(q, conn, params=modes + [season_start, season_end])
    flagged = analysis.flag_main_lines(analysis.add_devig(d))
    cons = analysis.consensus(flagged, min_books=min_books)
    cons = cons.dropna(subset=['consensus_fair_over'])
    keys = d[['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw', 'game_id', 'player_id', 'mode']].dropna(subset=['player_id']).drop_duplicates(
        ['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw'])
    cons = cons.merge(keys, on=['snapshot_ts_utc', 'event_id_vendor', 'player_name_raw'], how='inner', validate='many_to_one')
    cons['minutes_before'] = cons['mode'].str.replace('hist_T-', '', regex=False).astype(int)
    cons['player_id'] = cons['player_id'].astype(int)
    return cons[['game_id', 'player_id', 'minutes_before', 'market', 'consensus_line', 'consensus_fair_over', 'n_at_line', 'book_spread']]


def compare(rows, pmfs, cons, stats=None):
    """One row per (model row, stat) that has a consensus quote at its decision time and is not a push: model P(over | no push), market fair P(over), outcome."""
    out = []
    key = rows[['game_id', 'player_id', 'minutes_before']].copy()
    key['i'] = np.arange(len(rows))
    for stat in stats or list(pmfs):
        c = cons[cons['market'] == stat]
        m = key.merge(c, on=['game_id', 'player_id', 'minutes_before'], how='inner', validate='one_to_one')
        if m.empty:
            continue
        i = m['i'].to_numpy()
        y = observed(rows.iloc[i], stat)
        line = m['consensus_line'].to_numpy(float)
        pm = pmfs[stat][i]
        ok = ~np.isnan(y) & (y != line)                                                  # a push has no winner
        out.append(pd.DataFrame({'game_id': m['game_id'].to_numpy()[ok], 'player_id': m['player_id'].to_numpy()[ok], 'minutes_before': m['minutes_before'].to_numpy()[ok],
                                 'stat': stat, 'line': line[ok], 'p_model': oc.prob_over_no_push(pm[ok], line[ok]), 'p_market': m['consensus_fair_over'].to_numpy(float)[ok],
                                 'hit': (y[ok] > line[ok]).astype(float), 'n_at_line': m['n_at_line'].to_numpy()[ok], 'book_spread': m['book_spread'].to_numpy()[ok],
                                 'push_share_at_line': oc.prob_push(pm[ok], line[ok])}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _ll(p, h, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return -(h * np.log(p) + (1 - h) * np.log(1 - p))


def score_table(cmp, n_boot=ev.N_BOOT):
    """Log loss and Brier of model vs market on the same rows, with the paired difference (model minus market; positive = the market is better) and its game-clustered interval."""
    out = []
    for stat, g in [('all', cmp)] + list(cmp.groupby('stat')):
        d_ll = _ll(g['p_model'].to_numpy(), g['hit'].to_numpy()) - _ll(g['p_market'].to_numpy(), g['hit'].to_numpy())
        d_br = (g['p_model'] - g['hit']) ** 2 - (g['p_market'] - g['hit']) ** 2
        p, lo, hi = ev.cluster_bootstrap(d_ll, g['game_id'], n_boot=n_boot)
        pb, lob, hib = ev.cluster_bootstrap(d_br.to_numpy(), g['game_id'], n_boot=n_boot)
        out.append({'stat': stat, 'n': len(g), 'logloss_model': _ll(g['p_model'].to_numpy(), g['hit'].to_numpy()).mean(), 'logloss_market': _ll(g['p_market'].to_numpy(), g['hit'].to_numpy()).mean(),
                    'diff': p, 'lo': lo, 'hi': hi, 'brier_diff': pb, 'brier_lo': lob, 'brier_hi': hib, 'mean_p_model': g['p_model'].mean(), 'mean_p_market': g['p_market'].mean(),
                    'hit_rate': g['hit'].mean()})
    return pd.DataFrame(out)


def disagreement_table(cmp, edges=(-1, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 1), n_boot=ev.N_BOOT):
    """Calibration at the book's own line, split by how far the model is from the market (the region a bettor acts in). If the model carries information the
    market lacks, the realized hit rate moves from the market's probability toward the model's as the gap grows; if the gap is noise (or bias), it does not."""
    d = cmp['p_model'] - cmp['p_market']
    b = pd.cut(d, list(edges))
    out = []
    for k, g in cmp.groupby(b, observed=True):
        gap = g['p_model'] - g['p_market']
        slope_pt, lo, hi = ev.cluster_bootstrap((g['hit'] - g['p_market']).to_numpy(), g['game_id'], n_boot=n_boot)
        out.append({'model minus market': str(k), 'n': len(g), 'mean p_market': g['p_market'].mean(), 'mean p_model': g['p_model'].mean(), 'hit rate': g['hit'].mean(),
                    'hit - market': slope_pt, 'lo': lo, 'hi': hi, 'hit - model': (g['hit'] - g['p_model']).mean(), 'mean gap': gap.mean()})
    return pd.DataFrame(out)


def reliability_at_book_line(cmp, bins=10):
    """Model probability deciles at the book line against realized hit rate (and the market's own decile for reference)."""
    q = pd.qcut(cmp['p_model'], bins, duplicates='drop')
    return cmp.groupby(q, observed=True).agg(n=('hit', 'size'), mean_p_model=('p_model', 'mean'), mean_p_market=('p_market', 'mean'), hit_rate=('hit', 'mean')).reset_index(drop=True)
