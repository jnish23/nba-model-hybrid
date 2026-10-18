"""
Where do prop lines and player coverage change relative to tip? Reads the density pilot from data/props_odds.db
(games with a full grid of hist_T-* snapshots) and prints:
  1. players priced per snapshot (and how many first appear / disappear in each interval)
  2. consensus line movement per interval (mean |change|, share of player-markets that moved, per hour)
  3. residual movement to the close: how far the consensus line still moves after each snapshot until the last one
  4. book dispersion by snapshot (how far apart the books are)
Main line per (snapshot, book, player, market) = the line whose de-vigged over probability is closest to 50%.
Consensus line = median of the books' main lines (at least MIN_BOOKS books).
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import devig  # noqa: E402

MIN_BOOKS = 2


def load(db, min_snapshots):
    c = sqlite3.connect(db)
    # minutes to tip use the reference clock (NBA scheduled tip + 0 to 10 min, see props/schedule.py); snapshots at or after it are
    # in-game and dropped. Run scripts/annotate_props.py first.
    log = pd.read_sql("SELECT l.snapshot_ts_utc, l.event_id_vendor, l.mode, r.minutes_to_ref AS minutes_to_tip "
                      "FROM snapshot_log l JOIN snapshot_ref r USING (snapshot_ts_utc, event_id_vendor) "
                      "WHERE l.mode LIKE 'hist_T-%' AND r.in_game = 0", c)
    log['off'] = log['mode'].str.replace('hist_T-', '', regex=False).astype(int)
    full = log.groupby('event_id_vendor').size()
    events = full[full >= min_snapshots].index
    log = log[log.event_id_vendor.isin(events)]
    p = pd.read_sql("SELECT snapshot_ts_utc, event_id_vendor, book, player_name_raw, market, line, over_price, "
                    "under_price FROM prop_snapshots", c)
    p = p.merge(log[['snapshot_ts_utc', 'event_id_vendor', 'off', 'minutes_to_tip']],
                on=['snapshot_ts_utc', 'event_id_vendor'])
    return p, log


def main_lines(p):
    fo, _, _ = devig.devig_prices(p.over_price, p.under_price, 'multiplicative')
    p = p.assign(fair_over=fo, dist=np.abs(fo - 0.5)).dropna(subset=['fair_over'])
    key = ['event_id_vendor', 'off', 'book', 'player_name_raw', 'market']
    return p.sort_values('dist').groupby(key, as_index=False).first()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/props_odds.db')
    ap.add_argument('--min-snapshots', type=int, default=12)
    args = ap.parse_args()
    p, log = load(args.db, args.min_snapshots)
    games = p.event_id_vendor.nunique()
    offs = sorted((int(o) for o in p.off.unique()), reverse=True)
    print(f"{games} games, snapshots at T-{offs}\n")
    ml = main_lines(p)
    cons = (ml.groupby(['event_id_vendor', 'off', 'player_name_raw', 'market'])
              .agg(line=('line', 'median'), books=('book', 'nunique'), spread=('line', lambda s: s.max() - s.min()))
              .reset_index())
    cons = cons[cons.books >= MIN_BOOKS]

    # 1. coverage
    pl = ml[ml.market == 'points'].groupby(['event_id_vendor', 'off']).player_name_raw.nunique().unstack('off')[offs]
    first = {}
    print('1. players with a points line (any book), mean per game; first appearances and drop-outs per interval')
    rows = []
    for a, b in zip(offs[:-1], offs[1:]):
        sa = ml[(ml.market == 'points') & (ml.off == a)].groupby('event_id_vendor').player_name_raw.agg(set)
        sb = ml[(ml.market == 'points') & (ml.off == b)].groupby('event_id_vendor').player_name_raw.agg(set)
        new = sum(len(sb.get(e, set()) - sa.get(e, set())) for e in sb.index)
        gone = sum(len(sa.get(e, set()) - sb.get(e, set())) for e in sa.index)
        rows.append((f'T-{a} -> T-{b}', new / games, gone / games))
    print(f"   priced: " + ', '.join(f"T-{o}: {pl[o].mean():.1f}" for o in offs))
    print(pd.DataFrame(rows, columns=['interval', 'new_players/game', 'dropped/game']).round(2).to_string(index=False))

    # 2. movement per interval
    print('\n2. consensus line movement per interval (player-markets priced by >= 2 books at both ends)')
    rows = []
    for mkt in ('points', 'rebounds'):
        cm = cons[cons.market == mkt]
        for a, b in zip(offs[:-1], offs[1:]):
            m = cm[cm.off == a].merge(cm[cm.off == b], on=['event_id_vendor', 'player_name_raw'], suffixes=('_a', '_b'))
            hours = (log[log.off == a].minutes_to_tip.mean() - log[log.off == b].minutes_to_tip.mean()) / 60
            d = (m.line_b - m.line_a)
            rows.append((mkt, f'T-{a} -> T-{b}', len(m), d.abs().mean(), (d != 0).mean(), d.abs().mean() / hours, d.mean()))
    mv = pd.DataFrame(rows, columns=['market', 'interval', 'n', 'mean|dLine|', 'share_moved', '|dLine|/hour', 'mean_dLine'])
    print(mv.round(3).to_string(index=False))

    # 2b. price movement at an unchanged line (a book moves the juice but not the number)
    print('\n2b. same book, same main line at both ends: mean |change in de-vigged over probability| (percentage points)')
    rows = []
    key = ['event_id_vendor', 'book', 'player_name_raw', 'market']
    for mkt in ('points', 'rebounds'):
        mm = ml[ml.market == mkt]
        for a, b in zip(offs[:-1], offs[1:]):
            m = mm[mm.off == a].merge(mm[mm.off == b], on=key, suffixes=('_a', '_b'))
            same = m[m.line_a == m.line_b]
            hours = (log[log.off == a].minutes_to_tip.mean() - log[log.off == b].minutes_to_tip.mean()) / 60
            dp = (same.fair_over_b - same.fair_over_a) * 100
            rows.append((mkt, f'T-{a} -> T-{b}', len(same), dp.abs().mean(), dp.abs().mean() / hours, (m.line_a != m.line_b).mean()))
    print(pd.DataFrame(rows, columns=['market', 'interval', 'n_quotes', 'mean|dP|pp', 'pp/hour', 'share_line_changed']).round(3).to_string(index=False))

    # 3. residual movement to the close
    print(f'\n3. residual movement to the close (T-{offs[-1]}): mean |close consensus line - consensus line at snapshot|')
    rows = []
    for mkt in ('points', 'rebounds'):
        cm = cons[cons.market == mkt]
        close = cm[cm.off == offs[-1]]
        for a in offs[:-1]:
            m = cm[cm.off == a].merge(close, on=['event_id_vendor', 'player_name_raw'], suffixes=('_a', '_c'))
            d = (m.line_c - m.line_a)
            rows.append((mkt, f'T-{a}', len(m), d.abs().mean(), (d != 0).mean()))
    print(pd.DataFrame(rows, columns=['market', 'snapshot', 'n', 'mean|close-line|', 'share_still_to_move']).round(3).to_string(index=False))

    # 4. dispersion
    print('\n4. book dispersion: mean (max-min) of books\' main lines, and mean books per player-market')
    d = cons.groupby('off').agg(range_across_books=('spread', 'mean'), books=('books', 'mean')).reindex(offs)
    print(d.round(3).to_string())


if __name__ == '__main__':
    main()
