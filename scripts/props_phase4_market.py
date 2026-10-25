"""
Model versus de-vigged market at the book's own consensus line (props/models/market_bench.py), for one saved version and season.

    python scripts/props_phase4_market.py --version v2_base --season 2023-24 --out reports/props/phase4_market_2023-24.md

Descriptive: it selects no betting rule. Prices are consensus snapshots stamped at or before each decision time; pushes are dropped.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.models import evaluate as ev, market_bench as mb  # noqa: E402
from props.quality import md_table  # noqa: E402

SEASON_DATES = {'2023-24': ('2023-10-01', '2024-06-30'), '2024-25': ('2024-10-01', '2025-06-30'), '2025-26': ('2025-10-01', '2026-06-30')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--season', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--props-db', default='data/props_odds.db')
    ap.add_argument('--n-boot', type=int, default=500)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    rows, pmfs = ev.load_rows_light(a.version, a.season)
    start, end = SEASON_DATES[a.season]
    cons = mb.load_consensus(a.props_db, start, end)
    logging.info(f'{len(cons)} consensus quotes')
    cmp = mb.compare(rows, pmfs, cons)
    logging.info(f'{len(cmp)} comparable rows')
    sc = mb.score_table(cmp, n_boot=a.n_boot)
    dis = mb.disagreement_table(cmp, n_boot=a.n_boot)
    dis_stat = {s: mb.disagreement_table(g, n_boot=a.n_boot) for s, g in cmp.groupby('stat') if s in ('points', 'rebounds', 'assists', 'pra')}
    rel = mb.reliability_at_book_line(cmp)
    slope = mb.information_slope(cmp, n_boot=a.n_boot)
    by_t = pd.concat([mb.score_table(g, n_boot=a.n_boot).assign(minutes_before=t).query("stat == 'all'") for t, g in cmp.groupby('minutes_before')])
    r = lambda d, n=4: md_table(d.round(n), n)
    md = [f'# Model versus market at the book line: {a.season} ({a.version})\n',
          f'{len(cmp)} (player, stat, decision time) comparisons with a consensus line (at least {mb.MIN_BOOKS} agreeing books; stale, one-sided and wide quotes excluded), pushes dropped. '
          'Market probability = de-vigged consensus P(over) at the snapshot stamped at or before T; model probability = P(over | no push) from its pmf at the same line. '
          'Intervals: 95% bootstrap over whole games. DESCRIPTIVE: no betting rule is selected here.\n',
          '## 1. Log loss and Brier, model vs market (same rows; diff = model minus market, positive means the market is better)\n', r(sc),
          '## 2. Same, by decision time (all stats pooled)\n', r(by_t.drop(columns=['stat'])),
          '## 2b. Information slope: realized hit - market probability regressed on (model - market), through the origin\n',
          "Slope 0 = the disagreement is noise, 1 = the model is right about all of it. `mean hit - market` is the sample's own bias in the market (negative = overs are overpriced).\n", r(slope),
          '## 3. Calibration at the book line, by how far the model is from the market\n',
          'If the model held information the market lacks, `hit - market` would move with the gap and `hit - model` would stay near 0. If the gap is bias or noise, `hit - model` moves instead.\n', r(dis)]
    for s, d in dis_stat.items():
        md += [f'### {s}\n', r(d)]
    md += ['## 4. Reliability of the model at the book line (deciles of model probability)\n', r(rel)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(md), encoding='utf-8')
    print(sc.round(4).to_string(index=False))
    print(dis.round(4).to_string(index=False))
    print(slope.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
