"""
Phase 5, DEVELOPMENT stage (one season, 2023-24): information test, shrinkage factors k, CLV and the pre-registered rule family.

    python scripts/props_phase5_dev.py --season 2023-24 --p5-version v3_dev_p5 --full-version v3_dev --carry-season 2022-23 --out reports/props/phase5_dev_2023-24.md

k (per stat, the information slope floored at 0 and capped at 0.5) and every threshold are chosen HERE and written to a JSON next to the report; the validation and final seasons
only read that file. Results on this season are in-sample for k and for the choice of rule, so they are optimistic by construction; the validation season is the check.
"""
import argparse
import json
import logging
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.betting import evaluate as be, rules  # noqa: E402
from props.models import evaluate as ev, market_bench as mb, registry  # noqa: E402
from props.quality import md_table  # noqa: E402

PRIMARY = {'e': 0.03, 'time': 'earliest', 'min_books': 3}
GRID_E, GRID_TIME, GRID_BOOKS = (0.02, 0.03, 0.05), ('earliest', '240', '60', '5'), (3, 5)


def load_labels(db):
    with sqlite3.connect(db) as conn:
        lab = pd.read_sql('SELECT game_id, player_id, lbl_played, lbl_minutes, y_points, y_rebounds, y_assists, y_threes, y_pra FROM player_game_labels', conn)
    lab['game_id'] = lab['game_id'].astype(str)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', required=True)
    ap.add_argument('--p5-version', required=True)
    ap.add_argument('--full-version', required=True)
    ap.add_argument('--carry-season', default=None)
    ap.add_argument('--prices', default=None)
    ap.add_argument('--features-db', default='data/props_features_v3.db')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-boot', type=int, default=1000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    prices_path = a.prices or f'data/props_betting/prices_{a.season}.pkl'
    cons, bets = pickle.load(open(prices_path, 'rb'))
    labels = load_labels(a.features_db)

    idx5, pm5 = registry.load_predictions(a.p5_version, a.season)
    full_rows, full_pm = ev.load_rows_light(a.full_version, a.season, a.features_db)
    carry = ev.load_rows_light(a.full_version, a.carry_season, a.features_db) if a.carry_season else None
    ratios = be.r2b_ratio_by_date(full_rows, full_pm, carry)
    pm5 = be.apply_r2b(idx5, pm5, ratios)                                   # the frozen configuration: R2b on top of the saved 20,000-draw pmfs
    logging.info('R2b applied')

    # --- 1. information test at the consensus line
    rows5 = idx5.merge(labels, on=['game_id', 'player_id'], how='left', validate='many_to_one')
    cmp = mb.compare(rows5, pm5, cons.rename(columns={'stat': 'market'}))
    score = mb.score_table(cmp, n_boot=a.n_boot)
    slope = mb.information_slope(cmp, n_boot=a.n_boot)
    k_by_stat = {r['stat']: rules.clip_k(r['slope']) for _, r in slope.iterrows() if r['stat'] != 'all'}
    logging.info(f'k by stat: {k_by_stat}')
    dis = mb.disagreement_table(cmp, n_boot=a.n_boot)

    # --- 2. quotes with model probabilities, EV, selection
    q = be.quote_table(bets, idx5, pm5)
    q = be.attach_outcomes(q, labels)
    q = rules.add_ev(q, k_by_stat)
    logging.info(f'{len(q)} bettable quotes with a model probability')

    results, sel_primary = [], None
    for e in GRID_E:
        for tm in GRID_TIME:
            for mbk in GRID_BOOKS:
                sel = be.settle_all(rules.select(q, e, tm, mbk))
                label = f'e={e}, time={tm}, min_books={mbk}'
                r = be.roi_table(sel, label, n_boot=a.n_boot, alpha_adj=0.05 / 32)
                results.append(r.query("treatment == 'stands'"))
                if (e, tm, mbk) == (PRIMARY['e'], PRIMARY['time'], PRIMARY['min_books']):
                    sel_primary = sel
    for stat in sorted(q['stat'].unique()):
        sel = be.settle_all(rules.select(q[q['stat'] == stat], PRIMARY['e'], PRIMARY['time'], PRIMARY['min_books']))
        if len(sel):
            results.append(be.roi_table(sel, f'primary, {stat} only', n_boot=a.n_boot, alpha_adj=0.05 / 32).query("treatment == 'stands'"))
    grid = pd.concat(results, ignore_index=True)
    prim_all = be.roi_table(sel_primary, 'PRIMARY', n_boot=a.n_boot)
    clv, clv_counts = be.clv_table(sel_primary, cons, n_boot=a.n_boot)
    by_side = sel_primary.groupby('best_side').agg(bets=('best_ev', 'size'), mean_ev=('best_ev', 'mean')).reset_index()
    by_stat = sel_primary.groupby('stat').agg(bets=('best_ev', 'size'), mean_ev=('best_ev', 'mean'), profit=('profit_stands', 'sum')).reset_index()
    by_book = sel_primary.groupby('book').agg(bets=('best_ev', 'size'), profit=('profit_stands', 'sum')).reset_index()
    sel_primary.drop(columns=['snapshot_ts_utc']).to_csv(Path(a.out).with_suffix('.primary_bets.csv'), index=False)
    Path(a.out).with_suffix('.frozen_rule.json').write_text(json.dumps({'k_by_stat': k_by_stat, 'primary': PRIMARY, 'grid': {'e': GRID_E, 'time': GRID_TIME, 'min_books': GRID_BOOKS},
                                                                       'developed_on': a.season, 'tried_rules': int(len(grid))}, indent=2), encoding='utf-8')

    r = lambda d, n=4: md_table(d.round(n), n)
    md = [f'# Phase 5 development stage: {a.season} (frozen configuration + R2b, 20,000-draw simulations)\n',
          'Everything here is in-sample for k and for the choice among the tried rules: results are optimistic by construction. The validation season (2024-25) is the check, run once with the frozen file '
          f'`{Path(a.out).with_suffix(".frozen_rule.json").name}`. Bettable books: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Unibet (Fanatics is absent before 2025). Prices are quotes stamped at or before each decision time, '
          'main line on the consensus line, no stale quotes; pushes refund; the market probability is the de-vigged consensus (at least 3 agreeing books, all books counted).\n',
          '## 1. Information test (model vs market at the book line)\n', r(score), '### Information slope and k\n', r(slope),
          f'k per stat (slope floored at 0, capped at {rules.K_CAP}): {k_by_stat}\n', '### Calibration by model-minus-market gap\n', r(dis),
          '## 2. Primary rule (e = 0.03, earliest time with positive EV, at least 3 books, pooled over stats), all settlement treatments\n', r(prim_all),
          f'CLV to T-5 (bets placed at T-240 or T-60): {clv_counts}\n', r(clv) if len(clv) else 'not enough bets with a T-5 quote\n',
          '### Primary rule by side, stat and book\n', r(by_side), r(by_stat), r(by_book),
          f'## 3. All {len(grid)} tried rules (settlement: bet stands). `lo_adj/hi_adj` = interval at the Bonferroni level for 32 rules\n', r(grid.drop(columns=['treatment']))]
    Path(a.out).write_text('\n'.join(md), encoding='utf-8')
    print(score.round(4).to_string(index=False)); print(slope.round(3).to_string(index=False)); print(prim_all.round(4).to_string(index=False)); print(clv.round(4).to_string(index=False), clv_counts)


if __name__ == '__main__':
    main()
