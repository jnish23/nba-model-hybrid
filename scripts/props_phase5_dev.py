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


def market_level(q):
    """One row per (game, player, stat, T, line): the median (average bettor) and best (line shopper) bettable price per side."""
    g = q.groupby(['game_id', 'player_id', 'stat', 'minutes_before', 'line'])
    return g.agg(over_med=('over_dec', 'median'), under_med=('under_dec', 'median'), over_best=('over_dec', 'max'), under_best=('under_dec', 'max'),
                 y=('y', 'first'), played=('played', 'first'), lbl_minutes=('lbl_minutes', 'first'), expected_minutes=('expected_minutes', 'first')).reset_index()


def naive(m, side, price, label, n_boot):
    """A no-model strategy: one fixed side on every market at T-60, at the median or the best price."""
    b = m[m['minutes_before'] == 60].copy()
    b['best_side'] = side
    b['dec_price'] = b[f'{side}_{price}']
    b['book'] = 'avg'
    b['best_ev'] = 0.0
    return be.roi_table(be.settle_all(b), label, n_boot=n_boot).query("treatment == 'stands'")


def decompose(q, k_by_stat, n_boot):
    """Where does the primary rule's ROI come from? No-model sides, the same selection at the median price, the market-only (k = 0) rule, and by quote age."""
    m = market_level(q)
    rows = [naive(m, 'under', 'med', 'ALL UNDERS at T-60, median price (no model)', n_boot), naive(m, 'under', 'best', 'ALL UNDERS at T-60, best price (no model)', n_boot),
            naive(m, 'over', 'med', 'ALL OVERS at T-60, median price (no model)', n_boot), naive(m, 'over', 'best', 'ALL OVERS at T-60, best price (no model)', n_boot)]
    prim = be.settle_all(rules.select(q, PRIMARY['e'], PRIMARY['time'], PRIMARY['min_books']))
    key = ['game_id', 'player_id', 'stat', 'minutes_before', 'line']
    rep = prim.merge(m[key + ['over_med', 'under_med']], on=key, how='left')
    rep['dec_price'] = np.where(rep['best_side'] == 'over', rep['over_med'], rep['under_med'])
    rows.append(be.roi_table(be.settle_all(rep.drop(columns=[c for c in rep.columns if c.startswith(('profit_', 'status_'))])), 'PRIMARY bets repriced at the MEDIAN bettable price', n_boot=n_boot).query("treatment == 'stands'"))
    q0 = rules.add_ev(q.drop(columns=['p_star', 'ev_over', 'ev_under', 'best_side', 'best_ev', 'dec_price']), {s: 0.0 for s in k_by_stat})
    rows.append(be.roi_table(be.settle_all(rules.select(q0, PRIMARY['e'], PRIMARY['time'], PRIMARY['min_books'])), 'MARKET-ONLY rule (k = 0): best price vs consensus fair probability, no model', n_boot=n_boot).query("treatment == 'stands'"))
    for lo_, hi_ in ((0, 5), (5, 15), (15, 30), (30, 60.1)):
        sub = prim[(prim['quote_age_min'] >= lo_) & (prim['quote_age_min'] < hi_)]
        if len(sub) > 200:
            rows.append(be.roi_table(sub, f'PRIMARY bets whose quote was {lo_}-{int(hi_)} minutes old', n_boot=n_boot).query("treatment == 'stands'"))
    for side in ('over', 'under'):
        sub = prim[prim['best_side'] == side]
        rows.append(be.roi_table(sub, f'PRIMARY bets, {side}s only', n_boot=n_boot).query("treatment == 'stands'"))
    q15 = q[q['quote_age_min'] <= 15]
    rows.append(be.roi_table(be.settle_all(rules.select(q15, PRIMARY['e'], PRIMARY['time'], PRIMARY['min_books'])), 'PRIMARY rule using only quotes at most 15 minutes old', n_boot=n_boot).query("treatment == 'stands'"))
    return pd.concat(rows, ignore_index=True).drop(columns=['treatment'])


PLACEBO_SEEDS = (11, 12, 13, 14)
TRIED = 35                                                   # 24 grid rules + 8 per-stat + 3 exposure caps


def placebo_and_caps(q, cons, k_by_stat, sel_primary, real_clv, n_boot, alpha_adj):
    """(1) Placebo model: each market's model probability swapped for a random other market's of the same stat, same k, same selection, same prices. Its ROI and CLV are the zero point.
    (2) Exposure caps on the primary selection: the same player-game often carries points, pr, pa and pra on one side."""
    rows = []
    for sd in PLACEBO_SEEDS:
        qp = rules.add_ev(be.placebo_quotes(q, sd), k_by_stat)
        sel = be.settle_all(rules.select(qp, PRIMARY['e'], PRIMARY['time'], PRIMARY['min_books']))
        roi = be.roi_table(sel, f'placebo {sd}', n_boot=n_boot).query("treatment == 'stands'").iloc[0]
        clv, counts = be.clv_table(sel, cons, n_boot=n_boot)
        same = clv[clv['measure'].str.startswith('consensus')].iloc[0]
        moved = clv[clv['measure'].str.startswith('line moved')].iloc[0]
        rows.append({'run': f'placebo {sd}', 'bets': int(roi['bets']), 'roi': roi['roi'], 'roi_lo': roi['lo'], 'roi_hi': roi['hi'], 'clv_prob_pts': 100 * same['mean'], 'clv_lo': 100 * same['lo'], 'clv_hi': 100 * same['hi'],
                     'line_move_pts': moved['mean']})
        del qp, sel
    pl = pd.DataFrame(rows)
    real_same = real_clv[real_clv['measure'].str.startswith('consensus')].iloc[0]
    real_move = real_clv[real_clv['measure'].str.startswith('line moved')].iloc[0]
    excess = pd.DataFrame([{'measure': 'CLV, consensus fair probability to T-5, line unchanged (points)', 'real': 100 * real_same['mean'], 'real_lo': 100 * real_same['lo'], 'placebo_mean': pl['clv_prob_pts'].mean(),
                            'placebo_max_hi': pl['clv_hi'].max(), 'excess': 100 * real_same['mean'] - pl['clv_prob_pts'].mean(), 'excess_conservative_lower': 100 * real_same['lo'] - pl['clv_hi'].max()},
                           {'measure': 'line move in the bettor\'s favour by T-5 (points)', 'real': real_move['mean'], 'real_lo': real_move['lo'], 'placebo_mean': pl['line_move_pts'].mean(),
                            'placebo_max_hi': np.nan, 'excess': real_move['mean'] - pl['line_move_pts'].mean(), 'excess_conservative_lower': np.nan}])
    caps = []
    for label, sel in (('primary, max 1 bet per player-game', rules.cap_per_player_game(sel_primary, 1)), ('primary, max 2 bets per player-game', rules.cap_per_player_game(sel_primary, 2)),
                       ('primary, max 2 bets per game', rules.cap_per_game(sel_primary, 2))):
        caps.append(be.roi_table(sel, label, n_boot=n_boot, alpha_adj=alpha_adj).query("treatment == 'stands'").assign(distinct_player_games=sel.groupby(['game_id', 'player_id']).ngroups))
    return pl, excess, pd.concat(caps, ignore_index=True)


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
                r = be.roi_table(sel, label, n_boot=a.n_boot, alpha_adj=0.05 / TRIED)
                results.append(r.query("treatment == 'stands'"))
                if (e, tm, mbk) == (PRIMARY['e'], PRIMARY['time'], PRIMARY['min_books']):
                    sel_primary = sel
    for stat in sorted(q['stat'].unique()):
        sel = be.settle_all(rules.select(q[q['stat'] == stat], PRIMARY['e'], PRIMARY['time'], PRIMARY['min_books']))
        if len(sel):
            results.append(be.roi_table(sel, f'primary, {stat} only', n_boot=a.n_boot, alpha_adj=0.05 / TRIED).query("treatment == 'stands'"))
    grid = pd.concat(results, ignore_index=True)
    prim_all = be.roi_table(sel_primary, 'PRIMARY', n_boot=a.n_boot)
    decomp = decompose(q, k_by_stat, a.n_boot)
    clv, clv_counts = be.clv_table(sel_primary, cons, n_boot=a.n_boot)
    placebo, excess, caps = placebo_and_caps(q, cons, k_by_stat, sel_primary, clv, a.n_boot, 0.05 / TRIED)
    expo = be.exposure(sel_primary)
    by_side = sel_primary.groupby('best_side').agg(bets=('best_ev', 'size'), mean_ev=('best_ev', 'mean')).reset_index()
    by_stat = sel_primary.groupby('stat').agg(bets=('best_ev', 'size'), mean_ev=('best_ev', 'mean'), profit=('profit_stands', 'sum')).reset_index()
    by_book = sel_primary.groupby('book').agg(bets=('best_ev', 'size'), profit=('profit_stands', 'sum')).reset_index()
    sel_primary.drop(columns=['snapshot_ts_utc']).to_csv(Path(a.out).with_suffix('.primary_bets.csv'), index=False)
    Path(a.out).with_suffix('.frozen_rule.json').write_text(json.dumps({'k_by_stat': k_by_stat, 'primary': PRIMARY, 'grid': {'e': GRID_E, 'time': GRID_TIME, 'min_books': GRID_BOOKS},
                                                                       'developed_on': a.season, 'tried_rules': TRIED}, indent=2), encoding='utf-8')

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
          "### Decomposition: where the primary rule's ROI comes from (settlement: bet stands)\n", r(decomp),
          f'## 3. {len(grid)} tried rules (+3 exposure caps above) (settlement: bet stands). `lo_adj/hi_adj` = interval at the Bonferroni level for {TRIED} rules\n', r(grid.drop(columns=['treatment']))]
    Path(a.out).write_text('\n'.join(md), encoding='utf-8')
    print(score.round(4).to_string(index=False)); print(slope.round(3).to_string(index=False)); print(prim_all.round(4).to_string(index=False)); print(clv.round(4).to_string(index=False), clv_counts); print(decomp.round(4).to_string(index=False)); print(placebo.round(3).to_string(index=False)); print(excess.round(3).to_string(index=False)); print(caps.round(4).to_string(index=False)); print(expo)


if __name__ == '__main__':
    main()
