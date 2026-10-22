"""
The D1.1 parity test (reports/props/d1_parity_preregistration.md and addendum1): rebuilds the model inputs of historical dates as if live and compares them with the stored training rows.

    python scripts/props_d1_parity.py --attempt 1 --dates reports/props/d1_parity_dates.json

Writes data/props_live_parity/attempt<N>/ (results.json, one csv per criterion table, the failure lists) and prints the criteria. It never modifies the dataset, a model or the rule file.
"""
import argparse
import hashlib
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props import schedule as S  # noqa: E402
from props.betting import evaluate as be, export, rules  # noqa: E402
from props.features import build, history, rows as rows_mod  # noqa: E402
from props.live import at_t as A, parity, roster  # noqa: E402
from props.live.base import build_live_base  # noqa: E402
from props.live.frame import live_frame  # noqa: E402
from props.live.predict import predict_rows  # noqa: E402
from props.models import data, outcome as oc, registry  # noqa: E402

DB, FDB = 'data/nba_stats.db', 'data/props_features_v3.db'
RULE = 'reports/props/phase5_revised_rule.frozen_rule.json'
SEASON = '2025-26'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_live_frames(dates, hist, pos, reports, tips, tables, sched, cons, rule):
    bases, blocks, unresolved_all = [], [], []
    for date in dates:
        t0 = time.time()
        g = sched[sched['et_game_date'] == date].rename(columns={'et_game_date': 'game_date'}).assign(season=SEASON)
        gids = set(g['game_id'])
        h = hist[hist['game_date'] < date]
        per_t = {}
        for T in build.GRID:
            rep = A.asof_reports(reports, tips, T)
            assert (pd.to_datetime(rep['report_ts']) <= rep['gameId'].map(tips) - pd.Timedelta(minutes=T + 15)).all()
            rp = A.report_players(rep, tips, T)
            rp = rp[rp['game_id'].isin(gids)]
            qp = cons[(cons['minutes_before'] == T) & cons['game_id'].isin(gids)][['game_id', 'player_id']].drop_duplicates()
            per_t[T] = (rep, roster.rule_members(rule, g, hist, date, rp, qp), pd.concat([rp, qp], ignore_index=True))
        extra = pd.concat([per_t[T][2] for T in build.GRID], ignore_index=True)
        base_all, unresolved = build_live_base(hist, pos, g, extra, date, DB)       # the superset: the registered roster plus every reported and quoted player
        unresolved_all += [(date,) + u for u in unresolved]
        for T, (rep, members, _) in per_t.items():
            base_T = base_all.merge(members, on=['game_id', 'player_id'], how='inner', validate='one_to_one')
            blocks.append(A.live_at_t(base_T, rep, tips, T, tables[T]))
        bases.append(base_all)
        logging.info(f'{date}: {len(g)} games, {len(base_all)} live rows, {time.time() - t0:.0f}s')
    return pd.concat(bases, ignore_index=True), blocks, unresolved_all


def roi_diagnostic(sel_live, sel_train):
    """Realised ROI per unit staked (settlement 'stands', the selected quote's price) of the bets made with live features, with training features, and of the bets only one side makes."""
    lab = export.labels_table(FDB)
    key = ['game_id', 'player_id', 'stat']

    def roi(df):
        b = be.settle_all(be.attach_outcomes(df.reset_index(drop=True), lab))
        st, pr = b['status_stands'].to_numpy(), b['profit_stands'].to_numpy(float)
        stake = np.isin(st, ['win', 'loss'])
        return {'bets': int(stake.sum()), 'roi': float(pr[stake].sum() / stake.sum()) if stake.sum() else float('nan'), 'units': float(pr.sum())}
    only = lambda a, b: a.merge(b[key], on=key, how='left', indicator=True).query("_merge == 'left_only'").drop(columns='_merge')
    return {'live': roi(sel_live), 'training': roi(sel_train), 'live_shared': roi(sel_live.merge(sel_train[key], on=key)), 'live_only': roi(only(sel_live, sel_train)), 'training_only': roi(only(sel_train, sel_live))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True)
    ap.add_argument('--dates', default='reports/props/d1_parity_dates.json')
    ap.add_argument('--season', default='2025-26')
    ap.add_argument('--roster-rule', default='registered', choices=roster.RULES)
    ap.add_argument('--no-model', action='store_true', help='development: only A1-A4')
    ap.add_argument('--model', default='v3_final_2025-26')
    ap.add_argument('--prices', default=None)
    ap.add_argument('--n-sims', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--limit-dates', type=int, default=None, help='development only: use the first N dates')
    a = ap.parse_args()
    global SEASON
    SEASON = a.season
    a.prices = a.prices or f'data/props_betting/prices_{SEASON}.pkl'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    out = Path(f'data/props_live_parity/attempt{a.attempt}')
    out.mkdir(parents=True, exist_ok=True)
    dates = sorted(sum(json.loads(Path(a.dates).read_text())['dates'].values(), []))[:a.limit_dates]
    logging.info(f'{len(dates)} dates; rule file sha256 {sha(RULE)[:12]}')
    hist = rows_mod.load_history(DB)
    pos = history.position_groups(DB)
    reports = build.load_linked_reports(DB)
    tips = build.reference_tips_et([SEASON])
    tables = A.play_tables(FDB, build.GRID)
    sched = S.load_schedule(SEASON)
    cons, bets = pickle.load(open(a.prices, 'rb'))
    cons['game_id'], bets['game_id'] = cons['game_id'].astype(str), bets['game_id'].astype(str)
    gids = set(sched[sched['et_game_date'].isin(dates)]['game_id'])
    cons, bets = cons[cons['game_id'].isin(gids)].reset_index(drop=True), bets[bets['game_id'].isin(gids)].reset_index(drop=True)

    base_live, blocks, unresolved = build_live_frames(dates, hist, pos, reports, tips, tables, sched, cons, a.roster_rule)
    live = live_frame(base_live, blocks)
    train = data.load_frame(FDB, gids)
    train['game_id'] = train['game_id'].astype(str)
    logging.info(f'live frame {live.shape}, training frame {train.shape}')
    res, cols = {'attempt': a.attempt, 'roster_rule': a.roster_rule, 'season': SEASON, 'dates': dates, 'model': a.model, 'rule_sha256': sha(RULE), 'n_sims': a.n_sims, 'seed': a.seed}, list(train.columns)

    traded = cons[parity.KEY].drop_duplicates()
    r1 = parity.a1_coverage(traded, live, train, cols)
    r1['failures'].to_csv(out / 'a1_failures.csv', index=False)
    m = parity.merge_frames(live, train)
    r2 = parity.a2_identical(m, cols)
    r2['table'].to_csv(out / 'a2_columns.csv', index=False)
    r3 = parity.a3_injury(m)
    r4 = parity.a4_availability(m)
    r4['table'].to_csv(out / 'a4_columns.csv', index=False)
    res.update({'unresolved_players': len(unresolved), 'intersection_rows': int(len(m)), 'live_rows': int(len(live)), 'training_rows': int(len(train))})

    if a.no_model:
        res.update({'A1': {k: v for k, v in r1.items() if k != 'failures'}, 'A2': {k: v for k, v in r2.items() if k != 'table'}, 'A3': r3, 'A4': {k: v for k, v in r4.items() if k != 'table'}})
        (out / 'results.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
        for k in ('A1', 'A2', 'A3', 'A4'):
            print(k, 'PASS' if res[k]['pass'] else 'FAIL', json.dumps({x: y for x, y in res[k].items() if x != 'pass'}, default=float)[:500])
        return
    # model rows: players who played, with a consensus quote, in both frames (A5) or in the training frame only (A6)
    quoted = traded
    tr_m = train[(train['lbl_played'] == 1) & (train['lbl_minutes'] > 0)].merge(quoted, on=parity.KEY, how='inner').sort_values(parity.KEY).reset_index(drop=True)
    both_keys = tr_m[parity.KEY].merge(live[parity.KEY], on=parity.KEY, how='inner')
    tr_i = tr_m.merge(both_keys, on=parity.KEY, how='inner').sort_values(parity.KEY).reset_index(drop=True)
    tr_only = tr_m.merge(both_keys, on=parity.KEY, how='left', indicator=True).query("_merge == 'left_only'").drop(columns='_merge').reset_index(drop=True)
    lv_i = live.merge(both_keys, on=parity.KEY, how='inner').sort_values(parity.KEY).reset_index(drop=True)
    assert len(tr_i) == len(lv_i)
    bundle = registry.load_bundle(a.model, SEASON)
    t0 = time.time()
    mm_t, pm_t = predict_rows(bundle, tr_i, a.n_sims, a.seed)
    mm_l, pm_l = predict_rows(bundle, lv_i, a.n_sims, a.seed)
    mm_o, pm_o = predict_rows(bundle, tr_only, a.n_sims, a.seed + 1)
    logging.info(f'predicted {len(tr_i)} shared + {len(tr_only)} training-only rows in {time.time() - t0:.0f}s')
    means = {s: (oc.pmf_mean(pm_l[s]), oc.pmf_mean(pm_t[s])) for s in parity.A5_MEAN_STATS}
    key = tr_i[parity.KEY].copy()
    key['i'] = np.arange(len(key))
    qc = cons.merge(key, on=parity.KEY, how='inner', validate='many_to_one')
    p = qc[parity.KEY + ['stat', 'consensus_line']].copy()
    p['p_live'], p['p_train'] = np.nan, np.nan
    for stat, gq in qc.groupby('stat'):
        i, line = gq['i'].to_numpy(), gq['consensus_line'].to_numpy(float)
        p.loc[gq.index, 'p_live'] = oc.prob_over_no_push(pm_l[stat][i], line)
        p.loc[gq.index, 'p_train'] = oc.prob_over_no_push(pm_t[stat][i], line)
    r5 = parity.a5_model(means, p)
    r5['p_over']['over_10pp'].to_csv(out / 'a5_over_10pp.csv', index=False)

    rule = json.loads(Path(RULE).read_text())
    prim = rule['primary']

    def decisions(idx_rows, mm, pm):
        idx = idx_rows[parity.KEY + ['season', 'game_date']].reset_index(drop=True).assign(minutes_mean=mm)
        q = be.quote_table(bets, idx, pm)
        q = q[q['stat'].isin(rule['stats'])].reset_index(drop=True)
        return rules.select(rules.add_ev(q, rule['k_by_stat']), prim['e'], prim['time'], prim['min_books'])

    sel_live = decisions(lv_i, mm_l, pm_l)
    idx_all = pd.concat([tr_i, tr_only], ignore_index=True)
    sel_train = decisions(idx_all, np.concatenate([mm_t, mm_o]), {s: np.concatenate([pm_t[s], pm_o[s]]) for s in oc.CAPS})
    r6 = parity.a6_decisions(sel_live, sel_train)
    sel_live.to_csv(out / 'a6_bets_live.csv', index=False)
    res['roi_diagnostic'] = roi_diagnostic(sel_live, sel_train)           # declared in addendum 2 as information, not a criterion
    sel_train.to_csv(out / 'a6_bets_train.csv', index=False)

    def clean(r):
        return {k: (v if not isinstance(v, (pd.DataFrame, pd.Series)) else None) for k, v in r.items() if k not in ('table', 'failures')}
    res.update({'A1': clean(r1), 'A2': clean(r2), 'A3': clean(r3), 'A4': clean(r4),
                'A5': {'means': r5['means'], 'p_over': {k: v for k, v in r5['p_over'].items() if k != 'over_10pp'} | {'n_over_10pp': int(len(r5['p_over']['over_10pp']))}, 'pass': r5['pass']},
                'A6': clean(r6)})
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    for k in ('A1', 'A2', 'A3', 'A4', 'A5', 'A6'):
        print(k, 'PASS' if res[k]['pass'] else 'FAIL', json.dumps({x: y for x, y in res[k].items() if x != 'pass'}, default=float)[:600])


if __name__ == '__main__':
    main()
