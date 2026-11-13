"""
Step S7 of reports/props/audit_2026-11/WP6_WP7_experiment_plan.md: experiments 7c (per-player-game caps and quarter-Kelly sizing) and 7d (unders on players with expected
minutes under 20), with the rule of the 6b choice (k_walkforward.json: candidate and parameters fitted on 2023-24; 2024-25 out of sample, 2023-24 in sample).

    PY scripts/a11_rule_eval.py --p5-version a11_r0_p5 --registry-root $REG --prices-dir $OUT/prices --k-file $OUT/results/6b/k_walkforward.json
                                --experiments 7c 7d --out $OUT/results/7cd --report $RPT/7cd/report.md [--features-db data/props_features_v3.db] [--quotes-dir $OUT/quotes]

S16 re-report on the final model (descriptive only): the same command with the final --p5-version and --rule-file <candidate_rule_a11.json> instead of --k-file.
Inputs and refusals as scripts/a11_k_estimation.py (shares its loaders). Writes only under --out and --report.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
sys.path.append(str(REPO / 'scripts'))
from props.betting import a11_rules as ar, evaluate as be, rules  # noqa: E402
import a11_k_estimation as ke  # noqa: E402

SEASONS = ('2023-24', '2024-25')
SCORE = '2024-25'
TRAIN = '2023-24'


def variants(sel):
    """flat (the rule's own bets, 1 unit), cap 1 and cap 2 per player-game (rules.cap_per_player_game, highest EV kept), and quarter-Kelly with the per-game cap (units)."""
    k = sel.copy()
    k['units'] = ar.kelly_units(k)
    return {'flat': sel, 'cap1': rules.cap_per_player_game(sel, 1), 'cap2': rules.cap_per_player_game(sel, 2), 'kelly': k}


def metrics_row(name, season, b):
    m = ar.daily_metrics(b, 'med')
    mb_ = ar.daily_metrics(b, 'best')
    return {'variant': name, 'season': season, 'bets': len(b), **m, 'roi_best': mb_['roi'], 'sharpe_best': mb_['sharpe']}


def by_stat(name, season, b):
    rows = []
    for s, g in b.groupby('stat'):
        rows.append({'variant': name, 'season': season, 'stat': s, 'bets': len(g), 'stakes': int(g['stake_stands'].sum()), 'roi_med': ar.roi(g, 'med'), 'roi_best': ar.roi(g, 'best')})
    return rows


def share_shared(sel):
    per = sel.groupby(['game_id', 'player_id'])['stat'].transform('size')
    return {'bets': int(len(sel)), 'share_of_bets_sharing_a_player_game': float((per > 1).mean()) if len(sel) else np.nan, **be.exposure(sel)} if len(sel) else {'bets': 0}


def load_rule(a):
    if a.k_file:
        kf = json.loads(Path(a.k_file).read_text(encoding='utf-8'))
        if kf.get('stop'):
            raise SystemExit(f'{a.k_file}: the C.5 stop condition was met; rule work stops there (plan C.5 item 6, F.9)')
        return kf['pick'], kf['params'], (kf['pmarket']['devig_method'], kf['pmarket']['consensus_set']), f'k_walkforward.json ({kf.get("train")} fit)'
    rf = json.loads(Path(a.rule_file).read_text(encoding='utf-8'))
    cand = rf['candidate']
    p = {'b': rf['k_by_stat']}
    if cand in ('C2', 'C4'):
        p['a'] = rf['intercept_by_stat']
    if cand == 'C3':
        p['a_pos'] = {s: v['a_pos'] for s, v in rf['intercept_by_stat'].items()}
        p['a_neg'] = {s: v['a_neg'] for s, v in rf['intercept_by_stat'].items()}
    if cand == 'C4':
        p['c'] = rf['c']
    return cand, p, (rf['pmarket']['devig_method'], rf['pmarket']['consensus_set']), f'candidate rule file (fit on {rf.get("developed_on")}; descriptive only)'


def run_7c(sels, n_boot, seed):
    met, stat_rows, pair_rows, decisions, shared = [], [], [], {}, {}
    m = {}
    for season, (sel, q) in sels.items():
        vs = variants(sel)
        shared[season] = share_shared(sel)
        bd = ar.PairedBoot(q['game_date'], n_boot, seed)
        for name, b in vs.items():
            m[(name, season)] = metrics_row(name, season, b)
            met.append(m[(name, season)])
            stat_rows += by_stat(name, season, b)
            bd.add_bets(name, b, 'med')
        for name in ('cap1', 'cap2', 'kelly'):
            d, db_ = bd.diff(name, 'flat'), bd.diff(name, 'flat', ar.ALPHA_BONF)
            pair_rows.append({'season': season, 'comparison': f'{name} - flat (median ROI)', **d, 'lo_bonf': db_['lo'], 'hi_bonf': db_['hi']})
    if all((v, s) in m for v in ('flat', 'cap1') for s in (SCORE, TRAIN)):
        for cap in ('cap1', 'cap2'):
            decisions[cap] = ar.c7_cap_decide(cap, m[(cap, SCORE)], m[('flat', SCORE)], m[(cap, TRAIN)], m[('flat', TRAIN)])
        decisions['kelly'] = ar.c7_kelly_decide(m[('kelly', SCORE)], m[('flat', SCORE)], m[('kelly', TRAIN)], m[('flat', TRAIN)])
    return pd.DataFrame(met), pd.DataFrame(stat_rows), pd.DataFrame(pair_rows), decisions, shared


def minutes_split(b, n_boot, seed, alpha=ar.ALPHA, universe=None):
    st = b['units'] * b['stake_stands']
    pr = b['units'] * b['profit_med_stands']
    em = b['expected_minutes'].to_numpy(float)
    lo_m, hi_m = em < ar.MINUTES_SPLIT, em >= ar.MINUTES_SPLIT
    uni = b['game_date'] if universe is None else universe
    pb, d = ar.subset_diff(uni, b['game_date'].to_numpy(), pr.to_numpy(), st.to_numpy(), lo_m, hi_m, n_boot, seed, alpha)
    return pb, d


def run_7d(sels, qs, n_boot, seed):
    rows, dec_in = [], {}
    pooled = {}
    for side in ('under', 'over'):
        parts = {s: sel[sel['best_side'] == side] for s, (sel, _) in sels.items()}
        allb = pd.concat(list(parts.values()), ignore_index=True)
        uni = pd.concat([q['game_date'] for _, q in sels.values()])
        for label, b, u in [('pooled', allb, uni)] + [(s, parts[s], sels[s][1]['game_date']) for s in sorted(parts)]:
            if not len(b):
                continue
            pb, d = minutes_split(b, n_boot, seed, universe=u)
            _, dbf = minutes_split(b, n_boot, seed, ar.ALPHA_BONF, universe=u)
            ra, rb = pb.roi('a'), pb.roi('b')
            row = {'side': side, 'sample': label, 'stakes_lt20': ra['stakes'], 'roi_lt20': ra['point'], 'lt20_lo': ra['lo'], 'lt20_hi': ra['hi'],
                   'stakes_ge20': rb['stakes'], 'roi_ge20': rb['point'], 'ge20_lo': rb['lo'], 'ge20_hi': rb['hi'],
                   'diff': d['point'], 'diff_lo': d['lo'], 'diff_hi': d['hi'], 'diff_lo_bonf': dbf['lo'], 'diff_hi_bonf': dbf['hi']}
            rows.append(row)
            if side == 'under':
                pooled[label] = d
    decision = ar.d7_decide(pooled['pooled'], {s: pooled[s] for s in pooled if s != 'pooled'}) if 'pooled' in pooled else None
    # no-model version: every under at T-60, median price, split at 20 expected minutes (descriptive)
    nm = []
    for s, q in qs.items():
        m = q.groupby(ar.MARKET_KEY).agg(under_med=('under_dec', 'median'), y=('y', 'first'), played=('played', 'first'), lbl_minutes=('lbl_minutes', 'first'),
                                         expected_minutes=('expected_minutes', 'first'), game_date=('game_date', 'first')).reset_index()
        m = m[m['minutes_before'] == 60].copy()
        m['best_side'], m['dec_price'], m['book'] = 'under', m['under_med'], 'avg'
        m = be.settle_all(m)
        stake = np.isin(m['status_stands'], ('win', 'loss')).astype(float)
        for lab, msk in (('<20', m['expected_minutes'] < ar.MINUTES_SPLIT), ('>=20', m['expected_minutes'] >= ar.MINUTES_SPLIT)):
            st = stake[msk.to_numpy()].sum()
            nm.append({'season': s, 'expected_minutes': lab, 'stakes': int(st), 'roi_med': float(m.loc[msk, 'profit_stands'].sum() / st) if st else np.nan})
    return pd.DataFrame(rows), decision, pd.DataFrame(nm)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--p5-version', required=True)
    ap.add_argument('--registry-root', required=True)
    ap.add_argument('--prices-dir', required=True)
    ap.add_argument('--features-db', default='data/props_features_v3.db')
    ap.add_argument('--k-file', default=None)
    ap.add_argument('--rule-file', default=None, help='S16 re-report: the candidate rule file instead of --k-file (descriptive)')
    ap.add_argument('--pmarket', default=None, help='choice.json; must agree with the k-file / rule file (default: theirs)')
    ap.add_argument('--quotes-dir', default=None)
    ap.add_argument('--seasons', nargs='+', default=list(SEASONS))
    ap.add_argument('--experiments', nargs='+', default=['7c', '7d'], choices=('7c', '7d'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', default=None)
    ap.add_argument('--n-boot', type=int, default=ar.N_BOOT)
    ap.add_argument('--seed', type=int, default=ar.SEED)
    a = ap.parse_args(argv)
    ar.check_seasons(a.seasons)
    for p in (a.out, a.report):
        if p:
            ar.guard_output(p, REPO)
    if bool(a.k_file) == bool(a.rule_file):
        ap.error('give exactly one of --k-file and --rule-file')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cand, params, pm_rule, source = load_rule(a)
    if a.pmarket:
        _, pm_arg = ke.read_choice(a.pmarket)
        if tuple(pm_arg) != tuple(pm_rule):
            raise SystemExit(f'--pmarket {pm_arg} differs from the rule\'s p_market {pm_rule}')
    labels = ke.load_labels(a.features_db)
    sels, qs = {}, {}
    for s in a.seasons:
        d = ke.load_season(s, a.p5_version, a.registry_root, a.prices_dir, labels, pm_rule, a.quotes_dir)
        q = d['q']
        sel = ar.price_and_settle(ar.select_candidate(q, cand, params), ar.market_prices(q))
        sels[s], qs[s] = (sel, q), q
    out_dir = ar.guard_output(a.out, REPO)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {'step': 'S7', 'candidate': cand, 'params': params, 'source': source, 'pmarket': list(pm_rule), 'seasons': a.seasons, 'n_boot': a.n_boot, 'seed': a.seed,
           'roles': {TRAIN: 'in sample (parameters fitted on it)', SCORE: 'out of sample'} if a.k_file else 'descriptive (final model)'}
    md = [f'# 7c / 7d on rule {cand} ({source}), model {a.p5_version}\n',
          f'Median-price ROI (primary), settlement `stands`, date-clustered bootstrap ({a.n_boot} resamples, seed {a.seed}), paired on the same resamples; Bonferroni level 0.05/{ar.N_COMPARISONS}. '
          f'{TRAIN} is in sample for the rule parameters, {SCORE} out of sample.\n']
    if '7c' in a.experiments:
        met, st, pairs, dec, shared = run_7c(sels, a.n_boot, a.seed)
        res['7c'] = {'metrics': met, 'by_stat': st, 'paired': pairs, 'decisions': dec, 'sharing': shared,
                     'parameters': {**ar.KELLY, 'cap_roi_tol': ar.CAP_ROI_TOL, 'kelly_sharpe_mult': ar.KELLY_SHARPE_MULT, 'kelly_roi_tol': ar.KELLY_ROI_TOL}}
        md += ['## 7c: caps and quarter-Kelly (f* = EV/(d-1) at the best price; stake = min(0.25 f*, 0.02) B, B = 100; per-game total at most 0.05 B, lower-EV bets cut first)\n',
               ke.r(met), '### Paired median-ROI differences against flat\n', ke.r(pairs), '### By stat\n', ke.r(st), f'### Bets sharing a player-game: {shared}\n']
        for name, d in dec.items():
            md += [f'### Keep rule, {name}: **{"ADOPT" if d["adopt"] else "not adopted"}**' + (' (Kelly: evaluation only, no adoption before the fill study, owner decision G9)' if name == 'kelly' else '') + '\n',
                   ke.r(ke.clause_table(d['clauses']))]
    if '7d' in a.experiments:
        tab, dec7d, nm = run_7d(sels, qs, a.n_boot, a.seed)
        res['7d'] = {'table': tab, 'decision': dec7d, 'no_model': nm}
        md += ['## 7d: under bets, expected minutes < 20 against >= 20 (the single test; unadjusted 95% bound as amended 2026-11-12; Bonferroni interval printed beside it)\n']
        if dec7d:
            md += [f'Keep (sub-hypothesis of the forward test): **{"KEEP" if dec7d["keep"] else "DROP"}**\n', ke.r(ke.clause_table(dec7d['clauses']))]
        md += ['### Splits (unders: the test; overs: descriptive)\n', ke.r(tab), '### No-model version: every under at T-60, median price (descriptive)\n', ke.r(nm),
               '\n`expected_minutes` = props.betting.evaluate.quote_table: idx["minutes_mean"] = MinutesModel.mean_of over the minutes distribution of the prediction rows.\n']
    for s, (sel, _) in sels.items():
        sel.drop(columns=[c for c in ('snapshot_ts_utc',) if c in sel.columns]).to_csv(out_dir / f'bets_{s}_{cand}.csv', index=False)
    (out_dir / 'rule_eval.json').write_text(json.dumps(ke._json(res), indent=2), encoding='utf-8')
    if a.report:
        ke._write(a.report, '\n'.join(md))
    print('\n'.join(md))
    return res


if __name__ == '__main__':
    main()
