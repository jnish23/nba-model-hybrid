"""
Steps S5, S6 and S16 of reports/props/audit_2026-11/WP6_WP7_experiment_plan.md (section C, WP6b: re-estimating k). Pure logic in props/betting/a11_rules.py.

    S5  reproduction check + baseline (frozen k):
        PY scripts/a11_k_estimation.py --mode reproduce --p5-version a11_r0_p5 --seasons 2023-24 2024-25 --registry-root $REG --prices-dir $OUT/prices
                                       --features-db data/props_features_v3.db --out $OUT/results/r0_reproduce.json [--frozen-rule config/props_frozen_rule.json]
    S6  walk-forward (fit 2023-24, score 2024-25):
        PY scripts/a11_k_estimation.py --mode walkforward --p5-version a11_r0_p5 --registry-root $REG --prices-dir $OUT/prices --pmarket $OUT/results/7a/choice.json
                                       --train 2023-24 --score 2024-25 --estimators E0 E1 E2 E3 E4 --out $OUT/results/6b --report $RPT/6b_k/report.md
    S16 final k and candidate rule file:
        PY scripts/a11_k_estimation.py --mode final --p5-version <final>_p5 --registry-root $REG --prices-dir $OUT/prices --pmarket $OUT/results/7a/choice.json
                                       --train 2023-24 2024-25 --k-file $OUT/results/6b/k_walkforward.json --write-rule $RPT/candidate_rule_a11.json

Inputs (read-only): <registry-root>/<p5-version>/<season>/index.csv + pred_<stat>.npy (registry.save_predictions); <prices-dir>/prices_<season>.pkl = pickle (cons, bets) as
props.betting.prices.load_season returns them; labels from player_game_labels of --features-db (opened mode=ro); --pmarket = choice.json of scripts/a11_devig_calibration.py
(devig_method, consensus_set, passed, reason); a non-current choice reads <quotes-dir>/<season>/cons_<devig>_<set>.pkl (scripts/a11_cache_quotes.py).
Writes only --out, --report and --write-rule (+ .sha256). Refuses 2025-26 and later, and any output under config/.
"""
import argparse
import json
import logging
import pickle
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
from props.betting import a11_rules as ar, evaluate as be, rules  # noqa: E402
from props.live import monitor as mon  # noqa: E402
from props.models import evaluate as ev, market_bench as mb, registry  # noqa: E402
from props.quality import md_table  # noqa: E402

FROZEN_RULE = REPO / 'config' / 'props_frozen_rule.json'
REPRO_TOL = 0.05


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Loading (read-only)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_labels(db):
    with sqlite3.connect(f'file:{Path(db).resolve().as_posix()}?mode=ro', uri=True) as conn:
        lab = pd.read_sql('SELECT game_id, player_id, lbl_played, lbl_minutes, y_points, y_rebounds, y_assists, y_threes, y_pra FROM player_game_labels', conn)
    lab['game_id'] = lab['game_id'].astype(str)
    return lab


def read_choice(path):
    if not path:
        return None, ar.CURRENT_PMARKET
    choice = json.loads(Path(path).read_text(encoding='utf-8'))
    for k in ('devig_method', 'consensus_set', 'passed'):
        if k not in choice:
            raise SystemExit(f'{path}: missing key {k!r} (expected devig_method, consensus_set, passed, reason)')
    return choice, ar.pmarket_from_choice(choice)


def load_season(season, p5_version, registry_root, prices_dir, labels, pmarket=ar.CURRENT_PMARKET, quotes_dir=None, stats=ar.BET_STATS):
    """Returns dict(season, cons, cmp, q, n_bets_dropped_by_pmarket, prices_path). cmp: market_bench.compare rows + game_date; q: bettable quotes with p_mod, push, outcomes, game_date."""
    ar.check_season(season)
    ppath = Path(prices_dir) / f'prices_{season}.pkl'
    with open(ppath, 'rb') as f:
        cons, bets = pickle.load(f)
    cons, bets = cons.copy(), bets.copy()
    cons['game_id'], bets['game_id'] = cons['game_id'].astype(str), bets['game_id'].astype(str)
    dropped = 0
    if tuple(pmarket) != ar.CURRENT_PMARKET:
        if not quotes_dir:
            raise SystemExit(f'p_market {pmarket} is not the current method: --quotes-dir (the a11_cache_quotes.py --out folder) is required')
        vpath = Path(quotes_dir) / season / f'cons_{pmarket[0]}_{pmarket[1]}.pkl'
        cons, bets, dropped = ar.apply_pmarket(cons, bets, pd.read_pickle(vpath))
        logging.info(f'{season}: p_market {pmarket} from {vpath}; {dropped} bettable quotes dropped (not on the variant consensus line of the same snapshot)')
    if 'game_date' in cons.columns:
        ar.check_dates(cons['game_date'], f'{ppath} consensus')
    idx, pm = registry.load_predictions(p5_version, season, root=registry_root)
    idx['game_id'] = idx['game_id'].astype(str)
    ar.check_dates(idx['game_date'], f'{p5_version}/{season} predictions')
    rows = idx.merge(labels, on=['game_id', 'player_id'], how='left', validate='many_to_one')
    cmp = mb.compare(rows, pm, cons.rename(columns={'stat': 'market'}))
    gdate = idx.drop_duplicates('game_id').set_index('game_id')['game_date'].astype(str).str[:10]
    cmp['game_date'] = cmp['game_id'].astype(str).map(gdate)
    q = be.attach_outcomes(be.quote_table(bets, idx, pm), labels)
    q['game_date'] = idx['game_date'].astype(str).str[:10].to_numpy()[q['i'].to_numpy()]
    q = q[q['stat'].isin(stats)].reset_index(drop=True)
    logging.info(f'{season}: {len(cmp)} comparison rows, {len(q)} bettable quotes of {list(stats)}')
    return {'season': season, 'cons': cons, 'cmp': cmp, 'q': q, 'n_bets_dropped_by_pmarket': dropped, 'prices_path': str(ppath)}


def code_commit():
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10, cwd=REPO).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'props/betting/a11_rules.py', 'scripts/a11_k_estimation.py', 'scripts/a11_rule_eval.py'],
                                    capture_output=True, text=True, timeout=10, cwd=REPO).stdout.strip())
        return head, dirty
    except Exception:
        return '', None


def manifest_of(version, root):
    p = Path(root) / version / 'manifest.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def slope_frame(cmp):
    """props.live.monitor.information_slope input from compare rows."""
    return cmp.rename(columns={'hit': 'hit_over'})[['stat', 'game_date', 'hit_over', 'p_market', 'p_model']]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Scoring candidates on one season
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def score_rules(q, cand_params, n_boot=ar.N_BOOT, seed=ar.SEED):
    """Every candidate selected and settled on the same quotes; one PairedBoot over all game dates (primary) and one over all games (secondary)."""
    m = ar.market_prices(q)
    bets = {c: ar.price_and_settle(ar.select_candidate(q, c, p), m) for c, p in cand_params.items()}
    bd = ar.PairedBoot(q['game_date'], n_boot, seed)
    bg = ar.PairedBoot(q['game_id'], n_boot, seed)
    for c, b in bets.items():
        for price in ('med', 'best', 'flat'):
            bd.add_bets(f'{c}|{price}', b, price)
        bd.add_bets(f'{c}|med|ee_flagged', b, 'med', 'early_exit_flagged_books')
        bd.add_bets(f'{c}|med|ee_all', b, 'med', 'early_exit_all_books')
        bg.add_bets(f'{c}|med', b, 'med', cluster='game_id')
        bg.add_bets(f'{c}|best', b, 'best', cluster='game_id')
    return bets, bd, bg


def rule_table(bets, bd, bg):
    rows = []
    for c, b in bets.items():
        r = {'rule': c, 'bets': len(b), 'stakes': int(b['stake_stands'].sum()), 'overs': int((b['best_side'] == 'over').sum()), 'unders': int((b['best_side'] == 'under').sum()),
             'mean_claimed_ev': float(b['best_ev'].mean()) if len(b) else np.nan}
        for price in ('med', 'best'):
            x, xb, xg = bd.roi(f'{c}|{price}'), bd.roi(f'{c}|{price}', ar.ALPHA_BONF), bg.roi(f'{c}|{price}')
            r.update({f'roi_{price}': x['point'], f'{price}_lo': x['lo'], f'{price}_hi': x['hi'], f'{price}_lo_bonf': xb['lo'], f'{price}_hi_bonf': xb['hi'],
                      f'{price}_lo_game': xg['lo'], f'{price}_hi_game': xg['hi']})
        r['roi_flat'] = bd.point(f'{c}|flat')
        r['roi_med_ee_flagged'] = bd.point(f'{c}|med|ee_flagged')
        r['roi_med_ee_all'] = bd.point(f'{c}|med|ee_all')
        rows.append(r)
    return pd.DataFrame(rows)


def all_diffs(names, bd, bg, price='med'):
    out = {}
    for a in names:
        for b in names:
            if a != b:
                d = bd.diff(f'{a}|{price}', f'{b}|{price}')
                db_ = bd.diff(f'{a}|{price}', f'{b}|{price}', ar.ALPHA_BONF)
                dg = bg.diff(f'{a}|{price}', f'{b}|{price}')
                out[(a, b)] = {**d, 'lo_bonf': db_['lo'], 'hi_bonf': db_['hi'], 'lo_game': dg['lo'], 'hi_game': dg['hi']}
    return out


def diff_table(diffs, pairs):
    return pd.DataFrame([{'comparison': f'{a} - {b}', **diffs[(a, b)]} for a, b in pairs if (a, b) in diffs])


def segment_table(bets, col):
    rows = []
    for c, b in bets.items():
        if not len(b):
            continue
        key = ar.ev_bucket(b['best_ev']).astype(str) if col == 'ev_bucket' else b[col].astype(str)
        for k, g in b.groupby(key.to_numpy()):
            rows.append({'rule': c, col: k, 'bets': len(g), 'stakes': int(g['stake_stands'].sum()), 'roi_med': ar.roi(g, 'med'), 'roi_best': ar.roi(g, 'best')})
    return pd.DataFrame(rows)


def calib_table(bets):
    rows = []
    for c, b in bets.items():
        if not len(b):
            continue
        for side, x in ar.calibration_by_side(b).items():
            rows.append({'rule': c, 'side': side, 'n': x.get('n', 0), 'actual_wins': x.get('actual_wins'), 'expected_wins_pstar': x.get('expected_wins_pstar'), 'diff': x.get('diff'),
                         'diff_se': x.get('diff_se'), 'z': x.get('z'), 'within_2se': x.get('within_2se'), 'brier_pstar': x.get('brier_pstar')})
    return pd.DataFrame(rows)


def clv_rows(bets, cons):
    rows = []
    for c, b in bets.items():
        if len(b):
            t, counts = be.clv_table(b, cons, n_boot=500)
            for _, r in t.iterrows():
                rows.append({'rule': c, **r.to_dict(), **{f'n_{k}': v for k, v in counts.items()}})
    return pd.DataFrame(rows)


def placebo_clv(q, cand, params, cons, seeds=(11, 12, 13, 14)):
    """A.4.4: CLV of the same candidate on placebo quotes (p_mod, push swapped with a random other market of the same stat; props.betting.evaluate.placebo_quotes)."""
    rows = []
    m = ar.market_prices(q)
    for sd in seeds:
        sel = ar.price_and_settle(ar.select_candidate(be.placebo_quotes(q, sd), cand, params), m)
        t, _ = be.clv_table(sel, cons, n_boot=200)
        same = t[t['measure'].str.startswith('consensus')]
        rows.append({'seed': sd, 'bets': len(sel), 'roi_med': ar.roi(sel, 'med'), 'clv_prob_pts': 100 * float(same['mean'].iloc[0]) if len(same) else np.nan})
    return pd.DataFrame(rows)


def fit_tables(fits):
    out = []
    for e, f in fits.items():
        t = f['by_stat'].copy()
        t.insert(0, 'estimator', e)
        if f['c'] is not None:
            t['c'] = f['c']
        out.append(t)
    return out


def _json(o):
    if isinstance(o, dict):
        return {(f'{k[0]} - {k[1]}' if isinstance(k, tuple) else str(k)): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return _json(o.to_dict(orient='records'))
    return o


def _write(path, text):
    p = ar.guard_output(path, REPO)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    return p


def r(d, n=4):
    return md_table(d.round(n) if len(d) else d, n)


def clause_table(clauses):
    return pd.DataFrame([{'item': c['item'], 'clause': c['clause'], 'value': c['value'], 'threshold': c['threshold'],
                          'result': 'not reached' if not c['reached'] else ('decision' if c['value'] is None and c['threshold'] is None else ('PASS' if c['holds'] else 'FAIL'))}
                         for c in clauses])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def reproduce(a, labels):
    frozen = json.loads(Path(a.frozen_rule).read_text(encoding='utf-8'))
    stats = frozen.get('stats') or list(ar.BET_STATS)
    data = [load_season(s, a.p5_version, a.registry_root, a.prices_dir, labels, stats=stats) for s in a.seasons]
    cmp = pd.concat([d['cmp'].assign(season=d['season']) for d in data], ignore_index=True)
    pooled = mb.information_slope(cmp, n_boot=a.n_boot)                                   # the frozen method (game-clustered interval), as scripts/props_phase5_revise.py
    pooled_date = mon.information_slope(slope_frame(cmp), n_boot=a.n_boot, seed=a.seed)
    rows, gate = [], True
    for _, x in pooled.iterrows():
        s = x['stat']
        fz = frozen.get('pooled_slopes', {}).get(s, [np.nan])[0]
        k_new = rules.clip_k(x['slope'])
        k_fz = frozen['k_by_stat'].get(s, np.nan)
        bet = s in stats
        ok_k = bool(abs(k_new - k_fz) <= REPRO_TOL) if bet else None
        ok_s = bool(abs(x['slope'] - fz) <= REPRO_TOL) if bet else None
        if bet:
            gate = gate and ok_k and ok_s
        dd = pooled_date[pooled_date['stat'] == ('pooled' if s == 'all' else s)]
        rows.append({'stat': s, 'bet_stat': bet, 'n': int(x['n']), 'slope_new': x['slope'], 'lo_game': x['slope_lo'], 'hi_game': x['slope_hi'],
                     'lo_date': float(dd['so_lo'].iloc[0]) if len(dd) else np.nan, 'hi_date': float(dd['so_hi'].iloc[0]) if len(dd) else np.nan,
                     'slope_frozen': fz, 'slope_diff': x['slope'] - fz, 'k_new': k_new, 'k_frozen': k_fz, 'k_diff': k_new - k_fz if bet else np.nan,
                     'within_0.05': (ok_k and ok_s) if bet else None})
    table = pd.DataFrame(rows)
    per_season = pd.concat([mb.information_slope(d['cmp'], n_boot=500).assign(season=d['season']) for d in data])
    k_by_stat = {s: float(v) for s, v in frozen['k_by_stat'].items() if s in stats}
    base_rows, base_by = [], {}
    params = {'b': k_by_stat}
    for d in data:
        bets, bd, bg = score_rules(d['q'], {'C0': params}, a.n_boot, a.seed)
        t = rule_table(bets, bd, bg).assign(season=d['season'])
        base_rows.append(t)
        base_by[d['season']] = t.to_dict(orient='records')
        # cross-check: the same selection as the frozen path (rules.add_ev + rules.select)
        ref = rules.select(rules.add_ev(d['q'], k_by_stat), ar.PRIMARY['e'], ar.PRIMARY['time'], ar.PRIMARY['min_books'])
        kk = ['game_id', 'player_id', 'stat', 'minutes_before', 'book', 'best_side']
        mine = bets['C0'][kk].sort_values(kk).reset_index(drop=True)
        if not mine.equals(ref[kk].sort_values(kk).reset_index(drop=True)):
            raise SystemExit('C0 selection with the frozen k differs from rules.add_ev + rules.select')
    base = pd.concat(base_rows, ignore_index=True)
    out = {'step': 'S5', 'mode': 'reproduce', 'gate': 'PASS' if gate else 'FAIL', 'tolerance': REPRO_TOL, 'p5_version': a.p5_version, 'seasons': a.seasons,
           'frozen_rule': str(a.frozen_rule), 'table': table, 'per_season': per_season, 'baseline_frozen_k': base_by, 'n_comparisons': int(len(cmp)),
           'code_commit': code_commit(), 'p5_manifest': manifest_of(a.p5_version, a.registry_root), 'prices': [d['prices_path'] for d in data]}
    _write(a.out, json.dumps(_json(out), indent=2))
    md = [f'# S5 reproduction check ({a.p5_version}, {" + ".join(a.seasons)} pooled)\n',
          f'Gate (plan E.1 S5 / F.4): for every bet stat, the pooled through-origin slope within +/-{REPRO_TOL} of the frozen `pooled_slopes` and clip_k(slope) within +/-{REPRO_TOL} of the '
          f'frozen `k_by_stat`. **{out["gate"]}**\n', r(table), '### Per season (through origin, game-clustered)\n', r(per_season),
          '### Baseline: frozen k, primary rule, best-price selection; ROI at median (primary) and best price; date-clustered (primary), game-clustered, Bonferroni (0.05/23)\n', r(base)]
    rep = a.report or str(Path(a.out).with_suffix('.md'))
    _write(rep, '\n'.join(md))
    print(table.round(4).to_string(index=False))
    print(base.round(4).to_string(index=False))
    print(f'S5 reproduction gate: {out["gate"]}')
    return out


def walkforward(a, labels):
    for e in ('E0', 'E1', 'E2', 'E3'):
        if e not in a.estimators:
            raise SystemExit(f'--estimators must include E0, E1, E2 and E3 (candidates C0-C4 and P0 need them); got {a.estimators}')
    choice, pmarket = read_choice(a.pmarket)
    tr = load_season(a.train[0], a.p5_version, a.registry_root, a.prices_dir, labels, pmarket, a.quotes_dir)
    sc = load_season(a.score, a.p5_version, a.registry_root, a.prices_dir, labels, pmarket, a.quotes_dir)
    xy_tr, xy_sc = ar.xy_frame(tr['cmp']), ar.xy_frame(sc['cmp'])
    fits = {e: ar.fit_estimator(xy_tr, e, n_boot=a.n_boot, seed=a.seed) for e in a.estimators}
    pars = {e: ar.params_from_fit(f) for e, f in fits.items()}
    cands = list(ar.CANDIDATES) + (['C1_E4'] if 'E4' in pars else [])
    cpar = {c: ar.candidate_params(c, pars) for c in cands}
    # 2024-25 out of sample
    bets, bd, bg = score_rules(sc['q'], cpar, a.n_boot, a.seed)
    names = list(ar.CANDIDATES)
    diffs = all_diffs(names, bd, bg)
    cal = ar.calibration_by_side(bets['C2']) if len(bets['C2']) else {'over': {'n': 0, 'within_2se': False}, 'under': {'n': 0, 'within_2se': False}}
    bucket7 = {c: ar.bucket_roi(bets[c]) for c in names}
    stop_fits = {'E1': ar.fit_estimator(xy_sc, 'E1', stats=list(ar.BET_STATS), n_boot=a.n_boot, seed=a.seed),
                 'E2': ar.fit_estimator(xy_sc, 'E2', stats=list(ar.BET_STATS), n_boot=a.n_boot, seed=a.seed),
                 'E3': ar.fit_estimator(xy_sc, 'E3', stats=list(ar.BET_STATS), c=fits['E3']['c'], n_boot=a.n_boot, seed=a.seed)}
    stop_slopes = {e: f['pooled'] for e, f in stop_fits.items()}
    dec = ar.c5_decide(diffs, cal, bucket7, stop_slopes)
    # 2023-24 in sample (reported, decides nothing)
    bets_in, bd_in, bg_in = score_rules(tr['q'], cpar, a.n_boot, a.seed)
    # secondaries
    slope_sc = mon.information_slope(slope_frame(sc['cmp']), n_boot=a.n_boot, seed=a.seed)
    slope_sc4 = mon.information_slope(slope_frame(sc['cmp'][sc['cmp']['stat'].isin(ar.BET_STATS)]), n_boot=a.n_boot, seed=a.seed)
    score_sc = mb.score_table(sc['cmp'], n_boot=500)
    tab = rule_table(bets, bd, bg)
    tab_in = rule_table(bets_in, bd_in, bg_in)
    key_pairs = [('C1', 'C0'), ('C2', 'C1'), ('C2', 'P0'), ('C3', 'C0'), ('C3', 'C1'), ('C3', 'C2'), ('C4', 'C0'), ('C4', 'C1'), ('C4', 'C2'), ('C4', 'C3'),
                 ('C0', 'P0'), ('C1', 'P0'), ('C3', 'P0'), ('C4', 'P0'), ('P0', 'P00'), ('C0', 'P00')]
    dtab = diff_table(diffs, key_pairs)
    if 'C1_E4' in bets:
        e4 = bd.diff('C1_E4|med', 'C1|med')
        dtab = pd.concat([dtab, pd.DataFrame([{'comparison': 'C1_E4 - C1 (per-slot k, reported only)', **e4}])], ignore_index=True)
    dtab_in = diff_table(all_diffs(names, bd_in, bg_in), key_pairs)
    calt = calib_table(bets)
    seg = {col: segment_table(bets, col) for col in ('ev_bucket', 'stat', 'minutes_before', 'best_side')}
    clv = clv_rows({c: bets[c] for c in names}, sc['cons'])
    pick = dec['pick']
    plc = placebo_clv(sc['q'], pick, cpar[pick], sc['cons'])
    head, dirty = code_commit()
    out_dir = Path(a.out)
    ar.guard_output(out_dir, REPO)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c, b in bets.items():
        b.drop(columns=[x for x in ('snapshot_ts_utc',) if x in b.columns]).to_csv(out_dir / f'bets_{a.score}_{c}.csv', index=False)
    kw = {'step': 'S6', 'mode': 'walkforward', 'train': a.train, 'score': a.score, 'p5_version': a.p5_version, 'registry_root': str(a.registry_root),
          'pmarket': {'devig_method': pmarket[0], 'consensus_set': pmarket[1], 'choice_file': a.pmarket, 'choice': choice},
          'pick': pick, 'estimator': dec['estimator'], 'stop': dec['stop'], 'params': cpar[pick], 'c': pars.get('E3', {}).get('c'),
          'candidate_params': cpar, 'params_by_estimator': pars, 'pooled_train': {e: f['pooled'] for e, f in fits.items()}, 'stop_slopes': stop_slopes,
          'clauses': dec['clauses'], 'diffs_med': diffs, 'calibration_C2': cal, 'bucket7_med': bucket7,
          'thresholds': {'c1_worse_point': ar.C1_WORSE_POINT, 'adopt_margin': ar.ADOPT_MARGIN, 'c3_min_lo': ar.C3_MIN_LO, 'cal_max_se': ar.CAL_MAX_SE, 'amended': '2026-11-12'},
          'n_boot': a.n_boot, 'seed': a.seed, 'code_commit': head, 'code_dirty': dirty, 'p5_manifest': manifest_of(a.p5_version, a.registry_root),
          'prices': [tr['prices_path'], sc['prices_path']], 'bets_dropped_by_pmarket': {tr['season']: tr['n_bets_dropped_by_pmarket'], sc['season']: sc['n_bets_dropped_by_pmarket']}}
    (out_dir / 'k_walkforward.json').write_text(json.dumps(_json(kw), indent=2), encoding='utf-8')
    md = [f'# 6b: re-estimating k, walk-forward (fit {a.train[0]}, score {a.score}), model {a.p5_version}\n',
          f'p_market: {pmarket[0]} de-vig, {pmarket[1]} consensus (from {a.pmarket or "default: current method"}). Primary metric: ROI per unit staked at the MEDIAN bettable price, '
          f'settlement `stands`, bets selected on best-price EV (e = {ar.PRIMARY["e"]}, earliest slot, {ar.PRIMARY["min_books"]} books). Intervals: {a.n_boot} date resamples, '
          f'seed {a.seed}, percentile; differences are paired on the same resamples (ROI of each rule within each resample). Bonferroni level 0.05/{ar.N_COMPARISONS}. '
          'Thresholds as amended 2026-11-12 (+0.25 points in C.5 items 2-4).\n',
          f'## Decision (C.5): pick **{pick}** (estimator {dec["estimator"]}); stop condition: **{"STOP" if dec["stop"] else "not met"}**\n', r(clause_table(dec['clauses'])),
          f'## Estimators fitted on {a.train[0]} (slopes clipped by rules.clip_k; intercepts re-fitted at the clipped slope)\n',
          *[r(t) for t in fit_tables(fits)], '### E3 profile (total residual sum of squares over the bet stats)\n', r(fits['E3']['profile']),
          f'### Pooled slopes on {a.train[0]} (bet stats and others as fitted)\n', r(pd.DataFrame([{'estimator': e, **{k: v for k, v in f['pooled'].items() if k != 'stats'}} for e, f in fits.items()])),
          f'## {a.score} out of sample: rules\n', r(tab), '### Paired differences, median price (date 95%, Bonferroni, game 95%)\n', r(dtab),
          '### Calibration by side (expected wins under p* vs actual, date-clustered SE)\n', r(calt),
          '### By claimed-EV bucket, stat, slot, side\n', *[r(t) for t in seg.values()],
          f'### Stop-condition slopes on {a.score} (pooled over the bet stats, with intercepts)\n', r(pd.DataFrame([{'estimator': e, **{k: v for k, v in s.items() if k != 'stats'}} for e, s in stop_slopes.items()])),
          f'### Information slope with intercept on {a.score} (props.live.monitor.information_slope; all stats, then the bet stats)\n', r(slope_sc), r(slope_sc4),
          f'### Log loss and Brier at the book line, {a.score}\n', r(score_sc),
          '### CLV to the T-5 consensus (game-clustered, props.betting.evaluate.clv_table)\n', r(clv), f'### Placebo CLV/ROI for the pick {pick} (seeds 11-14)\n', r(plc),
          f'## {a.train[0]} IN-SAMPLE (k fitted on this season; decides nothing)\n', r(tab_in), r(dtab_in),
          '\nNot reported here: ROI by blowout (|final margin| >= 20) needs the final margin, which the loaded inputs do not carry.\n']
    if a.report:
        _write(a.report, '\n'.join(md))
    print(clause_table(dec['clauses']).to_string(index=False))
    print(tab.round(4).to_string(index=False))
    print(f'pick {pick}, estimator {dec["estimator"]}, stop {dec["stop"]}')
    return kw


def final(a, labels):
    choice, pmarket = read_choice(a.pmarket)
    kf = json.loads(Path(a.k_file).read_text(encoding='utf-8')) if a.k_file else None
    if kf is not None:
        if kf.get('stop'):
            raise SystemExit(f'{a.k_file}: the C.5 stop condition was met; no candidate rule is written (plan F.9)')
        cand = kf['pick']
        if a.candidate and a.candidate != cand:
            raise SystemExit(f'--candidate {a.candidate} differs from the k-file pick {cand}')
        kp = (kf['pmarket']['devig_method'], kf['pmarket']['consensus_set'])
        if tuple(kp) != tuple(pmarket):
            raise SystemExit(f'--pmarket {pmarket} differs from the k-file p_market {kp}')
    else:
        if not a.candidate:
            raise SystemExit('--mode final needs --k-file or --candidate (C0-C4)')
        cand = a.candidate
    est = ar.CAND_ESTIMATOR[cand]
    if a.estimator and a.estimator != est:
        raise SystemExit(f'--estimator {a.estimator} does not match candidate {cand} (estimator {est})')
    data = [load_season(s, a.p5_version, a.registry_root, a.prices_dir, labels, pmarket, a.quotes_dir) for s in a.train]
    cmp = pd.concat([d['cmp'] for d in data], ignore_index=True)
    xy = ar.xy_frame(cmp)
    need = {'E0': ['E0'], 'E1': ['E1'], 'E2': ['E2'], 'E3': ['E3']}[est]
    fits = {e: ar.fit_estimator(xy, e, n_boot=a.n_boot, seed=a.seed) for e in need}
    pars = {e: ar.params_from_fit(f) for e, f in fits.items()}
    params = ar.candidate_params(cand, pars)                                           # reads only the chosen estimator's parameters
    origin = mb.information_slope(cmp, n_boot=a.n_boot)
    pooled_slopes = {x['stat']: [round(x['slope'], 4), round(x['slope_lo'], 4), round(x['slope_hi'], 4)] for _, x in origin.iterrows()}
    head, dirty = code_commit()
    man = manifest_of(a.p5_version, a.registry_root)
    fsha = registry.sha256_file(a.features_db)
    if man.get('features_db_sha256') and man['features_db_sha256'] != fsha:
        logging.warning(f'labels DB sha256 {fsha[:12]} differs from the prediction manifest features_db_sha256 {man["features_db_sha256"][:12]}')
    rule = ar.candidate_rule(cand, params, ar.BET_STATS, a.train, pooled_slopes, head, man.get('features_db_sha256') or fsha, len(cmp), a.p5_version, pmarket,
                             extra={'code_dirty': dirty, 'labels_db_sha256': fsha, 'fit_by_stat': fits[est]['by_stat'].to_dict(orient='records'),
                                    'k_file': a.k_file, 'source_step': 'S16'})
    p, side, h = ar.write_rule_file(a.write_rule, _json(rule), REPO, overwrite=a.overwrite)
    print(json.dumps(_json({k: rule[k] for k in ('candidate', 'estimator', 'k_by_stat')}), indent=2))
    print(f'wrote {p} (sha256 {h}) and {side}')
    return rule


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', required=True, choices=('reproduce', 'walkforward', 'final'))
    ap.add_argument('--p5-version', required=True)
    ap.add_argument('--registry-root', required=True)
    ap.add_argument('--prices-dir', required=True)
    ap.add_argument('--features-db', default='data/props_features_v3.db')
    ap.add_argument('--seasons', nargs='+', default=['2023-24', '2024-25'], help='reproduce: seasons pooled')
    ap.add_argument('--train', nargs='+', default=['2023-24'])
    ap.add_argument('--score', default='2024-25')
    ap.add_argument('--estimators', nargs='+', default=['E0', 'E1', 'E2', 'E3', 'E4'])
    ap.add_argument('--estimator', default=None, help='final: E0-E3 (checked against the candidate)')
    ap.add_argument('--candidate', default=None, choices=('C0', 'C1', 'C2', 'C3', 'C4'), help='final without --k-file')
    ap.add_argument('--k-file', default=None)
    ap.add_argument('--pmarket', default=None, help='choice.json of a11_devig_calibration.py; default the current method (multiplicative, all-books mean)')
    ap.add_argument('--quotes-dir', default=None, help='a11_cache_quotes.py --out folder (needed only for a non-current p_market)')
    ap.add_argument('--frozen-rule', default=str(FROZEN_RULE))
    ap.add_argument('--out', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--write-rule', default=None)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--n-boot', type=int, default=ar.N_BOOT)
    ap.add_argument('--seed', type=int, default=ar.SEED)
    a = ap.parse_args(argv)
    ar.check_seasons(a.seasons + a.train + [a.score])                                 # refuses 2025-26 and later before anything is read
    for p in (a.out, a.report, a.write_rule):
        if p:
            ar.guard_output(p, REPO)
    if a.mode in ('reproduce', 'walkforward') and not a.out:
        ap.error(f'--out is required in mode {a.mode}')
    if a.mode == 'walkforward' and len(a.train) != 1:
        ap.error('walkforward: --train takes one season (k for a season comes only from seasons before it)')
    if a.mode == 'walkforward' and a.train[0] >= a.score:
        ap.error('walkforward: --train must precede --score')
    if a.mode == 'final' and not a.write_rule:
        ap.error('--mode final needs --write-rule')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    labels = load_labels(a.features_db)
    return {'reproduce': reproduce, 'walkforward': walkforward, 'final': final}[a.mode](a, labels)


if __name__ == '__main__':
    main()
