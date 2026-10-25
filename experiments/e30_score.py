"""E30 step 2 - score the forward test of the four E28 opening-line candidates (see PREREGISTRATION.md, E30). Read-only: nothing in the deployed
model, the live log or the database changes.
python -m experiments.e30_score [--start YYYY-MM-DD --dry-run] [--allow-code-change]
  --dry-run  pipeline check on games before the registered start (output dry_run.json; decides nothing)"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.odds_clean import _game_results, load_clean_odds, load_entry_odds   # noqa: E402
from experiments import e28_audit as audit                                      # noqa: E402
from experiments import e28_screen as sc                                        # noqa: E402
from experiments import e30_signals as sg                                       # noqa: E402

DIR = sg.ROOT / 'reports' / 'experiments' / 'E30_forward_tracking'
FROZEN = DIR / 'frozen'
START = '2026-10-25'
LOOKS_INFO = [600, 1230]          # reported, decide nothing
DECISIVE = [2460, 4920]           # first N eligible games of each test; one-sided alpha 0.025 each, Holm over the four tests
ALPHA = 0.025
WIN = 100.0 / 110.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def holm(ps):
    keys = sorted(ps, key=ps.get)
    out, prev = {}, 0.0
    for i, k in enumerate(keys):
        prev = max(prev, min(1.0, (len(keys) - i) * ps[k]))
        out[k] = prev
    return out


def slope(y, s, dates):
    """OLS y ~ 1 + s with a day-cluster bootstrap; one-sided p = (1 + draws <= 0) / (B + 1)."""
    ok = np.isfinite(y) & np.isfinite(s)
    n = int(ok.sum())
    if n < 30:
        return {'n': n, 'note': 'too few games'}
    X = np.column_stack([np.ones(n), s[ok]])
    b, d = sc.fit_boot(X, y[ok], np.zeros(n), 'ols', np.asarray(dates)[ok])
    G = d[:, 1]
    return {'n': n, 'slope': float(b[1]), 'ci': [float(np.quantile(G, .025)), float(np.quantile(G, .975))], 'p_one_sided': float((1 + (G <= 0).sum()) / (len(G) + 1)),
            'se': float(G.std(ddof=1))}


def load_forward(start, end=None):
    with sqlite3.connect(sg.DB) as conn:
        games = _game_results(conn)
    games['gameId'] = games['gameId'].astype(str).str.zfill(10)
    ent = load_entry_odds(sg.DB)
    ent['gameId'] = ent['gameId'].astype(str).str.zfill(10)
    g = games.merge(ent, on='gameId', how='inner')
    g = g[g['gameDate'] >= start]
    if end:
        g = g[g['gameDate'] < end]
    close = load_clean_odds(sg.DB)[['gameId', 'close_home_spread', 'close_total']]
    close['gameId'] = close['gameId'].astype(str).str.zfill(10)
    g = g.merge(close, on='gameId', how='left')
    g['gameDate'] = pd.to_datetime(g['gameDate'])
    g['target_total_pts'] = g['home_score'] + g['away_score']
    g['target_point_diff'] = g['home_score'] - g['away_score']
    g['regular'] = g['gameId'].str[:3] == '002'
    return g.sort_values(['gameDate', 'gameId']).reset_index(drop=True)


def live_feasible(feat, log_path=audit.LOG):
    """True when, for both teams, the previous game's team tracking was already in the database after the sync of the morning of the game
    (from the daily log), or that game predates the log. Games with no previous tracking row at all are False."""
    runs, adds, errs, found = audit.parse_log(log_path)
    first = adds[adds['table'] == audit.TRACK_TEAM].groupby('gameId')['run_date'].min() if len(adds) else pd.Series(dtype=str)
    log_start = pd.Timestamp(runs['run_date'].min()) if len(runs) else pd.Timestamp.max
    with sqlite3.connect(f'file:{sg.DB}?mode=ro', uri=True) as c:
        tg = pd.read_sql('SELECT teamId, gameId, gameDate FROM team_game_full', c)
        have = set(pd.read_sql('SELECT DISTINCT gameId FROM team_track_stats_v3', c)['gameId'].astype(str).str.zfill(10))
    tg['gameId'] = tg['gameId'].astype(str).str.zfill(10)
    tg['gameDate'] = pd.to_datetime(tg['gameDate'])
    tg = tg.sort_values(['teamId', 'gameDate', 'gameId'])
    tg['prev_gid'] = tg.groupby('teamId')['gameId'].shift(1)
    tg['prev_date'] = tg.groupby('teamId')['gameDate'].shift(1)
    tg['first_added'] = pd.to_datetime(tg['prev_gid'].map(first))

    def ok(r):
        if pd.isna(r['prev_gid']) or r['prev_gid'] not in have:
            return False
        if r['prev_date'] < log_start:
            return True
        return bool(pd.notna(r['first_added']) and r['first_added'] <= r['gameDate'])
    tg['ok'] = tg.apply(ok, axis=1)
    m = tg.set_index(['gameId', 'teamId'])['ok']
    f = feat[['gameId', 'teamId_home', 'teamId_away']].copy()
    return pd.Series([bool(m.get((g, h), False) and m.get((g, a), False)) for g, h, a in zip(f['gameId'], f['teamId_home'], f['teamId_away'])], index=feat.index)


def betting(y, s, move, thr, margin):
    """Bet the side of the signal when |signal| >= thr (frozen 75th percentile): -110 on the entry line, flat 1 unit; CLV in points."""
    ok = np.isfinite(y) & np.isfinite(s) & (np.abs(s) >= thr)
    if ok.sum() == 0:
        return {'bets': 0}
    side = np.sign(s[ok])
    win = (np.sign(y[ok]) == side)
    push = y[ok] == 0
    profit = np.where(push, 0.0, np.where(win, WIN, -1.0))
    clv = side * move[ok]
    return {'bets': int(ok.sum()), 'hit_rate_excl_pushes': float(win[~push].mean()), 'roi_per_bet': float(profit.mean()),
            'mean_clv_points': float(np.nanmean(clv))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default=START)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-code-change', action='store_true')
    a = ap.parse_args()
    if a.start < START and not a.dry_run:
        raise SystemExit(f'games before {START} are not part of the test; use --dry-run to check the pipeline on them')
    params = json.loads((FROZEN / 'params.json').read_text())
    changed = [f for f, h in params['code_sha256'].items() if sha(sg.ROOT / f) != h]
    if changed and not a.allow_code_change:
        raise SystemExit(f'code changed since the freeze: {changed}. Score from a checkout of the registration commit, or pass --allow-code-change and report it.')
    feat = sg.feature_games()
    g = load_forward(a.start, end=START if a.dry_run else None)
    feat = feat[feat['gameId'].isin(g['gameId'])].reset_index(drop=True)
    feat['live_feasible'] = live_feasible(feat)
    g = g.merge(feat, on='gameId', how='inner', suffixes=('', '_f'))
    res = {'dry_run': a.dry_run, 'start': a.start, 'code_changed': changed, 'games_with_result_and_entry': int(len(g)), 'regular_season': int(g['regular'].sum()),
           'live_feasible_share': float(g['live_feasible'].mean()) if len(g) else None, 'tests': {}}
    prim = {}
    for tid, (fam, tg_) in sg.TESTS.items():
        spec = params['tests'][tid]['primary']
        alt = params['tests'][tid]['all_seasons']
        margin = tg_ == 'margin'
        d = g.copy()
        d['sig'] = sg.signal_points(d, spec)
        d['sig_alt'] = sg.signal_points(d, alt)
        if margin:
            d['y_entry'] = d['target_point_diff'] + d['entry_home_spread']
            d['y_close'] = d['target_point_diff'] + d['close_home_spread']
            d['move'] = d['entry_home_spread'] - d['close_home_spread']
        else:
            d['y_entry'] = d['target_total_pts'] - d['entry_total']
            d['y_close'] = d['target_total_pts'] - d['close_total']
            d['move'] = d['close_total'] - d['entry_total']
        if tid == 'T3':
            d['live_feasible'] = True                       # schedule only: no tracking needed
        m = d[d['regular'] & d['sig'].notna()].reset_index(drop=True)
        t = {'family': fam, 'target': tg_, 'eligible_games': int(len(m)), 'games_without_signal': int((d['regular'] & d['sig'].isna()).sum()),
             'frozen_signal_sd': spec['signal_sd'], 'forward_signal_sd': float(m['sig'].std()) if len(m) > 1 else None}
        full = slope(m['y_entry'].to_numpy(float), m['sig'].to_numpy(float), m['gameDate'])
        t['running_all_eligible'] = full
        t['decisive'] = {}
        for N in DECISIVE:
            if len(m) >= N:
                sub = m.iloc[:N]
                t['decisive'][str(N)] = slope(sub['y_entry'].to_numpy(float), sub['sig'].to_numpy(float), sub['gameDate'])
        t['info_looks'] = {str(N): slope(m.iloc[:N]['y_entry'].to_numpy(float), m.iloc[:N]['sig'].to_numpy(float), m.iloc[:N]['gameDate']) for N in LOOKS_INFO if len(m) >= N}
        t['vs_close'] = slope(m['y_close'].to_numpy(float), m['sig'].to_numpy(float), m['gameDate'])
        t['clv_move_on_signal'] = slope(m['move'].to_numpy(float), m['sig'].to_numpy(float), m['gameDate'])
        t['all_season_fit'] = slope(m['y_entry'].to_numpy(float), m['sig_alt'].to_numpy(float), m['gameDate'])
        lf = m[m['live_feasible']]
        t['live_feasible_only'] = {'games': int(len(lf)), **slope(lf['y_entry'].to_numpy(float), lf['sig'].to_numpy(float), lf['gameDate'])}
        pm = d[d['sig'].notna()].reset_index(drop=True)
        t['with_playoffs'] = slope(pm['y_entry'].to_numpy(float), pm['sig'].to_numpy(float), pm['gameDate'])
        t['betting_readout'] = betting(m['y_entry'].to_numpy(float), m['sig'].to_numpy(float), m['move'].to_numpy(float), spec['abs_signal_p75'], margin)
        res['tests'][tid] = t
        for N in DECISIVE:
            if str(N) in t['decisive'] and 'p_one_sided' in t['decisive'][str(N)]:
                prim.setdefault(N, {})[tid] = t['decisive'][str(N)]['p_one_sided']
    res['holm_at_decisive_looks'] = {}
    for N, ps in prim.items():
        if len(ps) == len(sg.TESTS):
            h = holm(ps)
            res['holm_at_decisive_looks'][str(N)] = {k: {'p': ps[k], 'holm_p': h[k], 'supported': bool(h[k] <= ALPHA)} for k in ps}
        else:
            res['holm_at_decisive_looks'][str(N)] = {'note': f'only {len(ps)} of {len(sg.TESTS)} tests have {N} eligible games yet'}
    name = 'dry_run.json' if a.dry_run else f'results_{pd.Timestamp.now():%Y-%m-%d}.json'
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / name).write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(json.dumps({k: (v if k != 'tests' else {t: {kk: vv for kk, vv in tv.items() if kk in ('eligible_games', 'games_without_signal', 'running_all_eligible', 'live_feasible_only')} for t, tv in v.items()}) for k, v in res.items()}, indent=1, default=str))


if __name__ == '__main__':
    main()
