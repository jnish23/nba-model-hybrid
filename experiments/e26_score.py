"""E26 step 2 - score the forward test (see PREREGISTRATION.md, E26). Read-only: nothing in the deployed model or the live log changes.
python -m experiments.e26_score [--features PATH] [--start YYYY-MM-DD --dry-run]
  --features  a training set rebuilt with scripts/generate_training_set.py AFTER the games (needed for B-total only)
  --dry-run   pipeline check on games before the registered start; output goes to dry_run.json and decides nothing"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import e25_decorrelation as d25                      # noqa: E402
from experiments import metrics as M                                   # noqa: E402
from experiments.lib import OUT, ROOT                                  # noqa: E402
from core.odds_clean import _game_results, load_entry_odds, load_clean_odds   # noqa: E402

DIR = OUT / 'E26_forward_shrink'
FROZEN = DIR / 'frozen'
DB = ROOT / 'data' / 'nba_stats.db'
START = '2026-10-25'
DECISIVE = {'A-total': 1300, 'A-spread': 2500, 'A-ml': 2500, 'B-total': 2500}
LOOKS = [500, 1000, 1300]
DEPLOYED_SET_HASH = '0165d3361e4228ad2551a787c66e1cd82ce11bd5b9a4a258b7e79467e7cd02e0'   # _with_odds_cand_20261019 at registration


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frame(params, start, end=None):
    with sqlite3.connect(DB) as conn:
        log = pd.read_sql("SELECT game_id, game_date, prediction_date, ml_prob, spread_pred, total_pred, model_set_hash FROM predictions_log_v2", conn)
        games = _game_results(conn)
    games['gameId'] = games['gameId'].astype(str).str.zfill(10)
    ent = load_entry_odds(DB)
    ent['gameId'] = ent['gameId'].astype(str).str.zfill(10)
    g = games.merge(ent, on='gameId', how='inner')
    g = g[g['gameDate'] >= start]
    if end:
        g = g[g['gameDate'] < end]
    close = load_clean_odds(DB)[['gameId', 'close_home_spread', 'close_total', 'close_home_prob']]
    close['gameId'] = close['gameId'].astype(str).str.zfill(10)
    g = g.merge(close, on='gameId', how='left')
    log['gameId'] = log['game_id'].astype(str).str.zfill(10)
    g = g.merge(log.drop(columns=['game_id', 'game_date']), on='gameId', how='left')
    g['gameDate'] = pd.to_datetime(g['gameDate'])
    g['regular'] = g['gameId'].str[2] == '2'
    g['target_total_pts'] = g['home_score'] + g['away_score']
    g['target_point_diff'] = g['home_score'] - g['away_score']
    g['target_home_win'] = (g['target_point_diff'] > 0).astype(float)
    counts = {'games_with_result_and_entry': int(len(g)), 'no_logged_prediction': int(g['model_set_hash'].isna().sum())}
    same = g['model_set_hash'] == params['deployed_set_hash']
    timely = pd.to_datetime(g['prediction_date']) <= pd.to_datetime(g['entry_fetched_at'])
    counts['other_model_set'] = int((g['model_set_hash'].notna() & ~same).sum())
    counts['predicted_after_entry'] = int((same & ~timely).sum())
    g['eligible_a'] = same & timely
    counts['eligible_a'] = int(g['eligible_a'].sum())
    return g.reset_index(drop=True), counts


def edge(g, market, pred_col, close=False):
    if market == 'total':
        L = g['close_total' if close else 'entry_total'].to_numpy(float)
        y = g['target_total_pts'].to_numpy(float)
        return g[pred_col].to_numpy(float) - L, y - L, L, y
    if market == 'spread':
        L = -g['close_home_spread' if close else 'entry_home_spread'].to_numpy(float)
        y = g['target_point_diff'].to_numpy(float)
        return g[pred_col].to_numpy(float) - L, y - L, L, y
    L = logit(np.clip(g['close_home_prob' if close else 'entry_home_prob'].to_numpy(float), 1e-4, 1 - 1e-4))
    y = g['target_home_win'].to_numpy(float)
    return logit(np.clip(g[pred_col].to_numpy(float), 1e-4, 1 - 1e-4)) - L, y, L, y


def slope_test(g, market, pred_col, null, side, close=False):
    """Forward beta with a day-bootstrap interval; one-sided p = share of draws on the wrong side of `null`."""
    e, r, L, _ = edge(g, market, pred_col, close)
    ok = np.isfinite(e) & np.isfinite(r) & np.isfinite(L)
    gg, e, r, L = g[ok].reset_index(drop=True), e[ok], r[ok], L[ok]
    if len(gg) < 30:
        return {'n': int(len(gg)), 'note': 'too few games'}
    b = d25.beta(market, e, r, L)
    boots = np.array([d25.beta(market, e[i], r[i], L[i]) for i in d25.day_draws(gg['gameDate'])])
    p = float((boots >= null).mean()) if side == 'below' else float((boots <= null).mean())
    return {'n': int(len(gg)), 'beta': b, 'ci': d25.ci(boots), 'p_one_sided': p, 'edge_sd': float(e.std())}


def losses(g, market, pred_col, factor):
    """Per-game loss: line, unshrunk model, model shrunk by `factor`; paired day-bootstrap differences."""
    e, r, L, _ = edge(g, market, pred_col)
    ok = np.isfinite(e) & np.isfinite(r) & np.isfinite(L)
    gg, e, r, L = g[ok].reset_index(drop=True), e[ok], r[ok], L[ok]
    if len(gg) < 30:
        return {'n': int(len(gg))}
    ln, raw, shr = (d25.loss(market, r, L, s) for s in (0.0 * e, e, factor * e))
    out = {'n': int(len(gg)), 'line': float(ln.mean()), 'unshrunk': float(raw.mean()), 'shrunk': float(shr.mean())}
    for lab, d in (('shrunk_minus_unshrunk', shr - raw), ('shrunk_minus_line', shr - ln)):
        boots = [d[i].mean() for i in d25.day_draws(gg['gameDate'])]
        out[lab] = {'diff': float(d.mean()), 'ci': d25.ci(boots)}
    return out


def betting(g, params, beta):
    """Descriptive: unified 5% probability-edge rule on shrunk probabilities (spread/total at -110, ML at entry prices)."""
    rows = g.copy()
    rows['odds_total'], rows['odds_home_spread'] = rows['entry_total'], rows['entry_home_spread']
    rows['odds_home_ml'], rows['odds_away_ml'] = rows['entry_home_ml'], rows['entry_away_ml']
    rows['total_pred'] = rows['entry_total'] + beta['total'] * (rows['total_pred'] - rows['entry_total'])
    rows['spread_pred'] = -rows['entry_home_spread'] + beta['spread'] * (rows['spread_pred'] + rows['entry_home_spread'])
    lp = logit(np.clip(rows['entry_home_prob'], 1e-4, 1 - 1e-4))
    rows['ml_prob'] = expit(lp + beta['ml'] * (logit(np.clip(rows['ml_prob'], 1e-4, 1 - 1e-4)) - lp))
    rows['sigma_spread'], rows['sigma_total'] = params['sigma_shrunk']['spread'], params['sigma_shrunk']['total']
    try:
        bets = M.make_bets(rows, thr=0.05)
        return {mk: M.bet_metrics(bets[bets['market'] == mk]) for mk in ('ml', 'spread', 'total')}
    except Exception as exc:                      # the readout must never block the registered tests
        return {'error': repr(exc)}


def no_odds_predictions(g, params, features_csv, allow_code_change):
    changed = [f for f, h in params['feature_code_sha256'].items() if sha(ROOT / f) != h]
    if changed and not allow_code_change:
        raise SystemExit(f'feature code changed since the freeze: {changed}. Score B-total from a checkout of the registration commit.')
    feats = params['no_odds_total']['features']
    fs = pd.read_csv(features_csv, low_memory=False, usecols=lambda c: c in set(feats) | {'gameId'})
    fs['gameId'] = fs['gameId'].astype(str).str.zfill(10)
    model = joblib.load(FROZEN / 'no_odds_total.joblib')
    assert sha(FROZEN / 'no_odds_total.joblib') == params['no_odds_total']['model_sha256']
    fs['no_odds_total_pred'] = model.predict(fs[feats])
    inj = [c for c in feats if c.startswith('inj_')]
    out = g.merge(fs[['gameId', 'no_odds_total_pred'] + inj], on='gameId', how='left')
    info = {'code_changed': changed, 'rows_matched': int(out['no_odds_total_pred'].notna().sum()),
            'rows_missing_injury_features': int(out[inj].isna().any(axis=1).sum()) if inj else 0}
    return out.drop(columns=inj), info


def decide(res, key, n):
    if 'beta' not in res:
        return 'too few games'
    if n < DECISIVE[key]:
        return f'interim ({n} of {DECISIVE[key]} games): no decision'
    return 'decisive look'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features')
    ap.add_argument('--start', default=START)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-code-change', action='store_true')
    a = ap.parse_args()
    if a.start < START and not a.dry_run:
        raise SystemExit(f'games before {START} are not part of the test; use --dry-run to check the pipeline on them')
    params = json.loads((FROZEN / 'params.json').read_text())
    params['deployed_set_hash'] = DEPLOYED_SET_HASH
    beta = {mk: params['beta'][mk]['all_seasons']['beta'] for mk in ('spread', 'total', 'ml')}
    beta_disc = {mk: params['beta'][mk]['discovery']['beta'] for mk in ('spread', 'total', 'ml')}
    g, counts = load_frame(params, a.start, end=START if a.dry_run else None)
    res = {'dry_run': a.dry_run, 'start': a.start, 'counts': counts, 'frozen_beta': beta}
    ga = g[g['eligible_a']].reset_index(drop=True)
    cols = {'total': 'total_pred', 'spread': 'spread_pred', 'ml': 'ml_prob'}
    for mk, col in cols.items():
        key = f'A-{mk}'
        t = slope_test(ga, mk, col, 1.0, 'below')
        t['frozen_beta_inside_ci'] = bool('ci' in t and t['ci'][0] <= beta[mk] <= t['ci'][1])
        t['supported_if_decisive'] = bool('ci' in t and t['ci'][1] < 1)
        t['status'] = decide(t, key, t['n'])
        res[key] = {'primary': t, 'loss_frozen_all_seasons': losses(ga, mk, col, beta[mk]),
                    'loss_frozen_discovery': losses(ga, mk, col, beta_disc[mk]),
                    'vs_close': slope_test(ga, mk, col, 1.0, 'below', close=True),
                    'regular_season_only': slope_test(ga[ga['regular']].reset_index(drop=True), mk, col, 1.0, 'below')}
    res['betting_readout'] = betting(ga, params, beta)
    if a.features:
        gb, info = no_odds_predictions(g, params, a.features, a.allow_code_change)
        k = params['k_total_blend']['all_seasons']
        t = slope_test(gb, 'total', 'no_odds_total_pred', 0.0, 'above')
        t['supported_if_decisive'] = bool('ci' in t and t['ci'][0] > 0)
        t['status'] = decide(t, 'B-total', t['n'])
        res['B-total'] = {'primary': t, 'features': info, 'loss_blend': losses(gb, 'total', 'no_odds_total_pred', k),
                          'vs_close': slope_test(gb, 'total', 'no_odds_total_pred', 0.0, 'above', close=True),
                          'regular_season_only': slope_test(gb[gb['regular']].reset_index(drop=True), 'total', 'no_odds_total_pred', 0.0, 'above')}
    else:
        res['B-total'] = {'note': 'not scored: pass --features (a training set rebuilt after the games)'}
    ps = {k: res[k]['primary']['p_one_sided'] for k in DECISIVE if 'primary' in res[k] and 'p_one_sided' in res[k]['primary']}
    res['holm'] = d25.holm(ps) if ps else {}
    name = 'dry_run.json' if a.dry_run else f'results_{pd.Timestamp.now():%Y-%m-%d}.json'
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / name).write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(json.dumps({k: (v['primary'] if isinstance(v, dict) and 'primary' in v else v) for k, v in res.items()
                      if k not in ('betting_readout',)}, indent=1, default=str))


if __name__ == '__main__':
    main()
