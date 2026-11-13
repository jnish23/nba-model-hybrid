"""
Experiment 7e (reports/props/audit_2026-11/WP6_WP7_experiment_plan.md, D "7e"): a negative binomial in place of `dispersion.widen`.

    python scripts/a11_nb_dispersion.py --raw-version a11_r0_raw --base-version a11_r0 --registry-root $REG --folds 2021-22 2022-23 2023-24 --seen 2024-25 \
        --prices-dir $OUT/prices --out $OUT/results/7e --report $RPT/7e_nb/report.md

Inputs (read only):
  <registry-root>/<raw-version>/<season>/raw_<stat>.npy   PRE-dispersion pmfs, float32, (rows, cap + 1), for rebounds, assists, threes and ra (the others are not needed)
  <registry-root>/<raw-version>/<season>/index.csv         one line per row, same order: game_id, player_id, minutes_before, season, game_date, minutes_mean (the predicted-minutes
                                                           bucket of the dispersion factor); all played rows of the fold
  <registry-root>/<base-version>/<season>/bundle.pkl       its ['dispersion'] DispersionScaler holds the fitted c per (stat, minutes bucket); a sidecar dispersion_c.json
                                                           ({"stat:bucket": c}) in the same folder is read instead when present (no unpickling)
  <registry-root>/<base-version>/<season>/index.csv + pred_<stat>.npy   optional: only to check that the rows line up (means of the dispersed base pmfs against the raw ones)
  <features-db>                                            player_game_labels (outcomes), opened mode=ro
  <prices-dir>/prices_<season>.pkl                         (consensus, bets) as prices.load_season returns, for the actual book lines (2023-24; 2024-25 as seen)

For each dev fold (and each seen season) the raw pmfs are dispersed two ways with the SAME fitted c: widen (the frozen pipeline) and nb (dispersion.nb_match). Nothing is re-fitted. Every metric of
plan 7e item 5 is computed per row for both arms; differences are paired (nb minus widen, negative = nb better) with the game-clustered bootstrap of props.models.evaluate.cluster_bootstrap.
The keep rule of item 6 is evaluated clause by clause on the pooled dev folds; the Bonferroni interval (0.05 / 23, plan A.5) is printed beside the 95% one. 2024-25 is reported as seen and never
enters the rule. 2025-26 is refused. Writes only under --out and --report.
"""
import argparse
import json
import logging
import pickle
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from props.betting import rules  # noqa: E402
from props.models import dispersion, evaluate as ev, outcome as oc, registry  # noqa: E402
from props.models.pipeline import observed  # noqa: E402

HOLDOUT = '2025-26'                         # plan A.1: not trained, predicted or scored; every season >= this is refused
HOLDOUT_LAST_DATE = '2025-06-30'            # and so is any row dated after the end of the last seen season
DEV_FOLDS = ('2021-22', '2022-23', '2023-24')
SEEN = ('2024-25',)
BET_STATS = ('rebounds', 'assists', 'threes', 'ra')
LOW_STATS = ('threes', 'assists', 'rebounds')       # stats scored at the low lines
LINES = (0.5, 1.5)
N_COMPARISONS = 23                                  # plan A.5
ALPHA_BONF = 0.05 / N_COMPARISONS
CRPS_MARGIN = 0.0005                                # keep rule clause 3: relative upper bound of the paired CRPS difference, +0.05%
LL_EPS = 1e-4                                       # probability clip of the log score (market_bench._ll)
ROW_KEYS = ['game_id', 'player_id', 'minutes_before']
LABEL_COLS = 'game_id, player_id, lbl_played, lbl_minutes, y_points, y_rebounds, y_assists, y_threes, y_pra'
SEASON_RE = re.compile(r'^\d{4}-\d{2}$')


# ------------------------------------------------------------------------------------------------ guards

def refuse_holdout(seasons, what='season'):
    """Plan A.1: the 2025-26 holdout (and anything later) is never loaded, fitted, scored or priced here."""
    seasons = [str(s) for s in seasons]
    for s in seasons:
        if not SEASON_RE.match(s):
            raise SystemExit(f'{what} {s!r} is not a season like 2023-24')
    bad = sorted({s for s in seasons if s >= HOLDOUT})
    if bad:
        raise SystemExit(f'{what} {bad}: this script refuses {HOLDOUT} and later seasons (plan A.1: the holdout is not trained, predicted or scored)')


def refuse_late_dates(idx, season):
    if 'game_date' in idx.columns and len(idx) and str(idx['game_date'].astype(str).max())[:10] > HOLDOUT_LAST_DATE:
        raise SystemExit(f'{season}: the index holds game dates after {HOLDOUT_LAST_DATE} (max {str(idx["game_date"].astype(str).max())[:10]}): refused (plan A.1)')


def _inside(child, parent):
    child, parent = Path(child).resolve(), Path(parent).resolve()
    return child == parent or parent in child.parents


def check_output_paths(out, report, inputs):
    """Outputs go under --out and --report only: neither may sit inside, or be, an input."""
    for p, name in ((out, '--out'), (report, '--report')):
        for i in inputs:
            if i is not None and (_inside(p, i) or _inside(i, p)):
                raise SystemExit(f'{name} {p} overlaps the input {i}: outputs must not be written into inputs')


# ------------------------------------------------------------------------------------------------ loading (read only)

def read_labels(features_db):
    with sqlite3.connect(f'file:{Path(features_db).resolve().as_posix()}?mode=ro', uri=True, timeout=120) as conn:
        lab = pd.read_sql(f'SELECT {LABEL_COLS} FROM player_game_labels', conn)
    lab['game_id'] = lab['game_id'].astype(str)
    return lab


def read_prices(path):
    with open(path, 'rb') as f:
        cons, bets = pickle.load(f)
    for d in (cons, bets):
        d['game_id'] = d['game_id'].astype(str)
        d['player_id'] = d['player_id'].astype(int)
    return cons, bets


def load_scaler(base_version, season, root):
    """The fitted DispersionScaler of the base version's fold: dispersion_c.json if present, else the bundle's. Returns (scaler, source)."""
    d = Path(root) / base_version / season
    js = d / 'dispersion_c.json'
    if js.exists():
        sc = dispersion.DispersionScaler()
        sc.c = {(k.rsplit(':', 1)[0], int(k.rsplit(':', 1)[1])): float(v) for k, v in json.loads(js.read_text(encoding='utf-8')).items()}
        return sc, 'dispersion_c.json'
    if not (d / 'bundle.pkl').exists():
        raise FileNotFoundError(f'{d}: neither dispersion_c.json nor bundle.pkl: the base version {base_version!r} has no fitted dispersion factors for {season}')
    return registry.load_bundle(base_version, season, root)['dispersion'], 'bundle.pkl'


def load_raw(raw_version, season, root, stats=BET_STATS):
    """(index frame, {stat: raw pmf array opened read-only}) in the layout described in the module docstring."""
    d = Path(root) / raw_version / season
    ip = next((d / n for n in ('index.csv', 'raw_index.csv') if (d / n).exists()), None)
    if ip is None:
        raise FileNotFoundError(f'{d}: no index.csv (expected beside raw_<stat>.npy)')
    idx = pd.read_csv(ip, dtype={'game_id': str})
    if 'season' in idx.columns and set(idx['season'].astype(str)) != {season}:
        raise SystemExit(f'{ip}: season column {sorted(set(idx["season"].astype(str)))} is not {season}')
    refuse_late_dates(idx, season)
    miss = [c for c in ROW_KEYS + ['minutes_mean'] if c not in idx.columns]
    if miss:
        raise SystemExit(f'{ip}: missing columns {miss}')
    if idx.duplicated(ROW_KEYS).any():
        raise SystemExit(f'{ip}: duplicate (game_id, player_id, minutes_before) rows')
    raws = {}
    for s in stats:
        p = d / f'raw_{s}.npy'
        if not p.exists():
            raise FileNotFoundError(f'{p} missing: the raw version must hold raw_{s}.npy (props_phase5_predict.py --save-raw)')
        a = np.load(p, mmap_mode='r')
        if a.ndim != 2 or a.shape[0] != len(idx) or a.shape[1] != oc.CAPS[s] + 1:
            raise SystemExit(f'{p}: shape {a.shape}, expected ({len(idx)}, {oc.CAPS[s] + 1})')
        raws[s] = a
    return idx, raws


def check_raw_pmfs(raws, tol=1e-3):
    """Each row must be a pmf: finite, non-negative, summing to 1 (float32 histograms, so to within a small tolerance)."""
    out = {}
    for s, a in raws.items():
        worst, neg = 0.0, False
        for lo in range(0, len(a), 50000):
            b = np.asarray(a[lo:lo + 50000], np.float64)
            if not np.isfinite(b).all():
                raise SystemExit(f'raw_{s}: non-finite values')
            neg |= bool((b < 0).any())
            worst = max(worst, float(np.abs(b.sum(axis=1) - 1).max())) if len(b) else worst
        if neg or worst > tol:
            raise SystemExit(f'raw_{s}: not a pmf (negative entries: {neg}; largest |row sum - 1| = {worst:.2e})')
        out[s] = worst
    return out


def base_alignment(base_version, season, root, idx, raws):
    """Optional: does row i of the raw version describe the same row as row i of the base version's dispersed pmfs? Compares the pmf means (dispersion keeps the mean; the draws differ, so not equal)."""
    d = Path(root) / base_version / season
    if not (d / 'index.csv').exists():
        return {'checked': False, 'note': 'base index.csv not found'}
    bi = pd.read_csv(d / 'index.csv', dtype={'game_id': str})
    if len(bi) != len(idx) or not all((bi[k].to_numpy() == idx[k].to_numpy()).all() for k in ROW_KEYS):
        return {'checked': True, 'same_rows_same_order': False, 'note': f'base index has {len(bi)} rows against {len(idx)}: raw and base rows do not line up (the script keys on the raw index only)'}
    res = {'checked': True, 'same_rows_same_order': True}
    for s in raws:
        p = d / f'pred_{s}.npy'
        if p.exists():
            a, b = oc.pmf_mean(np.asarray(raws[s], np.float64)), oc.pmf_mean(np.asarray(np.load(p, mmap_mode='r'), np.float64))
            res[f'{s}_mean_corr'] = float(np.corrcoef(a, b)[0, 1])
            res[f'{s}_mean_abs_diff'] = float(np.abs(a - b).mean())
    return res


# ------------------------------------------------------------------------------------------------ per-row metrics

def ll(p, hit, eps=LL_EPS):
    """Per-row log loss (lower is better) of a probability p for a 0/1 outcome."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    hit = np.asarray(hit, float)
    return -(hit * np.log(p) + (1 - hit) * np.log(1 - p))


def row_metrics(raw, c, y, chunk=20000):
    """
    Per row, for the two arms (suffix _w = widen, _n = nb): CRPS, P(Y > 0.5), P(Y > 1.5), P(Y = 0), the mean and variance of the returned pmf, and its total-variation distance to the raw pmf.
    Also the raw pmf's mean mu and variance s2, `exists` (the row got a negative binomial) and `tail` (that negative binomial's mass above the last bin, before folding).
    raw: (n, cap + 1) float32 (an open memmap is fine); c: (n,); y: (n,) observed counts.
    """
    c = np.asarray(c, float)
    y = np.asarray(y)
    parts = []
    for lo in range(0, len(c), chunk):
        hi = min(lo + chunk, len(c))
        r = np.asarray(raw[lo:hi], np.float32)
        w = dispersion.widen(r, np.sqrt(c[lo:hi]))
        nbp, info = dispersion._nb_match(r, c[lo:hi], widened=w)
        yi = y[lo:hi].astype(int)
        l05, l15 = np.full(hi - lo, 0.5), np.full(hi - lo, 1.5)                          # prob_over wants one line per row
        d = {'y': y[lo:hi], 'c': c[lo:hi], 'mu': info['mu'], 's2': info['s2'], 'exists': info['exists'], 'tail': info['tail']}
        r64 = r.astype(np.float64)
        for tag, pm in (('w', w), ('n', nbp)):
            m, v = dispersion.nb_moments(pm)
            d.update({f'crps_{tag}': oc.crps(pm, yi), f'p_over05_{tag}': oc.prob_over(pm, l05), f'p_over15_{tag}': oc.prob_over(pm, l15), f'p0_{tag}': pm[:, 0].astype(np.float64),
                      f'mean_{tag}': m, f'var_{tag}': v, f'tv_{tag}': 0.5 * np.abs(pm.astype(np.float64) - r64).sum(axis=1)})
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_season(season, raw_version, base_version, root, labels, log=logging.info):
    """All per-row frames of one season: ({stat: frame}, c table, info). Rows are the played rows of the raw index that have a label and an observed count."""
    idx, raws = load_raw(raw_version, season, root)
    sums = check_raw_pmfs(raws)
    scaler, c_source = load_scaler(base_version, season, root)
    align = base_alignment(base_version, season, root, idx, raws)
    rows = idx.merge(labels, on=['game_id', 'player_id'], how='left', validate='many_to_one')
    played = ((rows['lbl_played'] == 1) & (rows['lbl_minutes'] > 0)).to_numpy()
    mm = rows['minutes_mean'].to_numpy(float)
    frames = {}
    for s in BET_STATS:
        y = observed(rows, s)
        ok = played & ~np.isnan(y) & ~np.isnan(mm)
        c = scaler.c_for(s, mm[ok])
        f = row_metrics(raws[s][np.flatnonzero(ok)], c, y[ok])
        k = rows.loc[ok, ROW_KEYS + [c for c in ('game_date',) if c in rows.columns]].reset_index(drop=True)
        f = pd.concat([k, f], axis=1)
        f.insert(0, 'season', season)
        f['minutes_mean'] = mm[ok]
        frames[s] = f
        log(f'{season} {s}: {ok.sum()} rows of {len(idx)}')
    ct = pd.DataFrame([{'season': season, 'stat': k[0], 'minutes_bucket': k[1], 'c': v} for k, v in sorted(scaler.c.items()) if k[0] in BET_STATS])
    info = {'rows_in_index': int(len(idx)), 'rows_without_label_or_not_played': int((~played).sum()), 'c_source': c_source, 'max_abs_row_sum_error': sums, 'base_alignment': align}
    return frames, ct, info


# ------------------------------------------------------------------------------------------------ paired intervals

def paired(d, clusters, bonf=True, n_boot=ev.N_BOOT, n_boot_bonf=10000, seed=0):
    """Mean of d with the game-clustered 95% interval (ev.cluster_bootstrap), and the Bonferroni (1 - 0.05 / 23) interval beside it when bonf."""
    d = np.asarray(d, float)
    p, lo, hi = ev.cluster_bootstrap(d, clusters, n_boot=n_boot, seed=seed)
    out = {'diff': p, 'lo': lo, 'hi': hi, 'bonf_lo': np.nan, 'bonf_hi': np.nan}
    if bonf:
        _, out['bonf_lo'], out['bonf_hi'] = ev.cluster_bootstrap(d, clusters, n_boot=n_boot_bonf, seed=seed, alpha=ALPHA_BONF)
    return out


# ------------------------------------------------------------------------------------------------ tables

def logscore_row(df, stat, line, scope, source, bonf, **kw):
    """Log score of P(Y > line) for both arms on the rows of df (columns y, p_over05_*, p_over15_*, game_id) and the paired difference nb - widen."""
    col = 'p_over05' if line == 0.5 else 'p_over15'
    hit = (df['y'].to_numpy() > line).astype(float)
    lw, ln = ll(df[f'{col}_w'], hit), ll(df[f'{col}_n'], hit)
    r = paired(ln - lw, df['game_id'].to_numpy(), bonf, **kw)
    return {'scope': scope, 'source': source, 'stat': stat, 'line': line, 'n': len(df), 'hit_rate': hit.mean(), 'mean_p_widen': df[f'{col}_w'].mean(), 'mean_p_nb': df[f'{col}_n'].mean(),
            'logscore_widen': lw.mean(), 'logscore_nb': ln.mean(), 'diff_nb_minus_widen': r['diff'], 'lo95': r['lo'], 'hi95': r['hi'], 'bonf_lo': r['bonf_lo'], 'bonf_hi': r['bonf_hi']}


def logscore_table(pooled_by_scope, bonf_scopes, **kw):
    """pooled_by_scope: {scope: {stat: frame}}. One row per scope x stat x line (all played rows)."""
    rows = []
    for scope, fr in pooled_by_scope.items():
        for s in LOW_STATS:
            if s in fr and len(fr[s]):
                for line in LINES:
                    rows.append(logscore_row(fr[s], s, line, scope, 'all_played', scope in bonf_scopes, **kw))
    return pd.DataFrame(rows)


def book_line_frames(frames_by_season, cons_by_season):
    """For each season with a price file: {stat: rows of that season's frame that have a consensus quote at line 0.5 / 1.5} -> {(stat, line): frame}."""
    out = {}
    for season, cons in cons_by_season.items():
        scol = 'stat' if 'stat' in cons.columns else 'market'
        for s in LOW_STATS:
            f = frames_by_season.get(season, {}).get(s)
            if f is None:
                continue
            for line in LINES:
                c = cons[(cons[scol] == s) & (cons['consensus_line'] == line)][ROW_KEYS].drop_duplicates()
                out[(season, s, line)] = f.merge(c, on=ROW_KEYS, how='inner')
    return out


def booklines_table(book_frames, dev_seasons, seen_seasons, bonf_scopes, **kw):
    rows = []
    for (season, s, line), f in sorted(book_frames.items()):
        if len(f):
            rows.append(logscore_row(f, s, line, f'seen_{season}' if season in seen_seasons else f'dev_{season}', 'book_line', True, **kw))
    for s in LOW_STATS:
        for line in LINES:
            fs = [f for (season, st, ln), f in book_frames.items() if st == s and ln == line and season in dev_seasons and len(f)]
            if fs and len(fs) > 1:
                rows.append(logscore_row(pd.concat(fs, ignore_index=True), s, line, 'dev_pooled', 'book_line', 'dev_pooled' in bonf_scopes, **kw))
    return pd.DataFrame(rows)


def p0_table(pooled_by_scope, bonf_scopes, **kw):
    """Calibration of P(Y = 0): mean predicted against observed, per arm (with game-clustered intervals of the bias), and the paired Brier difference."""
    rows = []
    for scope, fr in pooled_by_scope.items():
        for s in LOW_STATS:
            df = fr.get(s)
            if df is None or not len(df):
                continue
            z = (df['y'].to_numpy() == 0).astype(float)
            g = df['game_id'].to_numpy()
            row = {'scope': scope, 'stat': s, 'n': len(df), 'observed_p0': z.mean()}
            for tag, name in (('w', 'widen'), ('n', 'nb')):
                b, lo, hi = ev.cluster_bootstrap(df[f'p0_{tag}'].to_numpy() - z, g, n_boot=kw.get('n_boot', ev.N_BOOT))
                row.update({f'pred_p0_{name}': df[f'p0_{tag}'].mean(), f'bias_{name}': b, f'bias_{name}_lo': lo, f'bias_{name}_hi': hi, f'abs_bias_{name}': abs(b)})
            row['nb_no_worse_abs_bias'] = bool(row['abs_bias_nb'] <= row['abs_bias_widen'])
            r = paired((df['p0_n'].to_numpy() - z) ** 2 - (df['p0_w'].to_numpy() - z) ** 2, g, scope in bonf_scopes, **kw)
            row.update({'brier_p0_diff_nb_minus_widen': r['diff'], 'brier_lo95': r['lo'], 'brier_hi95': r['hi'], 'brier_bonf_lo': r['bonf_lo'], 'brier_bonf_hi': r['bonf_hi']})
            rows.append(row)
    return pd.DataFrame(rows)


def p0_reliability(df, bins=10):
    """P(Y = 0) by decile of the widen P(0): mean predicted (both arms) against observed frequency."""
    z = (df['y'].to_numpy() == 0).astype(float)
    q = pd.qcut(df['p0_w'], bins, duplicates='drop')
    return pd.DataFrame({'q': q, 'n': 1, 'p0_widen': df['p0_w'], 'p0_nb': df['p0_n'], 'observed': z}).groupby('q', observed=True).agg(
        n=('n', 'sum'), p0_widen=('p0_widen', 'mean'), p0_nb=('p0_nb', 'mean'), observed=('observed', 'mean')).reset_index(drop=True)


def crps_table(pooled_by_scope, bonf_scopes, **kw):
    rows = []
    for scope, fr in pooled_by_scope.items():
        for s in BET_STATS:
            df = fr.get(s)
            if df is None or not len(df):
                continue
            r = paired(df['crps_n'].to_numpy() - df['crps_w'].to_numpy(), df['game_id'].to_numpy(), scope in bonf_scopes, **kw)
            m = df['crps_w'].mean()
            rows.append({'scope': scope, 'stat': s, 'n': len(df), 'crps_widen': m, 'crps_nb': df['crps_n'].mean(), 'diff_nb_minus_widen': r['diff'], 'lo95': r['lo'], 'hi95': r['hi'],
                         'bonf_lo': r['bonf_lo'], 'bonf_hi': r['bonf_hi'], 'rel_diff_pct': 100 * r['diff'] / m, 'rel_hi95_pct': 100 * r['hi'] / m, 'rel_bonf_hi_pct': 100 * r['bonf_hi'] / m})
    return pd.DataFrame(rows)


def existence_table(frames_by_season):
    """Item 7: the share of rows where the negative binomial does not exist, and what it does to the pmf (moments, folded tail, distance from the raw pmf)."""
    rows = []
    for season, fr in frames_by_season.items():
        for s, f in fr.items():
            if not len(f):
                continue
            floor = f['c'].to_numpy() <= 1.0
            exists = f['exists'].to_numpy()
            nb_rows = f[exists]
            rows.append({'season': season, 'stat': s, 'n': len(f), 'share_c_at_floor_(widen_used)': floor.mean(), 'share_nb_exists': exists.mean(),
                         'share_c_above_1_but_no_nb': (~floor & ~exists).mean(), 'mean_c': f['c'].mean(),
                         'mean_tail_folded_into_last_bin': nb_rows['tail'].mean() if len(nb_rows) else np.nan, 'max_tail_folded': nb_rows['tail'].max() if len(nb_rows) else np.nan,
                         'mean_abs_mean_shift_nb_vs_raw': (nb_rows['mean_n'] - nb_rows['mu']).abs().mean() if len(nb_rows) else np.nan,
                         'nb_var_over_target': (nb_rows['var_n'] / (nb_rows['c'] * nb_rows['s2'])).mean() if len(nb_rows) else np.nan,
                         'widen_var_over_target_same_rows': (nb_rows['var_w'] / (nb_rows['c'] * nb_rows['s2'])).mean() if len(nb_rows) else np.nan,
                         'mean_tv_raw_to_widen_nb_rows': nb_rows['tv_w'].mean() if len(nb_rows) else np.nan, 'mean_tv_raw_to_nb_nb_rows': nb_rows['tv_n'].mean() if len(nb_rows) else np.nan,
                         'mean_tv_raw_to_nb_where_c_below_1.02': nb_rows.loc[nb_rows['c'] < 1.02, 'tv_n'].mean() if (nb_rows['c'] < 1.02).any() else np.nan,
                         'mean_tv_raw_to_widen_where_c_below_1.02': nb_rows.loc[nb_rows['c'] < 1.02, 'tv_w'].mean() if (nb_rows['c'] < 1.02).any() else np.nan})
    return pd.DataFrame(rows)


def keep_rule(pooled, n_boot=ev.N_BOOT, n_boot_bonf=10000, seed=0):
    """
    Plan 7e item 6, clause by clause, on the pooled dev folds (pooled: {stat: frame}). Returns (DataFrame of clauses, verdict bool). Adopt only if every clause holds:
      1  threes: paired log-score difference (nb - widen) of P(Y > 0.5) has a 95% upper bound below 0
      2  threes: the same for P(Y > 1.5)
      3  each bet stat (rebounds, assists, threes, ra): the 95% upper bound of the paired CRPS difference is at most +0.05% of the widen mean CRPS
      4  threes: |mean predicted P(0) - observed P(0)| under nb is no larger than under widen
    A stat missing from `pooled` fails its clause.
    """
    kw = {'n_boot': n_boot, 'n_boot_bonf': n_boot_bonf, 'seed': seed}
    rows = []
    t = pooled.get('threes')
    for i, line in enumerate(LINES, start=1):
        if t is None or not len(t):
            rows.append({'clause': i, 'what': f'threes log score of P(Y > {line}): 95% upper bound of nb - widen below 0', 'passed': False, 'value': 'no threes rows'})
            continue
        r = logscore_row(t, 'threes', line, 'dev_pooled', 'all_played', True, **kw)
        rows.append({'clause': i, 'what': f'threes log score of P(Y > {line}): 95% upper bound of nb - widen below 0', 'passed': bool(r['hi95'] < 0),
                     'value': f"diff {r['diff_nb_minus_widen']:+.6f}, 95% [{r['lo95']:+.6f}, {r['hi95']:+.6f}], Bonferroni [{r['bonf_lo']:+.6f}, {r['bonf_hi']:+.6f}], n {r['n']}"})
    for s in BET_STATS:
        f = pooled.get(s)
        if f is None or not len(f):
            rows.append({'clause': 3, 'what': f'{s} CRPS: 95% upper bound of nb - widen at most +0.05% of the widen mean', 'passed': False, 'value': 'no rows'})
            continue
        r = paired(f['crps_n'].to_numpy() - f['crps_w'].to_numpy(), f['game_id'].to_numpy(), True, **kw)
        m = f['crps_w'].mean()
        rows.append({'clause': 3, 'what': f'{s} CRPS: 95% upper bound of nb - widen at most +0.05% of the widen mean', 'passed': bool(r['hi'] <= CRPS_MARGIN * m),
                     'value': f"diff {100 * r['diff'] / m:+.4f}%, 95% [{100 * r['lo'] / m:+.4f}%, {100 * r['hi'] / m:+.4f}%], Bonferroni [{100 * r['bonf_lo'] / m:+.4f}%, {100 * r['bonf_hi'] / m:+.4f}%], widen mean {m:.5f}, n {len(f)}"})
    if t is None or not len(t):
        rows.append({'clause': 4, 'what': 'threes abs(mean predicted P(0) - observed P(0)) under nb no larger than under widen', 'passed': False, 'value': 'no threes rows'})
    else:
        z = (t['y'].to_numpy() == 0).astype(float)
        bw, bn = t['p0_w'].mean() - z.mean(), t['p0_n'].mean() - z.mean()
        rows.append({'clause': 4, 'what': 'threes abs(mean predicted P(0) - observed P(0)) under nb no larger than under widen', 'passed': bool(abs(bn) <= abs(bw)),
                     'value': f'observed {z.mean():.5f}; widen predicts {t["p0_w"].mean():.5f} (bias {bw:+.5f}); nb predicts {t["p0_n"].mean():.5f} (bias {bn:+.5f})'})
    df = pd.DataFrame(rows)
    return df, bool(df['passed'].all())


def rule_bets_effect(season, frames, bets, k_by_stat, e, min_books, stats=BET_STATS):
    """
    Item 7, descriptive: the frozen selection rule (shrunk probability with the frozen k, earliest slot, best-EV bettable quote) applied at the consensus lines 0.5 and 1.5 only, once with
    the widen pmfs and once with the nb pmfs (the 4,000-draw raw pmfs, no R2b), and how the bets differ. ROI is at the selected quote and at the median bettable price of the same market and slot.
    """
    out = []
    b_all = bets[bets['stat'].isin(stats) & bets['line'].isin(LINES)]
    for s in stats:
        f, b = frames.get(s), b_all[b_all['stat'] == s]
        if f is None or not len(b):
            continue
        med = b.groupby(ROW_KEYS + ['line']).agg(over_med=('over_dec', 'median'), under_med=('under_dec', 'median')).reset_index()
        sels = {}
        for tag, arm in (('w', 'widen'), ('n', 'nb')):
            fr = f[ROW_KEYS + ['y', f'p_over05_{tag}', f'p_over15_{tag}']].rename(columns={f'p_over05_{tag}': 'p05', f'p_over15_{tag}': 'p15'})
            q = b.merge(fr, on=ROW_KEYS, how='inner', validate='many_to_one')
            q['p_mod'] = np.where(q['line'] == 0.5, q['p05'], q['p15'])
            q['push'] = 0.0
            sel = rules.select(rules.add_ev(q, k_by_stat), e, 'earliest', min_books)
            sel = sel.merge(med, on=ROW_KEYS + ['line'], how='left')
            over = sel['best_side'] == 'over'
            win = np.where(over, sel['y'] > sel['line'], sel['y'] < sel['line'])
            sel['profit_best'] = np.where(win, sel['dec_price'] - 1.0, -1.0)
            dm = np.where(over, sel['over_med'], sel['under_med'])
            sel['profit_med'] = np.where(win, dm - 1.0, -1.0)
            sel['key'] = sel['game_id'].astype(str) + '|' + sel['player_id'].astype(str)
            sels[arm] = sel
        w, n = sels['widen'], sels['nb']
        for line in list(LINES) + ['both']:
            ws, ns = (w, n) if line == 'both' else (w[w['line'] == line], n[n['line'] == line])
            both = ws.merge(ns[['key', 'best_side']], on='key', suffixes=('', '_nb'))
            only_w, only_n = ws[~ws['key'].isin(ns['key'])], ns[~ns['key'].isin(ws['key'])]

            def roi(d, col):
                return d[col].mean() if len(d) else np.nan
            out.append({'season': season, 'stat': s, 'line': line, 'n_bets_widen': len(ws), 'n_bets_nb': len(ns), 'n_both': len(both), 'n_only_widen': len(only_w), 'n_only_nb': len(only_n),
                        'n_side_flipped': int((both['best_side'] != both['best_side_nb']).sum()),
                        'claimed_ev_widen': roi(ws, 'best_ev'), 'claimed_ev_nb': roi(ns, 'best_ev'), 'roi_best_widen': roi(ws, 'profit_best'), 'roi_best_nb': roi(ns, 'profit_best'),
                        'roi_med_widen': roi(ws, 'profit_med'), 'roi_med_nb': roi(ns, 'profit_med'), 'roi_med_only_widen': roi(only_w, 'profit_med'), 'roi_med_only_nb': roi(only_n, 'profit_med')})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ report

def md_table(df, digits=4):
    if df is None or len(df) == 0:
        return '_none_\n'
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: '' if pd.isna(v) else f'{v:.{digits}f}')
    cols = [str(c) for c in d.columns]
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|'] + ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in d.itertuples(index=False)]) + '\n'


def write_report(path, ctx, keep, verdict, tables):
    L = ['# 7e: negative-binomial dispersion against `widen` (S8)', '',
         f"Raw version `{ctx['raw_version']}`, base version `{ctx['base_version']}` (fitted c), registry root `{ctx['registry_root']}`. Dev folds {', '.join(ctx['folds'])} decide; {', '.join(ctx['seen']) or 'no season'} reported as seen and never enters the rule. "
         f"2025-26 is refused. Differences are nb minus widen (negative = nb better), paired, game-clustered bootstrap ({ctx['n_boot']} draws; Bonferroni intervals {ctx['n_boot_bonf']} draws at level 0.05/{N_COMPARISONS} = {ALPHA_BONF:.4f}).", '',
         '## Keep rule (plan 7e item 6), pooled dev folds', '',
         md_table(keep[['clause', 'what', 'passed', 'value']]), f"**Verdict: {'ADOPT as a candidate component (all four clauses hold)' if verdict else 'DO NOT ADOPT (at least one clause fails)'}.** "
         'Adoption would only add nb to the single bundled candidate; the forward record confirms (plan A.2, A.6).', '']
    for title, key in (('Share of rows without a negative binomial, and what nb does to the pmf (item 7)', 'existence'), ('Log score of P(Y > line), all played rows (item 5)', 'logscore'),
                       ('Log score at the actual book lines 0.5 and 1.5 (item 5; price files)', 'booklines'), ('Calibration of P(Y = 0) (item 5)', 'p0'),
                       ('P(Y = 0) reliability by decile of the widen P(0), threes, pooled dev', 'p0_rel'), ('CRPS of the four bet stats (item 5)', 'crps'),
                       ('Effect on the rule\'s bets at the lines 0.5 and 1.5 (item 7, descriptive only)', 'rule_bets'), ('Fitted dispersion factors c (base version, unchanged)', 'c')):
        L += [f'## {title}', '', md_table(tables.get(key)), '']
    L += ['## Row checks', '', '```', json.dumps(ctx['info'], indent=2, default=str), '```', '',
          '## Notes', '', *[f'- {n}' for n in NOTES], '', f"Tables and per-row data are under `{ctx['out']}`.", '']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(L), encoding='utf-8')


NOTES = [
    'Scope of "pooled dev" in the keep rule: all played dev rows of the three dev folds, every decision slot, one pooled game-clustered bootstrap (game ids are unique across seasons). The book-line table is reported beside it and is not part of the rule (the book lines exist for 2023-24 only).',
    'Clause 3 reads "worse with a 95% upper bound above +0.05%" as: the upper bound of the paired CRPS difference, divided by the widen mean CRPS of the pooled dev rows, is at most 0.0005. Per-fold values are reported, not decisive.',
    'Clause 4 compares point estimates of the absolute bias of mean predicted P(0) on all played dev rows of threes; the intervals are printed, not used.',
    'nb replaces the SHAPE of every row whose bucket has c > 1 (a negative binomial with the raw pmf\'s mean and variance c times its variance). Rows whose bucket sits at the floor c = 1 are untouched, but a bucket with c just above 1 changes shape although widen would hardly move the row: see the distance columns of the first table.',
    'widen does not hit the target variance c times s2 exactly (it is a stretch of a piecewise-linear CDF, then discretised): the variance-over-target columns show it, so part of any nb-widen difference is a variance difference, not shape alone.',
    'The Bonferroni intervals use level 0.05/23 as in plan A.5 and are computed for the pooled dev scope and the seen season only; per-fold rows carry the 95% interval alone. The effect on the rule\'s bets uses the 4,000-draw raw pmfs and no R2b recalibration; it is descriptive.',
]


# ------------------------------------------------------------------------------------------------ main

def run(a):
    folds, seen = list(a.folds), [s for s in a.seen if s.lower() != 'none']
    refuse_holdout(folds, '--folds')
    refuse_holdout(seen, '--seen')
    if set(folds) & set(seen):
        raise SystemExit('--folds and --seen overlap')
    root = registry.resolve_root(a.registry_root) if hasattr(registry, 'resolve_root') else Path(a.registry_root)
    out, report = Path(a.out), Path(a.report)
    check_output_paths(out, report, [Path(root) / a.raw_version, Path(root) / a.base_version, a.prices_dir, a.features_db])
    out.mkdir(parents=True, exist_ok=True)
    logging.info(f'registry {root}; raw {a.raw_version}; base {a.base_version}; dev {folds}; seen {seen}')
    labels = read_labels(a.features_db)
    frames, ctabs, infos = {}, [], {}
    for season in folds + seen:
        frames[season], ct, infos[season] = build_season(season, a.raw_version, a.base_version, root, labels)
        ctabs.append(ct)
        if a.save_rows:
            for s, f in frames[season].items():
                f.to_csv(out / f'rows_{season}_{s}.csv.gz', index=False)
    ctab = pd.concat(ctabs, ignore_index=True)
    kw = {'n_boot': a.n_boot, 'n_boot_bonf': a.n_boot_bonf}
    pooled = {'dev_pooled': {s: pd.concat([frames[x][s] for x in folds], ignore_index=True) for s in BET_STATS}}
    for x in folds:
        pooled[f'dev_{x}'] = frames[x]
    for x in seen:
        pooled[f'seen_{x}'] = frames[x]
    bonf_scopes = {'dev_pooled'} | {f'seen_{x}' for x in seen}
    keep, verdict = keep_rule(pooled['dev_pooled'], **kw)
    tables = {'existence': existence_table(frames), 'logscore': logscore_table(pooled, bonf_scopes, **kw), 'p0': p0_table(pooled, bonf_scopes, **kw), 'crps': crps_table(pooled, bonf_scopes, **kw),
              'p0_rel': p0_reliability(pooled['dev_pooled']['threes']), 'c': ctab}
    cons_by, bets_by = {}, {}
    for season in folds + seen:
        pp = Path(a.prices_dir) / f'prices_{season}.pkl'
        if pp.exists():
            cons_by[season], bets_by[season] = read_prices(pp)
        else:
            logging.info(f'no price file for {season}: its book-line rows are skipped')
    book = book_line_frames(frames, cons_by)
    tables['booklines'] = booklines_table(book, set(folds), set(seen), bonf_scopes, **kw)
    frozen = json.loads(Path(a.frozen_rule).read_text(encoding='utf-8')) if Path(a.frozen_rule).exists() else None
    if frozen:
        eff = [rule_bets_effect(season, frames[season], bets_by[season], frozen['k_by_stat'], frozen['primary']['e'], frozen['primary']['min_books']) for season in bets_by]
        tables['rule_bets'] = pd.concat(eff, ignore_index=True) if eff else pd.DataFrame()
    else:
        tables['rule_bets'] = pd.DataFrame()
    for name, t in tables.items():
        t.to_csv(out / f'{name}.csv', index=False)
    keep.to_csv(out / 'keep_rule.csv', index=False)
    (out / 'keep_rule.json').write_text(json.dumps({'adopt': verdict, 'clauses': keep.to_dict('records'), 'folds': folds, 'seen': seen}, indent=2, default=str), encoding='utf-8')
    manifest = None
    mp = Path(root) / a.raw_version / 'manifest.json'
    if mp.exists():
        manifest = {k: v for k, v in json.loads(mp.read_text(encoding='utf-8')).items() if k in ('n_sims', 'seed', 'source_version', 'features_db_sha256', 'git_commit')}
    ctx = {'raw_version': a.raw_version, 'base_version': a.base_version, 'registry_root': str(root), 'folds': folds, 'seen': seen, 'n_boot': a.n_boot, 'n_boot_bonf': a.n_boot_bonf,
           'out': str(out), 'info': {'raw_manifest': manifest, **infos}}
    write_report(report, ctx, keep, verdict, tables)
    logging.info(f'keep rule: {"ADOPT" if verdict else "DO NOT ADOPT"}; report {report}')
    return verdict


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw-version', required=True, help='prediction version holding raw_<stat>.npy + index.csv per season (props_phase5_predict.py --all-played --save-raw)')
    ap.add_argument('--base-version', required=True, help='model version whose bundles hold the fitted dispersion factors')
    ap.add_argument('--registry-root', required=True)
    ap.add_argument('--folds', nargs='+', default=list(DEV_FOLDS))
    ap.add_argument('--seen', nargs='*', default=list(SEEN), help='seasons reported as seen (never decide); "none" for no season')
    ap.add_argument('--prices-dir', required=True, help='folder of prices_<season>.pkl (consensus, bets)')
    ap.add_argument('--features-db', default='data/props_features_v3.db')
    ap.add_argument('--frozen-rule', default='config/props_frozen_rule.json', help='k per stat and the primary selection thresholds, for the descriptive bets table (read only)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--n-boot', type=int, default=ev.N_BOOT)
    ap.add_argument('--n-boot-bonf', type=int, default=10000, help='replicates for the Bonferroni intervals (the 0.11% tail needs more than 2,000)')
    ap.add_argument('--no-save-rows', dest='save_rows', action='store_false', help='do not write the per-row frames')
    return ap.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run(parse(argv))


if __name__ == '__main__':
    main()
