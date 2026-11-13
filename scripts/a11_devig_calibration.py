"""
Step S2 of reports/props/audit_2026-11/WP6_WP7_experiment_plan.md: experiment 7a, de-vig method and consensus set (pre-registered; no model involved).

    PY scripts/a11_devig_calibration.py --quotes $OUT/quotes --features-db data/props_features_v3.db --out $OUT/results/7a --report $RPT/7a_devig/report.md

Inputs: the consensus tables of the 9 variants (3 de-vig methods x 3 consensus sets) written by scripts/a11_cache_quotes.py, and the outcomes in `player_game_labels` of the
features database (opened read-only; only that table is read).

Rows: for each season, the (game, player, slot, stat) keys that every one of the 9 variants prices at the SAME consensus line (item 6), joined to the outcome of players who
played (lbl_played = 1 and lbl_minutes > 0); a push (stat equals the line) has no winner and is dropped, as in props.models.market_bench.compare. hit = 1 when the stat is above the
line. All 8 markets are used for the keep rule, as the plan does not restrict them; the four bet stats alone are reported as a secondary evaluation that decides nothing.

Metrics (item 6): log loss (probabilities clipped to 1e-4, as market_bench) and Brier of p_market, mean bias (hit - p), logistic recalibration hit ~ logit(p) (intercept, slope; ideal 0, 1), bias by
p_market bucket, all by season and stat. Paired differences against the current method (multiplicative, mean over all books) are date-clustered bootstraps (props.live.monitor.cluster_bootstrap_mean:
2,000 replicates, seed 20261112, the same date resamples for every comparison); the 95% interval and the Bonferroni interval (0.05 / 23 comparisons, plan A.5) are both printed.

Keep rule (item 7): a variant replaces the current method if, in EACH of 2023-24 and 2024-25, its paired log-loss difference has a 95% upper bound below 0, |mean bias| is no larger than the
current method's (point), and its coverage is at least 95% of the current method's. Among passing variants the lowest pooled log loss wins; an exact tie goes to the earlier variant in
a11_cache_quotes.VARIANTS. Result: `choice.json` (devig_method, consensus_set, passed, reason) under --out.

Writes only under --out and to the --report file. Reads no season later than 2024-25 (refuses 2025-26 and any game date after 2025-06-30).
"""
import argparse
import contextlib
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
import a11_cache_quotes as cq  # noqa: E402
from props.live import monitor  # noqa: E402

N_BOOT = 2000
SEED = 20261112
N_COMPARISONS = 23                                   # plan A.5
ALPHA = 0.05
ALPHA_BONF = ALPHA / N_COMPARISONS
MIN_COVERAGE = 0.95
EPS = 1e-4
BUCKET_EDGES = [0.0, 0.35, 0.45, 0.50, 0.55, 0.65, 1.0 + 1e-9]
BUCKET_LABELS = ['<0.35', '0.35-0.45', '0.45-0.50', '0.50-0.55', '0.55-0.65', '>=0.65']
BET_STATS = ('rebounds', 'assists', 'threes', 'ra')
SEASONS_BOTH = tuple(cq.SEASONS)
OBS = {'points': 'y_points', 'rebounds': 'y_rebounds', 'assists': 'y_assists', 'threes': 'y_threes', 'pra': 'y_pra'}
COMBOS = {'pr': ('y_points', 'y_rebounds'), 'pa': ('y_points', 'y_assists'), 'ra': ('y_rebounds', 'y_assists')}
LABEL_COLUMNS = ['game_id', 'player_id', 'lbl_played', 'lbl_minutes', 'y_points', 'y_rebounds', 'y_assists', 'y_threes', 'y_pra']
VNAMES = [cq.variant_name(v) for v in cq.VARIANTS]
CURRENT = cq.variant_name(cq.CURRENT)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_tables(quotes_dir, seasons):
    """{season: {variant name: consensus table}} from the pickles of a11_cache_quotes."""
    out = {}
    for s in seasons:
        cq.check_season(s)
        out[s] = {}
        for v in cq.VARIANTS:
            path = Path(quotes_dir) / s / f'cons_{cq.variant_name(v)}.pkl'
            if not path.exists():
                raise FileNotFoundError(f'{path} not found: run scripts/a11_cache_quotes.py for {s} first')
            out[s][cq.variant_name(v)] = pd.read_pickle(path)
    return out


def load_labels(features_db):
    """Outcome columns of player_game_labels, read-only."""
    uri = 'file:' + Path(features_db).resolve().as_posix() + '?mode=ro'
    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as conn:
        conn.execute('PRAGMA query_only = ON')
        lab = pd.read_sql(f"SELECT {', '.join(LABEL_COLUMNS)} FROM player_game_labels", conn)
    lab['game_id'] = lab['game_id'].astype(str)
    return lab


def check_dates(tables):
    last = max((str(t['game_date'].max()) for per in tables.values() for t in per.values() if len(t)), default='')
    if last > cq.LAST_ALLOWED_DATE:
        raise cq.SeasonRefused(f'quote tables contain game dates up to {last}, after {cq.LAST_ALLOWED_DATE}: refused (plan A.1)')


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def common_sample(per_variant):
    """Keys priced by every variant at one and the same consensus line. Columns: KEY, game_date, line, p::<variant> for each variant."""
    cur = per_variant[CURRENT]
    out = cur[cq.KEY + ['game_date', 'consensus_line', 'consensus_fair_over']].rename(columns={'consensus_line': 'line', 'consensus_fair_over': f'p::{CURRENT}'})
    for name in VNAMES:
        if name == CURRENT:
            continue
        t = per_variant[name][cq.KEY + ['consensus_line', 'consensus_fair_over']].rename(columns={'consensus_line': f'line::{name}', 'consensus_fair_over': f'p::{name}'})
        out = out.merge(t, on=cq.KEY, how='inner', validate='one_to_one')
    same = np.ones(len(out), bool)
    for name in VNAMES:
        if name != CURRENT:
            same &= (out[f'line::{name}'].to_numpy() == out['line'].to_numpy())
    out = out[same]
    return out[cq.KEY + ['game_date', 'line'] + [f'p::{n}' for n in VNAMES]].reset_index(drop=True)


def observed(frame, stat):
    if stat in OBS:
        return frame[OBS[stat]].to_numpy(float)
    a, b = COMBOS[stat]
    return (frame[a] + frame[b]).to_numpy(float)


def attach_outcomes(common, labels):
    """Joins outcomes. Returns (frame with hit, counts). Keeps players who played; drops pushes and rows without an outcome."""
    counts = {'common_rows': int(len(common))}
    m = common.merge(labels, on=['game_id', 'player_id'], how='left', validate='many_to_one', indicator=True)
    counts['no_label'] = int((m['_merge'] == 'left_only').sum())
    m = m[m['_merge'] == 'both'].drop(columns='_merge')
    played = (m['lbl_played'] == 1) & (m['lbl_minutes'] > 0)
    counts['did_not_play'] = int((~played).sum())
    m = m[played].copy()
    y = np.full(len(m), np.nan)
    for stat, g in m.groupby('stat').groups.items():
        pos = m.index.get_indexer(g)
        y[pos] = observed(m.loc[g], stat)
    line = m['line'].to_numpy(float)
    keep = ~np.isnan(y) & (y != line)
    counts['missing_stat'] = int(np.isnan(y).sum())
    counts['push'] = int((~np.isnan(y) & (y == line)).sum())
    m = m[keep].copy()
    m['hit'] = (y[keep] > line[keep]).astype(float)
    counts['used'] = int(len(m))
    return m.drop(columns=[c for c in LABEL_COLUMNS if c in m.columns and c not in cq.KEY]).reset_index(drop=True), counts


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def ll(p, h, eps=EPS):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return -(h * np.log(p) + (1 - h) * np.log(1 - p))


def logit_recal(p, h, iters=40):
    """Intercept and slope of the logistic regression hit ~ logit(p) (Newton steps). Ideal: 0 and 1."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    h = np.asarray(h, float)
    if len(p) < 10:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    beta = np.zeros(2)
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = mu * (1 - mu)
        H = (X * w[:, None]).T @ X
        g = X.T @ (h - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def seg_metrics(p, h):
    p, h = np.asarray(p, float), np.asarray(h, float)
    if len(p) == 0:
        return {'n': 0, 'logloss': np.nan, 'brier': np.nan, 'mean_p': np.nan, 'hit_rate': np.nan, 'bias': np.nan, 'recal_intercept': np.nan, 'recal_slope': np.nan}
    icpt, slope = logit_recal(p, h)
    return {'n': int(len(p)), 'logloss': float(ll(p, h).mean()), 'brier': float(((p - h) ** 2).mean()), 'mean_p': float(p.mean()), 'hit_rate': float(h.mean()),
            'bias': float((h - p).mean()), 'recal_intercept': icpt, 'recal_slope': slope}


def metric_tables(df, seasons):
    """Pooled, by-stat and by-bucket metric tables for every variant; 'both' pools the seasons."""
    pooled, by_stat, by_bucket = [], [], []
    for season in list(seasons) + (['both'] if len(seasons) > 1 else []):
        d = df if season == 'both' else df[df['season'] == season]
        h = d['hit'].to_numpy(float)
        for name in VNAMES:
            p = d[f'p::{name}'].to_numpy(float)
            pooled.append({'season': season, 'variant': name, **seg_metrics(p, h)})
            for stat, g in d.groupby('stat'):
                by_stat.append({'season': season, 'variant': name, 'stat': stat, **seg_metrics(g[f'p::{name}'], g['hit'])})
            b = pd.cut(pd.Series(p, index=d.index), BUCKET_EDGES, labels=BUCKET_LABELS, right=False)
            for lab in BUCKET_LABELS:
                sel = (b == lab).to_numpy()
                by_bucket.append({'season': season, 'variant': name, 'bucket': lab, **seg_metrics(p[sel], h[sel])})
    return pd.DataFrame(pooled), pd.DataFrame(by_stat), pd.DataFrame(by_bucket)


def paired(diff, dates, n_boot=N_BOOT, seed=SEED):
    """Mean of per-row differences with date-clustered percentile intervals at the 95% and Bonferroni levels (same seed -> the same date resamples in every call)."""
    a = monitor.cluster_bootstrap_mean(diff, dates, n_boot=n_boot, seed=seed, alpha=ALPHA)
    b = monitor.cluster_bootstrap_mean(diff, dates, n_boot=n_boot, seed=seed, alpha=ALPHA_BONF)
    return {'diff': a['point'], 'lo95': a['lo'], 'hi95': a['hi'], 'lo_bonf': b['lo'], 'hi_bonf': b['hi'], 'se': a['se'], 'n': a['n'], 'n_dates': a['n_clusters']}


def paired_tables(df, seasons, n_boot=N_BOOT, seed=SEED, by_stat=True):
    rows, rows_stat = [], []
    for season in list(seasons) + (['both'] if len(seasons) > 1 else []):
        d = df if season == 'both' else df[df['season'] == season]
        h = d['hit'].to_numpy(float)
        dates = d['game_date'].to_numpy()
        base_ll, base_br = ll(d[f'p::{CURRENT}'].to_numpy(float), h), (d[f'p::{CURRENT}'].to_numpy(float) - h) ** 2
        for name in VNAMES:
            if name == CURRENT:
                continue
            p = d[f'p::{name}'].to_numpy(float)
            rows.append({'season': season, 'variant': name, 'metric': 'logloss', **paired(ll(p, h) - base_ll, dates, n_boot, seed)})
            rows.append({'season': season, 'variant': name, 'metric': 'brier', **paired((p - h) ** 2 - base_br, dates, n_boot, seed)})
            if by_stat:
                for stat in sorted(d['stat'].unique()):
                    m = (d['stat'] == stat).to_numpy()
                    rows_stat.append({'season': season, 'variant': name, 'stat': stat, 'metric': 'logloss',
                                      **paired((ll(p, h) - base_ll)[m], dates[m], n_boot, seed)})
    return pd.DataFrame(rows), pd.DataFrame(rows_stat)


def coverage_table(tables):
    """Rows with a consensus per variant, as a share of the current method's, by season and stat (all rows of the quote tables, before outcomes and the common-line restriction)."""
    rows = []
    for season, per in tables.items():
        cur = per[CURRENT]['stat'].value_counts()
        for name in VNAMES:
            n = per[name]['stat'].value_counts()
            for stat in sorted(cur.index):
                rows.append({'season': season, 'variant': name, 'stat': stat, 'n': int(n.get(stat, 0)), 'n_current': int(cur[stat]), 'share_of_current': float(n.get(stat, 0) / cur[stat])})
            rows.append({'season': season, 'variant': name, 'stat': 'ALL', 'n': int(n.sum()), 'n_current': int(cur.sum()), 'share_of_current': float(n.sum() / cur.sum()) if cur.sum() else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keep rule
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def keep_rule(df, cov, seasons, stats=None, n_boot=N_BOOT, seed=SEED, require_seasons=SEASONS_BOTH):
    """
    The 7a keep rule on the rows of df (optionally restricted to `stats`; coverage restricted the same way). Returns (per-season table, per-variant table, chosen variant or None).
    A variant passes only if every season in `require_seasons` is present and passes.
    """
    d0 = df if stats is None else df[df['stat'].isin(stats)]
    c0 = cov if stats is None else cov[cov['stat'].isin(stats)]
    rows = []
    for season in seasons:
        d = d0[d0['season'] == season]
        h = d['hit'].to_numpy(float)
        dates = d['game_date'].to_numpy()
        p_cur = d[f'p::{CURRENT}'].to_numpy(float)
        base = ll(p_cur, h)
        bias_cur = float((h - p_cur).mean()) if len(h) else np.nan
        cc = c0[(c0['season'] == season) & (c0['stat'] != 'ALL')]
        for name in VNAMES:
            if name == CURRENT:
                continue
            p = d[f'p::{name}'].to_numpy(float)
            pr = paired(ll(p, h) - base, dates, n_boot, seed) if len(h) else {'diff': np.nan, 'hi95': np.nan, 'lo95': np.nan, 'hi_bonf': np.nan, 'lo_bonf': np.nan}
            bias = float((h - p).mean()) if len(h) else np.nan
            ccv = cc[cc['variant'] == name]['n'].sum()
            ccc = cc[cc['variant'] == CURRENT]['n'].sum()
            coverage = float(ccv / ccc) if ccc else np.nan
            ok_ll = bool(pr['hi95'] < 0)
            ok_bias = bool(abs(bias) <= abs(bias_cur) + 1e-12)
            ok_cov = bool(coverage >= MIN_COVERAGE)
            rows.append({'season': season, 'variant': name, 'dlogloss': pr['diff'], 'lo95': pr['lo95'], 'hi95': pr['hi95'], 'lo_bonf': pr['lo_bonf'], 'hi_bonf': pr['hi_bonf'],
                         'bias': bias, 'bias_current': bias_cur, 'coverage': coverage, 'ok_logloss': ok_ll, 'ok_bias': ok_bias, 'ok_coverage': ok_cov, 'season_pass': ok_ll and ok_bias and ok_cov})
    per_season = pd.DataFrame(rows)
    pooled_ll = {n: float(ll(d0[f'p::{n}'], d0['hit'].to_numpy(float)).mean()) if len(d0) else np.nan for n in VNAMES}
    out = []
    for name in VNAMES:
        if name == CURRENT:
            continue
        s = per_season[per_season['variant'] == name] if len(per_season) else per_season
        have = set(s['season']) if len(s) else set()
        passed = set(require_seasons) <= have and bool(s['season_pass'].all())
        reasons = []
        for _, r in s.iterrows():
            if not r['ok_logloss']:
                reasons.append(f"{r['season']}: log-loss upper bound {r['hi95']:+.5f} not below 0")
            if not r['ok_bias']:
                reasons.append(f"{r['season']}: |bias| {abs(r['bias']) * 100:.3f} pts above the current method's {abs(r['bias_current']) * 100:.3f}")
            if not r['ok_coverage']:
                reasons.append(f"{r['season']}: coverage {r['coverage'] * 100:.1f}% of current, below 95%")
        if not set(require_seasons) <= have:
            reasons.append('both 2023-24 and 2024-25 are required and one is missing')
        out.append({'variant': name, 'passed': passed, 'pooled_logloss': pooled_ll[name], 'pooled_logloss_current': pooled_ll[CURRENT], 'why_not': '; '.join(reasons)})
    per_variant = pd.DataFrame(out)
    chosen = None
    if len(per_variant) and per_variant['passed'].any():
        cand = per_variant[per_variant['passed']].copy()
        cand['order'] = cand['variant'].map({n: i for i, n in enumerate(VNAMES)})
        chosen = cand.sort_values(['pooled_logloss', 'order'], kind='mergesort').iloc[0]['variant']
    return per_season, per_variant, chosen


def make_choice(per_variant, chosen, seasons):
    cur_method, cur_set = cq.CURRENT
    if chosen is None:
        why = '; '.join(f"{r.variant}: {r.why_not}" for r in per_variant.itertuples()) if len(per_variant) else 'no comparison was possible'
        return {'devig_method': cur_method, 'consensus_set': cur_set, 'passed': False, 'variant': CURRENT, 'seasons': list(seasons),
                'reason': 'current method kept: no variant meets the 7a keep rule. ' + why}
    method, sname = next(v for v in cq.VARIANTS if cq.variant_name(v) == chosen)
    n_pass = int(per_variant['passed'].sum())
    pl = float(per_variant.loc[per_variant['variant'] == chosen, 'pooled_logloss'].iloc[0])
    return {'devig_method': method, 'consensus_set': sname, 'passed': True, 'variant': chosen, 'seasons': list(seasons),
            'reason': f'{chosen} passes the 7a keep rule in {" and ".join(seasons)}; it has the lowest pooled log loss ({pl:.5f}) of {n_pass} passing variant(s)'}


def bias_removed(pooled):
    rows = []
    for season, g in pooled.groupby('season'):
        cur = g[g['variant'] == CURRENT].iloc[0]
        for _, r in g.iterrows():
            rows.append({'season': season, 'variant': r['variant'], 'bias_pts': 100 * r['bias'], 'bias_current_pts': 100 * cur['bias'],
                         'removed_pts': 100 * (abs(cur['bias']) - abs(r['bias'])), 'removed_share': (abs(cur['bias']) - abs(r['bias'])) / abs(cur['bias']) if cur['bias'] else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_frame(tables, labels, seasons):
    parts, counts = [], {}
    for s in seasons:
        common = common_sample(tables[s])
        d, c = attach_outcomes(common, labels)
        c['rows_current_method'] = int(len(tables[s][CURRENT]))
        parts.append(d.assign(season=s))
        counts[s] = c
    return pd.concat(parts, ignore_index=True), counts


def evaluate(tables, labels, seasons, n_boot=N_BOOT, seed=SEED):
    """Everything of 7a from in-memory inputs: tables {season: {variant: table}} and labels. Returns a dict of result tables, the keep-rule tables and the choice."""
    for s in seasons:
        cq.check_season(s)
    check_dates({s: tables[s] for s in seasons})
    df, counts = build_frame(tables, labels, seasons)
    cov = coverage_table({s: tables[s] for s in seasons})
    pooled, by_stat, by_bucket = metric_tables(df, seasons)
    pair, pair_stat = paired_tables(df, seasons, n_boot, seed)
    per_season, per_variant, chosen = keep_rule(df, cov, seasons, BET_STATS, n_boot, seed)     # decides (plan amendment 5 of 2026-11-12): the four bet stats
    per_season_b, per_variant_b, chosen_b = keep_rule(df, cov, seasons, None, n_boot, seed)           # secondary, decides nothing: all eight markets
    choice = make_choice(per_variant, chosen, seasons)
    return {'frame': df, 'counts': counts, 'coverage': cov, 'pooled': pooled, 'by_stat': by_stat, 'by_bucket': by_bucket, 'paired': pair, 'paired_by_stat': pair_stat,
            'keep_season': per_season, 'keep_variant': per_variant, 'keep_season_all8': per_season_b, 'keep_variant_all8': per_variant_b, 'chosen_all_markets': chosen_b,
            'bias_removed': bias_removed(pooled), 'choice': choice}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def md_table(df, fmt=None, index=False):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(([df.index.name or ''] if index else []) + [str(c) for c in cols]) + ' |', '|' + '---|' * (len(cols) + (1 if index else 0))]
    for idx, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append('yes' if v else 'no')
            elif isinstance(v, (float, np.floating)):
                cells.append('' if np.isnan(v) else (fmt.get(c, '{:.5f}')).format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(([str(idx)] if index else []) + cells) + ' |')
    return '\n'.join(lines)


def ci(lo, hi, f='{:+.5f}'):
    return f'[{f.format(lo)}, {f.format(hi)}]'


def write_report(res, seasons, args, path):
    ch = res['choice']
    pooled, kp, kv = res['pooled'], res['keep_season'], res['keep_variant']
    out = ['# 7a: de-vig method and consensus set', '',
           f"Step S2 of `WP6_WP7_experiment_plan.md`, produced by `scripts/a11_devig_calibration.py` on {time.strftime('%Y-%m-%d %H:%M')}. Seasons: {', '.join(seasons)}. Quotes: `{args.get('quotes')}`. "
           f"Outcomes: `player_game_labels` of `{args.get('features_db')}`. Date-clustered bootstrap: {args.get('n_boot')} replicates, seed {args.get('seed')}."
           + ('' if args.get('n_boot') == N_BOOT and args.get('seed') == SEED else ' **Non-default bootstrap settings (a smoke run, not the registered analysis).**'), '',
           '## Decision', '']
    if ch['passed']:
        out += [f"**Chosen p_market method: `{ch['variant']}` (de-vig {ch['devig_method']}, consensus set {ch['consensus_set']}).** {ch['reason']}.", '']
    else:
        out += [f"**Current method kept** (multiplicative de-vig, mean over all books). {ch['reason']}", '']
    out += ['`choice.json` carries devig_method, consensus_set, passed and reason for the later steps.', '',
            '## Keep rule, variant by variant', '',
            'Rule (plan 7a item 7, judged on the four bet stats by amendment 5 of 2026-11-12): in each of 2023-24 and 2024-25 the paired log-loss difference (variant minus current; negative is better) has a 95% upper bound below 0, |mean bias| is no larger than '
            "the current method's, and coverage is at least 95% of the current method's. Bonferroni intervals use 0.05/23 = 0.0022 (99.78%, plan A.5); they are printed for reference and the keep rule "
            'uses the 95% bound.', '']
    rows = []
    for r in kp.itertuples():
        rows.append({'variant': r.variant, 'season': r.season, 'd logloss': f'{r.dlogloss:+.5f}', '95% interval': ci(r.lo95, r.hi95), 'Bonferroni interval': ci(r.lo_bonf, r.hi_bonf),
                     'abs bias pts (current)': f'{abs(r.bias) * 100:.3f} ({abs(r.bias_current) * 100:.3f})', 'coverage': f'{r.coverage * 100:.1f}%',
                     'logloss ok': r.ok_logloss, 'bias ok': r.ok_bias, 'coverage ok': r.ok_coverage})
    out += [md_table(pd.DataFrame(rows)), '']
    out += ['| variant | keep rule passes | pooled log loss | reason if not |', '|---|---|---|---|']
    for r in kv.itertuples():
        out.append(f"| {r.variant} | {'yes' if r.passed else 'no'} | {r.pooled_logloss:.5f} (current {r.pooled_logloss_current:.5f}) | {r.why_not} |")
    out += ['', f"Secondary, decides nothing: the same rule on all eight markets (the decision above uses the four bet stats, {', '.join(BET_STATS)}) would choose "
            f"`{res['chosen_all_markets'] or 'the current method'}`.", '']
    c = res['counts']
    out += ['## Rows', '', 'Rows where all 9 variants price the same line, joined to outcomes of players who played, pushes dropped.', '',
            md_table(pd.DataFrame([{'season': s, **v} for s, v in c.items()]), {}), '']
    out += ['## Metrics of the 9 variants (common rows, pooled over stats and slots)', '',
            'bias = mean(hit - p_market); recalibration = logistic fit of hit on logit(p) (ideal intercept 0, slope 1).', '',
            md_table(pooled[['season', 'variant', 'n', 'logloss', 'brier', 'bias', 'mean_p', 'hit_rate', 'recal_intercept', 'recal_slope']], {'n': '{:.0f}'}), '']
    out += ['## Paired differences against the current method', '',
            'Negative is better for log loss and Brier. Date-clustered bootstrap; 95% and Bonferroni intervals.', '',
            md_table(res['paired'][['season', 'variant', 'metric', 'diff', 'lo95', 'hi95', 'lo_bonf', 'hi_bonf', 'n_dates']], {'n_dates': '{:.0f}'}), '']
    out += ['## How much of the over-bias each variant removes', '', 'Bias in points (hit - p_market, times 100); removed = |current| - |variant|.', '',
            md_table(res['bias_removed'], {}), '']
    cov = res['coverage']
    bet = cov[(cov['stat'] == 'ALL')]
    out += ['## Coverage (rows with a consensus, share of the current method)', '', md_table(bet[['season', 'variant', 'n', 'n_current', 'share_of_current']], {'share_of_current': '{:.4f}'}), '']
    bl = cov[cov['variant'].str.endswith('bettable_mean') & (cov['stat'] != 'ALL')]
    out += ['Bettable-only coverage loss by stat (share of the current method):', '', md_table(bl[['season', 'variant', 'stat', 'share_of_current']], {'share_of_current': '{:.4f}'}), '']
    bs = res['by_stat']
    for s in [x for x in bs['season'].unique()]:
        piv = bs[bs['season'] == s].pivot(index='stat', columns='variant', values='logloss')[VNAMES]
        piv.index.name = 'stat'
        out += [f'## Log loss by stat, {s}', '', md_table(piv.reset_index()), '']
        piv = bs[bs['season'] == s].pivot(index='stat', columns='variant', values='bias')[VNAMES]
        out += [f'Mean bias (hit - p) by stat, {s}:', '', md_table(piv.reset_index(), {k: '{:+.5f}' for k in VNAMES}), '']
    bb = res['by_bucket']
    for s in [x for x in bb['season'].unique()]:
        g = bb[bb['season'] == s]
        piv = g.pivot(index='bucket', columns='variant', values='bias')[VNAMES].reindex(BUCKET_LABELS)
        n = g[g['variant'] == CURRENT].set_index('bucket')['n'].reindex(BUCKET_LABELS)
        piv.insert(0, 'n (current)', n.to_numpy())
        out += [f'## Bias by p_market bucket, {s} (each variant bucketed on its own p)', '', md_table(piv.reset_index(), {**{k: '{:+.5f}' for k in VNAMES}, 'n (current)': '{:.0f}'}), '']
    out += ['## Choices where the plan was silent', '',
            '- The keep rule uses all eight markets in the quote tables (the plan names no stat restriction); the four-bet-stat version is printed above as a secondary.',
            '- A push (the stat equals the line) is dropped and a player who did not play has no outcome; both follow `props.models.market_bench.compare`.',
            '- Coverage counts every consensus row of a variant (any line), before outcomes and before the common-line restriction.',
            '- Log-loss probabilities are clipped to [1e-4, 1 - 1e-4], as `market_bench` does.',
            '- Game date for the cluster is the ET game date of the `events` table.', '']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(out), encoding='utf-8')


def write_results(res, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ('coverage', 'pooled', 'by_stat', 'by_bucket', 'paired', 'paired_by_stat', 'keep_season', 'keep_variant', 'keep_season_all8', 'keep_variant_all8', 'bias_removed'):
        res[name].to_csv(out_dir / f'{name}.csv', index=False)
    pd.DataFrame([{'season': s, **c} for s, c in res['counts'].items()]).to_csv(out_dir / 'sample_counts.csv', index=False)
    (out_dir / 'choice.json').write_text(json.dumps(res['choice'], indent=2), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--quotes', required=True, help='folder written by a11_cache_quotes.py (one sub-folder per season)')
    ap.add_argument('--features-db', required=True, help='features database with player_game_labels (opened read-only)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--seasons', nargs='+', default=list(SEASONS_BOTH))
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        for s in a.seasons:
            cq.check_season(s)
        if (Path(a.out) / 'choice.json').exists() and not a.overwrite:
            raise FileExistsError(f'{Path(a.out) / "choice.json"} exists; pass --overwrite')
        t0 = time.time()
        tables = load_tables(a.quotes, a.seasons)
        labels = load_labels(a.features_db)
        res = evaluate(tables, labels, list(a.seasons), a.n_boot, a.seed)
        write_results(res, a.out)
        write_report(res, list(a.seasons), {'quotes': a.quotes, 'features_db': a.features_db, 'n_boot': a.n_boot, 'seed': a.seed}, a.report)
        print(json.dumps(res['choice'], indent=2))
        print(f'done in {time.time() - t0:.0f}s; results in {a.out}, report {a.report}')
    except cq.SeasonRefused as e:
        sys.exit(f'REFUSED: {e}')
    return 0


if __name__ == '__main__':
    main()
