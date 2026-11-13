"""
Steps S14 (6a) and S15 (7b) of reports/props/audit_2026-11/WP6_WP7_experiment_plan.md: paired evaluation of prediction versions, the keep rule clause by clause, and the choice for the next step.

    S14  PY scripts/a11_eval_models.py --experiment 6a --reference a11_r0_on_pg1 --candidates a11_pg1 a11_pg2 --leaky a11_r0 --folds 2021-22 2022-23 2023-24 --seen 2024-25 \
             --registry-root $REG --prices-dir $OUT/prices --out $OUT/results/6a --report $RPT/6a_roster/report.md
         (only if A1 and A2 both fail: train a11_noav, then run again with --candidates a11_pg1 a11_pg2 a11_noav; the third candidate is read as A0)
    S15  PY scripts/a11_eval_models.py --experiment 7b --reference <6a winner version, from $OUT/results/6a/choice.json> --candidates a11_env --folds 2021-22 2022-23 2023-24 \
             --seen 2024-25 --registry-root $REG --prices-dir $OUT/prices --out $OUT/results/7b --report $RPT/7b_env/report.md
    optional: --pmarket $OUT/results/7a/choice.json [--quotes-dir $OUT/quotes]   the 7a p_market (default: the current method, as scripts/a11_k_estimation.py)

Inputs (all read only)
  <registry-root>/<version>/<season>/index.csv + pred_<stat>.npy  (game_id, player_id, minutes_before, season, game_date, minutes_mean; float32 pmfs, row i = row i of the index;
                                                                   opened with numpy memmap, only rebounds, assists, threes and ra are read). A training version holds the pmfs of every
                                                                   played test row of each fold; a prediction version made by props_phase5_predict.py holds the same layout.
  <registry-root>/<version>_p5/<season>/...                        quoted-row versions (20,000 draws). When EVERY version has one for a season (--quoted-suffix, default _p5) the information
                                                                   slope, the scores at the book line and the rule's ROI of that season use them; otherwise the all-played versions are used.
                                                                   Which was used is printed and written to choice.json.
  <registry-root>/<version>/manifest.json                          features_db and its sha256 (copied to choice.json for the winner)
  <prices-dir>/prices_<season>.pkl                                 (consensus, bets) as props.betting.prices.load_season returns them (scripts/a11_cache_quotes.py)
  --features-db (default data/props_features_v3.db, mode=ro)       player_game_labels of the dev seasons only (rows of later seasons are never selected), and for 7b the env_team_spread of
                                                                   player_game_at_T; both identical in every features database of the plan (same target rows)
  --frozen-rule (default config/props_frozen_rule.json)            k_by_stat for the descriptive ROI (B.10: median and best price with the frozen k)

Roles. 6a: --reference is R, the candidates are read in order as A1, A2, (A0); --candidate-roles overrides. 7b: one candidate, the with_env model. --leaky is L (6a, descriptive only).
Test rows are the box-score rows in every version, so the key sets of all versions must be identical in every season: a mismatch is printed, written to key_check.csv, and stops the run.
--allow-key-mismatch instead continues on the keys common to all versions, prints how many rows each version lost, and marks choice.json (key_mismatch_override).

Metrics (paired, candidate minus comparator, negative = candidate better; all game-clustered with props.models.evaluate.cluster_bootstrap, seed 0, 2,000 resamples, 95% interval, and
the Bonferroni interval at 0.05/23 with 10,000 resamples beside it on the decision scope): per-row CRPS of the four bet stats; the pooled-four-stat relative CRPS difference (equal weight per stat,
each stat's difference divided by the comparator's pooled-dev mean CRPS of that stat); predicted-minutes MAE (a diagnostic); the information slope with an intercept per stat and pooled on the
bet-stat rows at the book line (date-clustered, props.live.monitor.information_slope) with a paired slope difference; log loss and Brier at the book line; for the rule's bets at the frozen k:
ROI at the median and best price, flat -110, claimed-EV, stat, slot, side and blowout (|final margin| >= 20, from lbl_team_margin) segments, calibration by side, CLV, placebo CLV.
Writes only under --out and --report. Refuses 2025-26 and later seasons, and any game date after 2025-06-30.
"""
import argparse
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
sys.path.append(str(REPO / 'scripts'))
import a11_k_estimation as ke  # noqa: E402
from props.betting import a11_rules as ar, evaluate as be  # noqa: E402
from props.live import monitor as mon  # noqa: E402
from props.models import evaluate as ev, market_bench as mb, outcome as oc, registry  # noqa: E402
from props.models.pipeline import observed  # noqa: E402
from props.quality import md_table  # noqa: E402

BET_STATS = ('rebounds', 'assists', 'threes', 'ra')
DEV_FOLDS = ('2021-22', '2022-23', '2023-24')
SEEN = ('2024-25',)
SLOPE_SEASON = '2023-24'                       # B.9 item 1 and 7b item 6: the slope clause is read on 2023-24
HOLDOUT = '2025-26'
LAST_DATE = '2025-06-30'
N_COMPARISONS = 23
ALPHA_BONF = 0.05 / N_COMPARISONS
CRPS_HARM = 0.001                              # B.9 item 1, 7b item 6: upper bound at most +0.10% of the comparator's mean CRPS
A2_STAT_POINT = 0.0005                         # B.9 item 2: no bet stat's point difference above +0.05%
SLOPE_MARGIN = 0.02                            # B.9 item 1, 7b item 6: slope at least the reference's minus 0.02
BLOWOUT = 20                                   # A.4.6
SPREAD_BIG = 10.0                              # 7b item 5
ROLES = ('A1', 'A2', 'A0')
ROW_KEY = ['game_id', 'player_id', 'minutes_before']
CRPS_COLS = ['pair', 'scope', 'metric', 'n', 'ref_mean', 'cand_mean', 'diff', 'lo', 'hi', 'rel_diff', 'rel_lo', 'rel_hi', 'lo_bonf', 'hi_bonf', 'rel_hi_bonf']
LABEL_COLS = ['game_id', 'player_id', 'lbl_played', 'lbl_minutes', 'lbl_team_margin', 'y_points', 'y_rebounds', 'y_assists', 'y_threes', 'y_pra']
ROSTER_RULE = {'A1': 'v2nq', 'A2': 'pregame_all', 'A0': 'none (no teammate group)', 'no_change': 'v2 (live as it is)'}


class KeyMismatch(SystemExit):
    """The versions do not hold the same test rows (plan B.2: target rows are identical across arms)."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def refuse_seasons(seasons, what='season'):
    seasons = [str(s) for s in seasons]
    for s in seasons:
        if not re.fullmatch(r'\d{4}-\d{2}', s):
            raise SystemExit(f'{what} {s!r} is not a season like 2023-24')
    bad = sorted({s for s in seasons if s >= HOLDOUT})
    if bad:
        raise SystemExit(f'{what} {bad}: this script refuses {HOLDOUT} and later seasons (plan A.1: the holdout is not trained, predicted or scored)')


def refuse_late_dates(idx, what):
    if 'game_date' in idx.columns and len(idx):
        m = str(idx['game_date'].astype(str).max())[:10]
        if m > LAST_DATE:
            raise SystemExit(f'{what}: game dates up to {m}, after {LAST_DATE}: refused (plan A.1)')


def _inside(child, parent):
    child, parent = Path(child).resolve(), Path(parent).resolve()
    return child == parent or parent in child.parents


def check_output_paths(outs, inputs):
    for o in outs:
        ar.guard_output(o, REPO)
        for i in inputs:
            if i is not None and (_inside(o, i) or _inside(i, o)):
                raise SystemExit(f'output path {o} overlaps the input {i}: outputs go only under --out and --report, never into inputs')


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Loading (read only)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def yy(season):
    return season[2:4]


def read_labels(features_db, seasons):
    """player_game_labels of the given seasons only (game_id digits 4-5), mode=ro. lbl_team_margin is read when the table has it."""
    with sqlite3.connect(f'file:{Path(features_db).resolve().as_posix()}?mode=ro', uri=True, timeout=120) as conn:
        have = [r[1] for r in conn.execute('PRAGMA table_info(player_game_labels)').fetchall()]
        cols = [c for c in LABEL_COLS if c in have]
        ph = ','.join('?' * len(seasons))
        lab = pd.read_sql(f'SELECT {", ".join(cols)} FROM player_game_labels WHERE substr(game_id, 4, 2) IN ({ph})', conn, params=[yy(s) for s in seasons])
    lab['game_id'] = lab['game_id'].astype(str)
    if 'lbl_team_margin' not in lab.columns:
        lab['lbl_team_margin'] = np.nan
    return lab.drop_duplicates(['game_id', 'player_id'])


def read_env(features_db, seasons):
    with sqlite3.connect(f'file:{Path(features_db).resolve().as_posix()}?mode=ro', uri=True, timeout=120) as conn:
        ph = ','.join('?' * len(seasons))
        d = pd.read_sql(f'SELECT game_id, player_id, minutes_before, env_team_spread FROM player_game_at_T WHERE substr(game_id, 4, 2) IN ({ph})', conn,
                        params=[yy(s) for s in seasons])
    d['game_id'] = d['game_id'].astype(str)
    return d


def load_index(root, version, season):
    d = Path(root) / version / season
    ip = d / 'index.csv'
    if not ip.exists():
        raise SystemExit(f'{ip}: missing (version {version!r}, season {season})')
    idx = pd.read_csv(ip, dtype={'game_id': str})
    miss = [c for c in ROW_KEY + ['game_date', 'minutes_mean'] if c not in idx.columns]
    if miss:
        raise SystemExit(f'{ip}: missing columns {miss}')
    if 'season' in idx.columns and set(idx['season'].astype(str)) != {season}:
        raise SystemExit(f'{ip}: season column {sorted(set(idx["season"].astype(str)))} is not {season}')
    refuse_late_dates(idx, f'{version}/{season}')
    return idx


def load_arrays(root, version, season, n_rows, stats=BET_STATS):
    d = Path(root) / version / season
    out = {}
    for s in stats:
        p = d / f'pred_{s}.npy'
        if not p.exists():
            raise SystemExit(f'{p}: missing')
        a = np.load(p, mmap_mode='r')
        if a.ndim != 2 or a.shape[0] != n_rows or a.shape[1] != oc.CAPS[s] + 1:
            raise SystemExit(f'{p}: shape {a.shape}, expected ({n_rows}, {oc.CAPS[s] + 1})')
        out[s] = a
    return out


def key_index(df):
    return pd.MultiIndex.from_arrays([df['game_id'].astype(str).to_numpy(), df['player_id'].astype(int).to_numpy(), df['minutes_before'].astype(int).to_numpy()])


def align_keys(ref_keys, keys):
    """Position in `keys` of every reference key (-1 if absent) and the counts: rows of the reference not in `keys`, rows of `keys` not in the reference, duplicated keys of `keys`."""
    pos = keys.get_indexer(ref_keys) if keys.is_unique else pd.Index(keys.drop_duplicates()).get_indexer(ref_keys)
    return pos, {'only_in_reference': int((pos < 0).sum()), 'only_in_version': int(len(keys.drop_duplicates()) - (pos >= 0).sum()), 'duplicate_keys': int(len(keys) - len(keys.drop_duplicates()))}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Row metrics
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def row_crps(arr, pos, y, chunk=40000):
    """Per-row discrete CRPS of rows `pos` of the memmap against y (NaN where y is NaN), as evaluate.compare_versions computes it."""
    out = np.full(len(pos), np.nan)
    ok = np.flatnonzero(~np.isnan(y))
    for lo in range(0, len(ok), chunk):
        sel = ok[lo:lo + chunk]
        out[sel] = oc.crps(np.asarray(arr[pos[sel]]), y[sel].astype(int))
    return out


def build_metrics(versions, roots, folds_seen, labels, allow_mismatch, out_dir=None, log=logging.info):
    """
    Per season: load every version's index, check the key sets against the first version (the reference), align, and compute the per-row CRPS of the four bet stats and the minutes
    absolute error. Returns (rows frame in reference order, metrics {version: {metric: array}}, key_check DataFrame, dropped {version: rows lost when allow_mismatch},
    whether any key mismatch was seen).
    """
    ref = versions[0]
    frames, check = [], []
    mets = {v: {m: [] for m in list(BET_STATS) + ['minutes_mae']} for v in versions}
    dropped = {v: 0 for v in versions}
    for season in folds_seen:
        idxs = {v: load_index(roots, v, season) for v in versions}
        rk = key_index(idxs[ref])
        if not rk.is_unique:
            raise SystemExit(f'{ref}/{season}: duplicate (game_id, player_id, minutes_before) rows in the reference index')
        poss, season_bad = {}, False
        for v in versions:
            pos, c = align_keys(rk, key_index(idxs[v]))
            poss[v] = pos
            check.append({'version': v, 'season': season, 'rows_version': len(idxs[v]), 'rows_reference': len(rk), **c})
            season_bad = season_bad or any(c.values())
        if season_bad and not allow_mismatch:
            continue                                                                  # reported below, after every season is checked
        keep = np.ones(len(rk), bool)
        for v in versions:
            keep &= poss[v] >= 0
        rows = idxs[ref].iloc[np.flatnonzero(keep)].reset_index(drop=True)
        rows = rows.merge(labels, on=['game_id', 'player_id'], how='left', validate='many_to_one')
        for v in versions:
            dropped[v] += int(len(rk) - keep.sum())
        arrays = {v: load_arrays(roots, v, season, len(idxs[v])) for v in versions}
        yobs = {s: observed(rows, s) for s in BET_STATS}
        played = ((rows['lbl_played'] == 1) & (rows['lbl_minutes'] > 0)).to_numpy()
        for v in versions:
            pos = poss[v][keep]
            for s in BET_STATS:
                mets[v][s].append(row_crps(arrays[v][s], pos, yobs[s]))
            mm = idxs[v]['minutes_mean'].to_numpy(float)[pos]
            mets[v]['minutes_mae'].append(np.where(played, np.abs(mm - rows['lbl_minutes'].to_numpy(float)), np.nan))
        frames.append(rows)
        log(f'{season}: {len(rows)} rows x {len(versions)} versions')
    kc = pd.DataFrame(check)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        kc.to_csv(Path(out_dir) / 'key_check.csv', index=False)                      # written before any failure, so the mismatch is on disk
    bad = kc[(kc['only_in_reference'] > 0) | (kc['only_in_version'] > 0) | (kc['duplicate_keys'] > 0)]
    if len(bad) and not allow_mismatch:
        raise KeyMismatch('key mismatch between the versions and the reference (rows must be the same keys in every version; nothing was joined away):\n' + bad.to_string(index=False)
                          + '\nfix the versions, or pass --allow-key-mismatch to continue on the common keys')
    if len(bad):
        print('KEY MISMATCH (continuing on the common keys, --allow-key-mismatch):\n' + bad.to_string(index=False))
    rows = pd.concat(frames, ignore_index=True)
    metrics = {v: {m: np.concatenate(a) for m, a in d.items()} for v, d in mets.items()}
    return rows, metrics, kc, dropped, bool(len(bad))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Paired CRPS tables
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def weights_of(metrics, version, dev_mask):
    """w_s = 1 / (mean CRPS of the comparator for stat s over the pooled dev rows): the equal-weight relative scale of the pooled-four-stat difference."""
    w = {}
    for s in BET_STATS:
        x = metrics[version][s][dev_mask]
        x = x[np.isfinite(x)]
        w[s] = 1.0 / float(x.mean()) if len(x) else np.nan
    return w


def boot(d, clusters, n_boot, alpha=0.05, seed=0):
    return ev.cluster_bootstrap(d, clusters, n_boot=n_boot, seed=seed, alpha=alpha)


def paired_rows(pair, scope, mask, rows, metrics, ref, cand, w, n_boot, n_boot_bonf, bonf, stats=BET_STATS, extra=('minutes_mae',)):
    """Rows of the CRPS table for one pair on one scope: per bet stat, the pooled four stats (relative), and the extra diagnostics."""
    out = []
    gid = rows['game_id'].to_numpy()
    for s in list(stats) + list(extra):
        a, b = metrics[ref][s], metrics[cand][s]
        ok = mask & np.isfinite(a) & np.isfinite(b)
        if ok.sum() == 0:
            continue
        d = (b - a)[ok]
        p, lo, hi = boot(d, gid[ok], n_boot)
        m = float(a[ok].mean())
        r = {'pair': pair, 'scope': scope, 'metric': s, 'n': int(ok.sum()), 'ref_mean': m, 'cand_mean': float(b[ok].mean()), 'diff': p, 'lo': lo, 'hi': hi,
             'rel_diff': p / m, 'rel_lo': lo / m, 'rel_hi': hi / m, 'lo_bonf': np.nan, 'hi_bonf': np.nan, 'rel_hi_bonf': np.nan}
        if bonf:
            _, lb, hb = boot(d, gid[ok], n_boot_bonf, ALPHA_BONF)
            r.update(lo_bonf=lb, hi_bonf=hb, rel_hi_bonf=hb / m)
        out.append(r)
    okall = mask.copy()
    for s in stats:
        okall &= np.isfinite(metrics[ref][s]) & np.isfinite(metrics[cand][s])
    if okall.sum():
        rel = sum((metrics[cand][s] - metrics[ref][s])[okall] * w[s] for s in stats) / len(stats)
        raw = sum((metrics[cand][s] - metrics[ref][s])[okall] for s in stats)
        p, lo, hi = boot(rel, gid[okall], n_boot)
        r = {'pair': pair, 'scope': scope, 'metric': 'pooled4_relative', 'n': int(okall.sum()), 'ref_mean': np.nan, 'cand_mean': np.nan, 'diff': p, 'lo': lo, 'hi': hi,
             'rel_diff': p, 'rel_lo': lo, 'rel_hi': hi, 'lo_bonf': np.nan, 'hi_bonf': np.nan, 'rel_hi_bonf': np.nan}
        if bonf:
            _, lb, hb = boot(rel, gid[okall], n_boot_bonf, ALPHA_BONF)
            r.update(lo_bonf=lb, hi_bonf=hb, rel_hi_bonf=hb)
        out.append(r)
        pr, lr, hr = boot(raw, gid[okall], n_boot)
        out.append({'pair': pair, 'scope': scope, 'metric': 'pooled4_raw_sum', 'n': int(okall.sum()), 'ref_mean': np.nan, 'cand_mean': np.nan, 'diff': pr, 'lo': lr, 'hi': hr,
                    'rel_diff': np.nan, 'rel_lo': np.nan, 'rel_hi': np.nan, 'lo_bonf': np.nan, 'hi_bonf': np.nan, 'rel_hi_bonf': np.nan})
    return out


def make_scopes(rows, folds, seen, experiment, env=None):
    season = rows['season'].astype(str).to_numpy()
    slot = rows['minutes_before'].astype(int).to_numpy()
    dev = np.isin(season, list(folds))
    sc = {'dev_pooled': dev}
    for f in folds:
        sc[f'fold_{f}'] = season == f
    for t in sorted(set(slot)):
        sc[f'dev_pooled_T{t}'] = dev & (slot == t)
    for s in seen:
        sc[f'seen_{s}'] = season == s
    if experiment == '7b':
        for f in folds:
            for t in sorted(set(slot)):
                sc[f'fold_{f}_T{t}'] = (season == f) & (slot == t)
        if env is not None:
            big = np.abs(env) >= SPREAD_BIG
            sc[f'dev_pooled_spread>={SPREAD_BIG:g}'] = dev & big
            sc[f'dev_pooled_spread<{SPREAD_BIG:g}'] = dev & ~big & np.isfinite(env)
            for f in folds:
                sc[f'fold_{f}_spread>={SPREAD_BIG:g}'] = (season == f) & big
            for s in seen:
                sc[f'seen_{s}_spread>={SPREAD_BIG:g}'] = (season == s) & big
    return sc


def crps_tables(pairs, rows, metrics, scopes, dev_mask, n_boot, n_boot_bonf, log=logging.info):
    out = []
    for (cand, ref, label) in pairs:
        w = weights_of(metrics, ref, dev_mask)
        for name, mask in scopes.items():
            if not mask.any():
                continue
            out += paired_rows(label, name, mask, rows, metrics, ref, cand, w, n_boot, n_boot_bonf, bonf=(name == 'dev_pooled'))
        log(f'paired CRPS: {label} done')
    return pd.DataFrame(out, columns=CRPS_COLS)


def pick(t, pair, scope, metric):
    r = t[(t['pair'] == pair) & (t['scope'] == scope) & (t['metric'] == metric)]
    if r.empty:
        raise SystemExit(f'internal: no CRPS row for {pair} / {scope} / {metric}')
    return r.iloc[0]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Information slope (date-clustered; with an intercept per stat)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def slope_T(df, dates, stats):
    d = df[['stat', 'game_date', 'hit_over', 'p_market', 'p_model']].dropna()
    x = (d['p_model'] - d['p_market']).to_numpy(float)
    y = (d['hit_over'] - d['p_market']).to_numpy(float)
    codes = dates.get_indexer(d['game_date'].astype(str))
    sidx = d['stat'].map({s: i for i, s in enumerate(stats)}).to_numpy()
    T = np.zeros((len(dates), len(stats), 5))
    for j, v in enumerate((np.ones_like(x), x, y, x * x, x * y)):
        np.add.at(T[..., j], (codes, sidx), v)
    return T


def slope_pair(df_a, df_b, n_boot=2000, seed=0, alpha=0.05):
    """
    Paired difference of the information slope with a per-stat intercept (b minus a) on the same date resamples: per stat and pooled. Same estimator and resample counts as
    props.live.monitor.information_slope, so each version's own slope is reproduced; both frames must hold the same rows (the caller checks the keys).
    """
    stats = sorted(set(df_a['stat']) | set(df_b['stat']))
    dates = pd.Index(sorted(set(df_a['game_date'].astype(str)) | set(df_b['game_date'].astype(str))))
    Ta, Tb = slope_T(df_a, dates, stats), slope_T(df_b, dates, stats)
    g, k = len(dates), len(stats)
    counts = mon._boot_counts(g, n_boot, seed)
    pa, pb = mon._slope_estimates(Ta.sum(0))[1], mon._slope_estimates(Tb.sum(0))[1]
    ba = mon._slope_estimates((counts @ Ta.reshape(g, k * 5)).reshape(n_boot, k, 5))[1]
    bb = mon._slope_estimates((counts @ Tb.reshape(g, k * 5)).reshape(n_boot, k, 5))[1]
    d = bb - ba
    lo, hi = np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return pd.DataFrame({'stat': stats + ['pooled'], 'slope_a': pa, 'slope_b': pb, 'diff_b_minus_a': pb - pa, 'lo': lo, 'hi': hi})


def slope_tables(frames, ref, n_boot, n_boot_bonf):
    """frames: {version: slope frame (stat, game_date, hit_over, p_market, p_model)} of one season. Returns (slope table, paired table vs reference)."""
    rows, pairs = [], []
    for v, f in frames.items():
        t = mon.information_slope(f, n_boot=n_boot, seed=0)
        tb = mon.information_slope(f, n_boot=n_boot_bonf, seed=0, alpha=ALPHA_BONF)
        t['si_lo_bonf'], t['si_hi_bonf'] = tb['si_lo'].to_numpy(), tb['si_hi'].to_numpy()
        t.insert(0, 'version', v)
        rows.append(t)
    for v, f in frames.items():
        if v != ref:
            p = slope_pair(frames[ref], f, n_boot, 0)
            pb = slope_pair(frames[ref], f, n_boot_bonf, 0, ALPHA_BONF)
            p['lo_bonf'], p['hi_bonf'] = pb['lo'].to_numpy(), pb['hi'].to_numpy()
            p.insert(0, 'version', v)
            p.insert(1, 'reference', ref)
            pairs.append(p)
    return pd.concat(rows, ignore_index=True), (pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame())


def pooled_slope(table, version):
    r = table[(table['version'] == version) & (table['stat'] == 'pooled')]
    return float(r['slope_icpt'].iloc[0]) if len(r) else np.nan


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Quoted rows: slope input, scores at the book line, the rule's bets
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Quotes:
    """Per season: the consensus and the bettable quotes of prices_<season>.pkl (with the 7a p_market applied), read once."""

    def __init__(self, prices_dir, pmarket, quotes_dir):
        self.prices_dir, self.pmarket, self.quotes_dir = Path(prices_dir), pmarket, quotes_dir
        self.cache, self.dropped = {}, {}

    def available(self, season):
        return (self.prices_dir / f'prices_{season}.pkl').exists()

    def get(self, season):
        if season not in self.cache:
            ar.check_season(season)
            cons, bets = pd.read_pickle(self.prices_dir / f'prices_{season}.pkl')
            cons, bets = cons.copy(), bets.copy()
            cons['game_id'], bets['game_id'] = cons['game_id'].astype(str), bets['game_id'].astype(str)
            dropped = 0
            if tuple(self.pmarket) != ar.CURRENT_PMARKET:
                if not self.quotes_dir:
                    raise SystemExit(f'p_market {self.pmarket} is not the current method: --quotes-dir is required')
                cons, bets, dropped = ar.apply_pmarket(cons, bets, pd.read_pickle(Path(self.quotes_dir) / season / f'cons_{self.pmarket[0]}_{self.pmarket[1]}.pkl'))
            if 'game_date' in cons.columns:
                ar.check_dates(cons['game_date'], f'prices_{season}.pkl consensus')
            self.cache[season] = (cons, bets[bets['stat'].isin(BET_STATS)].reset_index(drop=True))
            self.dropped[season] = dropped
        return self.cache[season]


def load_quoted(season, version, roots, labels, quotes, with_bets=True):
    """The steps of scripts/a11_k_estimation.load_season for one version, on the four bet stats only (memmap): comparison rows (slope input) and, when asked, the quotes with p_mod."""
    cons, bets = quotes.get(season)
    idx = load_index(roots, version, season)
    pm = load_arrays(roots, version, season, len(idx))
    rows = idx.merge(labels, on=['game_id', 'player_id'], how='left', validate='many_to_one')
    cmp = mb.compare(rows, pm, cons.rename(columns={'stat': 'market'}), stats=list(BET_STATS))
    gdate = idx.drop_duplicates('game_id').set_index('game_id')['game_date'].astype(str).str[:10]
    cmp['game_date'] = cmp['game_id'].astype(str).map(gdate)
    q = None
    if with_bets:
        q = be.attach_outcomes(be.quote_table(bets, idx, pm), labels)
        q['game_date'] = idx['game_date'].astype(str).str[:10].to_numpy()[q['i'].to_numpy()]
        q = q.reset_index(drop=True)
    return cmp, q, idx


def cmp_keys(cmp):
    return pd.MultiIndex.from_arrays([cmp['game_id'].astype(str).to_numpy(), cmp['player_id'].astype(int).to_numpy(), cmp['minutes_before'].astype(int).to_numpy(), cmp['stat'].astype(str).to_numpy()])


def common_cmp(cmps, ref, allow, season):
    """The comparison rows of every version must be the same keys; a difference stops the run (or, with allow, restricts every version to the common keys, printed)."""
    rk = cmp_keys(cmps[ref])
    info, common = [], rk
    for v, c in cmps.items():
        if v == ref:
            continue
        k = cmp_keys(c)
        only_ref, only_v = int((~rk.isin(k)).sum()), int((~k.isin(rk)).sum())
        info.append({'version': v, 'season': season, 'cmp_rows_version': len(c), 'cmp_rows_reference': len(rk), 'only_in_reference': only_ref, 'only_in_version': only_v})
        common = common[common.isin(k)]
    bad = [i for i in info if i['only_in_reference'] or i['only_in_version']]
    if bad and not allow:
        raise KeyMismatch(f'{season}: the book-line comparison rows of the versions differ (rows must be the same keys; nothing was joined away): {bad}; --allow-key-mismatch continues on the common keys')
    if bad:
        print(f'KEY MISMATCH in the {season} book-line rows (continuing on the common keys): {bad}')
        cmps = {v: c[cmp_keys(c).isin(common)].reset_index(drop=True) for v, c in cmps.items()}
    return cmps, info


def frozen_k(path):
    fr = json.loads(Path(path).read_text(encoding='utf-8'))
    return {s: float(v) for s, v in fr['k_by_stat'].items() if s in BET_STATS}


def settle_bets(q, k):
    m = ar.market_prices(q)
    b = ar.price_and_settle(ar.select_candidate(q, 'C0', {'b': k}), m)
    return b


def add_blowout(b, labels):
    margin = labels.set_index(['game_id', 'player_id'])['lbl_team_margin']
    key = pd.MultiIndex.from_arrays([b['game_id'].astype(str), b['player_id'].astype(int)])
    mv = margin.reindex(key).to_numpy(float)
    b = b.copy()
    b['abs_margin'] = np.abs(mv)
    b['blowout'] = np.where(np.isnan(mv), 'margin unknown', np.where(np.abs(mv) >= BLOWOUT, f'|margin| >= {BLOWOUT}', f'|margin| < {BLOWOUT}'))
    return b


def roi_block(bets, universe, ref, extra_pairs, n_boot, n_boot_bonf):
    """ROI per version at the median, best and flat price (date-clustered, Bonferroni beside it) and paired differences on the same resamples."""
    bd, bb = ar.PairedBoot(universe, n_boot, ar.SEED), ar.PairedBoot(universe, n_boot_bonf, ar.SEED)
    for v, b in bets.items():
        for price in ('med', 'best', 'flat'):
            bd.add_bets(f'{v}|{price}', b, price)
            bb.add_bets(f'{v}|{price}', b, price)
    rows = []
    for v, b in bets.items():
        r = {'version': v, 'bets': len(b), 'stakes': int(b['stake_stands'].sum()), 'overs': int((b['best_side'] == 'over').sum()), 'unders': int((b['best_side'] == 'under').sum())}
        for price in ('med', 'best'):
            x, xb = bd.roi(f'{v}|{price}'), bb.roi(f'{v}|{price}', ALPHA_BONF)
            r.update({f'roi_{price}': x['point'], f'{price}_lo': x['lo'], f'{price}_hi': x['hi'], f'{price}_lo_bonf': xb['lo'], f'{price}_hi_bonf': xb['hi']})
        r['roi_flat'] = bd.point(f'{v}|flat')
        rows.append(r)
    diffs = []
    pairs = [(v, ref) for v in bets if v != ref] + [p for p in extra_pairs if p[0] in bets and p[1] in bets]
    for a, b in pairs:
        for price in ('med', 'best'):
            d, db = bd.diff(f'{a}|{price}', f'{b}|{price}'), bb.diff(f'{a}|{price}', f'{b}|{price}', ALPHA_BONF)
            diffs.append({'comparison': f'{a} - {b}', 'price': price, 'diff': d['point'], 'lo': d['lo'], 'hi': d['hi'], 'lo_bonf': db['lo'], 'hi_bonf': db['hi']})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def blowout_table(bets):
    rows = []
    for v, b in bets.items():
        for (seg, side), g in b.groupby(['blowout', 'best_side']):
            rows.append({'version': v, 'segment': seg, 'side': side, 'bets': len(g), 'stakes': int(g['stake_stands'].sum()), 'roi_med': ar.roi(g, 'med'), 'roi_best': ar.roi(g, 'best')})
    return pd.DataFrame(rows)


def book_line_table(cmps, n_boot):
    out = []
    for v, c in cmps.items():
        t = mb.score_table(c, n_boot=n_boot)
        t.insert(0, 'version', v)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def spread_calibration(cmps, env, ref):
    """7b item 5: calibration of p_model at the book line in games with |env_team_spread| >= 10 at T (and the rest), per version, with the paired log-loss difference to the reference."""
    rows = []
    key = env.set_index(ROW_KEY)['env_team_spread']
    base = {}
    for v, c in cmps.items():
        sp = key.reindex(pd.MultiIndex.from_arrays([c['game_id'].astype(str), c['player_id'].astype(int), c['minutes_before'].astype(int)])).to_numpy(float)
        base[v] = (c, sp)
    for v, (c, sp) in base.items():
        for name, m in ((f'|spread| >= {SPREAD_BIG:g}', np.abs(sp) >= SPREAD_BIG), (f'|spread| < {SPREAD_BIG:g}', np.abs(sp) < SPREAD_BIG), ('no env value', np.isnan(sp))):
            g = c[m]
            if not len(g):
                continue
            ll = mb._ll(g['p_model'].to_numpy(), g['hit'].to_numpy())
            r = {'version': v, 'subset': name, 'n': len(g), 'mean_p_model': g['p_model'].mean(), 'mean_p_market': g['p_market'].mean(), 'hit_rate': g['hit'].mean(),
                 'hit_minus_p_model': g['hit'].mean() - g['p_model'].mean(), 'logloss_model': ll.mean(), 'logloss_market': mb._ll(g['p_market'].to_numpy(), g['hit'].to_numpy()).mean()}
            if v != ref and name.startswith('|spread| >='):
                cr, spr = base[ref]
                mr = np.abs(spr) >= SPREAD_BIG
                if len(cr[mr]) == len(g):
                    d = ll - mb._ll(cr[mr]['p_model'].to_numpy(), cr[mr]['hit'].to_numpy())
                    p, lo, hi = ev.cluster_bootstrap(d, g['game_id'].to_numpy(), n_boot=500)
                    r.update(logloss_diff_vs_ref=p, ll_lo=lo, ll_hi=hi)
            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keep rules (pure: numbers in, clauses out)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _clause(item, text, value, threshold, holds):
    v = float(value) if value is not None and np.isfinite(value) else (None if value is None else float('nan'))
    return {'item': item, 'clause': text, 'value': v, 'threshold': threshold, 'holds': bool(holds)}


def _le(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and a <= b)


def _ge(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and a >= b)


def nonharm_clauses(label, per_stat, slope_arm, slope_ref, item='B.9.1'):
    """
    B.9 item 1 (and the 7b stat/slope clauses). per_stat: {stat: {'hi': upper end of the 95% interval of the paired CRPS difference (arm - comparator, CRPS units), 'ref_mean': comparator's mean CRPS}}.
    Holds when hi / ref_mean <= +0.10% for every bet stat, and slope_arm >= slope_ref - 0.02.
    """
    cl = []
    for s in BET_STATS:
        x = per_stat[s]
        ratio = x['hi'] / x['ref_mean'] if x['ref_mean'] else np.nan
        cl.append(_clause(item, f'{label} vs comparator, {s}: 95% upper bound of the paired CRPS difference at most +0.10% of comparator mean CRPS', ratio, CRPS_HARM, _le(ratio, CRPS_HARM)))
    dsl = slope_arm - slope_ref if np.isfinite(slope_arm) and np.isfinite(slope_ref) else np.nan
    cl.append(_clause(item, f'{label}: 2023-24 pooled slope with intercept at least comparator\'s minus 0.02 (value = arm minus comparator)', dsl, -SLOPE_MARGIN, _ge(dsl, -SLOPE_MARGIN)))
    return cl


def a2_over_a1_clauses(a2_item1_holds, pooled_hi, stat_points_rel):
    """B.9 item 2: A2 passes item 1; pooled-four-stat (A2 - A1) 95% upper bound below 0; no bet stat's point difference above +0.05% (of A1's mean CRPS)."""
    cl = [_clause('B.9.2', 'A2 passes item 1', None, None, a2_item1_holds)]
    cl.append(_clause('B.9.2', 'pooled four stats, A2 - A1: 95% upper bound below 0 (relative units)', pooled_hi, 0.0, bool(np.isfinite(pooled_hi) and pooled_hi < 0)))
    for s in BET_STATS:
        v = stat_points_rel[s]
        cl.append(_clause('B.9.2', f'A2 - A1, {s}: point difference at most +0.05% of A1 mean CRPS', v, A2_STAT_POINT, _le(v, A2_STAT_POINT)))
    return cl


def choose_6a(passed):
    """
    B.9 item 3, in order: A2 if items 1 and 2 hold; otherwise A1 if item 1 holds; otherwise A0 (if trained) when item 1 holds; otherwise no change.
    passed: {'A1': {'item1': bool}, 'A2': {'item1': bool, 'item2': bool}, 'A0': {'item1': bool}} (a role that was not evaluated is absent).
    If A1 and A2 both fail and A0 is absent the plan says to train A0 first: the answer is 'pending_A0', not 'no_change'.
    """
    a2, a1, a0 = passed.get('A2'), passed.get('A1'), passed.get('A0')
    if a2 and a2['item1'] and a2.get('item2'):
        return 'A2'
    if a1 and a1['item1']:
        return 'A1'
    if a0 is None:
        return 'pending_A0'
    return 'A0' if a0['item1'] else 'no_change'


def keep_7b(pooled_hi, fold_point, per_stat, slope_arm, slope_ref):
    """7b item 6: pooled-four-stat upper bound below 0; 2023-24 fold point below 0; no bet stat's 95% upper bound above +0.10%; 2023-24 slope at least the reference's minus 0.02."""
    cl = [_clause('7b.6', 'pooled four stats over the dev folds, with_env - reference: 95% upper bound below 0 (relative units)', pooled_hi, 0.0, bool(np.isfinite(pooled_hi) and pooled_hi < 0)),
          _clause('7b.6', 'pooled four stats, 2023-24 fold alone: point estimate below 0 (relative units)', fold_point, 0.0, bool(np.isfinite(fold_point) and fold_point < 0))]
    cl += nonharm_clauses('with_env', per_stat, slope_arm, slope_ref, item='7b.6')
    return cl


def clause_frame(cl):
    return pd.DataFrame([{'item': c['item'], 'clause': c['clause'], 'value': c['value'], 'threshold': c['threshold'], 'result': 'PASS' if c['holds'] else 'FAIL'} for c in cl])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def per_stat_inputs(table, pair, scope='dev_pooled'):
    return {s: {'hi': float(pick(table, pair, scope, s)['hi']), 'ref_mean': float(pick(table, pair, scope, s)['ref_mean'])} for s in BET_STATS}


def manifest(version, root):
    p = Path(root) / version / 'manifest.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def db_of(version, root, fallback=None):
    m = manifest(version, root)
    return {'database': m.get('features_db', fallback), 'database_sha256': m.get('features_db_sha256'), 'manifest': str(Path(root) / version / 'manifest.json')}


def assign_roles(candidates, roles, experiment):
    if experiment == '7b':
        if len(candidates) != 1:
            raise SystemExit('--experiment 7b takes exactly one candidate (the with_env model)')
        return {candidates[0]: 'with_env'}
    r = roles or list(ROLES[:len(candidates)])
    if len(r) != len(candidates) or any(x not in ROLES for x in r) or len(set(r)) != len(r):
        raise SystemExit(f'--candidate-roles {r} must be distinct names from {ROLES}, one per candidate; the default reads the candidates in the order A1, A2, A0')
    return dict(zip(candidates, r))


def run(a):
    folds, seen = list(a.folds), [s for s in (a.seen or []) if s.lower() != 'none']
    refuse_seasons(folds, '--folds')
    refuse_seasons(seen, '--seen')
    if a.experiment == '7b' and a.leaky:
        raise SystemExit('--leaky belongs to 6a')
    if SLOPE_SEASON not in folds:
        raise SystemExit(f'{SLOPE_SEASON} must be one of --folds: the slope clause is read on it (B.9 item 1, 7b item 6)')
    roles = assign_roles(a.candidates, a.candidate_roles, a.experiment)
    inputs = [a.registry_root, a.prices_dir, a.features_db, a.quotes_dir, a.pmarket, a.frozen_rule]
    check_output_paths([a.out, a.report], inputs)
    root = registry.resolve_root(a.registry_root)
    ref = a.reference
    versions = [ref] + list(a.candidates) + ([a.leaky] if a.leaky else [])
    if len(set(versions)) != len(versions):
        raise SystemExit(f'--reference, --candidates and --leaky must be distinct versions: {versions}')
    seasons = folds + [s for s in seen if s not in folds]
    labels = read_labels(a.features_db, seasons)
    choice_in, pmarket = ke.read_choice(a.pmarket)
    quotes = Quotes(a.prices_dir, pmarket, a.quotes_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- paired CRPS
    rows, metrics, key_check, dropped, had_mismatch = build_metrics(versions, root, seasons, labels, a.allow_key_mismatch, out)
    env = None
    if a.experiment == '7b':
        e = read_env(a.env_db or a.features_db, seasons)
        env = pd.MultiIndex.from_arrays([rows['game_id'].astype(str), rows['player_id'].astype(int), rows['minutes_before'].astype(int)])
        env = e.set_index(ROW_KEY)['env_team_spread'].reindex(env).to_numpy(float)
    scopes = make_scopes(rows, folds, seen, a.experiment, env)
    dev_mask = scopes['dev_pooled']
    pairs = [(c, ref, f'{roles[c]}_vs_R' if a.experiment == '6a' else 'with_env_vs_reference') for c in a.candidates]
    inv = {r: c for c, r in roles.items()}
    if a.experiment == '6a' and 'A1' in inv and 'A2' in inv:
        pairs.append((inv['A2'], inv['A1'], 'A2_vs_A1'))
    if a.leaky:
        pairs.append((a.leaky, ref, 'L_vs_R'))
    crps = crps_tables(pairs, rows, metrics, scopes, dev_mask, a.n_boot, a.n_boot_bonf)
    crps.to_csv(out / 'crps_paired.csv', index=False)
    mean_scopes = [n for n in scopes if n == 'dev_pooled' or (n.startswith(('fold_', 'seen_')) and '_T' not in n and 'spread' not in n)]
    means = pd.DataFrame([{'version': v, 'scope': n, 'metric': m, 'n': int((scopes[n] & np.isfinite(metrics[v][m])).sum()), 'mean': float(metrics[v][m][scopes[n] & np.isfinite(metrics[v][m])].mean())}
                          for v in versions for n in mean_scopes for m in metrics[v] if (scopes[n] & np.isfinite(metrics[v][m])).any()])
    means.to_csv(out / 'crps_means.csv', index=False)

    # ---- slopes, scores at the book line, bets
    slope_tabs, pair_tabs, score_tabs, key_infos, roi_tabs, roi_diffs, calib_t, seg_t, clv_t, plc_t, blow_t, env_t = ([] for _ in range(12))
    slope_source = {}
    q_seasons = [s for s in seasons if s >= '2023-24' and quotes.available(s)]
    if SLOPE_SEASON not in q_seasons:
        raise SystemExit(f'{Path(a.prices_dir) / f"prices_{SLOPE_SEASON}.pkl"} is missing: the slope clause needs the {SLOPE_SEASON} quotes')
    kfz = frozen_k(a.frozen_rule) if not a.skip_roi else None
    for season in q_seasons:
        names = {v: v for v in versions}
        use_q = bool(a.quoted_suffix) and all((root / (v + a.quoted_suffix) / season / 'index.csv').exists() for v in versions)
        if use_q:
            names = {v: v + a.quoted_suffix for v in versions}
        slope_source[season] = a.quoted_suffix if use_q else '(all-played versions)'
        print(f'{season}: slope / scores / bets from {slope_source[season]}')
        cmps, bets = {}, {}
        cons_s = quotes.get(season)[0]
        for v in versions:
            cmp, q, idx = load_quoted(season, names[v], root, labels, quotes, with_bets=not a.skip_roi)
            cmps[v] = cmp
            if q is not None:
                bets[v] = add_blowout(settle_bets(q, kfz), labels)
                pq = ke.placebo_clv(q, 'C0', {'b': kfz}, cons_s)                       # A.4.4: the same rule on placebo quotes (seeds 11-14)
                pq.insert(0, 'version', v)
                pq.insert(0, 'season', season)
                plc_t.append(pq)
            del q
        cmps, info = common_cmp(cmps, ref, a.allow_key_mismatch, season)
        key_infos += info
        frames = {v: ke.slope_frame(c) for v, c in cmps.items()}
        st, sp = slope_tables(frames, ref, a.n_boot, a.n_boot_bonf)
        st.insert(0, 'season', season)
        slope_tabs.append(st)
        if len(sp):
            sp.insert(0, 'season', season)
            pair_tabs.append(sp)
        if a.experiment == '6a' and 'A1' in inv and 'A2' in inv:
            p = slope_pair(frames[inv['A1']], frames[inv['A2']], a.n_boot, 0)
            pb = slope_pair(frames[inv['A1']], frames[inv['A2']], a.n_boot_bonf, 0, ALPHA_BONF)
            p['lo_bonf'], p['hi_bonf'] = pb['lo'].to_numpy(), pb['hi'].to_numpy()
            p.insert(0, 'version', inv['A2'])
            p.insert(1, 'reference', inv['A1'])
            p.insert(0, 'season', season)
            pair_tabs.append(p)
        sc = book_line_table(cmps, min(a.n_boot, 500))
        sc.insert(0, 'season', season)
        score_tabs.append(sc)
        if env is not None:
            ee = read_env(a.env_db or a.features_db, [season])
            t = spread_calibration(cmps, ee, ref)
            t.insert(0, 'season', season)
            env_t.append(t)
        if bets:
            universe = sorted(set().union(*[set(b['game_date'].astype(str)) for b in bets.values()]))
            extra = [(inv['A2'], inv['A1'])] if a.experiment == '6a' and 'A1' in inv and 'A2' in inv else []
            t, d = roi_block(bets, universe, ref, extra, a.n_boot, a.n_boot_bonf)
            t.insert(0, 'season', season)
            d.insert(0, 'season', season)
            roi_tabs.append(t)
            roi_diffs.append(d)
            bt = blowout_table(bets)
            bt.insert(0, 'season', season)
            blow_t.append(bt)
            ct = ke.calib_table(bets)
            ct.insert(0, 'season', season)
            calib_t.append(ct)
            for col in ('ev_bucket', 'stat', 'minutes_before'):
                s_ = ke.segment_table(bets, col)
                s_.insert(0, 'season', season)
                s_.insert(1, 'segment_by', col)
                s_ = s_.rename(columns={col: 'segment'})
                seg_t.append(s_)
            cl = ke.clv_rows(bets, cons_s)
            if len(cl):
                cl.insert(0, 'season', season)
                clv_t.append(cl)
        del cmps, frames, bets

    def cat(x):
        return pd.concat(x, ignore_index=True) if x else pd.DataFrame()

    slope, slope_p, scores = cat(slope_tabs), cat(pair_tabs), cat(score_tabs)
    tables = {'slope': slope, 'slope_paired': slope_p, 'scores_book_line': scores, 'roi': cat(roi_tabs), 'roi_paired': cat(roi_diffs), 'blowout_roi': cat(blow_t),
              'calibration_by_side': cat(calib_t), 'segments': cat(seg_t), 'clv': cat(clv_t), 'placebo_clv': cat(plc_t), 'spread_calibration': cat(env_t), 'book_line_key_check': pd.DataFrame(key_infos)}
    for n, t in tables.items():
        if len(t):
            t.to_csv(out / f'{n}.csv', index=False)

    # ---- keep rules
    s_ref = pooled_slope(slope[slope['season'] == SLOPE_SEASON], ref)
    clauses, decision = [], {}
    if a.experiment == '6a':
        passed = {}
        for c, role in roles.items():
            label = f'{role} ({c})'
            pair = f'{role}_vs_R'
            s_arm = pooled_slope(slope[slope['season'] == SLOPE_SEASON], c)
            cl1 = nonharm_clauses(label, per_stat_inputs(crps, pair), s_arm, s_ref)
            ok1 = all(x['holds'] for x in cl1)
            passed[role] = {'item1': ok1}
            clauses += cl1
        if 'A2' in inv and 'A1' in inv:
            pt = {s: float(pick(crps, 'A2_vs_A1', 'dev_pooled', s)['rel_diff']) for s in BET_STATS}
            cl2 = a2_over_a1_clauses(passed['A2']['item1'], float(pick(crps, 'A2_vs_A1', 'dev_pooled', 'pooled4_relative')['hi']), pt)
            passed['A2']['item2'] = all(x['holds'] for x in cl2)
            clauses += cl2
        elif 'A2' in inv:
            passed['A2']['item2'] = False
            clauses.append(_clause('B.9.2', 'A2 over A1: not computable, no A1 candidate was given', None, None, False))
        choice = choose_6a(passed)
        decision = {'passed': passed}
    else:
        c = a.candidates[0]
        s_arm = pooled_slope(slope[slope['season'] == SLOPE_SEASON], c)
        pair = 'with_env_vs_reference'
        clauses = keep_7b(float(pick(crps, pair, 'dev_pooled', 'pooled4_relative')['hi']), float(pick(crps, pair, f'fold_{SLOPE_SEASON}', 'pooled4_relative')['diff']),
                          per_stat_inputs(crps, pair), s_arm, s_ref)
        choice = 'with_env' if all(x['holds'] for x in clauses) else 'no_change'
        decision = {'adopt': choice == 'with_env'}
    cf = clause_frame(clauses)
    cf.to_csv(out / 'keep_rule.csv', index=False)

    # ---- choice.json
    if a.experiment == '6a':
        if choice in ('A1', 'A2', 'A0'):
            win = inv[choice]
            sel = {'version': win, **db_of(win, root)}
        elif choice == 'no_change':
            base = a.leaky or manifest(ref, root).get('source_version') or ref
            sel = {'version': base, **db_of(base, root, a.features_db), 'reference_for_7b': ref}
        else:
            sel = {'version': None, 'database': None, 'database_sha256': None, 'manifest': None, 'note': 'A1 and A2 failed item 1/2 and A0 was not among the candidates: B.9.3 says train A0 (S12/S13 for a11_noav) and run S14 again'}
        sel.setdefault('reference_for_7b', sel['version'])
        sel['roster_rule'] = ROSTER_RULE.get(choice)
        sel['needs_live_builder_change'] = choice in ('A1', 'A2')
    else:
        win = a.candidates[0] if choice == 'with_env' else ref
        sel = {'version': win, **db_of(win, root, a.features_db), 'needs_live_builder_change': choice == 'with_env'}
    choice_json = {'experiment': a.experiment, 'choice': choice, **sel, 'reference': ref, 'candidates': roles, 'leaky': a.leaky, 'folds': folds, 'seen': seen,
                   'clauses': clauses, 'decision_detail': decision, 'slope_source': slope_source, 'pmarket': {'devig_method': pmarket[0], 'consensus_set': pmarket[1], 'choice_file': a.pmarket},
                   'key_mismatch_override': bool(a.allow_key_mismatch and (had_mismatch or any(i['only_in_reference'] or i['only_in_version'] for i in key_infos))),
                   'rows_dropped_by_key_mismatch': dropped, 'preconditions': 'L1-L6 (B.9) and the control check (B.6.3) are not verifiable from these inputs; this script assumes they passed',
                   'n_boot': a.n_boot, 'n_boot_bonf': a.n_boot_bonf, 'bonferroni_alpha': ALPHA_BONF, 'code_commit': ke.code_commit()[0]}
    (out / 'choice.json').write_text(json.dumps(ke._json(choice_json), indent=2), encoding='utf-8')
    (out / 'results.json').write_text(json.dumps(ke._json({'keep_rule': cf, 'crps': crps, 'slope': slope, 'slope_paired': slope_p}), indent=2), encoding='utf-8')
    write_report(a, versions, roles, crps, means, tables, cf, choice, choice_json, slope_source, key_check, dropped, pmarket)
    print(cf.to_string(index=False))
    print(f'{a.experiment} choice: {choice}  (version {choice_json.get("version")}, database {choice_json.get("database")})')
    return {'choice': choice, 'choice_json': choice_json, 'clauses': cf, 'crps': crps, 'slope': slope, 'slope_paired': slope_p, 'tables': tables, 'key_check': key_check, 'rows': rows,
            'metrics': metrics, 'decision': decision}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _fmt(df, n=4):
    return md_table(df.round(n) if len(df) else df, n)


def write_report(a, versions, roles, crps, means, tables, cf, choice, cj, slope_source, key_check, dropped, pmarket):
    exp = a.experiment
    title = {'6a': '6a: pre-game roster in training', '7b': '7b: game environment as a model input'}[exp]
    md = [f'# {title}\n',
          f'Reference `{a.reference}`; candidates {", ".join(f"`{c}` ({r})" for c, r in roles.items())}' + (f'; leaky `{a.leaky}` (L, descriptive)' if a.leaky else '') + '.\n',
          f'Dev folds (decide): {", ".join(a.folds)}. Seen (reported, never decides): {", ".join(a.seen or [])}. 2025-26 and later: refused. p_market: {pmarket[0]} de-vig, {pmarket[1]} consensus. '
          f'Registry root `{a.registry_root}`. Slope, book-line scores and ROI source by season: {slope_source}.\n',
          f'Intervals: game-clustered bootstrap for CRPS ({a.n_boot} resamples, seed 0, 2.5th-97.5th percentile), date-clustered for slopes and ROI. Bonferroni intervals at 0.05/{N_COMPARISONS} = {ALPHA_BONF:.5f} '
          f'({a.n_boot_bonf} resamples) beside them on the decision scope. Differences are candidate minus comparator: negative CRPS = better.\n',
          f'## Decision: **{choice}**\n', f'Version `{cj.get("version")}`; database `{cj.get("database")}`; roster rule `{cj.get("roster_rule", "")}`. Preconditions: {cj["preconditions"]}.\n']
    if cj.get('note'):
        md.append(f'**{cj["note"]}**\n')
    md += ['## Keep rule, clause by clause\n', _fmt(cf, 6)]
    if exp == '6a':
        md.append('B.9.3 order: A2 (items 1 and 2), else A1 (item 1), else A0 (item 1; trained only if A1 and A2 fail), else no change. Item 4: the correction is adopted for non-harm, not for a gain; '
                  'lower backtest numbers against L are not a reason to reject it.\n')
    kc = key_check[(key_check['only_in_reference'] > 0) | (key_check['only_in_version'] > 0) | (key_check['duplicate_keys'] > 0)]
    md += ['## Row keys\n', 'Every version holds the same test rows in every season.\n' if kc.empty else _fmt(kc, 0), f'Rows lost to --allow-key-mismatch: {dropped}.\n',
           '## Paired CRPS, decision scope (pooled dev folds)\n', _fmt(crps[crps['scope'] == 'dev_pooled'], 6),
           '`pooled4_relative` = mean over the four bet stats of each stat\'s difference divided by the comparator\'s pooled-dev mean CRPS of that stat (relative units, equal weight); '
           '`pooled4_raw_sum` = sum of the four raw CRPS differences (CRPS units: dominated by ra, shown for reference only). `minutes_mae` is a diagnostic.\n',
           '## Paired CRPS by fold, slot and seen season\n', _fmt(crps[(crps['scope'] != 'dev_pooled') & (crps['metric'].isin(['pooled4_relative'] + list(BET_STATS)))], 6)]
    if exp == '7b':
        md += ['(7b item 7: per fold and per T; T-60 has no environment before 2023-24 and is read apart. Scopes named spread>=10 / spread<10 use env_team_spread at T.)\n']
    md += ['## Mean CRPS by version\n', _fmt(means, 5)]
    for n, t in tables.items():
        if len(t):
            md += [f'## {n}\n', _fmt(t, 5)]
    md += ['## Not computed here\n',
           '- Final-margin segments use `lbl_team_margin` of player_game_labels (blowout = |margin| >= 20, descriptive); the plan text says the inputs carry no margin, but this column exists in data/props_features_v3.db.',
           '- B.10: the share of rows whose member set differs between `box` and each rule, the `av_n_out > 0` share under each rule, the parity tables and the quote-component effect (S11) need the '
           'feature databases and parity runs: `scripts/a11_compare_dbs.py` reports differing rows and the `av_n_out > 0` share per season; the parity numbers come from S11.',
           '- A.4 settlement treatments other than `stands` and the ROI of 2023-24 in-sample k are not produced; ROI here uses the frozen k, as B.10 asks.',
           '- Metrics of stats other than rebounds, assists, threes and ra are not read (not decision stats).',
           '- Preconditions L1-L6 and the `ctl` control check (B.6.3) cannot be checked from these inputs.\n']
    p = Path(a.report)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(md), encoding='utf-8')


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiment', required=True, choices=('6a', '7b'))
    ap.add_argument('--reference', required=True, help='R (6a) or the 6a winner / reference model (7b): a prediction version under --registry-root')
    ap.add_argument('--candidates', nargs='+', required=True, help='6a: A1 A2 [A0] in this order (or --candidate-roles); 7b: the with_env version')
    ap.add_argument('--candidate-roles', nargs='*', default=None, help='6a: roles of the candidates, from A1 A2 A0')
    ap.add_argument('--leaky', default=None, help='6a: L, the box-score-roster model scored on its own database (descriptive)')
    ap.add_argument('--folds', nargs='+', default=list(DEV_FOLDS))
    ap.add_argument('--seen', nargs='*', default=list(SEEN), help='reported, never decide; "none" for no season')
    ap.add_argument('--registry-root', required=True)
    ap.add_argument('--prices-dir', required=True, help='folder of prices_<season>.pkl')
    ap.add_argument('--pmarket', default=None, help='choice.json of scripts/a11_devig_calibration.py (default: the current method)')
    ap.add_argument('--quotes-dir', default=None, help='a11_cache_quotes.py --out folder (only for a non-current p_market)')
    ap.add_argument('--features-db', default='data/props_features_v3.db', help='labels (and env_team_spread for 7b), read only')
    ap.add_argument('--env-db', default=None, help='7b: database holding env_team_spread (default --features-db)')
    ap.add_argument('--frozen-rule', default=str(ke.FROZEN_RULE))
    ap.add_argument('--quoted-suffix', default='_p5', help='slope / book-line / ROI use <version><suffix> when every version has it for the season; "" to always use the all-played versions')
    ap.add_argument('--skip-roi', action='store_true', help='skip the rule\'s bets, ROI, calibration, CLV (slope and CRPS only)')
    ap.add_argument('--allow-key-mismatch', action='store_true', help='continue on the keys common to all versions (printed, and flagged in choice.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--n-boot', type=int, default=ev.N_BOOT)
    ap.add_argument('--n-boot-bonf', type=int, default=10000)
    return ap.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    return run(parse(argv))


if __name__ == '__main__':
    main()
