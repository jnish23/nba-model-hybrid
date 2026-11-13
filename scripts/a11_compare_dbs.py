"""
Step S10 of reports/props/audit_2026-11/WP6_WP7_experiment_plan.md (B.6): compare two props feature databases row by row on their shared keys.

    PY scripts/a11_compare_dbs.py --a data/props_features_v3.db --b $OUT/features/props_features_a11_ctl.db --out $OUT/results/ctl_vs_v3.json
    PY scripts/a11_compare_dbs.py --a $OUT/features/props_features_a11_ctl.db --b $OUT/features/props_features_a11_pg1.db --out $OUT/results/ctl_vs_pg1.json \
        --expect-identical --allow-columns group:teammates

What it does
  Both databases are opened read-only (SQLite mode=ro). Every table present in either is compared (the three real tables are player_game_base and player_game_labels, keyed by
  game_id + player_id, and player_game_at_T, keyed by game_id + player_id + minutes_before). The data is read one season at a time (the season is the two digits after '00x' in
  game_id), so memory stays small. Per table and season: row counts, keys only in A, keys only in B, duplicated keys. Per column (every non-key column of the table, so also the
  label columns and team_id, game_date and the like): rows compared, rows that differ exactly, rows that differ beyond REL_TOL = 1e-9 relative (|a - b| > 1e-9 * max(|a|, |b|), no
  absolute floor), their share, the largest absolute difference, and the first differing keys with both values. NaN equals NaN; a NaN against a number is a difference.
  Each column is assigned its feature group by props/features/dictionary.py (key, label, or the dictionary group; 'undocumented' when the dictionary has no entry).
  Seasons: 2019-20 to 2024-25 by default. 2025-26 and later are refused in --seasons and are never read (plan A.1); the rows of such seasons in the files are left alone.

Outputs (only under --out and --report; the inputs are never written)
  --out ending in .json : that JSON file, plus the CSV files in <out folder>/<out stem>_tables/ and a markdown report at --report (default <out>.md).
  --out otherwise       : a folder holding summary.json, the CSV files and report.md.
  CSV: one file per table (<table>.csv, one row per column), row_counts.csv, keys_only.csv (a sample of keys present in one database only), group_summary.csv.

--expect-identical   exit status 1 (and a list of the violations) if anything differs outside --allow-columns: a table or column present in one database only, keys present in one
                     only, duplicated keys, or a column whose rows differ beyond 1e-9 relative (--exact: any exact difference). --allow-columns takes column names, name prefixes
                     (a column is allowed if it equals a token or starts with it), 'group:<name>' for a whole feature group (for example group:teammates), and comma-separated lists.
Without --expect-identical the exit status is 0 whatever the differences are; the report is the product.
Also reported (plan B.6.3 and B.10): the share of base cells that differ (the plan re-trains R and L on the control when it exceeds 0.1%), the share of rows with any differing
column of each feature group (a lower bound on the share of rows whose roster member set differs, when the group is 'teammates'), and the share of rows with av_n_out > 0 in A and in B.
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
from props.features import dictionary  # noqa: E402
from props.models import registry  # noqa: E402
from props.quality import md_table  # noqa: E402

HOLDOUT_SEASON = '2025-26'                  # plan A.1
FIRST_SEASON = '2019-20'
LAST_DEFAULT_SEASON = '2024-25'
REL_TOL = 1e-9
BASE_SHARE_THRESHOLD = 0.001                # plan B.6.3: more than 0.1% of base cells differing re-bases the reference database
KNOWN_KEYS = {'player_game_base': ['game_id', 'player_id'], 'player_game_labels': ['game_id', 'player_id'],
              'player_game_at_T': ['game_id', 'player_id', 'minutes_before']}
GENERIC_KEYS = ('game_id', 'player_id', 'minutes_before')
POSITIVE_SHARE_COLUMNS = ('av_n_out', 'opp_av_n_out')
N_FIRST = 5                                 # differing keys shown per column
N_KEYS_SAMPLE = 20                          # keys-only sample per table, side and season


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Seasons and guards
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def default_seasons():
    return [f'{y}-{(y + 1) % 100:02d}' for y in range(2019, 2025)]


def yy_of(season):
    m = re.fullmatch(r'(\d{4})-(\d{2})', str(season))
    if not m or int(m[2]) != (int(m[1]) + 1) % 100:
        raise SystemExit(f'season {season!r} is not a season like 2023-24')
    return m[1][2:]


def check_seasons(seasons):
    for s in seasons:
        yy_of(s)
    bad = sorted({str(s) for s in seasons if str(s) >= HOLDOUT_SEASON})
    if bad:
        raise SystemExit(f'--seasons {bad}: this script refuses {HOLDOUT_SEASON} and later seasons (plan A.1: the holdout is not read, trained or scored)')
    return [str(s) for s in seasons]


def _same(p, q):
    return Path(p).resolve() == Path(q).resolve()


def check_output_paths(paths, inputs):
    for p in paths:
        if p is None:
            continue
        for i in inputs:
            if _same(p, i):
                raise SystemExit(f'output path {p} is an input database: refused (the inputs are read-only)')


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reading (read-only, one season at a time)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def open_ro(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f'{path}: no such database')
    return sqlite3.connect(f'file:{p.resolve().as_posix()}?mode=ro', uri=True, timeout=120)


def table_columns(conn):
    """{table: [(column, declared type), ...]} for every ordinary table (internal sqlite_* tables skipped)."""
    out = {}
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall():
        out[t] = [(r[1], r[2]) for r in conn.execute(f'PRAGMA table_info("{t}")').fetchall()]
    return out


def key_columns(table, columns):
    names = [c for c, _ in columns]
    k = KNOWN_KEYS.get(table) or [c for c in GENERIC_KEYS if c in names]
    return [c for c in k if c in names]


def read_chunk(conn, table, columns, season):
    cols = ', '.join(f'"{c}"' for c in columns)
    d = pd.read_sql(f'SELECT {cols} FROM "{table}" WHERE substr(game_id, 4, 2) = ?', conn, params=[yy_of(season)])
    d['game_id'] = d['game_id'].astype(str)
    return d


def feature_group(col):
    if col in dictionary.KEYS:
        return 'key'
    if col.startswith(dictionary.LABEL_PREFIXES):
        return 'label'
    info = dictionary.describe(col)
    return info[0] if info else 'undocumented'


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Cell comparison
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _as_number(s):
    """float array of a series when it is numeric (or an object column holding only numbers and nulls), else None."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(float)
    if s.dtype == object:
        c = pd.to_numeric(s, errors='coerce')
        if int(c.notna().sum()) == int(s.notna().sum()):
            return c.to_numpy(float)
    return None


def compare_values(sa, sb, rel_tol=REL_TOL):
    """
    Aligned series (same length). Returns dict(exact, tol, abs_diff, numeric): boolean arrays 'exact' (not equal; NaN equals NaN, NaN against a number differs) and 'tol'
    (differs beyond rel_tol * max(|a|, |b|), or the NaN pattern differs, or one side is infinite and the values differ); abs_diff is |a - b| where both are numbers (else 0).
    """
    na, nb = sa.isna().to_numpy(), sb.isna().to_numpy()
    a, b = _as_number(sa), _as_number(sb)
    if a is not None and b is not None:
        both = ~na & ~nb
        with np.errstate(invalid='ignore', over='ignore'):
            diff = np.where(both, np.abs(a - b), 0.0)
            thr = rel_tol * np.maximum(np.abs(a), np.abs(b))
            neq = both & (a != b)
            tol = (na != nb) | (both & (diff > thr)) | (neq & ~(np.isfinite(a) & np.isfinite(b)))
        return {'exact': (na != nb) | neq, 'tol': tol, 'abs_diff': np.nan_to_num(diff, nan=0.0, posinf=np.inf), 'numeric': True}
    oa, ob = sa.to_numpy(object), sb.to_numpy(object)
    eq = np.fromiter((x == y for x, y in zip(oa, ob)), dtype=bool, count=len(oa))
    exact = ~((na & nb) | (~na & ~nb & eq))
    return {'exact': exact, 'tol': exact.copy(), 'abs_diff': np.zeros(len(oa)), 'numeric': False}


class ColumnStat:
    def __init__(self, column, group):
        self.column, self.group = column, group
        self.n = self.n_exact = self.n_tol = 0
        self.max_abs = np.nan
        self.first = []                                     # [(key tuple, value a, value b)]
        self.by_season = {}
        self.numeric = True

    def add(self, season, res, keys_df, va, vb):
        n = len(va)
        self.n += n
        self.n_exact += int(res['exact'].sum())
        self.n_tol += int(res['tol'].sum())
        self.numeric = self.numeric and res['numeric']
        self.by_season[season] = {'n': n, 'n_tol': int(res['tol'].sum()), 'n_exact': int(res['exact'].sum())}
        if res['numeric'] and n and (~(va.isna().to_numpy() | vb.isna().to_numpy())).any():
            m = float(np.max(res['abs_diff']))
            self.max_abs = m if np.isnan(self.max_abs) else max(self.max_abs, m)
        if len(self.first) < N_FIRST:
            hit = np.flatnonzero(res['tol'] if res['tol'].any() else res['exact'])[:N_FIRST - len(self.first)]
            for i in hit:
                self.first.append((tuple(keys_df.iloc[i].tolist()), va.iloc[i], vb.iloc[i]))

    def row(self, allowed, status='both'):
        return {'column': self.column, 'group': self.group, 'status': status, 'n_compared': self.n, 'n_diff_exact': self.n_exact,
                'share_diff_exact': self.n_exact / self.n if self.n else np.nan, 'n_diff_tol': self.n_tol, 'share_diff_tol': self.n_tol / self.n if self.n else np.nan,
                'max_abs_diff': self.max_abs, 'allowed': bool(allowed), 'first_differing_keys': '; '.join(f'{k}: a={_fmt(a)} b={_fmt(b)}' for k, a, b in self.first)}


def _fmt(v):
    return 'NaN' if (v is None or (isinstance(v, float) and np.isnan(v))) else (f'{v:.12g}' if isinstance(v, (float, np.floating)) else str(v))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# One table
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compare_table(conn_a, conn_b, table, cols_a, cols_b, seasons, log=logging.info):
    ka, kb = key_columns(table, cols_a), key_columns(table, cols_b)
    keys = [k for k in ka if k in kb]
    names_a, names_b = [c for c, _ in cols_a], [c for c, _ in cols_b]
    common = [c for c in names_a if c in names_b and c not in keys]
    out = {'table': table, 'keys': keys, 'columns_only_a': [c for c in names_a if c not in names_b], 'columns_only_b': [c for c in names_b if c not in names_a],
           'common': common, 'row_counts': [], 'keys_only': [], 'stats': {c: ColumnStat(c, feature_group(c)) for c in common}, 'group_rows_any': {}, 'group_rows': {},
           'positive_share': []}
    if 'game_id' not in keys:
        out['note'] = 'no game_id key: only row counts are possible'
        return out
    if ka != kb:
        out['note'] = f'the key columns differ (A {ka}, B {kb}); the common keys {keys} are used'
    groups = {}
    for c in common:
        groups.setdefault(out['stats'][c].group, []).append(c)
    for season in seasons:
        da, db = read_chunk(conn_a, table, names_a, season), read_chunk(conn_b, table, names_b, season)
        dup_a, dup_b = int(da.duplicated(keys).sum()), int(db.duplicated(keys).sum())
        da, db = da.drop_duplicates(keys), db.drop_duplicates(keys)
        for side, d in (('A', da), ('B', db)):
            for c in POSITIVE_SHARE_COLUMNS:
                if c in d.columns and len(d):
                    out['positive_share'].append({'table': table, 'season': season, 'side': side, 'column': c, 'share_gt0': float((d[c] > 0).mean()), 'n': int(len(d))})
        ra = da[keys + common].rename(columns={c: f'{c}__a' for c in common})
        rb = db[keys + common].rename(columns={c: f'{c}__b' for c in common})
        m = ra.merge(rb, on=keys, how='outer', indicator=True)
        both = m[m['_merge'] == 'both'].reset_index(drop=True)
        only_a, only_b = m[m['_merge'] == 'left_only'], m[m['_merge'] == 'right_only']
        out['row_counts'].append({'table': table, 'season': season, 'rows_a': len(da) + dup_a, 'rows_b': len(db) + dup_b, 'shared': len(both), 'only_a': len(only_a),
                                  'only_b': len(only_b), 'duplicate_keys_a': dup_a, 'duplicate_keys_b': dup_b})
        for side, d in (('A', only_a), ('B', only_b)):
            for _, r in d.head(N_KEYS_SAMPLE).iterrows():
                out['keys_only'].append({'table': table, 'season': season, 'only_in': side, 'key': ' | '.join(str(r[k]) for k in keys)})
        any_rows = {g: np.zeros(len(both), bool) for g in groups}
        for c in common:
            va, vb = both[f'{c}__a'], both[f'{c}__b']
            res = compare_values(va, vb)
            out['stats'][c].add(season, res, both[keys], va, vb)
            any_rows[out['stats'][c].group] |= res['tol']
        for g, mask in any_rows.items():
            out['group_rows_any'][(season, g)] = int(mask.sum())
            out['group_rows'][(season, g)] = len(both)
        log(f'{table} {season}: A {len(da)} rows, B {len(db)} rows, {len(both)} shared')
        del da, db, ra, rb, m, both
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Allowed columns, verdict
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def parse_allow(tokens):
    out = []
    for t in tokens or []:
        out += [x.strip() for x in str(t).split(',') if x.strip()]
    return out


def is_allowed(col, allow, group=None):
    for t in allow:
        if t.startswith('group:'):
            if group is not None and group == t[len('group:'):]:
                return True
        elif col == t or col.startswith(t):
            return True
    return False


def verdict(tables, tables_only_a, tables_only_b, allow, exact=False):
    """The list of violations of 'identical outside the allowed columns' (empty = identical)."""
    v = []
    for t in tables_only_a:
        v.append(f'table {t} exists in A only')
    for t in tables_only_b:
        v.append(f'table {t} exists in B only')
    for t in tables:
        name = t['table']
        for c in t['columns_only_a']:
            if not is_allowed(c, allow, feature_group(c)):
                v.append(f'{name}: column {c} exists in A only')
        for c in t['columns_only_b']:
            if not is_allowed(c, allow, feature_group(c)):
                v.append(f'{name}: column {c} exists in B only')
        for r in t['row_counts']:
            if r['only_a'] or r['only_b']:
                v.append(f'{name} {r["season"]}: {r["only_a"]} keys in A only, {r["only_b"]} keys in B only')
            if r['duplicate_keys_a'] or r['duplicate_keys_b']:
                v.append(f'{name} {r["season"]}: duplicated keys (A {r["duplicate_keys_a"]}, B {r["duplicate_keys_b"]})')
        for c, s in t['stats'].items():
            n = s.n_exact if exact else s.n_tol
            if n and not is_allowed(c, allow, s.group):
                v.append(f'{name}.{c} ({s.group}): {n} of {s.n} rows differ' + (' exactly' if exact else f' beyond {REL_TOL:g} relative'))
    return v


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def table_frame(t, allow):
    rows = [s.row(is_allowed(c, allow, s.group)) for c, s in t['stats'].items()]
    rows += [{'column': c, 'group': feature_group(c), 'status': 'only_in_A', 'allowed': is_allowed(c, allow, feature_group(c))} for c in t['columns_only_a']]
    rows += [{'column': c, 'group': feature_group(c), 'status': 'only_in_B', 'allowed': is_allowed(c, allow, feature_group(c))} for c in t['columns_only_b']]
    return pd.DataFrame(rows)


def group_frame(tables):
    rows = []
    for t in tables:
        by = {}
        for c, s in t['stats'].items():
            g = by.setdefault(s.group, {'columns': 0, 'columns_differing': 0, 'cells': 0, 'cells_differing': 0})
            g['columns'] += 1
            g['columns_differing'] += int(s.n_tol > 0)
            g['cells'] += s.n
            g['cells_differing'] += s.n_tol
        for g, x in by.items():
            any_rows = sum(v for (sn, gg), v in t['group_rows_any'].items() if gg == g)
            rows_n = sum(v for (sn, gg), v in t['group_rows'].items() if gg == g)
            rows.append({'table': t['table'], 'group': g, **x, 'share_cells_differing': x['cells_differing'] / x['cells'] if x['cells'] else np.nan, 'rows': rows_n,
                         'rows_any_column_differing': any_rows, 'share_rows_any_differing': any_rows / rows_n if rows_n else np.nan})
    return pd.DataFrame(rows)


def season_group_frame(tables):
    rows = []
    for t in tables:
        for (season, g), n in sorted(t['group_rows_any'].items()):
            tot = t['group_rows'][(season, g)]
            rows.append({'table': t['table'], 'season': season, 'group': g, 'rows': tot, 'rows_any_column_differing': n, 'share': n / tot if tot else np.nan})
    return pd.DataFrame(rows)


def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return _json(o.to_dict(orient='records'))
    return o


def resolve_outputs(out, report):
    o = Path(out)
    if o.suffix.lower() == '.json':
        return o, o.parent / f'{o.stem}_tables', Path(report) if report else o.with_suffix('.md')
    return o / 'summary.json', o, Path(report) if report else o / 'report.md'


def run(a):
    seasons = check_seasons(a.seasons)
    inputs = [a.a, a.b]
    json_path, csv_dir, report_path = resolve_outputs(a.out, a.report)
    check_output_paths([a.out, json_path, report_path, csv_dir], inputs)
    allow = parse_allow(a.allow_columns)
    ca, cb = open_ro(a.a), open_ro(a.b)
    try:
        info_a, info_b = table_columns(ca), table_columns(cb)
        shared = [t for t in info_a if t in info_b]
        tables = [compare_table(ca, cb, t, info_a[t], info_b[t], seasons) for t in shared]
    finally:
        ca.close()
        cb.close()
    only_a, only_b = [t for t in info_a if t not in info_b], [t for t in info_b if t not in info_a]
    viol = verdict(tables, only_a, only_b, allow, exact=a.exact)
    identical = not verdict(tables, only_a, only_b, [], exact=False)
    csv_dir.mkdir(parents=True, exist_ok=True)
    frames = {t['table']: table_frame(t, allow) for t in tables}
    for name, f in frames.items():
        f.to_csv(csv_dir / f'{name}.csv', index=False)
    counts = pd.DataFrame([r for t in tables for r in t['row_counts']])
    counts.to_csv(csv_dir / 'row_counts.csv', index=False)
    ko = pd.DataFrame([r for t in tables for r in t['keys_only']], columns=['table', 'season', 'only_in', 'key'])
    ko.to_csv(csv_dir / 'keys_only.csv', index=False)
    gf, sgf = group_frame(tables), season_group_frame(tables)
    gf.to_csv(csv_dir / 'group_summary.csv', index=False)
    sgf.to_csv(csv_dir / 'group_by_season.csv', index=False)
    pos = pd.DataFrame([r for t in tables for r in t['positive_share']])
    if len(pos):
        pos.to_csv(csv_dir / 'positive_share.csv', index=False)
    base = {}
    for t in tables:
        cells = sum(s.n for s in t['stats'].values())
        diff = sum(s.n_tol for s in t['stats'].values())
        base[t['table']] = {'cells': cells, 'cells_differing': diff, 'share': diff / cells if cells else np.nan}
    bshare = base.get('player_game_base', {}).get('share', np.nan)
    summary = {'a': str(a.a), 'b': str(a.b), 'seasons': seasons, 'rel_tol': REL_TOL, 'exact_mode': bool(a.exact), 'tables_compared': shared, 'tables_only_in_a': only_a,
               'tables_only_in_b': only_b, 'identical': bool(identical), 'expect_identical': bool(a.expect_identical), 'allow_columns': allow, 'violations': viol,
               'passed': not viol, 'cell_share_differing_by_table': base, 'base_cell_share_differing': bshare,
               'base_share_over_0.1pct': bool(np.isfinite(bshare) and bshare > BASE_SHARE_THRESHOLD),
               'columns_differing': {n: [r['column'] for _, r in f.iterrows() if r.get('n_diff_tol', 0) and r['n_diff_tol'] > 0] for n, f in frames.items()},
               'groups_differing': {n: sorted({r['group'] for _, r in f.iterrows() if r.get('n_diff_tol', 0) and r['n_diff_tol'] > 0}) for n, f in frames.items()},
               'row_counts': counts, 'group_summary': gf, 'positive_share': pos, 'notes': {t['table']: t['note'] for t in tables if t.get('note')},
               'sha256': {'a': None if a.no_hash else registry.sha256_file(a.a), 'b': None if a.no_hash else registry.sha256_file(a.b)}}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_json(summary), indent=2), encoding='utf-8')
    md = ['# Feature database comparison\n', f'A: `{a.a}`\n\nB: `{a.b}`\n', f'Seasons compared: {", ".join(seasons)} ({HOLDOUT_SEASON} and later are never read, plan A.1). Tolerance {REL_TOL:g} relative; NaN equals NaN. '
          f'Allowed columns: {allow or "none"}.\n']
    if summary['sha256']['a']:
        md.append(f'sha256 A `{summary["sha256"]["a"]}`; B `{summary["sha256"]["b"]}`\n')
    md += [f'**Identical (no difference at all): {identical}.**', f'**Violations outside the allowed columns: {len(viol)}.**\n']
    md += [f'- {v}' for v in viol[:60]] + ([f'- ... {len(viol) - 60} more'] if len(viol) > 60 else [])
    md += ['', f'Share of player_game_base cells that differ beyond tolerance: {bshare:.6f} (plan B.6.3: more than {BASE_SHARE_THRESHOLD:.3f} re-bases the reference database: '
           f'{summary["base_share_over_0.1pct"]}).\n', '## Row counts by table and season\n', md_table(counts, 4),
           '## By feature group (cells and rows that differ beyond tolerance)\n', md_table(gf, 6),
           '## Rows with any differing column of a group, by season\n', md_table(sgf, 6)]
    if len(pos):
        md += ['## Share of rows with av_n_out > 0\n', md_table(pos, 4)]
    for n, f in frames.items():
        d = f[(f['status'] != 'both') | (f.get('n_diff_tol', pd.Series(0, index=f.index)).fillna(0) > 0) | (f.get('n_diff_exact', pd.Series(0, index=f.index)).fillna(0) > 0)]
        md += [f'## {n}: differing or one-sided columns ({len(d)} of {len(f)}; every column is in {n}.csv)\n', md_table(d.head(80), 6)]
        if len(d) > 80:
            md.append(f'... {len(d) - 80} more in the CSV\n')
    if len(ko):
        md += ['## Keys present in one database only (sample)\n', md_table(ko.head(40), 3)]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(md), encoding='utf-8')
    print(f'identical: {identical}; violations: {len(viol)}; base cells differing: {bshare:.6f}')
    for v in viol[:20]:
        print('  ', v)
    print(f'wrote {json_path}, {csv_dir}, {report_path}')
    exit_code = 1 if (a.expect_identical and viol) else 0
    summary['exit_code'] = exit_code
    summary['frames'] = frames
    return summary


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', required=True, help='first database (the reference: v3 or ctl)')
    ap.add_argument('--b', required=True, help='second database')
    ap.add_argument('--out', required=True, help='a .json file (tables go to <stem>_tables/) or a folder')
    ap.add_argument('--report', default=None, help='markdown report (default: beside --out)')
    ap.add_argument('--seasons', nargs='+', default=default_seasons(), help='default 2019-20 ... 2024-25; 2025-26 and later are refused')
    ap.add_argument('--expect-identical', action='store_true', help='exit 1 on any difference outside --allow-columns')
    ap.add_argument('--allow-columns', nargs='*', default=[], help='column names, prefixes, group:<name>, or comma-separated lists')
    ap.add_argument('--exact', action='store_true', help='with --expect-identical: any exact difference counts, not only differences beyond 1e-9 relative')
    ap.add_argument('--no-hash', action='store_true', help='do not compute the sha256 of the two files')
    return ap.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    return run(parse(argv))


if __name__ == '__main__':
    sys.exit(main()['exit_code'])
