"""
The acceptance criteria A1-A6 of reports/props/d1_parity_preregistration.md as functions of already-aligned frames, with the operational definitions of
reports/props/d1_parity_preregistration_addendum1.md. Each returns a dict with the measured numbers and a boolean `pass`. Nothing here depends on the live builder.
"""
import numpy as np
import pandas as pd

from props.features import dictionary
from props.models import data

CONTEXT_GROUPS = {'team_context', 'opponent', 'schedule'}
TOL_CONTEXT, TOL_HISTORY = 1e-6, 1e-9
KEY = ['game_id', 'player_id', 'minutes_before']
REQUIRED_MISSING_RATE = 0.01
A3_MIN_SHARE, A4_MIN_SHARE = 0.99, 0.95
A5_MEAN_STATS = ('points', 'rebounds', 'assists', 'threes')
A6_MIN_JACCARD, A6_MIN_SIDE, A6_MIN_EV_CORR = 0.85, 0.98, 0.98


def spec_columns(columns):
    return data.FeatureSpec('base').columns(columns)


def group_of(col):
    info = dictionary.describe(col)
    return info[0] if info else None


def _same(a, b, tol):
    """Elementwise agreement of two aligned Series: within `tol` for numbers (both missing agree), equal for anything else."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    if a.dtype.kind in 'fiub' and b.dtype.kind in 'fiub':
        x, y = a.astype(float), b.astype(float)
        return ((x - y).abs() <= tol) | (x.isna() & y.isna())
    return (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())


def merge_frames(live, train):
    """Rows present in both frames; training columns keep their names, the live ones get the suffix __live."""
    return train.merge(live, on=KEY, how='inner', suffixes=('', '__live'), validate='one_to_one')


def a1_coverage(traded, live, train, columns):
    """
    traded: distinct (game, player, T) with a quote. live / train: model-input frames (train = the stored rows of the same games). A live row is complete when it has no missing value in
    a base-spec column that is missing in fewer than REQUIRED_MISSING_RATE of the training rows. Returns the share of traded rows with a complete live row and the failures.
    """
    cols = [c for c in spec_columns(columns) if c in live.columns and c in train.columns]
    required = [c for c in cols if c not in KEY and train[c].isna().mean() < REQUIRED_MISSING_RATE]
    t = traded[KEY].drop_duplicates().merge(live[KEY + required], on=KEY, how='left', indicator=True)
    has = t['_merge'] == 'both'
    missing_cols = t[required].isna()
    complete = has & ~missing_cols.any(axis=1)
    fail = t[~complete].copy()
    which = [', '.join(missing_cols.columns[row][:3]) for row in missing_cols[~complete].to_numpy()]
    fail['reason'] = ['no live row' if not h else f'incomplete: {w}' for h, w in zip(has[~complete].to_numpy(), which)]
    share = float(complete.mean()) if len(t) else float('nan')
    return {'traded': int(len(t)), 'complete': int(complete.sum()), 'share': share, 'required_columns': len(required), 'failures': fail[KEY + ['reason']], 'pass': share >= 0.99}


def a2_identical(m, columns):
    """m: merge_frames output. Every base-spec column outside the injury and teammate groups must agree on 100% of the compared rows."""
    out = []
    for c in spec_columns(columns):
        g = group_of(c)
        if g in ('injury', 'teammates') or c == 'minutes_before' or c + '__live' not in m.columns:
            continue
        tol = TOL_CONTEXT if g in CONTEXT_GROUPS else TOL_HISTORY
        comp = m[c].notna().to_numpy()
        ok = _same(m.loc[comp, c], m.loc[comp, c + '__live'], tol).to_numpy()
        d = (pd.to_numeric(m.loc[comp, c], errors='coerce') - pd.to_numeric(m.loc[comp, c + '__live'], errors='coerce')).abs()
        out.append({'column': c, 'group': g, 'tol': tol, 'compared': int(comp.sum()), 'disagree': int((~ok).sum()), 'max_abs_diff': float(d.max()) if d.notna().any() else 0.0})
    t = pd.DataFrame(out)
    return {'table': t, 'columns': len(t), 'failing_columns': int((t['disagree'] > 0).sum()), 'pass': bool((t['disagree'] == 0).all())}


def a3_injury(m):
    code = _same(m['inj_status_code'], m['inj_status_code__live'], 0.0)
    pp = _same(m['inj_p_play'], m['inj_p_play__live'], 1e-9)
    return {'rows': int(len(m)), 'status_share': float(code.mean()), 'p_play_share': float(pp.mean()), 'pass': bool(code.mean() >= A3_MIN_SHARE and pp.mean() >= A3_MIN_SHARE)}


def av_tolerance(col):
    if col.endswith(('_n_out', '_n_questionable')):
        return 1.0
    if col.endswith('miss_starter_minutes'):
        return 3.0
    return 0.03


def a4_availability(m):
    out = []
    for c in [c for c in m.columns if c.startswith(('av_', 'opp_av_')) and not c.endswith('__live')]:
        tol = av_tolerance(c)
        ok = _same(m[c], m[c + '__live'], tol)
        d = (m[c] - m[c + '__live']).abs()
        out.append({'column': c, 'tol': tol, 'share_within': float(ok.mean()), 'median_abs_diff': float(d.median()), 'p95_abs_diff': float(d.quantile(0.95))})
    t = pd.DataFrame(out)
    return {'table': t, 'worst_share': float(t['share_within'].min()), 'failing_columns': int((t['share_within'] < A4_MIN_SHARE).sum()), 'pass': bool((t['share_within'] >= A4_MIN_SHARE).all())}


def a5_model(means, p_over):
    """
    means: {stat: (live_mean, training_mean)} aligned arrays. p_over: DataFrame with columns key (game_id, player_id, minutes_before, stat), p_live, p_train in probability units.
    Mean criterion per stat: the relative change is under 1% in at least 95% of rows. P(over) criterion: the change in percentage points has mean < 0.5, p95 < 2, max < 10.
    """
    rows = {}
    for s, (a, b) in means.items():
        rel = np.abs(np.asarray(a, float) - np.asarray(b, float)) / np.maximum(np.asarray(b, float), 1e-9)
        rows[s] = {'rows': int(len(rel)), 'share_under_1pct': float((rel < 0.01).mean()), 'median_rel_change': float(np.median(rel)), 'p95_rel_change': float(np.quantile(rel, 0.95))}
    d = (p_over['p_live'] - p_over['p_train']).abs() * 100.0
    po = {'quotes': int(len(d)), 'mean_abs_pp': float(d.mean()), 'p95_pp': float(d.quantile(0.95)), 'max_pp': float(d.max()), 'over_10pp': p_over.loc[d > 10]}
    ok = all(v['share_under_1pct'] >= 0.95 for v in rows.values()) and po['mean_abs_pp'] < 0.5 and po['p95_pp'] < 2 and po['max_pp'] < 10
    return {'means': rows, 'p_over': po, 'pass': bool(ok)}


def a6_decisions(sel_live, sel_train):
    """Qualified bets (game, player, stat) from the frozen rule with live and with training features."""
    key = ['game_id', 'player_id', 'stat']
    both = sel_live[key + ['best_side', 'best_ev']].merge(sel_train[key + ['best_side', 'best_ev']], on=key, how='inner', suffixes=('_live', '_train'), validate='one_to_one')
    union = len(sel_live) + len(sel_train) - len(both)
    jac = len(both) / union if union else float('nan')
    side = float((both['best_side_live'] == both['best_side_train']).mean()) if len(both) else float('nan')
    corr = float(np.corrcoef(both['best_ev_live'], both['best_ev_train'])[0, 1]) if len(both) > 2 else float('nan')
    return {'bets_live': int(len(sel_live)), 'bets_train': int(len(sel_train)), 'shared': int(len(both)), 'jaccard': float(jac), 'side_agreement': side, 'ev_corr': corr,
            'pass': bool(jac >= A6_MIN_JACCARD and side >= A6_MIN_SIDE and corr >= A6_MIN_EV_CORR)}
