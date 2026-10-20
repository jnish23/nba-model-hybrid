"""
Phase 4 evaluation against outcomes (reports/props/phase4_plan.md).

Everything here works on saved predictive pmfs (registry) joined to the realized labels. Uncertainty is a cluster bootstrap: whole games (every player and
decision time of a game) are resampled together; a date-clustered run is the sensitivity check. Differences are PAIRED (same rows, model minus comparator).
"""
import numpy as np
import pandas as pd

from props.models import calibrate, outcome as oc, registry, report
from props.models.pipeline import observed

STATS = list(oc.CAPS)
N_BOOT = 2000
HIST_GAMES = 20                    # the empirical baseline looks back this many played games
MIN_HIST = 5                       # fewer prior games than this: that row falls back to the Poisson baseline
RATE_COLS = ['m_min_h15', 'r_points_h15', 'r_rebounds_h15', 'r_assists_h15', 'r_fg3m_h15']


def load_rows(version, season, frame):
    idx, pmfs = registry.load_predictions(version, season)
    keep = [c for c in frame.columns if c.startswith(('y_', 'lbl_'))] + RATE_COLS + ['pos_group']
    f = frame[['game_id', 'player_id', 'minutes_before'] + list(dict.fromkeys(keep))]
    rows = idx.merge(f, on=['game_id', 'player_id', 'minutes_before'], how='left', validate='one_to_one')
    return rows, pmfs


# ------------------------------------------------------------------------------------------------ bootstrap

def _cluster_sums(values, clusters):
    codes, uniq = pd.factorize(pd.Series(np.asarray(clusters)))
    g = len(uniq)
    return np.bincount(codes, weights=np.asarray(values, float), minlength=g), np.bincount(codes, minlength=g).astype(float), g


def cluster_bootstrap(num, clusters, den=None, n_boot=N_BOOT, seed=0):
    """Point estimate and 95% interval of sum(num)/sum(den) (den defaults to the row count, i.e. the mean) resampling whole clusters."""
    sn, cnt, g = _cluster_sums(num, clusters)
    sd = cnt if den is None else _cluster_sums(den, clusters)[0]
    point = sn.sum() / sd.sum()
    w = np.random.default_rng(seed).multinomial(g, np.full(g, 1.0 / g), size=n_boot).astype(np.float64)
    draws = (w @ sn) / (w @ sd)
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


# ------------------------------------------------------------------------------------------------ baselines

def empirical_pmfs(rows, frame, stat, cap, n_hist=HIST_GAMES, min_hist=MIN_HIST):
    """The player's own distribution over his last n_hist played games BEFORE the target date, per-minute values rescaled to the row's expected minutes.
    Rows with fewer than min_hist prior games get NaN (the caller falls back to the Poisson baseline)."""
    h = frame[(frame['minutes_before'] == 60) & (frame['lbl_played'] == 1) & (frame['lbl_minutes'] > 0)]
    hy = observed(h, stat)
    vpm = hy / h['lbl_minutes'].to_numpy(float)
    hist = pd.DataFrame({'player_id': h['player_id'].to_numpy(), 'date': pd.to_datetime(h['game_date']).to_numpy(), 'vpm': vpm}).dropna().sort_values(['player_id', 'date'])
    by_player = {p: (g['date'].to_numpy(), g['vpm'].to_numpy()) for p, g in hist.groupby('player_id')}
    out = np.full((len(rows), cap + 1), np.nan, dtype=np.float32)
    exp_min = rows['m_min_h15'].to_numpy(float)
    dates = pd.to_datetime(rows['game_date']).to_numpy()
    for i, (p, d, m) in enumerate(zip(rows['player_id'].to_numpy(), dates, exp_min)):
        if p not in by_player or np.isnan(m):
            continue
        hd, hv = by_player[p]
        k = np.searchsorted(hd, d, side='left')                                       # strictly earlier dates only
        if k < min_hist:
            continue
        vals = np.clip(np.rint(hv[max(0, k - n_hist):k] * m), 0, cap).astype(int)
        out[i] = np.bincount(vals, minlength=cap + 1) / len(vals)
    return out


def baseline_set(rows, frame, stat):
    """Poisson, Normal and empirical last-20 pmfs for the rows' observed outcomes."""
    cap = oc.CAPS[stat]
    y = observed(rows, stat)
    ok = ~np.isnan(y) & rows['m_min_h15'].notna().to_numpy()
    pois, nrm = report.baseline_pmfs(rows[ok], stat, y[ok].astype(int), cap)
    emp = empirical_pmfs(rows[ok], frame, stat, cap)
    miss = np.isnan(emp).any(axis=1)
    emp[miss] = pois[miss]
    return ok, {'poisson': pois, 'normal': nrm, 'empirical20': emp}, float(miss.mean())


# ------------------------------------------------------------------------------------------------ scores

def per_row_crps(pm, y):
    return oc.crps(pm, y.astype(int))


def ladder_scores(stat, pm, y):
    """Log loss, Brier and ECE of P(X > line) at the ladder lines around each row's own mean (the calibration table's lines)."""
    lines, raw, _ = calibrate.over_prob_table(pm)
    hit = (y[:, None] > lines).astype(float)
    p = np.clip(raw, calibrate.EPS, 1 - calibrate.EPS)
    ll = float(-np.mean(hit * np.log(p) + (1 - hit) * np.log(1 - p)))
    q = np.quantile(raw, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(q, raw.ravel(), side='right') - 1, 0, 9)
    fp, fh = raw.ravel(), hit.ravel()
    ece = float(sum(abs(fp[b == k].mean() - fh[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any()))
    return {'logloss': ll, 'brier': float(np.mean((raw - hit) ** 2)), 'ece': ece}


def accuracy_table(rows, pmfs, frame, stats=STATS, n_boot=N_BOOT):
    """Per stat: mean CRPS of the model and of each baseline, and the paired model-minus-best-baseline difference with game- and date-clustered intervals."""
    out = []
    for stat in stats:
        ok, bases, fallback = baseline_set(rows, frame, stat)
        y = observed(rows, stat)[ok]
        sub = rows[ok]
        c_model = per_row_crps(pmfs[stat][ok], y)
        c_base = {k: per_row_crps(v, y) for k, v in bases.items()}
        best = min(c_base, key=lambda k: c_base[k].mean())
        d = c_model - c_base[best]
        g = cluster_bootstrap(d, sub['game_id'], n_boot=n_boot)
        dt = cluster_bootstrap(d, sub['game_date'], n_boot=n_boot)
        r = {'stat': stat, 'n': int(ok.sum()), 'crps_model': c_model.mean(), **{f'crps_{k}': v.mean() for k, v in c_base.items()}, 'best_baseline': best,
             'diff_vs_best': g[0], 'diff_lo_game': g[1], 'diff_hi_game': g[2], 'diff_lo_date': dt[1], 'diff_hi_date': dt[2],
             'rel_improvement_%': -100 * g[0] / c_base[best].mean(), 'empirical_fallback_share': fallback}
        out.append(r)
    return pd.DataFrame(out)


def bias_table(rows, pmfs, stats=STATS, n_boot=N_BOOT, by=None):
    """Observed / predicted - 1 of the mean, with a game-clustered interval. `by` is an optional column to split on."""
    out = []
    groups = [('all', np.ones(len(rows), bool))] if by is None else [(k, (rows[by] == k).to_numpy()) for k in sorted(rows[by].dropna().unique())]
    for stat in stats:
        y = observed(rows, stat)
        mu = oc.pmf_mean(pmfs[stat])
        for name, m in groups:
            m = m & ~np.isnan(y)
            if m.sum() < 500:
                continue
            p, lo, hi = cluster_bootstrap(y[m], rows['game_id'].to_numpy()[m], den=mu[m], n_boot=n_boot)
            out.append({'stat': stat, 'segment': name, 'n': int(m.sum()), 'obs_over_pred_minus_1_%': 100 * (p - 1), 'lo_%': 100 * (lo - 1), 'hi_%': 100 * (hi - 1)})
    return pd.DataFrame(out)


def calibration_scores(rows, pmfs, stats=STATS):
    out = []
    for stat in stats:
        y = observed(rows, stat)
        ok = ~np.isnan(y)
        out.append({'stat': stat, **ladder_scores(stat, pmfs[stat][ok], y[ok])})
    return pd.DataFrame(out)


def by_decision_time(rows, pmfs, stats=STATS, n_boot=N_BOOT):
    """Mean CRPS per decision time, and the paired T-5 minus T-240 difference per player-game (negative = later information helps)."""
    out = []
    for stat in stats:
        y = observed(rows, stat)
        ok = ~np.isnan(y)
        c = np.full(len(rows), np.nan)
        c[ok] = per_row_crps(pmfs[stat][ok], y[ok])
        df = pd.DataFrame({'g': rows['game_id'], 'p': rows['player_id'], 't': rows['minutes_before'], 'c': c}).dropna()
        w = df.pivot_table(index=['g', 'p'], columns='t', values='c').dropna()
        d = (w[5] - w[240]).reset_index()
        pt, lo, hi = cluster_bootstrap(d[0].to_numpy(), d['g'], n_boot=n_boot)
        out.append({'stat': stat, 'crps_T240': w[240].mean(), 'crps_T60': w[60].mean(), 'crps_T5': w[5].mean(), 'T5_minus_T240': pt, 'lo': lo, 'hi': hi})
    return pd.DataFrame(out)


def compare_versions(rows_a, pm_a, rows_b, pm_b, stats=STATS, n_boot=N_BOOT):
    """Paired CRPS difference B minus A (negative = B better) with a game-clustered interval, and each version's overall bias. Rows must be the same rows."""
    assert (rows_a['game_id'].to_numpy() == rows_b['game_id'].to_numpy()).all() and (rows_a['player_id'].to_numpy() == rows_b['player_id'].to_numpy()).all()
    out = []
    for stat in stats:
        y = observed(rows_a, stat)
        ok = ~np.isnan(y)
        ca, cb = per_row_crps(pm_a[stat][ok], y[ok]), per_row_crps(pm_b[stat][ok], y[ok])
        p, lo, hi = cluster_bootstrap(cb - ca, rows_a['game_id'].to_numpy()[ok], n_boot=n_boot)
        ba = 100 * (y[ok].mean() / oc.pmf_mean(pm_a[stat][ok]).mean() - 1)
        bb = 100 * (y[ok].mean() / oc.pmf_mean(pm_b[stat][ok]).mean() - 1)
        out.append({'stat': stat, 'crps_A': ca.mean(), 'crps_B': cb.mean(), 'B_minus_A': p, 'lo': lo, 'hi': hi, 'bias_A_%': ba, 'bias_B_%': bb,
                    'bias_shrink_share': 1 - abs(bb) / abs(ba) if ba != 0 else np.nan})
    return pd.DataFrame(out)
