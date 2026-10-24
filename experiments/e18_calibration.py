"""E18 - calibration of the totals model by total level (angle 4 of 8). Registered in reports/experiments/PREREGISTRATION.md ("E18") before
any residual was computed. Uses the saved baseline walk-forward predictions; nothing is retrained. python -m experiments.e18_calibration"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import metrics as M                                    # noqa: E402
from experiments.e14_blowout_under import DayBoot                        # noqa: E402
from experiments.lib import TEST_SEASONS, label_calib_folds, load_games, load_run   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports' / 'experiments' / 'E18_calibration'
HIGH = 242.0
FIT_PARAMS = {}


def trailing_level(all_games, dates):
    """Mean opening total over games dated in [d - 365 days, d), past lines only."""
    g = all_games.dropna(subset=['odds_total']).assign(d=lambda x: pd.to_datetime(x['gameDate']).dt.normalize())
    daily = g.groupby('d')['odds_total'].agg(['sum', 'count']).sort_index()
    cs, cc = np.concatenate([[0], daily['sum'].cumsum().to_numpy()]), np.concatenate([[0], daily['count'].cumsum().to_numpy()])
    days = daily.index.to_numpy()
    d = pd.to_datetime(pd.Series(dates)).dt.normalize().to_numpy()
    hi = np.searchsorted(days, d, side='left')                       # days strictly before d
    lo = np.searchsorted(days, d - np.timedelta64(365, 'D'), side='left')
    n = cc[hi] - cc[lo]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)


def make_fix(kind, cal, level='odds_total'):
    """Fit on the calibration frame; return f(frame) -> corrected predicted totals."""
    ok = cal[level].notna() & cal['total_pred'].notna() & cal['target_total_pts'].notna()
    c = cal[ok]
    if kind == 'a':
        iso = IsotonicRegression(out_of_bounds='clip').fit(c['total_pred'], c['target_total_pts'])
        return lambda f: np.where(f['total_pred'].notna(), iso.predict(f['total_pred'].fillna(0)), np.nan)
    mu = float(c[level].mean())
    if kind == 'b':
        x = c[level] - mu
        A = np.column_stack([np.ones(len(c)), x])
        coef = np.linalg.lstsq(A, c['target_total_pts'] - c['total_pred'], rcond=None)[0]
        f_b = lambda f: np.where(f[level].notna(), f['total_pred'] + coef[0] + coef[1] * (f[level] - mu), f['total_pred'])
        f_b.params = {'intercept': float(coef[0]), 'slope_per_total_point': float(coef[1]), 'mean_level': mu}
        return f_b
    # c: shrink toward the line, weight linear in the level, no intercept, clipped to [0, 1]
    gap = c['total_pred'] - c['odds_total']
    A = np.column_stack([gap, gap * (c[level] - mu)])
    b = np.linalg.lstsq(A, c['target_total_pts'] - c['odds_total'], rcond=None)[0]

    def f_(f):
        w = np.clip(b[0] + b[1] * (f[level] - mu), 0, 1)
        out = f['odds_total'] + w * (f['total_pred'] - f['odds_total'])
        return np.where(f[level].notna() & f['odds_total'].notna(), out, f['total_pred'])
    f_.params = {'w0': float(b[0]), 'w1': float(b[1]), 'mean_level': mu}
    return f_


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, games = load_games()
    base, calib = load_run('baseline')
    calib['fold'] = label_calib_folds(games, calib)
    for fr in (base, calib):
        fr['gameDate'] = pd.to_datetime(fr['gameDate'])
        fr['lvl_era'] = fr['odds_total'] - trailing_level(games, fr['gameDate'])
    base = base.reset_index(drop=True)
    db = DayBoot(base['gameDate'])
    S = base['seasonYear']
    ones = np.ones(len(base))
    act, line = base['target_total_pts'].to_numpy(float), base['odds_total'].to_numpy(float)
    res = {'n_games': int(len(base)), 'n_calibration_rows': int(len(calib))}

    # ---------------- step 1: descriptive
    resid = act - base['total_pred'].to_numpy(float)
    base['bias'] = resid
    ok = np.isfinite(resid)
    res['overall_bias'] = {'mean': float(np.nanmean(resid)), 'mae': float(np.nanmean(np.abs(resid)))}
    s1 = {'deciles': {}}
    for col in ('total_pred', 'odds_total'):
        edges = np.nanquantile(base[col], np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        base[f'dec_{col}'] = np.digitize(base[col], edges[1:-1])
        base[f'decws_{col}'] = base.groupby('seasonYear')[col].transform(lambda s: pd.qcut(s, 10, labels=False, duplicates='drop'))
        rows = []
        for d in range(10):
            m = (base[f'dec_{col}'] == d).to_numpy() & ok & base[col].notna().to_numpy()
            pt, lo, hi = db.ratio(base['bias'].fillna(0), ones, m)
            row = {'decile': d + 1, 'edge_lo': float(edges[d]) if d else None, 'games': int(m.sum()), 'bias': pt, 'lo': lo, 'hi': hi,
                   'by_season': {s: (float(base.loc[m & (S == s).to_numpy(), 'bias'].mean()) if (m & (S == s).to_numpy()).sum() else None) for s in TEST_SEASONS},
                   'n_by_season': {s: int((m & (S == s).to_numpy()).sum()) for s in TEST_SEASONS}}
            mw = (base[f'decws_{col}'] == d).to_numpy() & ok
            row['within_season_bias'] = db.ratio(base['bias'].fillna(0), ones, mw)[0]
            rows.append(row)
        s1['deciles'][col] = rows
    hi_tab = {}
    for col in ('odds_total', 'total_pred'):
        hm = (base[col] > HIGH).to_numpy() & ok
        pt, lo, hi = db.ratio(base['bias'].fillna(0), ones, hm)
        hi_tab[col] = {'games': int(hm.sum()), 'bias': pt, 'lo': lo, 'hi': hi, 'by_season': {}}
        for s in TEST_SEASONS:
            m = hm & (S == s).to_numpy()
            p_, l_, h_ = db.ratio(base['bias'].fillna(0), ones, m) if m.sum() else (None, None, None)
            hi_tab[col]['by_season'][s] = {'games': int(m.sum()), 'bias': p_, 'lo': l_, 'hi': h_, 'share_of_season': float(m.sum() / (S == s).sum())}
    s1['above_242'] = hi_tab
    lv = (base['odds_total'] - base['odds_total'].mean()).to_numpy()
    m_ok = ok & np.isfinite(lv)
    X1 = np.column_stack([np.ones(len(base)), np.nan_to_num(lv)])
    X2 = np.column_stack([X1] + [(S == s).to_numpy(float) for s in TEST_SEASONS[1:]])
    reg = {}
    for lab, X in (('without season dummies', X1), ('with season dummies', X2)):
        b_, lo_, hi_, _ = db.ols(X, np.nan_to_num(resid), m_ok)
        reg[lab] = {'slope_per_total_point': float(b_[1]), 'lo': float(lo_[1]), 'hi': float(hi_[1])}
    s1['level_regression'] = reg
    s1['level_effect_survives_season_control'] = bool(reg['with season dummies']['lo'] > 0 or reg['with season dummies']['hi'] < 0)
    s1['mean_total_by_season'] = {s: float(base.loc[S == s, 'odds_total'].mean()) for s in TEST_SEASONS}
    res['step1'] = s1

    # ---------------- step 2: fixes
    variants = {'a': 'odds_total', 'b': 'odds_total', 'c': 'odds_total'}
    corrected = {'baseline': (base['total_pred'].to_numpy(float), base['sigma_total'].to_numpy(float))}

    def run(tag, kind, level, recent):
        pred = base['total_pred'].to_numpy(float).copy()
        sig = np.zeros(len(base))
        for s in TEST_SEASONS:
            cal = calib[calib['fold'] == s].sort_values('gameDate')
            fit_on = cal.iloc[len(cal) // 2:] if recent else cal
            fx = make_fix(kind, fit_on, level)
            if hasattr(fx, 'params'):
                FIT_PARAMS.setdefault(tag, {})[s] = fx.params
            m = (S == s).to_numpy()
            pred[m] = fx(base[m])
            cres = cal['target_total_pts'] - fx(cal)
            sig[m] = float(np.nanstd(cres))
        corrected[tag] = (pred, sig)
    for kind in 'abc':
        run(f'{kind}', kind, 'odds_total', False)
        run(f'{kind}_R1', kind, 'odds_total', True)
    for kind in 'bc':
        run(f'{kind}_R2', kind, 'lvl_era', False)

    # ---------------- step 3: metrics
    nonpush = act != line
    over = (act > line).astype(float)

    def cover_ll(pred, sig):
        with np.errstate(invalid='ignore'):
            p = np.clip(norm.cdf((pred - line) / sig), 1e-6, 1 - 1e-6)
        ll = -(over * np.log(p) + (1 - over) * np.log(1 - p))
        return np.where(nonpush & np.isfinite(ll), ll, np.nan)
    b_pred, b_sig = corrected['baseline']
    base_ae, base_ll = np.abs(act - b_pred), cover_ll(b_pred, b_sig)
    hi_m = (base['odds_total'] > HIGH).to_numpy()
    out = {}
    for tag, (pred, sig) in corrected.items():
        ae = np.abs(act - pred)
        ll = cover_ll(pred, sig)
        row = {'mae': float(np.nanmean(ae)), 'mae_by_season': {s: float(np.nanmean(ae[(S == s).to_numpy()])) for s in TEST_SEASONS},
               'bias_above_242': float(np.nanmean((act - pred)[hi_m])), 'mae_above_242': float(np.nanmean(ae[hi_m])), 'cover_logloss': float(np.nanmean(ll))}
        if tag != 'baseline':
            d = np.where(np.isfinite(ae) & np.isfinite(base_ae), ae - base_ae, 0.0)
            valid = (np.isfinite(ae) & np.isfinite(base_ae)).astype(float)
            pt, lo, hi = db.ratio(d, valid, valid > 0)
            n_d, d_d = db.day_sum(d), db.day_sum(valid)
            boots = (db.W @ n_d) / (db.W @ d_d)
            row['paired_mae'] = {'diff': pt, 'lo': lo, 'hi': hi, 'p_one_sided_better': float((1 + (boots >= 0).sum()) / (len(boots) + 1))}
            per = {}
            for s in TEST_SEASONS:
                m = (S == s).to_numpy() & (valid > 0)
                p_, l_, h_ = db.ratio(d, valid, m)
                per[s] = {'diff': p_, 'lo': l_, 'hi': h_}
            row['paired_mae_by_season'] = per
            row['seasons_better'] = int(sum(v['diff'] < 0 for v in per.values()))
            row['seasons_ci_excludes_zero_in_favor'] = int(sum(v['hi'] < 0 for v in per.values()))
            dl = np.where(np.isfinite(ll) & np.isfinite(base_ll), ll - base_ll, 0.0)
            vl = (np.isfinite(ll) & np.isfinite(base_ll)).astype(float)
            row['paired_cover_logloss'] = dict(zip(('diff', 'lo', 'hi'), db.ratio(dl, vl, vl > 0)))
            mm = hi_m & (valid > 0)
            row['paired_mae_above_242'] = dict(zip(('diff', 'lo', 'hi'), db.ratio(d, valid, mm))) if mm.sum() else None
            for lab, sub in (('discovery', S.isin(TEST_SEASONS[:3]).to_numpy()), ('confirmation', S.isin(TEST_SEASONS[3:]).to_numpy())):
                row[f'paired_mae_{lab}'] = dict(zip(('diff', 'lo', 'hi'), db.ratio(d, valid, sub & (valid > 0))))
        fr = base[['gameId', 'gameDate', 'seasonYear', 'odds_total', 'target_total_pts', 'close_total']].copy()
        fr['total_pred'], fr['sigma_total'] = pred, sig
        bets = M.make_bets(fr, thr=0.05, markets=('total',))
        row['bets_5pct'] = {'n': int(len(bets)), 'hit_rate': float(bets['won'].mean() * 100) if len(bets) else None,
                            'roi_per_100': float(bets['profit'].mean()) if len(bets) else None, 'over_share': float((bets['side'] == 'over').mean() * 100) if len(bets) else None}
        out[tag] = row
    res['variants'] = out

    # ---------------- keep rule
    keep = {}
    for kind, name in (('a', 'isotonic'), ('b', 'linear in the opening total'), ('c', 'level-dependent shrinkage')):
        r = out[kind]
        cond_i = r['seasons_better'] >= 5
        cond_ii = r['paired_mae']['hi'] < 0
        r1 = out[f'{kind}_R1']['paired_mae']['diff'] < 0
        r2 = out.get(f'{kind}_R2', {}).get('paired_mae', {}).get('diff', None)
        cond_iii = bool(r1 and (r2 is None or r2 < 0))
        verdict = 'KEEP' if (cond_i and cond_ii and cond_iii) else ('era-confounded' if (cond_i and cond_ii) else 'not kept')
        keep[kind] = {'name': name, 'seasons_better': r['seasons_better'], 'pooled_ci_below_zero': bool(cond_ii), 'robustness_R1_negative': bool(r1),
                      'robustness_R2_negative': None if r2 is None else bool(r2 < 0), 'verdict': verdict,
                      'seasons_with_own_ci_excluding_zero': r['seasons_ci_excludes_zero_in_favor'], 'pooled_p_one_sided': r['paired_mae']['p_one_sided_better']}
    res['keep'] = keep
    # ---- EXPLORATORY additions (not registered): reference to the market, fitted parameters, the deployed OOS files
    line_ae = np.abs(act - line)
    both = np.isfinite(line_ae) & np.isfinite(base_ae)
    res['exploratory'] = {'fit_parameters': FIT_PARAMS,
                          'market_line_mae_same_games': float(line_ae[both].mean()), 'baseline_mae_same_games': float(base_ae[both].mean()),
                          'market_line_mae_above_242': float(line_ae[both & hi_m].mean()), 'baseline_mae_above_242': float(base_ae[both & hi_m].mean()),
                          'mae_c_same_games': float(np.abs(act - corrected['c'][0])[both].mean())}
    oos = pd.concat([pd.read_csv(p, dtype={'game_id': str}, usecols=['game_id', 'game_date', 'total_pred', 'odds_total', 'home_score', 'away_score']) for p in sorted((ROOT / 'reports').glob('oos_predictions_20??-??.csv'))])
    oos['bias'] = oos['home_score'] + oos['away_score'] - oos['total_pred']
    oos = oos.dropna(subset=['bias', 'odds_total']).reset_index(drop=True)
    dbo = DayBoot(oos['game_date'])
    hm = (oos['odds_total'] > HIGH).to_numpy()
    oos_season = pd.to_datetime(oos['game_date']).dt.year.where(pd.to_datetime(oos['game_date']).dt.month >= 9, pd.to_datetime(oos['game_date']).dt.year - 1)
    X = np.column_stack([np.ones(len(oos)), oos['odds_total'] - oos['odds_total'].mean()] + [(oos_season == y).to_numpy(float) for y in sorted(oos_season.unique())[1:]])
    bb, lo_, hi_, _ = dbo.ols(X, oos['bias'].to_numpy(float), np.ones(len(oos), bool))
    res['exploratory']['deployed_oos_2021_26'] = {'games': int(len(oos)), 'games_above_242': int(hm.sum()),
                                                 'bias_above_242': dict(zip(('mean', 'lo', 'hi'), dbo.ratio(oos['bias'], np.ones(len(oos)), hm))) if hm.sum() else None,
                                                 'level_slope_with_season_dummies': {'slope': float(bb[1]), 'lo': float(lo_[1]), 'hi': float(hi_[1])}}
    res['holm_input_p'] = float(min(1.0, 3 * min(k['pooled_p_one_sided'] for k in keep.values())))
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    print(json.dumps({'overall': res['overall_bias'], 'above_242': {k: {kk: vv for kk, vv in v.items() if kk != 'by_season'} for k, v in hi_tab.items()},
                      'level_reg': reg, 'keep': keep, 'holm_p': res['holm_input_p']}, indent=1, default=float))


if __name__ == '__main__':
    main()
