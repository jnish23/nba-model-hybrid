"""E27 - totals level calibration (E18 fix c) on the deployed form. Registered in reports/experiments/E27_candidate_form/REGISTRATION.md
before any outcome was computed. Re-scores saved walk-forward predictions; the deployed models are not retrained (the only fits are fix (c)'s
two coefficients per fold and, for the sigma question, E4's small per-game sigma model refitted on calibrated out-of-fold errors).

Primary run: reports/experiments/E13_combined_winners/predictions.csv (E1 spread, E3 ML stack, E4 per-game total sigma, baseline total forecast);
calibration blocks: reports/experiments/baseline/calibration_set_predictions.csv. Replication: reports/experiments/E25_with_odds (current harness).

The saved predictions are gitignored, so they are read from the production checkout (read only): set E27_SRC to override.
python -m experiments.e27_candidate_form   (run from the worktree root)
"""
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import metrics as M                                                    # noqa: E402
from experiments.e14_blowout_under import DayBoot                                       # noqa: E402
from experiments.e18_calibration import make_fix, trailing_level                        # noqa: E402
from experiments.e04_sigma import SIGMA_CFG, SIGMA_FEATURES                              # noqa: E402
from experiments.lib import GRID, SEED, TEST_SEASONS, label_calib_folds, load_games, make_est, split_fold   # noqa: E402
from team import betting as bt                                                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get('E27_SRC', 'C:/Users/jorda/nba-model-hybrid/reports/experiments'))
OUT = ROOT / 'reports' / 'experiments' / 'E27_candidate_form'
SEEDS = {'brief_42': 42, 'e18_20261017': 20261017}
BUCKETS = [(-np.inf, 215), (215, 225), (225, 235), (235, 242), (242, np.inf)]
BUCKET_NAMES = ['<215', '215-225', '225-235', '235-242', '>242']


# ------------------------------------------------------------------------------------------------ loading

def read_run(name, calib_name=None):
    t = pd.read_csv(SRC / name / 'predictions.csv', dtype={'gameId': str}, parse_dates=['gameDate'])
    c = pd.read_csv(SRC / (calib_name or name) / 'calibration_set_predictions.csv', dtype={'gameId': str}, parse_dates=['gameDate'])
    return t, c


# ------------------------------------------------------------------------------------------------ fix (c), fold by fold

def fit_fold_fixes(calib, kind='c', level='odds_total', recent=False):
    """{season: f(frame)} fitted on each fold's calibration block only (recent: the later half by date)."""
    out = {}
    for s in TEST_SEASONS:
        cal = calib[calib['fold'] == s].sort_values('gameDate')
        fit_on = cal.iloc[len(cal) // 2:] if recent else cal
        out[s] = make_fix(kind, fit_on, level)
    return out


def apply_fixes(frame, fixes, calib, level='odds_total'):
    """(corrected test forecasts, calibrated global sigma per game, per-fold params)."""
    pred = frame['total_pred'].to_numpy(float).copy()
    sig = np.full(len(frame), np.nan)
    S = frame['seasonYear'].to_numpy()
    for s, fx in fixes.items():
        m = S == s
        pred[m] = fx(frame[m])
        cal = calib[calib['fold'] == s]
        sig[m] = float(np.nanstd(cal['target_total_pts'] - fx(cal)))
    return pred, sig


# ------------------------------------------------------------------------------------------------ scoring helpers

class Scorer:
    def __init__(self, frame, seed):
        self.f = frame.reset_index(drop=True)
        self.db = DayBoot(self.f['gameDate'], seed=seed)
        self.S = self.f['seasonYear'].to_numpy()
        self.act = self.f['target_total_pts'].to_numpy(float)
        self.line = self.f['odds_total'].to_numpy(float)
        self.ones = np.ones(len(self.f))

    def level(self, pred):
        """MAE / RMSE / bias (actual - forecast), pooled and per season, with day-bootstrap intervals for MAE and bias."""
        e = self.act - pred
        ok = np.isfinite(e)
        res = {}
        for lab, m in [('pooled', ok)] + [(s, ok & (self.S == s)) for s in TEST_SEASONS]:
            mae = self.db.ratio(np.abs(np.where(ok, e, 0)), self.ones, m)
            bias = self.db.ratio(np.where(ok, e, 0), self.ones, m)
            res[lab] = {'n': int(m.sum()), 'mae': mae[0], 'mae_lo': mae[1], 'mae_hi': mae[2],
                        'rmse': float(np.sqrt(np.mean(e[m] ** 2))), 'bias': bias[0], 'bias_lo': bias[1], 'bias_hi': bias[2]}
        return res

    def paired(self, pa, pb, power=1):
        """mean(|err_a|^p - |err_b|^p) pooled and per season (negative: a better), day-bootstrap intervals."""
        ea, eb = np.abs(self.act - pa) ** power, np.abs(self.act - pb) ** power
        valid = np.isfinite(ea) & np.isfinite(eb)
        d = np.where(valid, ea - eb, 0.0)
        v = valid.astype(float)
        pt, lo, hi = self.db.ratio(d, v, valid)
        boots = (self.db.W @ self.db.day_sum(d)) / (self.db.W @ self.db.day_sum(v))
        per = {}
        for s in TEST_SEASONS:
            p_, l_, h_ = self.db.ratio(d, v, valid & (self.S == s))
            per[s] = {'diff': p_, 'lo': l_, 'hi': h_}
        return {'diff': pt, 'lo': lo, 'hi': hi, 'p_one_sided_a_better': float((1 + (boots >= 0).sum()) / (len(boots) + 1)), 'by_season': per,
                'seasons_a_better': int(sum(x['diff'] < 0 for x in per.values())), 'seasons_ci_excludes_zero_for_a': int(sum(x['hi'] < 0 for x in per.values()))}

    def cover_ll(self, pred, sig):
        nonpush = self.act != self.line
        over = (self.act > self.line).astype(float)
        with np.errstate(invalid='ignore'):
            p = np.clip(bt.cover_probabilities(pred - self.line, sig, lines=self.line), 1e-6, 1 - 1e-6)
        ll = -(over * np.log(p) + (1 - over) * np.log(1 - p))
        return np.where(nonpush & np.isfinite(ll), ll, np.nan)

    def paired_ll(self, la, lb):
        valid = np.isfinite(la) & np.isfinite(lb)
        d = np.where(valid, la - lb, 0.0)
        pt, lo, hi = self.db.ratio(d, valid.astype(float), valid)
        return {'diff': pt, 'lo': lo, 'hi': hi}

    def slope(self, pred):
        """OLS (intercept) of (actual - open) on (forecast - open), day-bootstrap interval."""
        x, y = pred - self.line, self.act - self.line
        ok = np.isfinite(x) & np.isfinite(y)
        X = np.column_stack([np.ones(len(x)), np.nan_to_num(x)])
        b, lo, hi, _ = self.db.ols(X, np.nan_to_num(y), ok)
        return {'slope': float(b[1]), 'lo': float(lo[1]), 'hi': float(hi[1]), 'intercept': float(b[0]), 'n': int(ok.sum())}

    def bucket_bias(self, pred):
        e = self.act - pred
        out = {}
        for name, (lo_, hi_) in zip(BUCKET_NAMES, BUCKETS):
            m = (self.line >= lo_) & (self.line < hi_) & np.isfinite(e)
            pt, lo, hi = self.db.ratio(np.where(m, e, 0), self.ones, m)
            out[name] = {'n': int(m.sum()), 'bias': pt, 'lo': lo, 'hi': hi, 'mae': float(np.abs(e[m]).mean()) if m.sum() else None}
        return out

    def bets(self, pred, sig):
        fr = self.f[['gameId', 'gameDate', 'seasonYear', 'odds_total', 'target_total_pts', 'close_total']].copy()
        fr['total_pred'], fr['sigma_total'] = pred, sig
        b = M.make_bets(fr, thr=0.05, markets=('total',))
        if not len(b):
            return {'n': 0}
        idx = pd.Series(np.arange(len(fr)), index=fr['gameId'].to_numpy())
        pos = idx.loc[b['gameId']].to_numpy()
        ind, prof, won, over = (np.zeros(len(fr)) for _ in range(4))
        ind[pos] = 1
        prof[pos] = b['profit'].to_numpy()
        won[pos] = b['won'].to_numpy(float)
        over[pos] = (b['side'] == 'over').to_numpy(float)
        roi = self.db.ratio(prof, ind, ind > 0)
        hit = self.db.ratio(won, ind, ind > 0)
        return {'n': int(len(b)), 'over': int(over.sum()), 'under': int(len(b) - over.sum()), 'hit_rate': hit[0] * 100, 'hit_lo': hit[1] * 100, 'hit_hi': hit[2] * 100,
                'roi_per_100': roi[0], 'roi_lo': roi[1], 'roi_hi': roi[2], 'by_season': {s: {'n': int(ind[self.S == s].sum()), 'hit_rate': float(won[self.S == s].sum() / max(ind[self.S == s].sum(), 1) * 100),
                                                                                         'roi_per_100': float(prof[self.S == s].sum() / max(ind[self.S == s].sum(), 1))} for s in TEST_SEASONS}}


# ------------------------------------------------------------------------------------------------ per-game sigma (E4) on calibrated errors

def hetero_sigma_refit(df, feats, info, fixes, base_ids):
    """E4's per-game total sigma re-fitted per fold, once on the raw out-of-fold |errors| (reproduction check against the saved sigma.csv)
    and once on the |errors| of fix (c)'s forecast (fix fitted on the fold's calibration block, applied to the out-of-fold predictions inside model_train).
    Returns two Series indexed by gameId (test games)."""
    sf = [c for c in SIGMA_FEATURES if c in df.columns]
    raw_out, cal_out = {}, {}
    for s in TEST_SEASONS:
        mt, calib, test, n_fl = split_fold(df, s, calib_cap=None)      # the baseline run's split (no split.json: pre-audit scheme)
        f_ = feats[s]['total']
        cfg = info[s]['total']['cfg']
        parts = np.array_split(np.arange(len(mt)), 5)
        idx, pred = [], []
        for k in range(1, 5):
            tr, te = np.concatenate(parts[:k]), parts[k]
            est = make_est('reg', GRID[cfg])
            est.fit(mt.iloc[tr][f_], mt.iloc[tr]['target_total_pts'])
            idx.append(te)
            pred.append(est.predict(mt.iloc[te][f_]))
        idx, pred = np.concatenate(idx), np.concatenate(pred)
        sub = mt.iloc[idx]
        y = sub['target_total_pts'].to_numpy(float)
        fr = pd.DataFrame({'odds_total': sub['odds_total'].to_numpy(float), 'total_pred': pred})
        for tag, p_ in (('raw', pred), ('cal', fixes[s](fr))):
            g = lgb.LGBMRegressor(**SIGMA_CFG, random_state=SEED, deterministic=True, force_row_wise=True, verbosity=-1)
            g.fit(sub[sf], np.abs(y - p_))
            sig = np.clip(g.predict(test[sf]), 1.0, None) * np.sqrt(np.pi / 2)
            (raw_out if tag == 'raw' else cal_out)[s] = pd.Series(sig, index=test['gameId'].astype(str).str.zfill(10).to_numpy())
    cat = lambda d: pd.concat(d.values())
    return cat(raw_out), cat(cal_out)


# ------------------------------------------------------------------------------------------------ main

def run_variants(frame, calib, games):
    """pred / global sigma for raw (as saved), fix c, c_R1, c_R2 on `frame`."""
    frame = frame.copy()
    frame['lvl_era'] = frame['odds_total'] - trailing_level(games, frame['gameDate'])
    cal = calib.copy()
    cal['lvl_era'] = cal['odds_total'] - trailing_level(games, cal['gameDate'])
    out = {}
    for tag, kw in (('c', {}), ('c_R1', {'recent': True}), ('c_R2', {'level': 'lvl_era'})):
        fixes = fit_fold_fixes(cal, 'c', **{k: v for k, v in kw.items()})
        lvl = kw.get('level', 'odds_total')
        pred, sig = apply_fixes(frame, fixes, cal, lvl)
        out[tag] = {'pred': pred, 'sig': sig, 'fixes': fixes, 'params': {s: getattr(fx, 'params', None) for s, fx in fixes.items()}}
    return frame, out


def keep_rule(sc, raw_pred, v):
    c_vs_raw = sc.paired(v['c']['pred'], raw_pred)
    r1 = sc.paired(v['c_R1']['pred'], raw_pred)
    r2 = sc.paired(v['c_R2']['pred'], raw_pred)
    i = c_vs_raw['seasons_a_better'] >= 5
    ii = c_vs_raw['hi'] < 0
    iii = bool(r1['diff'] < 0 and r2['diff'] < 0)
    verdict = 'KEEP' if (i and ii and iii) else ('era-confounded' if (i and ii) else 'not kept')
    return {'seasons_c_better_than_raw': c_vs_raw['seasons_a_better'], 'pooled_ci_below_zero': bool(ii), 'pooled_diff': c_vs_raw['diff'], 'pooled_ci': [c_vs_raw['lo'], c_vs_raw['hi']],
            'R1_diff': r1['diff'], 'R2_diff': r2['diff'], 'R1_R2_negative': iii, 'verdict': verdict,
            'seasons_with_own_ci_excluding_zero': c_vs_raw['seasons_ci_excludes_zero_for_a']}, c_vs_raw


def write_report(res):
    """RESULT.md from the results.json numbers (seed 42 is the brief's; the E18-seed keep-rule verdict is listed beside it)."""
    s = res['brief_42']
    L = ['# E27 - totals level calibration (E18 fix c) on the deployed form: result', '',
         f"Registration: `REGISTRATION.md` (sha256 `{res['decision']['registration_sha256']}`), written before any outcome. Run: `python -m experiments.e27_candidate_form`.",
         f"**Verdict: {res['decision']['verdict_seed42']}** (seed 42, the brief's seed) / {res['decision']['verdict_seed20261017']} (E18's seed 20261017). ML and spread unchanged: yes (columns carried through bit for bit; fix (c) only touches the total).", '',
         '## Data', '',
         f"- Primary run: `E13_combined_winners/predictions.csv`, {res['n_games_e13']} games, 7 seasons (spread = E1, ML = E3 stack, per-game total sigma = E4, total forecast = baseline total; equals the baseline run's `total_pred` on every game: {res['e13_total_pred_equals_baseline_total_pred']}).",
         '- Calibration blocks: `baseline/calibration_set_predictions.csv` (pre-audit uncapped last-20% blocks, 16,218 rows), labelled per fold.',
         '- Reproduction of E18 on the baseline run (8,916 games): ' + '; '.join(f"{k} MAE {v['mine']:.4f} vs E18 {v['e18']:.4f}" for k, v in res['reproduction_of_E18_on_baseline_run'].items() if k in ('baseline', 'c', 'c_R1', 'c_R2'))
         + '. Fix (c), R1 and the baseline reproduce E18 to the last digit; R2 does not (it uses the trailing-365-day mean line of ALL games, taken here from the rebuilt 2026-11-01 training set rather than the lost original), but its sign agrees (E18 saved R2 14.620 vs 14.772 baseline).', '',
         '- Registration erratum: `REGISTRATION.md` says `E25_with_odds` was made under the current harness. File evidence says otherwise: its calibration file has 16,218 rows (the uncapped last-20% layout), no fold column and no split.json, '
         'so it is a second walk-forward run of the same scheme as the baseline (different training-set rebuild and feature selection; `total_pred` differs from the baseline run by up to 24 points on single games). It is a replication, not a test of the deployed one-year-block scheme.',
         '- Limit: the deployed trainer fits on a one-year (about 1,300-game) calibration block and refits the base models on all data; no saved walk-forward run used that scheme together with the E1/E3/E4 pieces, so fix (c) was not scored under it.', '',
         '## Keep rule (E18 rule applied to fix (c), against the raw total forecast)', '']
    for sn, lab in (('brief_42', 'seed 42'), ('e18_20261017', 'seed 20261017')):
        k = res[sn]['keep_rule']
        L.append(f"- {lab}: seasons better {k['seasons_c_better_than_raw']}/7 (rule: at least 5); pooled paired MAE diff {k['pooled_diff']:+.3f} [{k['pooled_ci'][0]:+.3f}, {k['pooled_ci'][1]:+.3f}] (rule: CI below 0: {k['pooled_ci_below_zero']}); "
                 f"R1 {k['R1_diff']:+.3f}, R2 {k['R2_diff']:+.3f} (rule: both negative: {k['R1_R2_negative']}); seasons whose own CI excludes 0: {k['seasons_with_own_ci_excluding_zero']}/7 -> {k['verdict']}")
    rp = res['replication_E25_with_odds']['brief_42']
    L += [f"- Replication on `E25_with_odds` (independent re-run of the baseline total on the 2026-10-19 training set, global sigma only): {rp['keep_rule']['seasons_c_better_than_raw']}/7, "
          f"{rp['keep_rule']['pooled_diff']:+.3f} [{rp['keep_rule']['pooled_ci'][0]:+.3f}, {rp['keep_rule']['pooled_ci'][1]:+.3f}], R1 {rp['keep_rule']['R1_diff']:+.3f}, R2 {rp['keep_rule']['R2_diff']:+.3f} -> {rp['keep_rule']['verdict']}", '',
          '## Total MAE / RMSE / bias (actual - forecast), 8,899 games', '', '| | MAE pooled [95% CI] | RMSE | bias [95% CI] | ' + ' | '.join(TEST_SEASONS) + ' |', '|---|---|---|---|' + '---|' * 7]
    for k, nm in (('raw', 'raw model'), ('fix_c', 'fix (c)'), ('open_line', 'opening line')):
        v = s['levels'][k]
        p_ = v['pooled']
        L.append(f"| {nm} | {p_['mae']:.3f} [{p_['mae_lo']:.3f}, {p_['mae_hi']:.3f}] | {p_['rmse']:.3f} | {p_['bias']:+.3f} [{p_['bias_lo']:+.2f}, {p_['bias_hi']:+.2f}] | "
                 + ' | '.join(f"{v[x]['mae']:.2f} / {v[x]['bias']:+.2f}" for x in TEST_SEASONS) + ' |')
    L += ['', 'Per-season cells are MAE / bias. Paired differences (negative: first is better):', '',
          '| comparison | MAE diff [95% CI] | seasons first better | MSE diff [95% CI] | ' + ' | '.join(TEST_SEASONS) + ' (MAE diff) |', '|---|---|---|---|' + '---|' * 7]
    for k, nm in (('c_minus_raw', 'fix (c) - raw'), ('c_minus_line', 'fix (c) - line'), ('raw_minus_line', 'raw - line')):
        a_, b_ = s['paired_mae'][k], s['paired_mse'][k]
        L.append(f"| {nm} | {a_['diff']:+.3f} [{a_['lo']:+.3f}, {a_['hi']:+.3f}] | {a_['seasons_a_better']}/7 | {b_['diff']:+.2f} [{b_['lo']:+.2f}, {b_['hi']:+.2f}] | " + ' | '.join(f"{a_['by_season'][x]['diff']:+.2f}" for x in TEST_SEASONS) + ' |')
    L += ['', '## Cover log loss (p(over) = Phi((forecast - line)/sigma), pushes dropped; lower is better)', '', '| forecast + sigma | log loss |', '|---|---|']
    for k, v in s['cover_logloss'].items():
        L.append(f'| {k} | {v:.5f} |')
    L += ['', '| paired difference (first minus second) | diff [95% CI] | seasons first better |', '|---|---|---|']
    for k, v in s['cover_logloss_paired'].items():
        L.append(f"| {k} | {v['diff']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}] | {v['seasons_a_better']}/7 |")
    sr = res['sigma_refit_reproduction']
    L += ['', f"Sigma caveat: `c_E4_sigma_refit_calibrated` refits E4's per-game sigma model on forward-chained out-of-fold errors of the calibrated forecast, using the rebuilt 2026-11-01 training set; "
          f"the same refit on RAW errors correlates only {sr['corr_with_saved_E4']:.2f} with the saved E4 sigma (mean {sr['mean_refit_raw']:.2f} vs {sr['mean_saved']:.2f}), so it is a different sigma model from the one that was saved, not a reproduction.",
          '', '## 5%-rule total bets priced at the open, -110 (descriptive)', '', '| forecast | bets | over / under | hit rate [95% CI] | ROI per 100 [95% CI] |', '|---|---|---|---|---|']
    for k, nm in (('raw_E4_sigma', 'raw, E4 per-game sigma'), ('c', f"fix (c), {s['sigma_used_for_c_bets']}")):
        b = s['bets_5pct'][k]
        L.append(f"| {nm} | {b['n']} | {b['over']} / {b['under']} | {b['hit_rate']:.1f}% [{b['hit_lo']:.1f}, {b['hit_hi']:.1f}] | {b['roi_per_100']:+.2f} [{b['roi_lo']:+.1f}, {b['roi_hi']:+.1f}] |")
    e = s['edge_slope']
    L += ['', '## Edge slope: (actual - open) on (forecast - open), intercept included', '',
          f"- raw: {e['raw']['slope']:+.3f} [{e['raw']['lo']:+.3f}, {e['raw']['hi']:+.3f}]", f"- fix (c): {e['fix_c']['slope']:+.3f} [{e['fix_c']['lo']:+.3f}, {e['fix_c']['hi']:+.3f}]", '',
          '## Bias (actual - forecast) by opening-total bucket', '', '| bucket | games | raw bias [95% CI] | fix (c) bias [95% CI] | line bias [95% CI] |', '|---|---|---|---|---|']
    for b in BUCKET_NAMES:
        r_, c_, l_ = (s['bucket_bias'][k][b] for k in ('raw', 'fix_c', 'open_line'))
        L.append(f"| {b} | {r_['n']} | {r_['bias']:+.2f} [{r_['lo']:+.1f}, {r_['hi']:+.1f}] | {c_['bias']:+.2f} [{c_['lo']:+.1f}, {c_['hi']:+.1f}] | {l_['bias']:+.2f} [{l_['lo']:+.1f}, {l_['hi']:+.1f}] |")
    L += ['', '## Fitted fix (c) parameters per fold (weight w = w0 + w1 (line - mean_level), clipped to [0, 1])', '', '| fold | w0 | w1 | mean_level |', '|---|---|---|---|']
    for k, v in res['fit_params_c']['c'].items():
        L.append(f"| {k} | {v['w0']:.3f} | {v['w1']:+.4f} | {v['mean_level']:.1f} |")
    sn = res['sensitivity_last_1300_calibration_rows']
    L += ['', '## Sensitivity (not registered; no decision): fix (c) refitted on only the last 1,300 rows of each fold\'s calibration block', '',
          'The deployed trainer fits on a one-year block (about 1,300 games); the saved runs used 1,500-3,000. This approximates the deployed block size.', '',
          f"- pooled MAE {sn['mae_pooled']:.3f}, bias {sn['bias_pooled']:+.3f}; fix (c) - raw {sn['c_minus_raw']['diff']:+.3f} [{sn['c_minus_raw']['lo']:+.3f}, {sn['c_minus_raw']['hi']:+.3f}], {sn['c_minus_raw']['seasons_a_better']}/7 seasons better; "
          f"fix (c) - line {sn['c_minus_line']['diff']:+.3f} [{sn['c_minus_line']['lo']:+.3f}, {sn['c_minus_line']['hi']:+.3f}]",
          '- fitted weights (w0, w1 per total point, mean level): ' + '; '.join(f"{k}: {v['w0']:.2f}, {v['w1']:+.4f}, {v['mean_level']:.0f}" for k, v in sn['params_by_fold'].items()),
          '- bias by bucket (actual - forecast): ' + ', '.join(f"{k} {v['bias']:+.2f}" for k, v in sn['bucket_bias'].items())]
    (OUT / 'RESULT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m, games = load_games()
    res = {'src': str(SRC)}

    # ---- reproduction check on the baseline run (E18's own frame)
    base, calib = read_run('baseline')
    calib['fold'] = label_calib_folds(games, calib)
    fb, vb = run_variants(base, calib, games)
    e18 = json.loads((ROOT / 'reports' / 'experiments' / 'E18_calibration' / 'results.json').read_text())['variants']
    act, line = fb['target_total_pts'].to_numpy(float), fb['odds_total'].to_numpy(float)
    chk = {}
    for tag in ('c', 'c_R1', 'c_R2'):
        mine = float(np.nanmean(np.abs(act - vb[tag]['pred'])))
        chk[tag] = {'mine': mine, 'e18': e18[tag]['mae'], 'abs_diff': abs(mine - e18[tag]['mae'])}
    chk['baseline'] = {'mine': float(np.nanmean(np.abs(act - fb['total_pred']))), 'e18': e18['baseline']['mae']}
    chk['baseline']['abs_diff'] = abs(chk['baseline']['mine'] - chk['baseline']['e18'])
    chk['c_by_season'] = {s: {'mine': float(np.nanmean(np.abs(act - vb['c']['pred'])[(fb['seasonYear'] == s).to_numpy()])), 'e18': e18['c']['mae_by_season'][s]} for s in TEST_SEASONS}
    res['reproduction_of_E18_on_baseline_run'] = chk
    print('reproduction', json.dumps(chk, indent=1))

    # ---- primary: E13 (the deployed form)
    e13, _ = read_run('E13_combined_winners', calib_name='baseline')
    ids_common = e13['gameId'].isin(base['gameId'])
    bi = base.set_index('gameId')
    same_total = np.allclose(e13['total_pred'].to_numpy(float), bi.loc[e13['gameId'], 'total_pred'].to_numpy(float), equal_nan=True)
    res['e13_total_pred_equals_baseline_total_pred'] = bool(same_total)
    res['n_games_e13'] = int(len(e13))
    e13 = e13.reset_index(drop=True)
    # E13's spread/ML columns are carried through untouched below; keep a hash of them to show fix (c) never changes them
    ml0, sp0 = e13['ml_prob'].to_numpy(float).copy(), e13['spread_pred'].to_numpy(float).copy()
    fe, ve = run_variants(e13, calib, games)

    # per-game sigma on calibrated residuals (E4 procedure refitted)
    feats = json.loads((SRC / 'baseline' / 'features.json').read_text())
    info = json.loads((SRC / 'baseline' / 'fold_info.json').read_text())
    sig_raw_refit, sig_cal_refit = hetero_sigma_refit(games, feats, info, ve['c']['fixes'], None)
    saved = pd.read_csv(SRC / 'E04_sigma' / 'sigma.csv', dtype={'gameId': str}).set_index('gameId')['sigma_total_h']
    ids = fe['gameId']
    sh_saved = fe['sigma_total_h'].to_numpy(float)
    sh_raw_refit = sig_raw_refit.reindex(ids).to_numpy(float)
    sh_cal_refit = sig_cal_refit.reindex(ids).to_numpy(float)
    both = np.isfinite(sh_raw_refit) & np.isfinite(sh_saved)
    res['sigma_refit_reproduction'] = {'corr_with_saved_E4': float(np.corrcoef(sh_raw_refit[both], sh_saved[both])[0, 1]), 'mean_saved': float(np.nanmean(sh_saved)),
                                       'mean_refit_raw': float(np.nanmean(sh_raw_refit)), 'mean_refit_calibrated': float(np.nanmean(sh_cal_refit)),
                                       'max_abs_diff': float(np.nanmax(np.abs(sh_raw_refit - sh_saved))), 'n_missing_refit': int(np.isnan(sh_raw_refit).sum())}
    print('sigma repro', res['sigma_refit_reproduction'])

    for seed_name, seed in SEEDS.items():
        sc = Scorer(fe, seed)
        raw = fe['total_pred'].to_numpy(float)
        r = {}
        kr, c_vs_raw = keep_rule(sc, raw, ve)
        r['keep_rule'] = kr
        r['levels'] = {'raw': sc.level(raw), 'fix_c': sc.level(ve['c']['pred']), 'open_line': sc.level(sc.line)}
        r['paired_mae'] = {'c_minus_raw': c_vs_raw, 'c_minus_line': sc.paired(ve['c']['pred'], sc.line), 'raw_minus_line': sc.paired(raw, sc.line)}
        r['paired_mse'] = {'c_minus_raw': sc.paired(ve['c']['pred'], raw, 2), 'c_minus_line': sc.paired(ve['c']['pred'], sc.line, 2), 'raw_minus_line': sc.paired(raw, sc.line, 2)}
        r['robustness'] = {k: {'pooled_diff_vs_raw': sc.paired(ve[k]['pred'], raw)['diff'], 'ci': [sc.paired(ve[k]['pred'], raw)['lo'], sc.paired(ve[k]['pred'], raw)['hi']],
                               'seasons_better': sc.paired(ve[k]['pred'], raw)['seasons_a_better']} for k in ('c_R1', 'c_R2')}
        # cover log loss
        sig_raw_glob = fe['sigma_total'].to_numpy(float)
        ll = {'raw_E4_sigma': sc.cover_ll(raw, sh_saved), 'raw_global_sigma': sc.cover_ll(raw, sig_raw_glob),
              'c_global_calibrated_sigma': sc.cover_ll(ve['c']['pred'], ve['c']['sig']),
              'c_E4_sigma_as_saved': sc.cover_ll(ve['c']['pred'], sh_saved),
              'c_E4_sigma_refit_calibrated': sc.cover_ll(ve['c']['pred'], sh_cal_refit)}
        r['cover_logloss'] = {k: float(np.nanmean(v)) for k, v in ll.items()}
        r['cover_logloss_by_season'] = {k: {s: float(np.nanmean(v[sc.S == s])) for s in TEST_SEASONS} for k, v in ll.items()}
        pairs = {'c_global_minus_raw_E4': ('c_global_calibrated_sigma', 'raw_E4_sigma'), 'c_global_minus_raw_global': ('c_global_calibrated_sigma', 'raw_global_sigma'),
                 'raw_E4_minus_raw_global': ('raw_E4_sigma', 'raw_global_sigma'),
                 'c_E4refit_minus_c_global': ('c_E4_sigma_refit_calibrated', 'c_global_calibrated_sigma'), 'c_E4saved_minus_c_global': ('c_E4_sigma_as_saved', 'c_global_calibrated_sigma'),
                 'c_E4refit_minus_raw_E4': ('c_E4_sigma_refit_calibrated', 'raw_E4_sigma')}
        r['cover_logloss_paired'] = {k: {**sc.paired_ll(ll[a], ll[b]), 'seasons_a_better': int(sum(np.nanmean(ll[a][sc.S == s]) < np.nanmean(ll[b][sc.S == s]) for s in TEST_SEASONS))} for k, (a, b) in pairs.items()}
        # bets, edge slope, bucket bias
        better_sigma = sh_cal_refit if r['cover_logloss_paired']['c_E4refit_minus_c_global']['hi'] < 0 else ve['c']['sig']
        r['sigma_used_for_c_bets'] = 'E4 per-game sigma refit on calibrated errors' if better_sigma is sh_cal_refit else 'calibrated global sigma'
        r['bets_5pct'] = {'raw_E4_sigma': sc.bets(raw, sh_saved), 'c': sc.bets(ve['c']['pred'], better_sigma), 'c_global_sigma': sc.bets(ve['c']['pred'], ve['c']['sig'])}
        r['edge_slope'] = {'raw': sc.slope(raw), 'fix_c': sc.slope(ve['c']['pred'])}
        r['bucket_bias'] = {'raw': sc.bucket_bias(raw), 'fix_c': sc.bucket_bias(ve['c']['pred']), 'open_line': sc.bucket_bias(sc.line)}
        res[seed_name] = r
    res['fit_params_c'] = {k: ve[k]['params'] for k in ('c', 'c_R1', 'c_R2')}

    # ML / spread untouched
    assert np.array_equal(ml0, fe['ml_prob'].to_numpy(float), equal_nan=True) and np.array_equal(sp0, fe['spread_pred'].to_numpy(float), equal_nan=True)
    met = json.loads((SRC / 'E13_combined_winners' / 'metrics.json').read_text())
    act_w = fe['target_home_win'].to_numpy(float)
    p = np.clip(fe['ml_prob'].to_numpy(float), 1e-6, 1 - 1e-6)
    ok = np.isfinite(p)
    res['ml_spread_unchanged'] = {'ml_prob_and_spread_pred_columns_identical_to_E13_file': True, 'ml_logloss_from_file': float(-(act_w * np.log(p) + (1 - act_w) * np.log(1 - p))[ok].mean()),
                                  'spread_mae_from_file': float(np.nanmean(np.abs(fe['target_point_diff'] - fe['spread_pred']))), 'e13_metrics_keys': list(met)[:6]}

    # ---- replication on E25_with_odds (current harness; global sigma only)
    e25, c25 = read_run('E25_with_odds')
    c25['fold'] = label_calib_folds(games, c25)        # the file has no fold column; its 16,218 rows fit the uncapped last-20% layout
    f25, v25 = run_variants(e25, c25, games)
    rep = {}
    for seed_name, seed in SEEDS.items():
        sc = Scorer(f25, seed)
        raw = f25['total_pred'].to_numpy(float)
        kr, c_vs = keep_rule(sc, raw, v25)
        rep[seed_name] = {'n': int(len(f25)), 'keep_rule': kr, 'levels': {'raw': sc.level(raw), 'fix_c': sc.level(v25['c']['pred']), 'open_line': sc.level(sc.line)},
                          'paired_mae': {'c_minus_raw': c_vs, 'c_minus_line': sc.paired(v25['c']['pred'], sc.line), 'raw_minus_line': sc.paired(raw, sc.line)}}
    rep['fit_params_c'] = v25['c']['params']
    res['replication_E25_with_odds'] = rep

    # ---- sensitivity (NOT registered, no decision): the deployed trainer fits on a one-year block (about 1,300 games), the saved runs on 1,500-3,000.
    #      Refit fix (c) on only the LAST 1,300 rows of each fold's calibration block (the cap scheme's block, approximately: the base model that made those
    #      predictions was trained on a somewhat shorter history than the capped scheme's would be).
    cal_cap = calib.sort_values(['fold', 'gameDate']).groupby('fold', group_keys=False).tail(1300)
    sens = {}
    for tag, kw in (('c_last1300', {}), ('c_last1300_R1', {'recent': True})):
        fixes = fit_fold_fixes(cal_cap, 'c', **kw)
        sens[tag] = apply_fixes(fe, fixes, cal_cap)[0]
        sens[tag + '_params'] = {s: fx.params for s, fx in fixes.items()}
    sc = Scorer(fe, 42)
    raw = fe['total_pred'].to_numpy(float)
    p_ = sc.paired(sens['c_last1300'], raw)
    pl = sc.paired(sens['c_last1300'], sc.line)
    res['sensitivity_last_1300_calibration_rows'] = {
        'params_by_fold': sens['c_last1300_params'], 'mae_pooled': sc.level(sens['c_last1300'])['pooled']['mae'], 'bias_pooled': sc.level(sens['c_last1300'])['pooled']['bias'],
        'c_minus_raw': {k: p_[k] for k in ('diff', 'lo', 'hi', 'seasons_a_better')}, 'c_minus_line': {k: pl[k] for k in ('diff', 'lo', 'hi', 'seasons_a_better')},
        'bucket_bias': sc.bucket_bias(sens['c_last1300'])}

    # decision
    dec = {s: res[s]['keep_rule'] for s in SEEDS}
    verdict = res['brief_42']['keep_rule']['verdict']
    res['decision'] = {'experiment': 'E27', 'verdict_seed42': verdict, 'verdict_seed20261017': res['e18_20261017']['keep_rule']['verdict'],
                       'ml_spread_unchanged': True, 'passes': bool(verdict == 'KEEP'), 'keep_rule': dec,
                       'replication_E25_verdict_seed42': rep['brief_42']['keep_rule']['verdict']}
    reg = OUT / 'REGISTRATION.md'
    res['decision']['registration_sha256'] = __import__('hashlib').sha256(reg.read_bytes()).hexdigest() if reg.exists() else None
    (OUT / 'decision.json').write_text(json.dumps(res['decision'], indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    write_report(res)
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    print(json.dumps(res['decision'], indent=1, default=float))


if __name__ == '__main__':
    main()
