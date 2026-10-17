"""E13 - the winners rerun together (see PREREGISTRATION.md 'Winners combined'). Winners: E1 spread residual target, E2 Platt,
E3 logit-margin stack, E4 total-market heteroskedastic sigma, E12 best-price settlement (reported separately, not a model change).

How the combination is built (nothing is re-tuned):
  spread  : E1 spread_pred / sigma_spread
  total   : baseline total_pred, sigma_total_i from E4
  ML      : logistic stack on [logit(raw ML prob), E1 margin / E1 sigma] fitted on the fold's calibration set. This is E3 with E1's
            margin. E2 (Platt) is the same family restricted to the first regressor, so E3 contains E2 (nested); the two cannot
            be applied one after the other without calibrating twice, so the ML column is "E3 on top of E1", not "E1+E2+E3".
Additivity: a market is only touched by its own winners (spread by E1, total by E4), so those two are additive by construction
and the check just verifies the combined numbers equal the single-experiment numbers. ML is the only market with an interaction
(E1 -> margin input of E3); compare E1+E3 with E3 alone and with the sum of E3's gain and E1's own ML effect (zero, E1-spread
does not change the ML classifier).  python -m experiments.e13_combined
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import decide, metrics as M                                        # noqa: E402
from experiments.e04_sigma import cover_frame                                        # noqa: E402
from experiments.lib import OUT, TEST_SEASONS, label_calib_folds, load_games, load_run   # noqa: E402
from experiments.report import fmt_compare                                           # noqa: E402

NAME = 'E13_combined_winners'


def lg(p):
    return logit(np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4))


def main():
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    base, base_cal = load_run('baseline')
    e1, e1_cal = load_run('E01_residual_targets')
    e3 = pd.read_csv(OUT / 'E03_coherence' / 'predictions_stack_logit_margin.csv', dtype={'gameId': str}, parse_dates=['gameDate'])
    sig = pd.read_csv(OUT / 'E04_sigma' / 'sigma.csv', dtype={'gameId': str})
    _, df = load_games()
    base_cal['fold'] = label_calib_folds(df, base_cal)
    df1 = df.dropna(subset=['odds_home_spread', 'odds_total']).reset_index(drop=True)      # the frame E1 was run on
    e1_cal['fold'] = label_calib_folds(df1, e1_cal)
    # games common to the baseline and E1 (E1 requires both opening lines)
    common = base['gameId'].isin(e1['gameId'])
    b = base[common].reset_index(drop=True)
    e1i = e1.set_index('gameId')
    c = b.copy()
    c['spread_pred'] = e1i.loc[c['gameId'], 'spread_pred'].to_numpy()
    c['sigma_spread'] = e1i.loc[c['gameId'], 'sigma_spread'].to_numpy()
    for s in TEST_SEASONS:
        cal_b = base_cal[base_cal['fold'] == s].set_index('gameId')
        cal_e = e1_cal[e1_cal['fold'] == s].set_index('gameId')
        ids = cal_b.index.intersection(cal_e.index)
        Xc = np.column_stack([lg(cal_b.loc[ids, 'ml_prob_raw']), cal_e.loc[ids, 'spread_pred'] / cal_e.loc[ids, 'sigma_spread']])
        yc = cal_b.loc[ids, 'target_home_win']
        tm = (c['seasonYear'] == s).to_numpy()
        Xt = np.column_stack([lg(c.loc[tm, 'ml_prob_raw']), c.loc[tm, 'spread_pred'] / c.loc[tm, 'sigma_spread']])
        c.loc[tm, 'ml_prob'] = LogisticRegression(C=1.0).fit(Xc, yc).predict_proba(Xt)[:, 1]
    sg = sig.set_index('gameId')
    c['sigma_total_h'] = sg.loc[c['gameId'], 'sigma_total_h'].to_numpy()
    c.to_csv(OUT / NAME / 'predictions.csv', index=False)

    e3c = e3[e3['gameId'].isin(b['gameId'])].reset_index(drop=True)
    e1c = e1[e1['gameId'].isin(b['gameId'])].reset_index(drop=True)
    bl, cl = M.game_losses(b), M.game_losses(c)
    r = {k: M.paired(bl, cl, k) for k in ('ll', 'brier', 'ae_spread', 'se_spread', 'ae_total', 'se_total')}
    lines = [f'# {NAME}\n', f'Common games with the baseline: {len(b)} of {len(base)} (E1 needs both opening lines).\n',
             '## Combined vs baseline (paired, day-bootstrap)\n', fmt_compare(r).to_markdown(index=False), '']
    # single-experiment effects on the same games
    single = {'E3 stack (baseline margin)': M.paired(bl, M.game_losses(e3c), 'll'),
              'E1 spread-only (ML unchanged)': M.paired(bl, M.game_losses(e1c.assign(ml_prob=b['ml_prob'].values)), 'll'),
              'E1 spread target': M.paired(bl, M.game_losses(e1c), 'ae_spread'),
              'combined ML (E1 margin in the stack)': r['ll'], 'combined spread': r['ae_spread']}
    # ML additivity: combined vs E3 alone, per-game paired
    l3 = M.game_losses(e3c)
    d_ml = M.paired(l3, cl, 'll')
    d_sp = M.paired(M.game_losses(e1c), cl, 'ae_spread')
    ece = {k: M.ece(t['target_home_win'], t['ml_prob']) for k, t in (('baseline', b), ('E3 alone', e3c), ('combined', c))}
    contra = {k: float(((t['ml_prob'] > .5) != (t['spread_pred'] > 0)).mean()) for k, t in (('baseline', b), ('E3 alone', e3c), ('combined', c))}
    sum_expected = single['E3 stack (baseline margin)']['diff']
    lines += ['## Additivity\n',
              f"- ML log loss: E3 alone {single['E3 stack (baseline margin)']['diff']:+.5f}; E1-spread's own ML effect 0 (ML classifier untouched); expected additive "
              f"{sum_expected:+.5f}; combined {r['ll']['diff']:+.5f}; combined - E3 alone = {d_ml['diff']:+.5f} (95% CI {d_ml['ci_lo']:+.5f}..{d_ml['ci_hi']:+.5f})",
              f"- ML seasons improved vs baseline: combined {r['ll']['seasons_improved']}/7 (E3 alone {single['E3 stack (baseline margin)']['seasons_improved']}/7)",
              f"- ECE: {json.dumps({k: round(v, 4) for k, v in ece.items()})}; contradiction rate (ML vs spread): {json.dumps({k: round(v, 4) for k, v in contra.items()})}",
              f"- spread MAE: E1 alone {single['E1 spread target']['diff']:+.5f}, combined {r['ae_spread']['diff']:+.5f}; combined - E1 alone = {d_sp['diff']:+.5f} "
              f"(identical by construction: the spread model is untouched by the other winners)"]
    # total: E4 cover log-likelihood, base global sigma vs hetero sigma
    g0 = cover_frame(b, 'total', b['sigma_total'].to_numpy(float))
    g1 = cover_frame(b, 'total', c['sigma_total_h'].to_numpy(float))
    ok = g0['ok'] & g1['ok']

    def cll(g):
        return pd.DataFrame({'cl': -(g['y'] * np.log(g['p']) + (1 - g['y']) * np.log(1 - g['p']))}).assign(
            seasonYear=b['seasonYear'].values, gameDate=b['gameDate'].values).set_index(b['gameId'].values)[ok.values]
    rt = M.paired(cll(g0), cll(g1), 'cl')
    lines += [f"- total cover log loss (E4): combined {rt['diff']:+.5f} (95% CI {rt['ci_lo']:+.5f}..{rt['ci_hi']:+.5f}), seasons better {rt['seasons_improved']}/7; "
              f"ECE {M.ece(g0['y'][ok], g0['p'][ok]):.4f} -> {M.ece(g1['y'][ok], g1['p'][ok]):.4f}; E4 alone is the same number (the point forecast is unchanged)", '']
    # bets under the unified 5% rule: baseline vs combined
    bb = M.make_bets(b.dropna(subset=['spread_pred', 'total_pred']))
    cb = M.make_bets(c.dropna(subset=['spread_pred', 'total_pred']), sigma_total=c['sigma_total_h'].to_numpy(float)[c[['spread_pred', 'total_pred']].notna().all(axis=1).to_numpy()])
    rows = []
    for mk in ('ml', 'spread', 'total'):
        for tag, bets in (('baseline', bb), ('combined', cb)):
            x = bets[bets['market'] == mk]
            m_ = M.bet_metrics(x)
            rows.append({'market': mk, 'run': tag, 'bets': m_.get('bets'), 'hit %': round(m_.get('hit_rate', np.nan), 2), 'ROI %': round(m_.get('roi', np.nan), 2),
                         'ROI 95% CI': f"[{m_['roi_ci'][0]:.1f}, {m_['roi_ci'][1]:.1f}]" if m_.get('bets') else '', 'max DD ($100 units)': round(m_.get('max_drawdown', np.nan), 1)})
        d = M.roi_diff_ci(cb[cb['market'] == mk], bb[bb['market'] == mk])
        rows[-1]['ROI diff vs baseline (95% CI)'] = f"{d['roi_diff'] * 100 / 100:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
    lines += ['## Betting, unified 5%-edge rule, opening prices (descriptive; ROI differences are not tested against a keep rule)\n',
              pd.DataFrame(rows).to_markdown(index=False), '']
    lines += ['## Per-season ML log loss difference (combined - baseline)\n', str(r['ll']['per_season']), '',
              '## Per-season spread MAE difference (combined - baseline)\n', str(r['ae_spread']['per_season']), '',
              '## Per-season total cover log-loss difference (E4)\n', str(rt['per_season']), '']
    (OUT / NAME / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'metrics.json').write_text(json.dumps({'combined_vs_baseline': r, 'ml_combined_minus_E3': d_ml, 'spread_combined_minus_E1': d_sp,
                                                         'total_cover': rt, 'ece': ece, 'contradiction': contra}, indent=1, default=float))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
