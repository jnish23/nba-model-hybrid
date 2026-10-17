"""E3 - cross-model coherence (see PREREGISTRATION.md). Uses the baseline models' raw predictions; only the final ML
probability changes: (b) blend with Phi(margin_pred/sigma), (c) logistic stack of [logit(raw ML), margin_pred/sigma]."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import decide, metrics as M                                         # noqa: E402
from experiments.lib import OUT, TEST_SEASONS, label_calib_folds, load_games, load_run   # noqa: E402

NAME = 'E03_coherence'
GRID_W = np.round(np.arange(0, 1.01, 0.1), 1)


def lg(p):
    return logit(np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4))


def contradiction(df, p_col='ml_prob'):
    return float(((df[p_col] > 0.5) != (df['spread_pred'] > 0)).mean())


def ll(y, p):
    p = M.clip_p(p)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    base, cal = load_run('baseline')
    _, df = load_games()
    cal['fold'] = label_calib_folds(df, cal)
    b, c = base.copy(), base.copy()
    ws = {}
    for s in TEST_SEASONS:
        tm, cm = (base['seasonYear'] == s).to_numpy(), (cal['fold'] == s).to_numpy()
        ct, te = cal[cm], base[tm]
        pc = norm.cdf(ct['spread_pred'] / ct['sigma_spread'])
        pt = norm.cdf(te['spread_pred'] / te['sigma_spread'])
        # (b) blend in the raw space, weight chosen on the calibration set (Platt-calibrated log loss), then the baseline isotonic
        best_w, best = None, np.inf
        for w in GRID_W:
            blend = w * ct['ml_prob_raw'] + (1 - w) * pc
            platt = LogisticRegression(C=1e6).fit(lg(blend).reshape(-1, 1), ct['target_home_win'])
            sc = ll(ct['target_home_win'].to_numpy(float), platt.predict_proba(lg(blend).reshape(-1, 1))[:, 1])
            if sc < best:
                best_w, best = w, sc
        ws[s] = float(best_w)
        blend_c = best_w * ct['ml_prob_raw'] + (1 - best_w) * pc
        blend_t = best_w * te['ml_prob_raw'] + (1 - best_w) * pt
        b.loc[tm, 'ml_prob'] = IsotonicRegression(out_of_bounds='clip').fit(blend_c, ct['target_home_win']).predict(blend_t)
        # (c) stack: logistic regression on [logit(raw ML), margin/sigma] fitted on the calibration set
        Xc = np.column_stack([lg(ct['ml_prob_raw']), ct['spread_pred'] / ct['sigma_spread']])
        Xt = np.column_stack([lg(te['ml_prob_raw']), te['spread_pred'] / te['sigma_spread']])
        c.loc[tm, 'ml_prob'] = LogisticRegression(C=1.0).fit(Xc, ct['target_home_win']).predict_proba(Xt)[:, 1]
    bl = M.game_losses(base)
    rows, decisions = [], {}
    base_contra = contradiction(base)
    for v, t in (('blend_with_spread_prob', b), ('stack_logit_margin', c)):
        t.to_csv(OUT / NAME / f'predictions_{v}.csv', index=False) if (OUT / NAME).mkdir(parents=True, exist_ok=True) is None else None
        r, rb = M.paired(bl, M.game_losses(t), 'll'), M.paired(bl, M.game_losses(t), 'brier')
        contra = contradiction(t)
        ok = decide.mc(r) and decide.nh(rb) and contra <= base_contra
        decisions[v] = bool(ok)
        rows.append({'variant': v, 'logloss diff': round(r['diff'], 5), 'CI': f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]",
                     'seasons better': f"{r['seasons_improved']}/7", 'Brier diff': round(rb['diff'], 5),
                     'ECE': round(M.ece(t['target_home_win'], t['ml_prob']), 4), 'contradiction rate': round(contra, 4), 'KEEP': ok})
    bmask = base['spread_pred'].notna()
    lines = [f'# {NAME}\n', '## (a) Contradictions in the baseline\n',
             f"- ML vs spread (ml_prob > .5 vs predicted margin > 0): **{base_contra:.2%}** of {len(base)} games",
             f"- ML vs opening line (ml_prob > .5 vs opening home spread < 0): {((base['ml_prob'] > .5) != (base['odds_home_spread'] < 0)).mean():.2%}",
             f"- predicted margin vs opening line direction: {((base['spread_pred'] > 0) != (base['odds_home_spread'] < 0)).mean():.2%}",
             f"- by season: {base.groupby('seasonYear').apply(lambda g: round(contradiction(g), 4), include_groups=False).to_dict()}",
             f"- baseline log loss {ll(base['target_home_win'].to_numpy(float), base['ml_prob']):.5f}; blend weights on the classifier by fold: {ws}", '',
             '## (b), (c) vs baseline\n', pd.DataFrame(rows).to_markdown(index=False), '', '## Decision\n'] + \
        [f'- {k}: **{"KEEP" if v else "REJECT"}**' for k, v in decisions.items()]
    for v, t in (('blend', b), ('stack', c)):
        lines += ['', f'### {v} per-season log loss diff', str(M.paired(bl, M.game_losses(t), 'll')['per_season'])]
    (OUT / NAME / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'decision.json').write_text(json.dumps({'decision': decisions, 'blend_weights': ws}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
