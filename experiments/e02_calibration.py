"""E2 - calibration method (see PREREGISTRATION.md). Baseline models/raw predictions are reused unchanged; only the
probability calibrator differs: isotonic on the last-20% split (baseline), Platt on log-odds (same split), and
cross-fitted isotonic over the last two training seasons (4 time blocks, out-of-fold raw predictions)."""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import decide, metrics as M                                  # noqa: E402
from experiments.lib import GRID, OUT, TEST_SEASONS, label_calib_folds, load_games, load_run, make_est   # noqa: E402
from experiments.report import evaluate, per_season_table                     # noqa: E402

NAME = 'E02_calibration'


def platt(calib, test):
    x = logit(np.clip(calib['ml_prob_raw'].to_numpy(float), 1e-4, 1 - 1e-4)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6).fit(x, calib['target_home_win'])
    return lr.predict_proba(logit(np.clip(test['ml_prob_raw'].to_numpy(float), 1e-4, 1 - 1e-4)).reshape(-1, 1))[:, 1]


def crossfit_isotonic(df, season, feats, cfg, test_raw, n_blocks=4):
    train = df[df['seasonYear'] < season].sort_values('gameDate', kind='stable').reset_index(drop=True)
    seasons = sorted(train['seasonYear'].unique())[-2:]
    idx = np.flatnonzero(train['seasonYear'].isin(seasons).to_numpy())
    blocks = np.array_split(idx, n_blocks)
    oof_p, oof_y = [], []
    for b in blocks:
        mask = np.ones(len(train), bool)
        mask[b] = False
        est = make_est('clf', GRID[cfg])
        est.fit(train.loc[mask, feats], train.loc[mask, 'target_home_win'])
        oof_p.append(est.predict_proba(train.loc[b, feats])[:, 1])
        oof_y.append(train.loc[b, 'target_home_win'].to_numpy())
    iso = IsotonicRegression(out_of_bounds='clip').fit(np.concatenate(oof_p), np.concatenate(oof_y))
    return iso.predict(test_raw)


def main():
    logging.basicConfig(level=logging.INFO)
    base, base_cal = load_run('baseline')
    m, df = load_games()
    feats = json.loads((OUT / 'baseline' / 'features.json').read_text())
    info = json.loads((OUT / 'baseline' / 'fold_info.json').read_text())
    cal_fold = label_calib_folds(df, base_cal)
    variants = {'platt': base.copy(), 'crossfit_isotonic': base.copy()}
    for s in TEST_SEASONS:
        tm = (base['seasonYear'] == s).to_numpy()
        cm = cal_fold == s
        variants['platt'].loc[tm, 'ml_prob'] = platt(base_cal[cm], base[tm])
        variants['crossfit_isotonic'].loc[tm, 'ml_prob'] = crossfit_isotonic(
            df, s, feats[s]['ml'], info[s]['ml']['cfg'], base.loc[tm, 'ml_prob_raw'].to_numpy(float))
    bl = M.game_losses(base)
    rows, lines = [], [f'# {NAME}\n']
    decisions = {}
    for v, t in variants.items():
        (OUT / NAME).mkdir(parents=True, exist_ok=True)
        t.to_csv(OUT / NAME / f'predictions_{v}.csv', index=False)
        r = M.paired(bl, M.game_losses(t), 'll')
        rb = M.paired(bl, M.game_losses(t), 'brier')
        ece_s = {s: M.ece(g['target_home_win'], g['ml_prob']) for s, g in t.groupby('seasonYear')}
        ece_b = {s: M.ece(g['target_home_win'], g['ml_prob']) for s, g in base.groupby('seasonYear')}
        ece_better = sum(ece_s[s] < ece_b[s] for s in ece_s)
        distinct = int(pd.Series(np.round(t['ml_prob'], 6)).nunique())
        d_base = int(pd.Series(np.round(base['ml_prob'], 6)).nunique())
        crit_i = ece_better >= 5
        crit_ii = r['ci_hi'] < 0.0005
        crit_iii = distinct >= 3 * d_base or decide.mc(r)
        decisions[v] = bool(crit_i and crit_ii and crit_iii)
        rows.append({'variant': v, 'ECE pooled': round(M.ece(t['target_home_win'], t['ml_prob']), 4),
                     'ECE baseline': round(M.ece(base['target_home_win'], base['ml_prob']), 4),
                     'seasons ECE better': f'{ece_better}/7', 'log loss': round(float(bl['ll'].mean() + r['diff']), 5),
                     'logloss diff': round(r['diff'], 5), 'CI': f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]",
                     'Brier diff': round(rb['diff'], 5), 'distinct probs': distinct, 'baseline distinct': d_base,
                     '(i)': crit_i, '(ii)': crit_ii, '(iii)': crit_iii, 'KEEP': decisions[v]})
        lines += [f'## {v}: reliability by decile\n', M.reliability(t['target_home_win'], t['ml_prob']).round(4).to_markdown(index=False), '',
                  f'per-season log loss diff: {r["per_season"]}', '']
    lines += ['## baseline reliability by decile\n', M.reliability(base['target_home_win'], base['ml_prob']).round(4).to_markdown(index=False), '']
    res = pd.DataFrame(rows)
    # tie-break (pre-registered): lower pooled ECE wins
    keepers = res[res['KEEP']]
    winner = keepers.sort_values('ECE pooled').iloc[0]['variant'] if len(keepers) else None
    lines = [lines[0], '## Summary\n', res.to_markdown(index=False), '', f'**Winner (pre-registered rule): {winner}**', ''] + lines[1:]
    (OUT / NAME / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'decision.json').write_text(json.dumps({'decision': decisions, 'winner': winner}, indent=1))
    print('\n'.join(lines[:12]))


if __name__ == '__main__':
    main()
