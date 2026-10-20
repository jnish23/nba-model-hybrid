"""E8 step 1 - direct test of 'fresh absences are underpriced, chronic ones are not' on the DISCOVERY seasons only
(2019-20..2021-22). Market residual r = home margin + opening spread (0 if the opening line is unbiased); regressors are the
home-minus-away value lost to fresh (played one of the last 3 team games) and chronic (missed >= 5 in a row) absences.
Supported iff the fresh coefficient is negative with a day-bootstrap CI excluding zero AND the chronic coefficient's CI
includes zero or |chronic| < 0.5 |fresh| (pre-registered). python -m experiments.e08_hypothesis"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import inj_variants as iv, metrics as M          # noqa: E402
from experiments.lib import OUT                                      # noqa: E402
from team.data_paths import latest_training_set                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'nba_stats.db'
NAME = 'E08_fresh_chronic'
DISCOVERY = ['2019-20', '2020-21', '2021-22']
FEATS = ['inj_fresh_lost', 'inj_chronic_lost', 'inj_chronic_lost_dec', 'inj_mid_lost']
VARIANT = ROOT / 'data' / 'processed' / 'experiments' / 'e08_training_set.csv'


def build_csv():
    games = pd.read_csv(latest_training_set(), low_memory=False)
    games['teamId_home'], games['teamId_away'] = games['teamId_home'].astype('int64'), games['teamId_away'].astype('int64')
    seasons = sorted(s for s in games['seasonYear'].unique() if s >= '2019-20')
    inj = iv.build_variant_features(DB, seasons, 'e8')
    out = iv.attach(games, inj, FEATS)
    VARIANT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(VARIANT, index=False)
    return out


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def day_bootstrap(df, cols, y, n=1000, seed=M.RNG_SEED):
    codes, uniques = pd.factorize(pd.to_datetime(df['gameDate']))
    groups = [np.flatnonzero(codes == i) for i in range(len(uniques))]
    rng = np.random.default_rng(seed)
    X, yy = df[cols].to_numpy(float), df[y].to_numpy(float)
    betas = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        betas.append(ols(X[idx], yy[idx]))
    return np.array(betas)


def main():
    games = build_csv()
    d = games[games['seasonYear'].isin(DISCOVERY)].copy()
    d = d.dropna(subset=['odds_home_spread', 'odds_total'])
    d['r_spread'] = d['target_point_diff'] + d['odds_home_spread']
    d['r_total'] = d['target_total_pts'] - d['odds_total']
    for c in ('inj_fresh_lost', 'inj_chronic_lost', 'inj_chronic_lost_dec', 'inj_mid_lost'):
        d[f'x_{c}'] = d[f'{c}_home'].fillna(0) - d[f'{c}_away'].fillna(0)
    d = d[(d['inj_missing_home'] == 0) & (d['inj_missing_away'] == 0)]       # games with a report for both teams
    lines = [f'# {NAME}: step 1 - hypothesis test on discovery seasons {DISCOVERY} ({len(d)} games with reports for both teams)\n']
    out = {}
    for label, y, cols in (('spread residual ~ fresh + chronic (registered)', 'r_spread', ['x_inj_fresh_lost', 'x_inj_chronic_lost']),
                           ('spread residual ~ fresh + chronic + mid (control)', 'r_spread', ['x_inj_fresh_lost', 'x_inj_chronic_lost', 'x_inj_mid_lost']),
                           ('total residual ~ sum of home+away fresh / chronic (descriptive)', 'r_total', None)):
        if cols is None:
            d['s_fresh'] = d['inj_fresh_lost_home'].fillna(0) + d['inj_fresh_lost_away'].fillna(0)
            d['s_chronic'] = d['inj_chronic_lost_home'].fillna(0) + d['inj_chronic_lost_away'].fillna(0)
            cols = ['s_fresh', 's_chronic']
        beta = ols(d[cols].to_numpy(float), d[y].to_numpy(float))
        boots = day_bootstrap(d, cols, y)
        lo, hi = np.quantile(boots, .025, axis=0), np.quantile(boots, .975, axis=0)
        rows = [{'term': t, 'coef': round(beta[i + 1], 4), '95% CI': f'[{lo[i + 1]:+.4f}, {hi[i + 1]:+.4f}]'} for i, t in enumerate(cols)]
        rows.insert(0, {'term': 'intercept', 'coef': round(beta[0], 4), '95% CI': f'[{lo[0]:+.3f}, {hi[0]:+.3f}]'})
        lines += [f'## {label}\n', pd.DataFrame(rows).to_markdown(index=False), '']
        out[label] = {'coef': dict(zip(['intercept'] + cols, beta.tolist())), 'lo': lo.tolist(), 'hi': hi.tolist()}
        if label.startswith('spread residual ~ fresh + chronic (registered)'):
            bf, bc = beta[1], beta[2]
            fresh_ok = bf < 0 and hi[1] < 0
            chronic_ok = (lo[2] <= 0 <= hi[2]) or abs(bc) < 0.5 * abs(bf)
            supported = bool(fresh_ok and chronic_ok)
    lines += ['## Mean fresh / chronic / mid lost value per team-game (discovery seasons)\n',
              d[[f'{c}_home' for c in FEATS]].describe().loc[['mean', 'std']].round(3).to_markdown(), '',
              f'## Result of the pre-registered test: hypothesis **{"SUPPORTED" if supported else "NOT SUPPORTED"}**',
              f'- fresh coef {bf:+.4f}, CI excludes zero & negative: {fresh_ok}',
              f'- chronic coef {bc:+.4f}, CI includes zero or |chronic| < 0.5|fresh|: {chronic_ok}', '']
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    (OUT / NAME / 'HYPOTHESIS.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'hypothesis.json').write_text(json.dumps({'supported': supported, **out}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
