"""E11 - same-game exposure (see PREREGISTRATION.md). Bets = the baseline 5%-edge rule; only the stake sizing differs.
(i) independent quarter-Kelly per bet [baseline], (ii) one best-EV market per game (quarter-Kelly), (iii) joint Kelly over the
game's bets with the exact ML/spread dependence (both are thresholds on the one game margin, so the events are nested or
complementary - joint_probs; totals independent), scaled by 1/4.
Bankroll simulation: bets settled by game day, fractions applied to the start-of-day bankroll; 1000 day-bootstrap resamples.
python -m experiments.e11_exposure"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import metrics as M                                                  # noqa: E402
from experiments.lib import OUT, load_run   # noqa: E402

NAME = 'E11_exposure'
KELLY_SCALE = 0.25
CAP = 0.25            # cap on the sum of FULL-Kelly fractions per game in the joint optimisation


def with_probs(base):
    """The 5%-edge bets with the win probability each was selected on (make_bets p_win: push-aware on whole-number lines)."""
    bets = M.make_bets(base, thr=0.05)
    bets['p'] = bets['p_win']
    bets['b'] = np.where(bets['price'] < 0, 100.0 / -bets['price'], bets['price'] / 100.0)
    bets['kelly'] = np.clip((bets['p'] * bets['b'] - (1 - bets['p'])) / bets['b'], 0, None)
    bets['ev'] = bets['p'] * bets['b'] - (1 - bets['p'])
    return bets


def joint_probs(p1, p2, same_side):
    """
    Exact joint distribution of two bet outcomes on the SAME game: {(1, 1): both win, (1, 0): only the first wins, (0, 1): only the
    second wins, (0, 0): neither}. A moneyline and a spread bet are thresholds on one random variable, the final margin M, so there is
    no copula to fit: the events are comonotonic.
      same side (both home or both away): both are upper (or both lower) tails of M, so one event contains the other and
        P(both) = min(p1, p2) - the favourite cannot cover -5.5 and lose the game;
      opposite sides (home ML with away spread, ...): an upper tail and a lower tail, so P(both) = max(0, p1 + p2 - 1), the 'middle'.
    (The earlier version used a bivariate normal with rho = sin(pi * phi / 2) < 1, which is only the median-split identity and gave
    positive probability to impossible outcomes.)
    """
    if same_side:
        p11 = min(p1, p2)
    else:
        p11 = max(0.0, p1 + p2 - 1.0)
    p10, p01 = p1 - p11, p2 - p11
    return {(1, 1): p11, (1, 0): p10, (0, 1): p01, (0, 0): 1.0 - (p11 + p10 + p01)}


def outcome_probs(g):
    """(all 0/1 outcome vectors of one game's bets, their joint probabilities)."""
    k = len(g)
    p = g['p'].to_numpy()
    side_team = np.where(g['market'] == 'total', 'x', np.where(g['side'] == 'home', 'H', 'A'))
    outcomes = np.array(list(itertools.product([0, 1], repeat=k)))
    probs = np.ones(len(outcomes))
    ml_i = [i for i in range(k) if g['market'].iloc[i] == 'ml']
    sp_i = [i for i in range(k) if g['market'].iloc[i] == 'spread']
    dependent = ml_i + sp_i if (ml_i and sp_i) else []
    indep = [i for i in range(k) if i not in dependent]
    for oi, o in enumerate(outcomes):
        pr = 1.0
        for i in indep:
            pr *= p[i] if o[i] else 1 - p[i]
        probs[oi] = pr
    if dependent:
        i, j = ml_i[0], sp_i[0]
        pj = joint_probs(p[i], p[j], same_side=side_team[i] == side_team[j])
        for oi, o in enumerate(outcomes):
            probs[oi] = probs[oi] * pj[(o[i], o[j])]
    return outcomes, probs


def joint_fractions(g):
    """Full-Kelly fractions maximising expected log growth over the joint outcomes of one game's bets."""
    k = len(g)
    b = g['b'].to_numpy()
    outcomes, probs = outcome_probs(g)
    returns = np.where(outcomes == 1, b, -1.0)

    def neg(f):
        wealth = 1 + returns @ f
        return -(probs * np.log(np.clip(wealth, 1e-9, None))).sum()
    res = minimize(neg, x0=np.full(k, 0.01), bounds=[(0, CAP)] * k, constraints=[{'type': 'ineq', 'fun': lambda f: CAP - f.sum()}],
                   method='SLSQP')
    return np.clip(res.x, 0, None)


def stakes(bets):
    bets = bets.copy()
    bets['f_ind'] = KELLY_SCALE * bets['kelly']
    best = bets.groupby('gameId')['ev'].transform('max') == bets['ev']
    bets['f_best'] = np.where(best, KELLY_SCALE * bets['kelly'], 0.0)
    bets['f_joint'] = 0.0
    for gid, g in bets.groupby('gameId'):
        if len(g) == 1:
            bets.loc[g.index, 'f_joint'] = KELLY_SCALE * g['kelly'].iloc[0]
            continue
        f = joint_fractions(g)
        bets.loc[g.index, 'f_joint'] = KELLY_SCALE * f
    return bets


def main():
    base, _ = load_run('baseline')
    bets = with_probs(base)
    bets = stakes(bets)
    cols = {'(i) independent 1/4 Kelly [baseline]': 'f_ind', '(ii) best-EV market per game': 'f_best', '(iii) joint Kelly (1/4)': 'f_joint'}
    n_boot = 1000
    nd = bets['gameDate'].nunique()
    rng = np.random.default_rng(M.RNG_SEED)
    draws = rng.integers(0, nd, size=(n_boot, nd))
    res = {k: {'growth': [], 'vol': [], 'dd': []} for k in cols}
    # day order index: sort unique days chronologically so "chron" matches
    days_sorted = np.sort(pd.unique(bets['gameDate']))
    day_pos = {d: i for i, d in enumerate(days_sorted)}
    bets['day'] = bets['gameDate'].map(day_pos)
    day_ret = {k: np.clip(np.bincount(bets['day'], weights=bets[c].to_numpy() * np.where(bets['won'], bets['b'], -1.0), minlength=len(days_sorted)), -0.99, None)
               for k, c in cols.items()}
    def stats(r):
        g = np.log1p(r)
        cum = np.cumsum(g)
        peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
        return cum[-1], g.std(), (1 - np.exp(cum - peak)).max()
    actual = {k: stats(day_ret[k]) for k in cols}
    boot = {k: np.array([stats(day_ret[k][d]) for d in draws]) for k in cols}
    rows = []
    for k in cols:
        gb = boot[k]
        rows.append({'strategy': k, 'bets': int((bets[cols[k]] > 0).sum()), 'avg stake % of bankroll': round(float(bets.loc[bets[cols[k]] > 0, cols[k]].mean() * 100), 2),
                     'actual log-growth (sum over days)': round(actual[k][0], 4), 'actual max DD': round(actual[k][2], 3),
                     'boot median log-growth': round(float(np.median(gb[:, 0])), 4), 'log-growth 95% CI': f'[{np.quantile(gb[:, 0], .025):.3f}, {np.quantile(gb[:, 0], .975):.3f}]',
                     'daily vol (median)': round(float(np.median(gb[:, 1])), 4), 'max DD median': round(float(np.median(gb[:, 2])), 3),
                     'max DD p95': round(float(np.quantile(gb[:, 2], .95)), 3)})
    base_k = '(i) independent 1/4 Kelly [baseline]'
    decisions = {}
    for k in cols:
        if k == base_k:
            continue
        d = boot[k][:, 0] - boot[base_k][:, 0]
        ok = (np.median(boot[k][:, 0]) > np.median(boot[base_k][:, 0])) and np.quantile(d, .025) > 0 and np.quantile(boot[k][:, 2], .95) <= np.quantile(boot[base_k][:, 2], .95)
        decisions[k] = bool(ok)
        rows[list(cols).index(k)].update({'paired growth diff vs (i)': round(float(np.median(d)), 4), 'diff 95% CI': f'[{np.quantile(d, .025):.3f}, {np.quantile(d, .975):.3f}]', 'KEEP': bool(ok)})
    lines = [f'# {NAME}\n', f'Bets: {len(bets)} (5% edge rule) on {bets["gameId"].nunique()} games, {int((bets.groupby("gameId").size() > 1).sum())} games with more than one bet.',
             'ML/spread dependence: exact (one margin, nested or complementary events); totals independent.', '',
             pd.DataFrame(rows).to_markdown(index=False), '', '## Decision (pre-registered)\n'] + \
        [f'- {k}: **{"KEEP" if v else "REJECT"}**' for k, v in decisions.items()]
    avg = bets.groupby('gameId').size().value_counts().sort_index().to_dict()
    lines += ['', f'Bets per game: {avg}']
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    (OUT / NAME / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / NAME / 'decision.json').write_text(json.dumps({'decision': decisions}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
