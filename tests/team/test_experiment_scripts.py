"""Fixes to individual experiment scripts (E2, E6, E10-E12, E17, E19, E11 joint Kelly)."""
import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ E11: exact joint distribution of an ML and a spread bet

def test_same_side_ml_and_spread_events_are_nested():
    from experiments.e11_exposure import joint_probs
    # home favourite: P(win) 0.80, P(cover -5.5) 0.55 -> covering implies winning, so "cover and lose" is impossible
    j = joint_probs(0.80, 0.55, same_side=True)          # (ml wins, spread wins)
    assert j[(0, 1)] == 0.0 and j[(1, 1)] == pytest.approx(0.55) and j[(1, 0)] == pytest.approx(0.25) and j[(0, 0)] == pytest.approx(0.20)
    # home underdog +5.5: winning implies covering, so "win and fail to cover" is impossible
    j = joint_probs(0.40, 0.55, same_side=True)
    assert j[(1, 0)] == 0.0 and j[(1, 1)] == pytest.approx(0.40) and j[(0, 1)] == pytest.approx(0.15) and j[(0, 0)] == pytest.approx(0.45)


def test_opposite_side_events_overlap_only_in_the_middle():
    from experiments.e11_exposure import joint_probs
    j = joint_probs(0.80, 0.45, same_side=False)          # home ML + away +5.5: the 'middle' is a home win by 1-5
    assert j[(1, 1)] == pytest.approx(0.25) and j[(1, 0)] == pytest.approx(0.55) and j[(0, 1)] == pytest.approx(0.20) and j[(0, 0)] == 0.0
    j = joint_probs(0.40, 0.45, same_side=False)          # disjoint tails (home +5.5 and away -5.5 cannot both hit)
    assert j[(1, 1)] == 0.0 and j[(0, 0)] == pytest.approx(0.15)


@pytest.mark.parametrize('p1,p2', [(0.5, 0.5), (0.9, 0.3), (0.3, 0.9), (0.6, 0.7), (0.05, 0.95), (0.7, 0.2)])
@pytest.mark.parametrize('same', [True, False])
def test_joint_probabilities_are_a_distribution_with_the_right_marginals(p1, p2, same):
    from experiments.e11_exposure import joint_probs
    j = joint_probs(p1, p2, same)
    assert all(v >= -1e-12 for v in j.values()) and sum(j.values()) == pytest.approx(1.0)
    assert j[(1, 1)] + j[(1, 0)] == pytest.approx(p1) and j[(1, 1)] + j[(0, 1)] == pytest.approx(p2)


def _game(p_ml, p_sp, ml_side='home', sp_side='home', price=-110):
    return pd.DataFrame({'market': ['ml', 'spread'], 'side': [ml_side, sp_side], 'p': [p_ml, p_sp], 'b': [0.9, 100 / 110]})


def test_outcome_probabilities_give_no_mass_to_impossible_outcomes():
    from experiments.e11_exposure import outcome_probs
    outcomes, probs = outcome_probs(_game(0.80, 0.55))           # outcome vector = (ml won, spread won)
    assert probs.sum() == pytest.approx(1.0)
    by = {tuple(o): p for o, p in zip(outcomes, probs)}
    assert by[(0, 1)] == 0.0                                      # lose the game yet cover -5.5
    outcomes, probs = outcome_probs(_game(0.80, 0.45, sp_side='away'))
    by = {tuple(o): p for o, p in zip(outcomes, probs)}
    assert by[(1, 1)] == pytest.approx(0.25) and by[(0, 0)] == 0.0


def test_totals_stay_independent_of_the_margin_bets():
    from experiments.e11_exposure import outcome_probs
    g = pd.concat([_game(0.80, 0.55), pd.DataFrame({'market': ['total'], 'side': ['over'], 'p': [0.6], 'b': [100 / 110]})], ignore_index=True)
    outcomes, probs = outcome_probs(g)
    assert probs.sum() == pytest.approx(1.0)
    assert sum(p for o, p in zip(outcomes, probs) if o[2] == 1) == pytest.approx(0.6)


def test_joint_kelly_fractions_are_feasible():
    from experiments.e11_exposure import CAP, joint_fractions
    f = joint_fractions(_game(0.80, 0.58))
    assert (f >= 0).all() and f.sum() <= CAP + 1e-9


# ------------------------------------------------------------------ E2: cross-fitted isotonic must be causal

def test_crossfit_isotonic_trains_each_block_only_on_earlier_games(monkeypatch):
    from experiments import e02_calibration as e2
    n = 800
    df = pd.DataFrame({'seasonYear': np.repeat(['2017-18', '2018-19', '2019-20', '2020-21'], n // 4),
                       'gameDate': pd.Timestamp('2017-10-20') + pd.to_timedelta(np.arange(n), unit='D'),
                       'target_home_win': (np.arange(n) % 3 == 0).astype(int), 'f': np.arange(n, dtype=float)})
    seen = []

    class Stub:
        def fit(self, X, y):
            self.max_train = X.index.max()
            self.n_train = len(X)
            return self

        def predict_proba(self, X):
            seen.append((self.max_train, X.index.min(), X.index.max(), self.n_train))
            return np.column_stack([1 - np.full(len(X), 0.4), np.full(len(X), 0.4)])

    monkeypatch.setattr(e2, 'make_est', lambda kind, cfg: Stub())
    monkeypatch.setattr(e2, 'GRID', [{}])
    out = e2.crossfit_isotonic(df, '2020-21', ['f'], 0, np.array([0.2, 0.5]))
    assert len(seen) == 4 and len(out) == 2
    assert all(trained_to < block_start for trained_to, block_start, _, _ in seen)        # never fits on a game it has not "lived through"
    assert [s[3] for s in seen] == sorted(s[3] for s in seen) and seen[0][3] == 200       # the window expands: block 0 sees the 2017-18 games only


# ------------------------------------------------------------------ E12: the totals decision uses the totals control

def _mats():
    ids = ['g1', 'g2', 'g3']
    cols = ['dk', 'fd', 'mgm']
    tot = pd.DataFrame([[218.0, 219.5, 221.0], [218.0, 219.5, 221.0], [np.nan, np.nan, np.nan]], index=ids, columns=cols)
    hs = pd.DataFrame([[-5.5, -6.0, -5.0], [-5.5, -6.0, -5.0], [np.nan] * 3], index=ids, columns=cols)
    return {'tot': tot, 'hs': hs, 'as': -hs}


def test_both_sides_control_for_totals_bets_over_at_the_lowest_and_under_at_the_highest_total():
    from experiments.e12_shopping import both_sides_control
    base = pd.DataFrame({'gameId': ['g1', 'g2', 'g3'], 'target_total_pts': [220.0, 230.0, 200.0], 'target_point_diff': [3.0, -9.0, 0.0]})
    r = both_sides_control(base, _mats(), 'total')
    win, loss = 100 / 110 * 100, -100.0
    # g1 (220): over 218 wins, under 221 wins -> a middle; g2 (230): over wins, under loses; g3 has no quote
    assert r['n'] == 2 and r['roi'] == pytest.approx((win + win + win + loss) / 400 * 100)


def test_both_sides_control_for_spreads_differs_from_the_totals_control():
    from experiments.e12_shopping import both_sides_control
    base = pd.DataFrame({'gameId': ['g1', 'g2', 'g3'], 'target_total_pts': [220.0, 230.0, 200.0], 'target_point_diff': [3.0, -9.0, 0.0]})
    sp, tot = both_sides_control(base, _mats(), 'spread'), both_sides_control(base, _mats(), 'total')
    # best home handicap -5.0, best away handicap +6.0: g1 (home by 3) home loses / away wins; g2 (away by 9) home loses / away wins
    assert sp['n'] == 2 and sp['roi'] == pytest.approx(2 * (-100 + 100 / 1.1) / 400 * 100)
    assert sp['roi'] != pytest.approx(tot['roi'])


def test_total_decision_is_judged_against_the_total_control():
    import inspect
    from experiments import e12_shopping
    src = inspect.getsource(e12_shopping.main)
    assert "('total', tot_roi)" in src and "('spread', sp_roi)" in src


# ------------------------------------------------------------------ cover probabilities: one definition everywhere (make_bets p_win)

def _lattice_games(n=60000, seed=0, line=-3.0, sigma=12.0, push_aware_pred=2.0):
    """Margins that really are round(Normal(pred, sigma)) on a whole-number line, so a calibrated model's P(win | no push) is exact."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(-6, 6, n) + push_aware_pred
    margin = np.round(rng.normal(pred, sigma))
    total_pred = rng.uniform(210, 230, n)
    total_line = np.round(total_pred + rng.normal(0, 3, n))                 # whole-number totals
    total = np.round(rng.normal(total_pred, 17.0))
    return pd.DataFrame({
        'gameId': [f'{i:010d}' for i in range(n)], 'gameDate': pd.Timestamp('2024-01-01') + pd.to_timedelta(np.arange(n) // 8, unit='D'),
        'seasonYear': '2023-24', 'ml_prob': 0.5, 'odds_home_ml': -110.0, 'odds_away_ml': -110.0, 'target_home_win': 1,
        'spread_pred': pred, 'odds_home_spread': line, 'target_point_diff': margin, 'sigma_spread': sigma,
        'total_pred': total_pred, 'odds_total': total_line, 'target_total_pts': total, 'sigma_total': 17.0})


def test_make_bets_p_win_is_the_probability_behind_the_edge():
    from experiments import metrics as M
    b = M.make_bets(_lattice_games(4000), thr=0.0, markets=('spread', 'total'))
    assert np.allclose(b['p_win'], b['edge'] + M.BREAK_EVEN_110) and b['p_win'].between(0, 1).all()
    ml = M.make_bets(pd.DataFrame({'gameId': ['a', 'b'], 'gameDate': pd.Timestamp('2024-01-01'), 'seasonYear': 's', 'ml_prob': [0.7, 0.3],
                                   'odds_home_ml': [-110.0, -110.0], 'odds_away_ml': [-110.0, -110.0], 'target_home_win': [1, 0]}), thr=0.0, markets=('ml',))
    assert sorted(ml['p_win'].round(3)) == [0.7, 0.7] and set(ml['side']) == {'home', 'away'}


def test_hit_rate_of_non_push_bets_matches_mean_p_win_on_whole_number_lines():
    from experiments import metrics as M
    b = M.make_bets(_lattice_games(), thr=0.02, markets=('spread', 'total'))
    for mk in ('spread', 'total'):
        d = b[b['market'] == mk]
        assert len(d) > 2000 and not d['push'].any()
        assert abs(d['won'].mean() - d['p_win'].mean()) < 0.006, (mk, d['won'].mean(), d['p_win'].mean())


def test_e11_stakes_and_e10_check_use_the_probability_the_bet_was_selected_on():
    from experiments import e10_rules, e11_exposure, metrics as M
    g = _lattice_games(3000)
    bets = e11_exposure.with_probs(g)
    assert np.allclose(bets['p'], bets['p_win'])
    f = (bets['p'] * bets['b'] - (1 - bets['p'])) / bets['b']
    assert np.allclose(bets['kelly'], np.clip(f, 0, None)) and np.allclose(bets['ev'], bets['p'] * bets['b'] - (1 - bets['p']))
    sp = M.make_bets(g, thr=0.05, markets=('spread',))
    assert e10_rules.predicted_win(sp, g) == pytest.approx(sp['p_win'].mean())
    assert np.isnan(e10_rules.predicted_win(sp.iloc[0:0], g))


def test_e04_cover_frame_is_push_aware_like_the_bets():
    from scipy.stats import norm
    from experiments.e04_sigma import cover_frame
    df = pd.DataFrame({'spread_pred': [4.0, 4.0], 'odds_home_spread': [-3.0, -3.5], 'target_point_diff': [5.0, 5.0],
                       'total_pred': [220.0, 220.0], 'odds_total': [218.0, 218.5], 'target_total_pts': [225.0, 225.0]})
    sig = np.array([12.0, 12.0])
    sp = cover_frame(df, 'spread', sig)
    assert sp['p'][1] == pytest.approx(norm.cdf(0.5 / 12))                            # half-point line: no push, plain Phi
    win, loss = norm.cdf((1.0 - 0.5) / 12), norm.cdf((-1.0 - 0.5) / 12)
    assert sp['p'][0] == pytest.approx(win / (win + loss)) and sp['p'][0] != pytest.approx(norm.cdf(1.0 / 12))
    tot = cover_frame(df, 'total', sig)
    win, loss = norm.cdf((2.0 - 0.5) / 12), norm.cdf((-2.0 - 0.5) / 12)
    assert tot['p'][0] == pytest.approx(win / (win + loss)) and tot['p'][1] == pytest.approx(norm.cdf(1.5 / 12))


# ------------------------------------------------------------------ E6: calibration rows stay in fold order

def test_e06_calibration_rows_are_in_fold_order_not_date_order():
    from experiments import e06_run
    # consecutive folds' calibration blocks overlap in gameDate (the last 20% of each training window)
    def calib(kappa):
        rows = []
        for i, s in enumerate(e06_run.TEST_SEASONS):
            d = pd.Timestamp('2018-01-01') + pd.Timedelta(days=200 * i) + pd.to_timedelta(np.arange(0, 600, 100), unit='D')
            rows.append(pd.DataFrame({'fold': s, 'gameDate': d, 'kappa': kappa}))
        return pd.concat(rows, ignore_index=True)
    cals = {1.0: calib(1.0), 0.75: calib(0.75), 0.5: calib(0.5)}
    chosen = {s: k for s, k in zip(e06_run.TEST_SEASONS, [1.0, 0.5, 0.75, 1.0, 0.5, 0.75, 1.0])}
    out = e06_run.assemble_calibration(cals, chosen)
    assert list(out['fold'].drop_duplicates()) == e06_run.TEST_SEASONS                    # fold blocks are contiguous and chronological
    assert not out['gameDate'].is_monotonic_increasing                                    # ...so a date sort would have scrambled them
    for s, g in out.groupby('fold'):
        assert set(g['kappa']) == {chosen[s]} and len(g) == 6 and g['gameDate'].is_monotonic_increasing


# ------------------------------------------------------------------ E17 / E19: bootstrap draws that miss a thin cell

@pytest.mark.parametrize('module,fn', [('e17_interaction', 'ratio_diff'), ('e19_tanking', 'e14_ratio_diff')])
def test_bootstrap_ratio_diff_ignores_draws_with_an_empty_cell(module, fn):
    import importlib
    from experiments.e14_blowout_under import DayBoot
    mod = importlib.import_module(f'experiments.{module}')
    dates = pd.Timestamp('2024-01-01') + pd.to_timedelta(np.repeat(np.arange(12), 5), unit='D')
    db = DayBoot(dates, n_boot=400, seed=3)
    n = len(dates)
    num, den = np.ones(n), np.ones(n)
    thin = np.zeros(n, bool)
    thin[:2] = True                                              # two games on one day: ~1/3 of the draws never pick that day
    rest = ~thin
    assert (db.W @ db.day_sum(np.where(thin, den, 0.0)) == 0).any()
    out = getattr(mod, fn)(db, num, den, thin, rest)
    assert np.isfinite([out['diff'], out['lo'], out['hi']]).all() and out['lo'] <= out['hi']


# ------------------------------------------------------------------ E31b: the no-selection arms against a post-audit reference

def _reference_dir(tmp_path, split, selection='residual'):
    d = tmp_path / 'E25_with_odds'
    d.mkdir()
    import json
    (d / 'split.json').write_text(json.dumps(split))
    (d / 'selection.json').write_text(json.dumps({'selection': selection}))
    return d


def test_e31b_refuses_a_reference_made_under_the_old_scheme(tmp_path, monkeypatch):
    from experiments import e31b_nosel as e31b, lib
    monkeypatch.setattr(lib, 'OUT', tmp_path)
    old = tmp_path / 'E25_with_odds'
    old.mkdir()                                                               # no split.json / selection.json: a pre-audit run
    with pytest.raises(SystemExit):
        e31b.require_new_defaults('E25_with_odds')
    import shutil
    shutil.rmtree(old)
    _reference_dir(tmp_path, {'calib_cap': 1300, 'refit_on_all': True, 'ml_calibrator': 'isotonic'})     # capped and refit, but the old calibrator
    with pytest.raises(SystemExit):
        e31b.require_new_defaults('E25_with_odds')
    shutil.rmtree(old)
    _reference_dir(tmp_path, {'calib_cap': 1300, 'refit_on_all': True, 'ml_calibrator': 'platt'}, selection='raw')
    with pytest.raises(SystemExit):
        e31b.require_new_defaults('E25_with_odds')
    shutil.rmtree(old)
    _reference_dir(tmp_path, {'calib_cap': 1300, 'refit_on_all': True, 'ml_calibrator': 'platt'})
    assert e31b.require_new_defaults('E25_with_odds')['calib_cap'] == 1300


def test_e31b_arm_lists_nest_reference_inside_stage1_inside_all(tmp_path, monkeypatch):
    import json
    from experiments import e31b_nosel as e31b, lib
    from tests.team.test_experiment_harness import _games
    m, df = _games(per_season=700)
    rng = np.random.default_rng(3)
    for j in range(4):                                                        # near-duplicates: clustering must drop them, 'pool' keeps them
        df[f'diff_dup{j}'] = df[f'diff_f{j}'] + rng.normal(scale=1e-3, size=len(df))
    m.features_diff = [c for c in df.columns if c.startswith('diff_')] + ['odds_home_spread']
    m.features_sep = [c for c in df.columns if c.startswith('diff_')] + ['odds_home_spread', 'odds_total']       # the lines are in the pools
    monkeypatch.setattr(lib, 'OUT', tmp_path)
    monkeypatch.setattr(e31b, 'TEST_SEASONS', ['2022-23'])
    monkeypatch.setattr(e31b, 'load_games', lambda: (m, df))
    ref_dir = _reference_dir(tmp_path, {'calib_cap': 100, 'refit_on_all': True, 'ml_calibrator': 'platt'})
    mt, calib, test, n_fl = lib.split_fold(df, '2022-23', calib_cap=100)
    ref = m.select_market_features(mt.iloc[:n_fl], seed=42)
    (ref_dir / 'features.json').write_text(json.dumps({'2022-23': ref}))
    counts = e31b.write_features('E31b_all', 'pool')
    c = counts['2022-23']['total']
    assert c['reference'] <= c['stage1'] <= c['all'] and c['stage1'] < c['all']
    got = json.loads((tmp_path / 'E31b_all' / 'features.json').read_text())['2022-23']
    assert set(got['ml']) == set(got['spread']) and 'odds_total' in got['total']
    stage1 = e31b.write_features('E31b_stage1', 'stage1')
    assert stage1 == counts and len(json.loads((tmp_path / 'E31b_stage1' / 'features.json').read_text())['2022-23']['total']) == c['stage1']
