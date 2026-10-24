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
