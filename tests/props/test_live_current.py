import numpy as np
import pandas as pd

from props.live import current

RULE = {'primary': {'e': 0.03, 'min_books': 3}}


def pmf(mean=7.0, n=30):
    from math import exp, factorial
    p = np.array([exp(-mean) * mean ** k / factorial(k) for k in range(n)])
    return p / p.sum()


def mkt(line=6.5, fair_over=0.45, over=(1.95, 2.0, 1.9), under=(1.9, 1.85, 1.95), n=5):
    return pd.DataFrame({'book': ['fanduel', 'draftkings', 'betmgm'], 'over_dec': over, 'under_dec': under, 'consensus_fair_over': fair_over, 'n_at_line': n, 'line': line})


def test_the_recorded_bet_is_valued_at_the_current_prices_and_line():
    bet = {'side': 'over', 'line': 6.5, 'pmf': pmf()}
    r = current.bet_now(bet, mkt(), RULE, 0.3)
    p_mod = float(pmf()[7:].sum())                                    # P(X > 6.5) for the Poisson(7) pmf (no push on a half line)
    p_star = 0.45 + 0.3 * (p_mod - 0.45)
    assert r['line_moved'] == 0 and np.isclose(r['dec_best_now'], 2.0) and r['best_book_now'] == 'draftkings' and np.isclose(r['dec_median_now'], 1.95)
    assert np.isclose(r['ev_best_now'], p_star * 1.0 - (1 - p_star)) and r['still_qualifies'] == int(r['ev_best_now'] >= 0.03)


def test_a_moved_line_is_flagged_and_revalued_from_the_stored_distribution():
    bet = {'side': 'over', 'line': 6.5, 'pmf': pmf()}
    r = current.bet_now(bet, mkt(line=7.5, fair_over=0.40), RULE, 0.3)
    assert r['line_moved'] == 1 and r['line_now'] == 7.5
    p_mod = float(pmf()[8:].sum())
    assert np.isclose(r['ev_best_now'], (0.4 + 0.3 * (p_mod - 0.4)) * 1.0 - (1 - (0.4 + 0.3 * (p_mod - 0.4))))


def test_an_under_uses_the_under_prices_and_a_missing_market_gives_nothing():
    bet = {'side': 'under', 'line': 6.5, 'pmf': pmf()}
    r = current.bet_now(bet, mkt(), RULE, 0.3)
    assert np.isclose(r['dec_best_now'], 1.95) and r['best_book_now'] == 'betmgm'
    assert current.bet_now(bet, mkt().iloc[0:0], RULE, 0.3) is None
