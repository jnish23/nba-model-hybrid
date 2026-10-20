import numpy as np
import pandas as pd

from team.backtest import Backtester


def test_games_without_a_moneyline_are_not_bet_at_an_invented_price():
    df = pd.DataFrame({'prob_home_win': [0.80, 0.80, 0.80], 'target_home_win': [1, 1, 1],
                       'odds_home_ml': [-150.0, np.nan, -150.0], 'odds_away_ml': [130.0, np.nan, np.nan]})
    pnl, roi, bets = Backtester(df).run_backtest(prob_col='prob_home_win', edge_threshold=0.05)
    # games 1 and 3 have a home price and win at -150; game 2 has no price at all. The old -110 fill "bet" game 2 at a price
    # nobody offered (3 bets, +$190 on an invented price)
    assert bets == 2
    assert np.isclose(pnl, 2 * 100 * (100 / 150))
