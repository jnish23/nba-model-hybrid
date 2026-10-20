import numpy as np
import pandas as pd

from team.data_checks import missingness_era_report


def frame():
    rng = np.random.default_rng(0)
    seasons = np.repeat(['2017-18', '2018-19', '2019-20', '2020-21'], 250)
    n = len(seasons)
    df = pd.DataFrame({'seasonYear': seasons, 'always': rng.normal(size=n), 'random_gaps': rng.normal(size=n)})
    df['odds_home_ml'] = np.where(df['seasonYear'] < '2019-20', np.nan, rng.normal(size=n))      # empty for the early era
    df.loc[rng.random(n) < 0.15, 'random_gaps'] = np.nan                                          # missing at random
    return df


def test_era_marking_features_are_flagged_and_random_gaps_are_not():
    rep = missingness_era_report(frame(), ['always', 'random_gaps', 'odds_home_ml', 'not_a_column'])
    assert rep['feature'].tolist() == ['odds_home_ml']
    assert rep.loc[0, 'corr_with_season'] < -0.5                                                   # missing for OLD seasons
    assert rep.loc[0, 'missing_share'] == 0.5


def test_fully_present_frame_gives_an_empty_report():
    df = frame().drop(columns=['odds_home_ml', 'random_gaps'])
    assert missingness_era_report(df, ['always']).empty
