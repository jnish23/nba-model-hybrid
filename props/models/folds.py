"""
Walk-forward folds (decision D8): expanding windows, one fold per test season.

For a test season S the training window is every season before S (from FIRST_SEASON), minus the 2019-20 bubble games, which are excluded from
training only. The window is split by DATE: the first (1 - CALIB_FRAC) of its game dates fit the models, the last CALIB_FRAC calibrate them.
Test rows are strictly later than every training and calibration row. Nothing here ever looks at labels.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from props.models.data import BUBBLE_START

FIRST_SEASON = '2019-20'
CALIB_FRAC = 0.20
DEV_TEST_SEASONS = ('2021-22', '2022-23', '2023-24')       # development folds
VALIDATION_SEASON = '2024-25'
FINAL_SEASON = '2025-26'                                    # touched only after the configuration is frozen


@dataclass
class Fold:
    test_season: str
    fit_idx: np.ndarray
    calib_idx: np.ndarray
    test_idx: np.ndarray
    cutoff_date: str

    def describe(self):
        return {'test_season': self.test_season, 'fit_rows': int(len(self.fit_idx)), 'calib_rows': int(len(self.calib_idx)),
                'test_rows': int(len(self.test_idx)), 'calibration_starts': self.cutoff_date}


def make_fold(frame, test_season, first_season=FIRST_SEASON, calib_frac=CALIB_FRAC, exclude_bubble=True):
    """frame needs season and game_date. Returns positional index arrays into the frame."""
    season = frame['season'].to_numpy()
    date = frame['game_date'].to_numpy()
    train = (season >= first_season) & (season < test_season)
    if exclude_bubble:
        train &= ~((season == '2019-20') & (date >= BUBBLE_START))
    dates = np.unique(date[train])
    if len(dates) < 10:
        raise ValueError(f'training window for {test_season} has only {len(dates)} game dates')
    cutoff = dates[int(np.floor(len(dates) * (1 - calib_frac)))]
    fold = Fold(test_season, np.flatnonzero(train & (date < cutoff)), np.flatnonzero(train & (date >= cutoff)),
                np.flatnonzero(season == test_season), str(cutoff))
    assert_fold_integrity(frame, fold)
    return fold


def assert_fold_integrity(frame, fold):
    d = frame['game_date'].to_numpy()
    if not (len(fold.fit_idx) and len(fold.calib_idx) and len(fold.test_idx)):
        raise AssertionError('an empty part of the fold')
    if not d[fold.fit_idx].max() < d[fold.calib_idx].min():
        raise AssertionError('fit rows are not strictly before calibration rows')
    if not d[fold.calib_idx].max() < d[fold.test_idx].min():
        raise AssertionError('calibration rows are not strictly before test rows')
    if set(fold.fit_idx) & set(fold.test_idx) or set(fold.calib_idx) & set(fold.test_idx):
        raise AssertionError('training rows overlap the test rows')
    s = frame['season'].to_numpy()
    if not (s[fold.test_idx] == fold.test_season).all() or not (s[fold.fit_idx] < fold.test_season).all():
        raise AssertionError('seasons are not walk-forward')
