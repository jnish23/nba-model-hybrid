"""
Stage A: probability that the player plays, at each decision time. Trained on every listed row. Baseline: the per-status table probability from
Phase 2 (`inj_p_play`); the model also sees role, rest, teammates, report age and the other features, so it can disagree with the table.
Its output is only used to report P(play) next to the conditional distributions; the outcome distribution itself is conditional on playing.
"""
import numpy as np

from props.models import common
from props.models.data import FeatureSpec


class Availability:
    def __init__(self, spec: FeatureSpec = FeatureSpec(), params=None):
        self.spec, self.params = spec, params
        self.booster = None
        self.cols = None

    def fit(self, df):
        """df: the fit rows (all listed rows)."""
        self.cols = self.spec.columns(df.columns)
        self.spec.check(self.cols)
        X, y, w = df[self.cols], df['lbl_played'].to_numpy(float), df['weight'].to_numpy(float)
        b, n = common.fit_booster('binary', X, y, w, df['game_date'].to_numpy(), self.params)
        self.booster = common.refit('binary', X, y, w, n, self.params)
        common.log_config('availability', spec=self.spec.name, exclude=sorted(self.spec.exclude_groups), rounds=n, n_rows=len(df))
        return self

    def predict(self, df):
        return np.clip(self.booster.predict(df[self.cols]), 1e-4, 1 - 1e-4)
