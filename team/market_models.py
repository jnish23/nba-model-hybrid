"""
Market-aware model components adopted from the E1 / E3 / E4 experiments (reports/experiments/FINAL_REPORT.md).

    E1  ResidualSpreadModel   the spread regressor learns margin + opening_home_spread and predicts
                              -opening_home_spread + f(X), i.e. it only has to explain what the opening line misses
    E3  MarginStackCalibrator the ML probability is a logistic stack of [logit(raw ML prob), spread margin / sigma]
                              fitted on the calibration block (replaces the isotonic step; contradictions with the
                              spread model drop from 6.6% to 2.7% of games)
    E4  HeteroSigma           the total's cover probability uses a per-game sigma from a second model on
                              out-of-fold absolute residuals instead of one global std

All three need the OPENING lines as inputs, so they belong to the deployed with-odds model set only. Everything here
is picklable (joblib) and a model set trained before this module existed (plain regressors + isotonic calibrator)
keeps working through predict_markets().
"""
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

SEED = 42
SPREAD_LINE_COL = 'odds_home_spread'
SIGMA_MIN_ABS_ERR = 1.0            # floor on the predicted mean |error| (points)
SIGMA_FOLDS = 5                    # forward-chaining parts inside model_train for the out-of-fold residuals
SIGMA_FEATURES = [
    'off_fieldGoalsAttempted_20g_home', 'off_fieldGoalsAttempted_20g_away', 'off_fieldGoalsAttempted_5g_home', 'off_fieldGoalsAttempted_5g_away',
    'off_percentageFieldGoalsAttempted3pt_20g_home', 'off_percentageFieldGoalsAttempted3pt_20g_away',
    'off_threePointersAttempted_20g_home', 'off_threePointersAttempted_20g_away',
    'inj_n_out_home', 'inj_n_out_away', 'inj_missing_home', 'inj_missing_away', 'inj_lost_home', 'inj_lost_away',
    'odds_home_spread', 'odds_total', 'days_rest_home', 'days_rest_away', 'season_game_number_home']
SIGMA_PARAMS = dict(learning_rate=0.05, n_estimators=150, num_leaves=4, min_child_samples=100,
                    reg_lambda=20.0, colsample_bytree=0.8)


def _logit(p):
    return logit(np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4))


# ------------------------------------------------------------------------------------------------ E1

class ResidualSpreadModel:
    """LightGBM regressor on the market residual `target_point_diff + odds_home_spread`.

    predict(X, line) = -line + f(X), where `line` is the opening home spread (home favoured = negative). The line is
    passed separately because it need not be one of the model's selected features; with line=None it is read from
    X[SPREAD_LINE_COL]. A missing line gives NaN (no spread prediction without an opening line).
    """

    def __init__(self, model):
        self.model = model

    @property
    def feature_name_(self):                       # lets team/model_registry.py check the config against the pickle
        return list(self.model.feature_name_)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @staticmethod
    def make_target(df):
        return df['target_point_diff'] + df[SPREAD_LINE_COL]

    def predict_residual(self, X):
        return self.model.predict(X)

    def predict(self, X, line=None):
        if line is None:
            if SPREAD_LINE_COL not in X.columns:
                raise ValueError(f"ResidualSpreadModel.predict needs the opening spread: pass line= or include {SPREAD_LINE_COL}")
            line = X[SPREAD_LINE_COL]
        return -np.asarray(line, float) + self.model.predict(X)


# ------------------------------------------------------------------------------------------------ E3

class MarginStackCalibrator:
    """P(home win) = logistic([logit(raw ML prob), margin / sigma]) fitted on the calibration block.

    A second logistic on logit(raw) alone (Platt scaling, E2) is kept for games without a margin (no opening spread).
    """
    uses_margin = True

    def __init__(self, C=1.0):
        self.C = C
        self.stack = None
        self.platt = None
        self.sigma = None

    def fit(self, raw_prob, margin, sigma, y):
        raw_prob, margin, y = (np.asarray(a, float) for a in (raw_prob, margin, y))
        self.sigma = float(sigma)
        ok = np.isfinite(margin)
        self.stack = LogisticRegression(C=self.C).fit(np.column_stack([_logit(raw_prob[ok]), margin[ok] / self.sigma]), y[ok])
        self.platt = LogisticRegression(C=1e6).fit(_logit(raw_prob).reshape(-1, 1), y)
        return self

    def predict(self, raw_prob, margin=None):
        raw_prob = np.asarray(raw_prob, float)
        out = self.platt.predict_proba(_logit(raw_prob).reshape(-1, 1))[:, 1]
        if margin is not None:
            margin = np.asarray(margin, float)
            ok = np.isfinite(margin)
            if ok.any():
                out[ok] = self.stack.predict_proba(np.column_stack([_logit(raw_prob[ok]), margin[ok] / self.sigma]))[:, 1]
        return out


# ------------------------------------------------------------------------------------------------ E4

class HeteroSigma:
    """Per-game residual std: sqrt(pi/2) * E[|error| | pre-game features], from a LightGBM on out-of-fold |errors|."""

    def __init__(self, model, features, fallback_std):
        self.model, self.features, self.fallback_std = model, list(features), float(fallback_std)

    def predict(self, X):
        Xf = X.reindex(columns=self.features).astype(float)
        e_abs = np.clip(self.model.predict(Xf), SIGMA_MIN_ABS_ERR, None)
        sigma = e_abs * np.sqrt(np.pi / 2)
        return np.where(np.isfinite(sigma), sigma, self.fallback_std)


def oof_abs_errors(df, features, target_col, params, n_parts=SIGMA_FOLDS):
    """Forward-chaining out-of-fold |error| inside the (date-ordered) model-train block: part k+1 is predicted by a model
    fitted on parts 1..k, so no row is scored by a model that saw it or its future."""
    parts = np.array_split(np.arange(len(df)), n_parts)
    idx, err = [], []
    for k in range(1, n_parts):
        tr, te = np.concatenate(parts[:k]), parts[k]
        est = lgb.LGBMRegressor(**params, verbosity=-1, random_state=SEED)
        est.fit(df.iloc[tr][features], df.iloc[tr][target_col])
        err.append(np.abs(df.iloc[te][target_col].to_numpy(float) - est.predict(df.iloc[te][features])))
        idx.append(te)
    return np.concatenate(idx), np.concatenate(err)


def fit_hetero_sigma(model_train_df, features, target_col, params, fallback_std, sigma_features=SIGMA_FEATURES):
    """Fit the sigma model on out-of-fold errors of a model with `params`/`features`. `model_train_df` must be sorted by date."""
    sf = [c for c in sigma_features if c in model_train_df.columns]
    if len(sf) < len(sigma_features):
        logging.warning(f"HeteroSigma: {len(sigma_features) - len(sf)} sigma features are not in the training set: "
                        f"{[c for c in sigma_features if c not in sf]}")
    idx, err = oof_abs_errors(model_train_df, features, target_col, params)
    g = lgb.LGBMRegressor(**SIGMA_PARAMS, verbosity=-1, random_state=SEED, deterministic=True, force_row_wise=True)
    g.fit(model_train_df.iloc[idx][sf].astype(float), err)
    return HeteroSigma(g, sf, fallback_std)


# ------------------------------------------------------------------------------------------------ inference

def predict_markets(bundle, frame, align=None):
    """Point forecasts and probabilities for the rows of `frame` (the live frame or a training-set slice).

    Works for both model-set generations: plain regressors + isotonic calibrator (pre-E1/E3/E4) and the market-aware
    set. `align(frame, name, cols)` selects a model's columns (live code passes align_features; default = plain
    selection). Returns a DataFrame: ml_prob_raw, ml_prob, spread_pred, total_pred, spread_std, total_std.
    """
    def cols(name, key):
        c = bundle.features[key]
        return align(frame, name, c) if align is not None else frame[c]

    raw = bundle.ml.predict_proba(cols('ML', 'ml'))[:, 1]
    X_spread = cols('Spread', 'spread')
    if isinstance(bundle.spread, ResidualSpreadModel):
        line = frame[SPREAD_LINE_COL].to_numpy(float) if SPREAD_LINE_COL in frame.columns else np.full(len(frame), np.nan)
        spread = bundle.spread.predict(X_spread, line=line)
    else:
        spread = bundle.spread.predict(X_spread)
    X_total = cols('Total', 'total')
    total = bundle.total.predict(X_total)
    meta = bundle.calib_meta
    spread_std = np.full(len(frame), float(meta.get('spread_std', np.nan)))
    total_std = np.full(len(frame), float(meta.get('total_std', np.nan)))
    sigma_model = getattr(bundle, 'sigma_total', None)
    if sigma_model is not None:
        total_std = sigma_model.predict(frame)
    if getattr(bundle.calibrator, 'uses_margin', False):
        prob = bundle.calibrator.predict(raw, margin=spread)
    else:
        prob = bundle.calibrator.predict(raw)
    return pd.DataFrame({'ml_prob_raw': raw, 'ml_prob': prob, 'spread_pred': spread, 'total_pred': total,
                         'spread_std': spread_std, 'total_std': total_std}, index=frame.index)
