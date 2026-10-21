"""
Stage C (decision D6): per-minute rates, in two model families that are compared out of sample.

Components (all conditional on playing, exposure = minutes):
  counts   fg2a (two-point attempts), fg3a, fta, reb (rebounds), ast (assists)         Poisson rate per minute
  makes    p2, p3, pft: make probabilities for two-point, three-point and free-throw attempts   logistic, weighted by attempts
Points are built from these in the outcome stage (2 x fg2 makes + 3 x fg3 makes + ft makes), so their discreteness and their dispersion come from the
structure; threes made come from the same simulation, consistent with points.

Families:
  lgb   LightGBM. Counts: Poisson objective on count/minutes with weight = minutes (identical to a log-minutes offset). Makes: cross-entropy on made/attempts
        with weight = attempts.
  hier  regularized log-linear model (scikit-learn): the shrunk player rates of Phase 2 (already partially pooled toward league priors by position and role)
        enter in logs, with opponent, environment and availability terms as multiplicative adjustments; L2 strength chosen on the latest part of the fit window.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from props.models import common
from props.models.data import CATEGORICAL, FeatureSpec

COUNT_COMPONENTS = {'fg2a': lambda d: d['lbl_fga'] - d['lbl_fg3a'], 'fg3a': lambda d: d['lbl_fg3a'], 'fta': lambda d: d['lbl_fta'],
                    'reb': lambda d: d['lbl_rebounds'], 'ast': lambda d: d['lbl_assists']}
MAKE_COMPONENTS = {'p2': (lambda d: d['lbl_fgm'] - d['lbl_fg3m'], lambda d: d['lbl_fga'] - d['lbl_fg3a']),
                   'p3': (lambda d: d['lbl_fg3m'], lambda d: d['lbl_fg3a']),
                   'pft': (lambda d: d['lbl_ftm'], lambda d: d['lbl_fta'])}
BACKBONE = {  # the shrunk, partially pooled player rate per minute that the hierarchical model keeps fixed (an offset) and adjusts multiplicatively
    'fg2a': lambda d: (d['r_fga_h15'] - d['r_fg3a_h15']).clip(lower=1e-3), 'fg3a': lambda d: d['r_fg3a_h15'].clip(lower=1e-3),
    'fta': lambda d: d['r_fta_h15'].clip(lower=1e-3), 'reb': lambda d: d['r_rebounds_h15'].clip(lower=1e-3),
    'ast': lambda d: d['r_assists_h15'].clip(lower=1e-3)}
FAMILIES = ('lgb', 'hier')
ALPHAS = (1.0, 10.0, 100.0)            # L2 strengths tried for the hierarchical Poisson model (sklearn alpha)
CS = (0.01, 0.1, 1.0)                  # inverse L2 strengths tried for the hierarchical logistic model


class HierDesign:
    """Median-impute, log-transform the positive rate features, one-hot the categoricals, standardize: fitted on the fit rows only."""

    def __init__(self, cols):
        self.cols = [c for c in cols if c not in CATEGORICAL]
        self.cats = [c for c in cols if c in CATEGORICAL]
        self.logcols = [c for c in self.cols if c.startswith(('r_', 'c_', 'm_min')) and not c.startswith('m_min_sd')]

    def _raw(self, df):
        x = df[self.cols].astype(float).copy()
        for c in self.logcols:
            x[c] = np.log(np.clip(x[c], 1e-3, None))
        return x

    def fit(self, df):
        x = self._raw(df)
        self.median = x.median().fillna(0.0)
        x = x.fillna(self.median)
        self.mean, self.std = x.mean(), x.std().replace(0, 1.0)
        self.dummy_cols = [f'{c}={v}' for c in self.cats for v in df[c].cat.categories]
        return self

    def transform(self, df):
        x = (self._raw(df).fillna(self.median) - self.mean) / self.std
        d = pd.DataFrame({f'{c}={v}': (df[c].astype(str) == str(v)).astype(float).to_numpy() for c in self.cats for v in df[c].cat.categories},
                         index=x.index)
        return pd.concat([x, d], axis=1).to_numpy(dtype=float)


class RateModels:
    def __init__(self, spec: FeatureSpec = FeatureSpec(), params=None):
        self.spec, self.params = spec, params
        self.cols = None
        self.models = {}                       # (component, family) -> fitted object
        self.nb_k = {}                         # component -> negative-binomial k given minutes (method of moments on the fit window)
        self.info = {}

    # ---- helpers
    @staticmethod
    def _targets(df):
        t = {name: fn(df).to_numpy(float) for name, fn in COUNT_COMPONENTS.items()}
        for name, (num, den) in MAKE_COMPONENTS.items():
            t[name] = (num(df).to_numpy(float), den(df).to_numpy(float))
        return t

    def fit(self, df, families=FAMILIES):
        df = df[(df['lbl_played'] == 1) & (df['lbl_minutes'] > 0)].sort_values('game_date')
        self.cols = self.spec.columns(df.columns)
        self.spec.check(self.cols)
        X, dates = df[self.cols], df['game_date'].to_numpy()
        minutes, w = df['lbl_minutes'].to_numpy(float), df['weight'].to_numpy(float)
        T = self._targets(df)
        tr, va = common.split_by_date(dates)
        self.design = HierDesign(self.cols).fit(df.iloc[tr])
        Z = self.design.transform(df)
        for name in COUNT_COMPONENTS:
            y = T[name] / minutes
            wt = w * minutes
            b = BACKBONE[name](df).to_numpy(float)
            if 'lgb' in families:
                _, n = common.fit_booster('poisson', X, y, wt, dates, self.params)
                self.models[(name, 'lgb')] = common.refit('poisson', X, y, wt, n, self.params)
            if 'hier' in families:
                best = None
                # Poisson with offset log(minutes x backbone): fit the ratio count / (minutes x backbone) with weight minutes x backbone
                for a in ALPHAS:
                    m = PoissonRegressor(alpha=a, max_iter=400).fit(Z[tr], y[tr] / b[tr], sample_weight=wt[tr] * b[tr])
                    pred = np.clip(m.predict(Z[va]), 1e-6, None)
                    dev = np.sum(wt[va] * b[va] * (pred - (y[va] / b[va]) * np.log(pred))) / (wt[va] * b[va]).sum()
                    if best is None or dev < best[0]:
                        best = (dev, a)
                self.models[(name, 'hier')] = PoissonRegressor(alpha=best[1], max_iter=400).fit(Z, y / b, sample_weight=wt * b)
                common.log_config('rates_hier', component=name, alpha=best[1], spec=self.spec.name)
        for name, (num, den) in MAKE_COMPONENTS.items():
            made, att = T[name]
            ok = att > 0
            y = made[ok] / att[ok]
            wt = w[ok] * att[ok]
            if 'lgb' in families:
                _, n = common.fit_booster('cross_entropy', X.iloc[np.flatnonzero(ok)], y, wt, dates[ok], self.params)
                self.models[(name, 'lgb')] = common.refit('cross_entropy', X.iloc[np.flatnonzero(ok)], y, wt, n, self.params)
            if 'hier' in families:
                Zo = Z[ok]
                tr_o, va_o = common.split_by_date(dates[ok])
                best = None
                for C in CS:
                    m = self._logit(Zo[tr_o], made[ok][tr_o], att[ok][tr_o], w[ok][tr_o], C)
                    p = np.clip(m.predict_proba(Zo[va_o])[:, 1], 1e-4, 1 - 1e-4)
                    ll = -np.sum(wt[va_o] * (y[va_o] * np.log(p) + (1 - y[va_o]) * np.log(1 - p))) / wt[va_o].sum()
                    if best is None or ll < best[0]:
                        best = (ll, C)
                self.models[(name, 'hier')] = self._logit(Zo, made[ok], att[ok], w[ok], best[1])
                common.log_config('rates_hier', component=name, C=best[1], spec=self.spec.name)
        self.info = {'rows': int(len(df))}
        return self

    @staticmethod
    def _logit(Z, made, att, w, C):
        Zd = np.vstack([Z, Z])
        yd = np.r_[np.ones(len(Z)), np.zeros(len(Z))]
        wd = np.r_[made * w, (att - made) * w]
        keep = wd > 0
        return LogisticRegression(C=C, max_iter=300).fit(Zd[keep], yd[keep], sample_weight=wd[keep])

    def fit_dispersion(self, df_calib, family_by_component):
        """
        Negative-binomial k per count component (Var = mu + mu^2 / k) from the CALIBRATION rows, which the rate models never saw, with the actual minutes as
        exposure. Pooled method of moments; k is capped at 5000 where the counts are no more variable than Poisson.
        """
        df = df_calib[(df_calib['lbl_played'] == 1) & (df_calib['lbl_minutes'] > 0)]
        rates = self.predict(df, family_by_component)
        minutes = df['lbl_minutes'].to_numpy(float)
        T = self._targets(df)
        for name in COUNT_COMPONENTS:
            mu = rates[name].to_numpy() * minutes
            excess = np.sum((T[name] - mu) ** 2 - mu)
            self.nb_k[name] = float(np.clip(np.sum(mu ** 2) / excess, 0.5, 5000.0)) if excess > 0 else 5000.0
        self.info['nb_k'] = {k: round(v, 1) for k, v in self.nb_k.items()}
        return self.nb_k

    # ---- prediction
    def _rate(self, name, family, X, Z, df):
        m = self.models[(name, family)]
        if family == 'lgb':
            return m.predict(X)
        if name in COUNT_COMPONENTS:
            return BACKBONE[name](df).to_numpy(float) * m.predict(Z)
        return m.predict_proba(Z)[:, 1]

    def predict(self, df, family_by_component):
        """Per-minute count rates and make probabilities for each row, using the chosen family per component."""
        X = df[self.cols]
        need_hier = any(f == 'hier' for f in family_by_component.values())
        Z = self.design.transform(df) if need_hier else None
        out = {}
        for name in list(COUNT_COMPONENTS) + list(MAKE_COMPONENTS):
            fam = family_by_component[name]
            r = self._rate(name, fam, X, Z, df)
            out[name] = np.clip(r, 1e-4, 0.999 if name in MAKE_COMPONENTS else 20.0)
        return pd.DataFrame(out, index=df.index)

    def score(self, df, family):
        """Out-of-sample component losses with the ACTUAL minutes as exposure (isolates the rate models from the minutes model)."""
        df = df[(df['lbl_played'] == 1) & (df['lbl_minutes'] > 0)]
        rates = self.predict(df, {c: family for c in list(COUNT_COMPONENTS) + list(MAKE_COMPONENTS)})
        minutes, w = df['lbl_minutes'].to_numpy(float), df['weight'].to_numpy(float)
        T = self._targets(df)
        out = {}
        for name in COUNT_COMPONENTS:
            mu = rates[name].to_numpy() * minutes
            y = T[name]
            dev = 2 * np.sum(w * (np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0.0) - (y - mu))) / w.sum()
            out[name] = dev
        for name in MAKE_COMPONENTS:
            made, att = T[name]
            ok = att > 0
            p = rates[name].to_numpy()[ok]
            ll = -np.sum(w[ok] * (made[ok] * np.log(p) + (att[ok] - made[ok]) * np.log(1 - p))) / np.sum(w[ok] * att[ok])
            out[name] = ll
        return out
