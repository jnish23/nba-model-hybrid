"""
Stage B (decision D5): minutes given that the player plays, as a two-regime mixture.

  regime "limited": minutes under LIMITED_RATIO x the player's trailing expectation (m_min_h15), defined for players with at least MIN_REF expected minutes
                    (8.9% of played games in 2019-24): early exits, injuries in game, rest or foul trouble.
  P(limited)       a LightGBM classifier.
  per regime       a LightGBM mean, a LightGBM spread (mean absolute error), and an empirical bank of standardized residuals (out-of-fold, inside the
                   training window) by predicted-minutes bucket; sampling the bank gives the regime's whole shape, tails included.

distribution(df) returns, per row, 2 x S support points with weights (S per regime, weights (1 - p) / S and p / S), so any later stage can take exact
weighted expectations over minutes without simulation noise.
"""
import numpy as np

from props.models import common
from props.models.data import FeatureSpec

LIMITED_RATIO = 0.5
MIN_REF = 8.0
S = 64                                         # support points per regime
BUCKET_EDGES = (15.0, 28.0)                    # predicted-minutes buckets for the residual banks: <15, 15-28, 28+
MIN_SD = 0.75
OOF_BLOCKS = 5                                  # forward-chaining blocks inside the fit window (block 0 only trains)
MIN_BANK_ROWS = 300


def _levels(n):
    return (np.arange(n) + 0.5) / n


class MinutesModel:
    def __init__(self, spec: FeatureSpec = FeatureSpec(), params=None):
        self.spec, self.params = spec, params
        self.cols = None
        self.clf = None
        self.reg = {}                                      # regime -> dict(mean=booster, spread=booster, bank=array(buckets, S))

    @staticmethod
    def limited_label(df):
        ref = df['m_min_h15'].to_numpy(float)
        defined = ref >= MIN_REF
        return defined & (df['lbl_minutes'].to_numpy(float) < LIMITED_RATIO * ref), defined

    def _bucket(self, mu):
        return np.digitize(mu, BUCKET_EDGES)

    def fit(self, df):
        """df: played rows of the fit window with some trailing history (m_min_h15 not null)."""
        df = df[(df['lbl_played'] == 1) & df['m_min_h15'].notna()].sort_values('game_date')
        self.cols = self.spec.columns(df.columns)
        self.spec.check(self.cols)
        X, y, w, dates = df[self.cols], df['lbl_minutes'].to_numpy(float), df['weight'].to_numpy(float), df['game_date'].to_numpy()
        lim, defined = self.limited_label(df)
        # P(limited): only players with an expectation worth comparing against
        idx = np.flatnonzero(defined)
        _, n = common.fit_booster('binary', X.iloc[idx], lim[idx].astype(float), w[idx], dates[idx], self.params)
        self.clf = common.refit('binary', X.iloc[idx], lim[idx].astype(float), w[idx], n, self.params)
        info = {'clf_rounds': n, 'limited_share': float(np.average(lim[idx], weights=w[idx]))}
        for name, sel in (('regular', np.flatnonzero(~lim)), ('limited', np.flatnonzero(lim))):
            Xs, ys, ws, ds = X.iloc[sel], y[sel], w[sel], dates[sel]
            _, nm = common.fit_booster('regression', Xs, ys, ws, ds, self.params)
            mean = common.refit('regression', Xs, ys, ws, nm, self.params)
            # forward-chaining residuals (each block predicted from earlier blocks only) and an OUT-OF-SAMPLE sd for each (the spread model that scales a residual
            # never saw that row or any later one), so the standardized residuals in the bank are not narrowed by in-sample fitting
            oof = common.forward_oof('regression', Xs, ys, ws, ds, nm, n_blocks=OOF_BLOCKS, params=self.params)
            res = ys - oof
            ok = ~np.isnan(res)
            _, ns = common.fit_booster('regression', Xs.iloc[np.flatnonzero(ok)], np.abs(res[ok]), ws[ok], ds[ok], self.params)
            spread = common.refit('regression', Xs.iloc[np.flatnonzero(ok)], np.abs(res[ok]), ws[ok], ns, self.params)
            sd_oof = common.forward_oof('regression', Xs, np.where(ok, np.abs(res), 0.0), ws, ds, ns, n_blocks=OOF_BLOCKS, params=self.params, usable=ok)
            ok = ok & ~np.isnan(sd_oof)
            if ok.sum() < MIN_BANK_ROWS:
                raise ValueError(f'regime {name}: only {int(ok.sum())} rows with forward-chaining residuals and out-of-sample sd (need {MIN_BANK_ROWS})')
            sd = np.maximum(sd_oof[ok], MIN_SD)
            z = res[ok] / sd
            mu_oof = oof[ok]
            bank = np.vstack([np.quantile(z[self._bucket(mu_oof) == b], _levels(S)) if (self._bucket(mu_oof) == b).sum() >= 50
                              else np.quantile(z, _levels(S)) for b in range(len(BUCKET_EDGES) + 1)])
            self.reg[name] = {'mean': mean, 'spread': spread, 'bank': bank}
            info[name] = {'rows': int(len(sel)), 'mean_rounds': nm, 'spread_rounds': ns}
        common.log_config('minutes', spec=self.spec.name, exclude=sorted(self.spec.exclude_groups), **info)
        self.info = info
        return self

    def components(self, df):
        X = df[self.cols]
        out = {'p_limited': np.where(df['m_min_h15'].to_numpy(float) >= MIN_REF, np.clip(self.clf.predict(X), 0.001, 0.95), 0.0)}
        for name in ('regular', 'limited'):
            m = self.reg[name]
            out[f'mu_{name}'] = np.clip(m['mean'].predict(X), 0.5, 60.0)
            out[f'sd_{name}'] = np.maximum(m['spread'].predict(X), MIN_SD)
        return out

    def distribution(self, df, comps=None):
        """(values, weights): arrays (n, 2S). Values are minutes in [0.5, 60]."""
        c = self.components(df) if comps is None else comps
        n = len(df)
        vals = np.empty((n, 2 * S))
        wts = np.empty((n, 2 * S))
        p = c['p_limited']
        for k, name in enumerate(('regular', 'limited')):
            mu, sd = c[f'mu_{name}'], c[f'sd_{name}']
            bank = self.reg[name]['bank'][self._bucket(mu)]                 # (n, S)
            v = np.clip(mu[:, None] + sd[:, None] * bank, 0.5, 60.0)
            vals[:, k * S:(k + 1) * S] = v
            wts[:, k * S:(k + 1) * S] = ((1 - p) if k == 0 else p)[:, None] / S
        return vals, wts

    @staticmethod
    def mean_of(vals, wts):
        return (vals * wts).sum(axis=1)
