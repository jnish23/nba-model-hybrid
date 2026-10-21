"""
R2: rolling recalibration of the predicted mean (pre-registered in reports/props/phase4_plan.md, constants fixed before any evaluation).

For a target game on date d and a stat, the ratio of observed to predicted over the TRAILING WINDOW of completed games is computed from the rows whose
game_date is in [d - WINDOW_DAYS, d - 1] (strictly earlier dates: nothing from the target date, so nothing that could have been unfinished at any decision
time), shrunk toward 1 with a prior worth PRIOR_PLAYER_GAMES player-games:

    r = (sum_w y + K m) / (sum_w mu + K m)          m = mean predicted value in the window (r = 1 when the window is empty)

where w = 1/3 per row (each player-game appears at three decision times). Every pmf of that date is then rescaled by r, a mean-scaling of the
piecewise-linear CDF (bin y spans y-0.5 .. y+0.5), which scales the mean by r and the sd by r.
"""
import numpy as np
import pandas as pd

from props.models import outcome as oc

WINDOW_DAYS = 21
PRIOR_PLAYER_GAMES = 2000.0
ROW_WEIGHT = 1.0 / 3.0


def scale_pmf(pmf, r):
    """Rescale each row's distribution by r[i] (r > 0), keeping the support 0..cap."""
    pmf = np.asarray(pmf, np.float64)
    n, width = pmf.shape
    r = np.broadcast_to(np.asarray(r, float), (n,)).copy()
    if np.all(r == 1.0):
        return pmf.astype(np.float32)
    grid = np.concatenate([np.zeros((n, 1)), np.cumsum(pmf, axis=1)], axis=1)
    grid[:, -1] = 1.0
    ys = np.arange(width)[None, :]
    pos = np.clip((ys + 0.5) / r[:, None] + 0.5, 0.0, width)                       # position of the scaled bin edge on the original grid (edge k is at k - 0.5)
    lo = np.minimum(np.floor(pos).astype(int), width - 1)
    frac = pos - lo
    cdf = np.take_along_axis(grid, lo, axis=1) * (1 - frac) + np.take_along_axis(grid, lo + 1, axis=1) * frac
    cdf[:, -1] = 1.0
    out = np.maximum(np.diff(np.concatenate([np.zeros((n, 1)), cdf], axis=1), axis=1), 0.0)
    return (out / out.sum(axis=1, keepdims=True)).astype(np.float32)


def rolling_ratio(dates, obs, mu, window_days=WINDOW_DAYS, prior=PRIOR_PLAYER_GAMES, w=ROW_WEIGHT):
    """Per-row shrunk ratio using only rows dated strictly before the row's own date and within window_days of it."""
    d = pd.to_datetime(pd.Series(np.asarray(dates))).dt.normalize().to_numpy()
    daily = pd.DataFrame({'d': d, 'y': np.asarray(obs, float), 'mu': np.asarray(mu, float)}).groupby('d').agg(y=('y', 'sum'), mu=('mu', 'sum'), n=('y', 'size'))
    full = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(full, fill_value=0.0)
    roll = daily.rolling(window_days, min_periods=1).sum().shift(1).fillna(0.0)   # window ends the day before
    sy, smu, n = roll['y'] * w, roll['mu'] * w, roll['n'] * w
    m = np.where(n > 0, smu / np.maximum(n, 1e-9), 0.0)
    ratio = (sy + prior * m) / (smu + prior * m).where((smu + prior * m) > 0, np.nan)
    ratio = ratio.fillna(1.0)
    return pd.Series(d).map(ratio).to_numpy(float)


class RollingRecalibrator:
    def __init__(self, window_days=WINDOW_DAYS, prior=PRIOR_PLAYER_GAMES):
        self.window_days, self.prior = window_days, prior

    def apply(self, stat, pmf, dates, obs):
        """Recalibrated pmfs for the rows of one test season (obs are the realized values of the rows; row i's ratio never uses a row dated on or after its own date)."""
        r = rolling_ratio(dates, obs, oc.pmf_mean(pmf), self.window_days, self.prior)
        return scale_pmf(pmf, r), r


# ---------------------------------------------------------------------------------------------------------------------------- R2b
# Pre-registered variant (user-approved 2026-10-20), constants fixed before any result: the window is the previous WINDOW_GAME_DAYS distinct dates with games
# (so it reaches across the off-season: the previous season's last completed game-days count at the start of a season), the prior is worth
# PRIOR_PLAYER_GAMES_B player-games, everything else as R2. Same adoption rule as R2.
WINDOW_GAME_DAYS = 21
PRIOR_PLAYER_GAMES_B = 1000.0


def rolling_ratio_gamedays(dates, obs, mu, carry=None, window_game_days=WINDOW_GAME_DAYS, prior=PRIOR_PLAYER_GAMES_B, w=ROW_WEIGHT):
    """
    Per-row shrunk ratio over the previous `window_game_days` distinct game dates strictly before the row's own date. `carry` = (dates, obs, mu) of earlier rows
    (the previous season's forecasts, made by that season's own models) that may enter the window; only the rows of `dates` get a ratio back.
    """
    d = pd.to_datetime(pd.Series(np.asarray(dates))).dt.normalize().to_numpy()
    frames = [pd.DataFrame({'d': d, 'y': np.asarray(obs, float), 'mu': np.asarray(mu, float)})]
    if carry is not None:
        cd, co, cm = carry
        frames.insert(0, pd.DataFrame({'d': pd.to_datetime(pd.Series(np.asarray(cd))).dt.normalize().to_numpy(), 'y': np.asarray(co, float), 'mu': np.asarray(cm, float)}))
    daily = pd.concat(frames).groupby('d').agg(y=('y', 'sum'), mu=('mu', 'sum'), n=('y', 'size')).sort_index()
    cs = daily.cumsum()
    pad = pd.DataFrame({'y': [0.0], 'mu': [0.0], 'n': [0.0]})
    cs = pd.concat([pad, cs.reset_index(drop=True)], ignore_index=True)           # cs[i] = totals of the first i game dates
    i = np.arange(len(daily))
    lo = np.maximum(i - window_game_days, 0)
    sy, smu, n = [(cs[c].to_numpy()[i] - cs[c].to_numpy()[lo]) * w for c in ('y', 'mu', 'n')]      # dates lo .. i-1: strictly before date i
    m = np.where(n > 0, smu / np.maximum(n, 1e-9), 0.0)
    num, den = sy + prior * m, smu + prior * m
    ratio = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0), index=daily.index)
    return pd.Series(d).map(ratio).to_numpy(float)


class RollingRecalibratorB:
    def apply(self, stat, pmf, dates, obs, carry=None):
        r = rolling_ratio_gamedays(dates, obs, oc.pmf_mean(pmf), carry)
        return scale_pmf(pmf, r), r
