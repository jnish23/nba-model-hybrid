"""Small synthetic game history (opening lines, drifting scoring level) shared by the training tests."""
import numpy as np
import pandas as pd

SEASONS = [('2020-21', '2020-10-20'), ('2021-22', '2021-10-20'), ('2022-23', '2022-10-20'), ('2023-24', '2023-10-20'),
           ('2024-25', '2024-10-20')]


def training_csv(path, seed=0, per_season=500, drift=10.0):
    """Games with opening spread / total; total level rises `drift` points per season; x1 predicts the total gap, x0 the spread gap."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, (season, start) in enumerate(SEASONS):
        n = per_season
        dates = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) // 4, unit='D')
        x = rng.normal(size=(n, 4))
        level = 205 + drift * k
        line_total = np.round(level + rng.normal(0, 6, n) + 3 * x[:, 1] * 0)
        line_spread = np.round(rng.normal(-2, 6, n) * 2) / 2
        margin = -line_spread + 2.5 * x[:, 0] + rng.normal(0, 11, n)
        total = line_total + 3.0 * x[:, 1] + rng.normal(0, 14, n)
        df = pd.DataFrame({'gameDate': dates, 'seasonYear': season, 'gameId': [f'{season}{i:05d}' for i in range(n)],
                           'teamId_home': 1, 'teamId_away': 2, 'target_point_diff': margin, 'target_total_pts': total,
                           'target_home_win': (margin > 0).astype(int), 'points_home': 100, 'points_away': 100,
                           'odds_home_spread': line_spread, 'odds_total': line_total,
                           'odds_home_ml': np.where(line_spread < 0, -150, 130), 'odds_away_ml': np.where(line_spread < 0, 130, -150)})
        for j in range(4):
            df[f'diff_f{j}'] = x[:, j]
            df[f'f{j}_home'] = x[:, j]
            df[f'f{j}_away'] = rng.normal(size=n)
        rows.append(df)
    pd.concat(rows).to_csv(path, index=False)
