"""A small synthetic league with known minutes and rates, shaped like the Phase 2 model frame (the columns the models read)."""
import numpy as np
import pandas as pd

from props.models import data


def synthetic_frame(n_players=40, days_per_season=45, seasons=('2019-20', '2020-21', '2021-22'), seed=0, limited_share=0.10):
    rng = np.random.default_rng(seed)
    base_min = rng.uniform(12, 36, n_players)
    skill = rng.uniform(0.6, 1.4, n_players)
    pos = rng.choice(data.POS_GROUPS, n_players)
    rows = []
    gid = 0
    for si, season in enumerate(seasons):
        start = pd.Timestamp(f'{2019 + si}-10-25')
        for day in range(days_per_season):
            date = (start + pd.Timedelta(days=2 * day)).strftime('%Y-%m-%d')
            for p in range(n_players):
                gid += 1
                playing = rng.random() < 0.9
                limited = playing and rng.random() < limited_share
                minutes = 0.0
                if playing:
                    minutes = float(np.clip(rng.normal(base_min[p], 2.5) * (rng.uniform(0.2, 0.5) if limited else 1.0), 1.0, 48.0))
                r2, r3, rf = 0.20 * skill[p], 0.08 * skill[p], 0.06 * skill[p]
                rr, ra = 0.15 * skill[p] ** 0.5, 0.09 * skill[p]
                fg2a, fg3a, fta = rng.poisson(r2 * minutes), rng.poisson(r3 * minutes), rng.poisson(rf * minutes)
                fg2m, fg3m, ftm = rng.binomial(fg2a, 0.52), rng.binomial(fg3a, 0.36), rng.binomial(fta, 0.78)
                reb, ast = rng.poisson(rr * minutes), rng.poisson(ra * minutes)
                pts = 2 * fg2m + 3 * fg3m + ftm
                nan = np.nan
                for t in (240, 60, 5):
                    rows.append({
                        'game_id': f'{gid:010d}', 'player_id': p, 'minutes_before': t, 'season': season, 'game_date': date, 'weight': 1 / 3,
                        'lbl_played': int(playing), 'lbl_minutes': minutes if playing else nan,
                        'lbl_fga': fg2a + fg3a if playing else nan, 'lbl_fg3a': fg3a if playing else nan, 'lbl_fta': fta if playing else nan,
                        'lbl_fgm': fg2m + fg3m if playing else nan, 'lbl_fg3m': fg3m if playing else nan, 'lbl_ftm': ftm if playing else nan,
                        'lbl_rebounds': reb if playing else nan, 'lbl_assists': ast if playing else nan,
                        'y_points': pts if playing else nan, 'y_rebounds': reb if playing else nan, 'y_assists': ast if playing else nan,
                        'y_threes': fg3m if playing else nan, 'y_pra': pts + reb + ast if playing else nan,
                        # features known before the game: noisy versions of the truth
                        'm_min_h15': base_min[p] + rng.normal(0, 1.0), 'role_p_played_h15': 0.9 + rng.normal(0, 0.02),
                        'inj_p_play': 0.9, 'inj_status_code': 0, 'inj_report_age_min': 60.0 + rng.normal(0, 5),
                        'r_fga_h15': (r2 + r3) * (1 + rng.normal(0, 0.05)), 'r_fg3a_h15': r3 * (1 + rng.normal(0, 0.05)),
                        'r_fta_h15': rf * (1 + rng.normal(0, 0.05)), 'r_rebounds_h15': rr * (1 + rng.normal(0, 0.05)),
                        'r_assists_h15': ra * (1 + rng.normal(0, 0.05)), 'r_points_h15': 2 * r2 * 0.52 + 3 * r3 * 0.36 + rf * 0.78,
                        'r_fg3m_h15': r3 * 0.36, 'c_fg3_pct_h15': 0.36 + rng.normal(0, 0.02), 'tm_off_rtg': 110 + rng.normal(0, 3),
                        'pos_group': pos[p], 'inj_reason_class': 'none'})
    df = pd.DataFrame(rows)
    df = data.encode_categoricals(df)
    return df.sort_values(['game_date', 'game_id', 'minutes_before']).reset_index(drop=True)
