# E29x_F1pace_total

Arm columns (18): ['trk_dist_per_min_last_home', 'trk_dist_per_min_last_away', 'diff_trk_dist_per_min_last', 'trk_dist_per_poss_last_home', 'trk_dist_per_poss_last_away', 'diff_trk_dist_per_poss_last', 'trk_xload_last_home', 'trk_xload_last_away', 'diff_trk_xload_last', 'trk_xload_4n_home', 'trk_xload_4n_away', 'diff_trk_xload_4n', 'trk_rot_xload_last_home', 'trk_rot_xload_last_away', 'diff_trk_rot_xload_last', 'trk_prev_opp_pace_home', 'trk_prev_opp_pace_away', 'diff_trk_prev_opp_pace']
Registered decision-bearing arm: False. Reference: `E25_with_odds`.

## Validity checks

{"same_games_same_order": true, "ml_identical": true, "spread_identical": true}

## Paired comparison vs the reference (exp - ref; negative = better)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |           0       | [0.00000, 0.00000]  | 2/7                | False                    |
| ML Brier    | 8916 |           0       | [-0.00000, 0.00000] | 5/7                | False                    |
| Spread MAE  | 8916 |           0       | [-0.00000, 0.00000] | 4/7                | False                    |
| Spread MSE  | 8916 |          -0       | [-0.00000, 0.00000] | 5/7                | False                    |
| Total MAE   | 8916 |           0.01757 | [0.00205, 0.03531]  | 1/7                | False                    |
| Total MSE   | 8916 |           1.24393 | [0.58573, 1.90722]  | 1/7                | False                    |

## Per season, total MAE difference

{"2019-20": 0.0471, "2020-21": -0.0179, "2021-22": 0.0033, "2022-23": 0.0481, "2023-24": 0.023, "2024-25": 0.0098, "2025-26": 0.0097}

Noise floor for total MAE (largest seed-only difference): 0.005. Pooled difference +0.01757: exceeds the floor.

## Share of total-model gain taken by the arm's columns, per fold

- 2019-20: 0.043  (top: [('odds_total', np.float64(0.7465)), ('off_points_5g_away', np.float64(0.0514)), ('def_OPP_PTS_5g_away', np.float64(0.0371))])
- 2020-21: 0.023  (top: [('odds_total', np.float64(0.7306)), ('off_points_5g_away', np.float64(0.0576)), ('def_OPP_PTS_5g_away', np.float64(0.038))])
- 2021-22: 0.016  (top: [('odds_total', np.float64(0.7001)), ('off_points_5g_home', np.float64(0.0493)), ('def_OPP_FG3A_20g_away', np.float64(0.042))])
- 2022-23: 0.008  (top: [('odds_total', np.float64(0.6575)), ('def_OPP_FG3M_20g_away', np.float64(0.0832)), ('off_points_5g_away', np.float64(0.0669))])
- 2023-24: 0.011  (top: [('odds_total', np.float64(0.7789)), ('off_points_20g_home', np.float64(0.0572)), ('def_OPP_FGA_20g_away', np.float64(0.0433))])
- 2024-25: 0.013  (top: [('odds_total', np.float64(0.7724)), ('def_OPP_FGA_20g_away', np.float64(0.0811)), ('def_OPP_PTS_5g_away', np.float64(0.0258))])
- 2025-26: 0.032  (top: [('odds_total', np.float64(0.6916)), ('def_OPP_FGA_20g_away', np.float64(0.0684)), ('off_points_5g_home', np.float64(0.0538))])
Folds with at least 5%: 0/7

## Verdict (registered rule; EXPLORATORY arm, decides nothing): **REJECT**
- decide.mc on total MAE: False; no-harm: {'total MSE': True, 'ML log loss': True, 'spread MAE': True}; candidate rule (beats the floor and >= 4/7 seasons): False