# E29x_F1pool_total

Arm columns (15): ['trk_dist_per_min_last_home', 'trk_dist_per_min_last_away', 'diff_trk_dist_per_min_last', 'trk_dist_per_poss_last_home', 'trk_dist_per_poss_last_away', 'diff_trk_dist_per_poss_last', 'trk_xload_last_home', 'trk_xload_last_away', 'diff_trk_xload_last', 'trk_xload_4n_home', 'trk_xload_4n_away', 'diff_trk_xload_4n', 'trk_rot_xload_last_home', 'trk_rot_xload_last_away', 'diff_trk_rot_xload_last']
Registered decision-bearing arm: False. Reference: `E25_with_odds`.

## Validity checks

{"same_games_same_order": true}

## Paired comparison vs the reference (exp - ref; negative = better)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |          -0.00148 | [-0.00856, 0.00604] | 3/7                | False                    |
| ML Brier    | 8916 |           0.00017 | [-0.00067, 0.00103] | 2/7                | False                    |
| Spread MAE  | 8916 |           0.0097  | [-0.00182, 0.02147] | 2/7                | False                    |
| Spread MSE  | 8916 |           0.13409 | [-0.19567, 0.47350] | 2/7                | False                    |
| Total MAE   | 8916 |          -0.01364 | [-0.04261, 0.01297] | 3/7                | False                    |
| Total MSE   | 8916 |          -0.96225 | [-2.10428, 0.04951] | 5/7                | False                    |

## Per season, total MAE difference

{"2019-20": -0.1457, "2020-21": -0.0122, "2021-22": -0.0219, "2022-23": 0.006, "2023-24": 0.0102, "2024-25": 0.0091, "2025-26": 0.0416}

Noise floor for total MAE (largest seed-only difference): 0.005. Pooled difference -0.01364: exceeds the floor.

## Share of total-model gain taken by the arm's columns, per fold

- 2019-20: 0.000  (top: [('odds_total', np.float64(0.743)), ('def_OPP_FGA_20g_away', np.float64(0.0409)), ('off_points_5g_home', np.float64(0.0326))])
- 2020-21: 0.000  (top: [('odds_total', np.float64(0.7167)), ('off_points_5g_away', np.float64(0.0524)), ('off_points_5g_home', np.float64(0.0481))])
- 2021-22: 0.000  (top: [('odds_total', np.float64(0.7709)), ('def_OPP_FGA_20g_away', np.float64(0.0632)), ('off_points_5g_home', np.float64(0.0249))])
- 2022-23: 0.000  (top: [('odds_total', np.float64(0.6788)), ('def_OPP_FGA_20g_away', np.float64(0.0836)), ('off_points_5g_home', np.float64(0.05))])
- 2023-24: 0.000  (top: [('odds_total', np.float64(0.7333)), ('def_OPP_PTS_20g_away', np.float64(0.0783)), ('off_points_20g_home', np.float64(0.0489))])
- 2024-25: 0.000  (top: [('odds_total', np.float64(0.6874)), ('def_OPP_FGA_20g_away', np.float64(0.0867)), ('def_OPP_PTS_5g_away', np.float64(0.0791))])
- 2025-26: 0.000  (top: [('odds_total', np.float64(0.6904)), ('def_OPP_FGA_20g_away', np.float64(0.0726)), ('off_points_5g_away', np.float64(0.0521))])
Folds with at least 5%: 0/7

## Verdict (registered rule; EXPLORATORY arm, decides nothing): **REJECT**
- decide.mc on total MAE: False; no-harm: {'total MSE': True, 'ML log loss': True, 'spread MAE': True}; candidate rule (beats the floor and >= 4/7 seasons): False