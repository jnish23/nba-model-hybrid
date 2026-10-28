# E29_F1_total

Arm columns (15): ['trk_dist_per_min_last_home', 'trk_dist_per_min_last_away', 'diff_trk_dist_per_min_last', 'trk_dist_per_poss_last_home', 'trk_dist_per_poss_last_away', 'diff_trk_dist_per_poss_last', 'trk_xload_last_home', 'trk_xload_last_away', 'diff_trk_xload_last', 'trk_xload_4n_home', 'trk_xload_4n_away', 'diff_trk_xload_4n', 'trk_rot_xload_last_home', 'trk_rot_xload_last_away', 'diff_trk_rot_xload_last']
Registered decision-bearing arm: True. Reference: `E25_with_odds`.

## Validity checks

{"same_games_same_order": true, "ml_identical": true, "spread_identical": true}

## Paired comparison vs the reference (exp - ref; negative = better)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |           0       | [0.00000, 0.00000]  | 2/7                | False                    |
| ML Brier    | 8916 |           0       | [-0.00000, 0.00000] | 5/7                | False                    |
| Spread MAE  | 8916 |           0       | [-0.00000, 0.00000] | 4/7                | False                    |
| Spread MSE  | 8916 |          -0       | [-0.00000, 0.00000] | 5/7                | False                    |
| Total MAE   | 8916 |           0.01395 | [-0.00298, 0.03194] | 2/7                | False                    |
| Total MSE   | 8916 |           0.91002 | [0.25558, 1.60908]  | 2/7                | False                    |

## Per season, total MAE difference

{"2019-20": 0.0776, "2020-21": 0.0119, "2021-22": -0.0187, "2022-23": 0.0187, "2023-24": -0.0101, "2024-25": 0.0231, "2025-26": 0.0036}

Noise floor for total MAE (largest seed-only difference): 0.005. Pooled difference +0.01395: exceeds the floor.

## Share of total-model gain taken by the arm's columns, per fold

- 2019-20: 0.032  (top: [('odds_total', np.float64(0.7422)), ('off_points_5g_away', np.float64(0.0649)), ('def_OPP_PTS_5g_away', np.float64(0.0323))])
- 2020-21: 0.012  (top: [('odds_total', np.float64(0.6774)), ('off_points_5g_away', np.float64(0.0778)), ('def_OPP_FGA_5g_home', np.float64(0.06))])
- 2021-22: 0.013  (top: [('odds_total', np.float64(0.7306)), ('off_points_5g_home', np.float64(0.0426)), ('def_OPP_FGM_20g_home', np.float64(0.0409))])
- 2022-23: 0.007  (top: [('odds_total', np.float64(0.722)), ('off_points_5g_away', np.float64(0.0552)), ('def_OPP_FG3M_20g_away', np.float64(0.0498))])
- 2023-24: 0.010  (top: [('odds_total', np.float64(0.7663)), ('off_points_20g_home', np.float64(0.0701)), ('def_OPP_FGA_20g_away', np.float64(0.0579))])
- 2024-25: 0.013  (top: [('odds_total', np.float64(0.7888)), ('def_OPP_FGA_20g_away', np.float64(0.0635)), ('off_fieldGoalsAttempted_5g_away', np.float64(0.0275))])
- 2025-26: 0.029  (top: [('odds_total', np.float64(0.6847)), ('def_OPP_FGA_20g_away', np.float64(0.0837)), ('off_points_5g_home', np.float64(0.0482))])
Folds with at least 5%: 0/7

## Verdict (registered rule): **REJECT**
- decide.mc on total MAE: False; no-harm: {'total MSE': True, 'ML log loss': True, 'spread MAE': True}; candidate rule (beats the floor and >= 4/7 seasons): False