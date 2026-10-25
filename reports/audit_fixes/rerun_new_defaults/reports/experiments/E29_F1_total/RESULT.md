# E29_F1_total

Arm columns (15): ['trk_dist_per_min_last_home', 'trk_dist_per_min_last_away', 'diff_trk_dist_per_min_last', 'trk_dist_per_poss_last_home', 'trk_dist_per_poss_last_away', 'diff_trk_dist_per_poss_last', 'trk_xload_last_home', 'trk_xload_last_away', 'diff_trk_xload_last', 'trk_xload_4n_home', 'trk_xload_4n_away', 'diff_trk_xload_4n', 'trk_rot_xload_last_home', 'trk_rot_xload_last_away', 'diff_trk_rot_xload_last']
Registered decision-bearing arm: True. Reference: `E25_with_odds`.

## Validity checks

{"same_games_same_order": true, "ml_identical": true, "spread_identical": true}

## Paired comparison vs the reference (exp - ref; negative = better)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |           0       | [-0.00000, 0.00000] | 3/7                | False                    |
| ML Brier    | 8916 |           0       | [-0.00000, 0.00000] | 3/7                | False                    |
| Spread MAE  | 8916 |           0       | [-0.00000, 0.00000] | 3/7                | False                    |
| Spread MSE  | 8916 |           0       | [0.00000, 0.00000]  | 2/7                | False                    |
| Total MAE   | 8916 |           0.00321 | [-0.01876, 0.02877] | 5/7                | False                    |
| Total MSE   | 8916 |           0.60146 | [-0.25179, 1.55578] | 3/7                | False                    |

## Per season, total MAE difference

{"2019-20": 0.1009, "2020-21": -0.0071, "2021-22": 0.0007, "2022-23": -0.002, "2023-24": -0.0065, "2024-25": -0.0093, "2025-26": -0.0423}

Noise floor for total MAE (largest seed-only difference): 0.005. Pooled difference +0.00321: is inside the floor.

## Share of total-model gain taken by the arm's columns, per fold

- 2019-20: 0.125  (top: [('odds_total', np.float64(0.6233)), ('off_fieldGoalsAttempted_5g_away', np.float64(0.0533)), ('def_OPP_FGA_5g_home', np.float64(0.0468))])
- 2020-21: 0.025  (top: [('odds_total', np.float64(0.7668)), ('def_OPP_FGA_5g_away', np.float64(0.0816)), ('off_fieldGoalsAttempted_20g_home', np.float64(0.0408))])
- 2021-22: 0.074  (top: [('odds_total', np.float64(0.6301)), ('def_OPP_PTS_20g_home', np.float64(0.1364)), ('def_OPP_FGA_5g_home', np.float64(0.0256))])
- 2022-23: 0.032  (top: [('odds_total', np.float64(0.7013)), ('off_threePointersAttempted_5g_away', np.float64(0.1164)), ('def_OPP_FGA_5g_home', np.float64(0.0559))])
- 2023-24: 0.017  (top: [('odds_total', np.float64(0.9068)), ('def_OPP_PTS_20g_home', np.float64(0.0287)), ('def_OPP_FGA_5g_home', np.float64(0.0132))])
- 2024-25: 0.019  (top: [('odds_total', np.float64(0.7764)), ('def_OPP_PTS_20g_home', np.float64(0.0976)), ('odds_home_ml', np.float64(0.0297))])
- 2025-26: 0.047  (top: [('odds_total', np.float64(0.6299)), ('def_OPP_PTS_20g_home', np.float64(0.2172)), ('inj_lost_new_home', np.float64(0.0201))])
Folds with at least 5%: 2/7

## Verdict (registered rule): **REJECT**
- decide.mc on total MAE: False; no-harm: {'total MSE': True, 'ML log loss': True, 'spread MAE': True}; candidate rule (beats the floor and >= 4/7 seasons): False