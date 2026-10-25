# E29x_F1pace_total

Arm columns (18): ['trk_dist_per_min_last_home', 'trk_dist_per_min_last_away', 'diff_trk_dist_per_min_last', 'trk_dist_per_poss_last_home', 'trk_dist_per_poss_last_away', 'diff_trk_dist_per_poss_last', 'trk_xload_last_home', 'trk_xload_last_away', 'diff_trk_xload_last', 'trk_xload_4n_home', 'trk_xload_4n_away', 'diff_trk_xload_4n', 'trk_rot_xload_last_home', 'trk_rot_xload_last_away', 'diff_trk_rot_xload_last', 'trk_prev_opp_pace_home', 'trk_prev_opp_pace_away', 'diff_trk_prev_opp_pace']
Registered decision-bearing arm: False. Reference: `E25_with_odds`.

## Validity checks

{"same_games_same_order": true, "ml_identical": true, "spread_identical": true}

## Paired comparison vs the reference (exp - ref; negative = better)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |           0       | [-0.00000, 0.00000] | 3/7                | False                    |
| ML Brier    | 8916 |           0       | [-0.00000, 0.00000] | 3/7                | False                    |
| Spread MAE  | 8916 |           0       | [-0.00000, 0.00000] | 3/7                | False                    |
| Spread MSE  | 8916 |           0       | [0.00000, 0.00000]  | 2/7                | False                    |
| Total MAE   | 8916 |           0.00213 | [-0.02152, 0.02934] | 5/7                | False                    |
| Total MSE   | 8916 |           0.89145 | [-0.01203, 1.91264] | 2/7                | False                    |

## Per season, total MAE difference

{"2019-20": 0.0881, "2020-21": 0.025, "2021-22": -0.0018, "2022-23": -0.0057, "2023-24": -0.0026, "2024-25": -0.0249, "2025-26": -0.0491}

Noise floor for total MAE (largest seed-only difference): 0.005. Pooled difference +0.00213: is inside the floor.

## Share of total-model gain taken by the arm's columns, per fold

- 2019-20: 0.129  (top: [('odds_total', np.float64(0.6445)), ('off_fieldGoalsAttempted_5g_away', np.float64(0.0446)), ('def_OPP_FGA_5g_home', np.float64(0.0445))])
- 2020-21: 0.077  (top: [('odds_total', np.float64(0.6881)), ('def_OPP_FGA_5g_away', np.float64(0.0632)), ('off_fieldGoalsAttempted_20g_home', np.float64(0.0366))])
- 2021-22: 0.072  (top: [('odds_total', np.float64(0.6815)), ('def_OPP_PTS_20g_home', np.float64(0.0947)), ('def_OPP_FGA_5g_home', np.float64(0.0283))])
- 2022-23: 0.027  (top: [('odds_total', np.float64(0.7829)), ('off_threePointersAttempted_5g_away', np.float64(0.0703)), ('def_OPP_FGA_5g_home', np.float64(0.041))])
- 2023-24: 0.027  (top: [('odds_total', np.float64(0.7754)), ('def_OPP_PTS_20g_home', np.float64(0.1228)), ('def_OPP_FGA_5g_home', np.float64(0.0216))])
- 2024-25: 0.021  (top: [('odds_total', np.float64(0.7893)), ('def_OPP_PTS_20g_home', np.float64(0.1218)), ('def_OPP_FGA_5g_home', np.float64(0.0184))])
- 2025-26: 0.047  (top: [('odds_total', np.float64(0.7193)), ('def_OPP_PTS_20g_home', np.float64(0.1235)), ('inj_lost_new_home', np.float64(0.0174))])
Folds with at least 5%: 3/7

## Verdict (registered rule; EXPLORATORY arm, decides nothing): **REJECT**
- decide.mc on total MAE: False; no-harm: {'total MSE': True, 'ML log loss': True, 'spread MAE': True}; candidate rule (beats the floor and >= 4/7 seasons): False