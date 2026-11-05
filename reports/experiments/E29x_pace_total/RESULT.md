# E29x_pace_total

Arm columns (3): ['trk_prev_opp_pace_home', 'trk_prev_opp_pace_away', 'diff_trk_prev_opp_pace']
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
| Total MAE   | 8916 |           0.01067 | [-0.00254, 0.02568] | 1/7                | False                    |
| Total MSE   | 8916 |           0.25368 | [-0.26857, 0.82034] | 2/7                | False                    |

## Per season, total MAE difference

{"2019-20": 0.0215, "2020-21": -0.0194, "2021-22": 0.0181, "2022-23": 0.0086, "2023-24": 0.0211, "2024-25": 0.0031, "2025-26": 0.0197}

Noise floor for total MAE (largest seed-only difference): 0.005. Pooled difference +0.01067: exceeds the floor.

## Share of total-model gain taken by the arm's columns, per fold

- 2019-20: 0.024  (top: [('odds_total', np.float64(0.7246)), ('off_points_5g_away', np.float64(0.0689)), ('def_OPP_PTS_5g_away', np.float64(0.0422))])
- 2020-21: 0.011  (top: [('odds_total', np.float64(0.763)), ('off_points_5g_away', np.float64(0.0454)), ('def_OPP_FGA_5g_home', np.float64(0.0344))])
- 2021-22: 0.004  (top: [('odds_total', np.float64(0.7503)), ('off_points_5g_home', np.float64(0.0374)), ('def_OPP_FGM_20g_home', np.float64(0.0353))])
- 2022-23: 0.003  (top: [('odds_total', np.float64(0.6671)), ('def_OPP_FG3M_20g_away', np.float64(0.0736)), ('off_points_5g_away', np.float64(0.0679))])
- 2023-24: 0.004  (top: [('odds_total', np.float64(0.6818)), ('off_points_20g_home', np.float64(0.1185)), ('def_OPP_FGA_20g_away', np.float64(0.0756))])
- 2024-25: 0.006  (top: [('odds_total', np.float64(0.7243)), ('def_OPP_FGA_20g_away', np.float64(0.1009)), ('def_OPP_PTS_5g_away', np.float64(0.0315))])
- 2025-26: 0.015  (top: [('odds_total', np.float64(0.6119)), ('def_OPP_FGA_20g_away', np.float64(0.0953)), ('off_points_5g_home', np.float64(0.0759))])
Folds with at least 5%: 0/7

## Verdict (registered rule; EXPLORATORY arm, decides nothing): **REJECT**
- decide.mc on total MAE: False; no-harm: {'total MSE': True, 'ML log loss': True, 'spread MAE': True}; candidate rule (beats the floor and >= 4/7 seasons): False