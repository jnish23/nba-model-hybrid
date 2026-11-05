# E29x_pace_total

Arm columns (3): ['trk_prev_opp_pace_home', 'trk_prev_opp_pace_away', 'diff_trk_prev_opp_pace']
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
| Total MAE   | 8916 |          -0.00041 | [-0.01252, 0.01210] | 3/7                | False                    |
| Total MSE   | 8916 |           0.20463 | [-0.21235, 0.64465] | 2/7                | False                    |

## Per season, total MAE difference

{"2019-20": 0.0222, "2020-21": -0.0006, "2021-22": 0.0082, "2022-23": -0.0134, "2023-24": 0.011, "2024-25": 0.0021, "2025-26": -0.0294}

Noise floor for total MAE (largest seed-only difference): 0.005. Pooled difference -0.00041: is inside the floor.

## Share of total-model gain taken by the arm's columns, per fold

- 2019-20: 0.009  (top: [('odds_total', np.float64(0.7923)), ('off_fieldGoalsAttempted_5g_away', np.float64(0.0687)), ('def_OPP_FGA_5g_home', np.float64(0.042))])
- 2020-21: 0.005  (top: [('odds_total', np.float64(0.7675)), ('def_OPP_FGA_5g_away', np.float64(0.0874)), ('off_fieldGoalsAttempted_20g_home', np.float64(0.043))])
- 2021-22: 0.015  (top: [('odds_total', np.float64(0.6983)), ('def_OPP_PTS_20g_home', np.float64(0.0783)), ('def_OPP_FGA_5g_home', np.float64(0.0361))])
- 2022-23: 0.003  (top: [('odds_total', np.float64(0.7893)), ('off_threePointersAttempted_5g_away', np.float64(0.0805)), ('def_OPP_FGA_5g_home', np.float64(0.0415))])
- 2023-24: 0.002  (top: [('odds_total', np.float64(0.7945)), ('def_OPP_PTS_20g_home', np.float64(0.0935)), ('inj_lost_new_home', np.float64(0.0269))])
- 2024-25: 0.004  (top: [('odds_total', np.float64(0.8685)), ('def_OPP_PTS_20g_home', np.float64(0.0729)), ('def_OPP_FGA_5g_home', np.float64(0.0119))])
- 2025-26: 0.008  (top: [('odds_total', np.float64(0.7215)), ('def_OPP_PTS_20g_home', np.float64(0.1481)), ('inj_lost_new_home', np.float64(0.0151))])
Folds with at least 5%: 0/7

## Verdict (registered rule; EXPLORATORY arm, decides nothing): **REJECT**
- decide.mc on total MAE: False; no-harm: {'total MSE': True, 'ML log loss': True, 'spread MAE': True}; candidate rule (beats the floor and >= 4/7 seasons): False