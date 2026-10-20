# E05_pace_adjusted: pace-normalised, opponent-adjusted team stats (added candidates)

## Paired comparison vs baseline (exp - base)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |           0.00383 | [-0.00221, 0.01084] | 3/7                | False                    |
| ML Brier    | 8916 |           0.00047 | [-0.00030, 0.00132] | 2/7                | False                    |
| Spread MAE  | 8916 |          -0.00887 | [-0.02353, 0.00622] | 5/7                | False                    |
| Spread MSE  | 8916 |          -0.21553 | [-0.61408, 0.20846] | 6/7                | False                    |
| Total MAE   | 8916 |          -0.01006 | [-0.04245, 0.01923] | 3/7                | False                    |
| Total MSE   | 8916 |          -0.12121 | [-1.42423, 0.95781] | 2/7                | False                    |

## Per season (value, difference vs baseline)

|         | ml_logloss       | ml_brier         | spread_mae        | total_mae         |
|:--------|:-----------------|:-----------------|:------------------|:------------------|
| 2019-20 | 0.6267 (-0.0052) | 0.2180 (-0.0025) | 10.0754 (-0.0210) | 15.1408 (-0.1518) |
| 2020-21 | 0.6337 (-0.0025) | 0.2213 (-0.0012) | 10.9467 (+0.0159) | 14.9317 (-0.0204) |
| 2021-22 | 0.6317 (+0.0029) | 0.2168 (+0.0014) | 10.9764 (-0.0238) | 14.6618 (-0.0180) |
| 2022-23 | 0.6365 (-0.0143) | 0.2224 (+0.0012) | 9.9401 (-0.0037)  | 14.5078 (+0.0005) |
| 2023-24 | 0.6442 (+0.0383) | 0.2071 (+0.0015) | 10.7748 (-0.0113) | 14.6139 (+0.0166) |
| 2024-25 | 0.5973 (+0.0014) | 0.2055 (+0.0004) | 10.8479 (-0.0217) | 14.6131 (+0.0311) |
| 2025-26 | 0.5975 (+0.0043) | 0.2063 (+0.0019) | 11.2296 (+0.0047) | 14.9352 (+0.0514) |

## Pooled primary metrics

|                   |   baseline |   E05_pace_adjusted |
|:------------------|-----------:|--------------------:|
| games             |  8916      |           8916      |
| ml_logloss        |     0.6199 |              0.6237 |
| ml_brier          |     0.2132 |              0.2137 |
| ml_ece            |     0.0107 |              0.0129 |
| ml_acc            |     0.6611 |              0.6629 |
| ml_distinct_probs |   256      |            250      |
| spread_mae        |    10.7011 |             10.6923 |
| spread_rmse       |    13.709  |             13.7012 |
| spread_bias       |    -0.154  |             -0.1311 |
| total_mae         |    14.7721 |             14.7621 |
| total_rmse        |    18.7114 |             18.7081 |
| total_bias        |     1.285  |              1.3173 |

## Selected-feature counts and tuned configs by fold

{"2019-20": {"ml": [20, 4], "spread": [20, 4], "total": [36, 0]}, "2020-21": {"ml": [20, 4], "spread": [20, 4], "total": [36, 2]}, "2021-22": {"ml": [21, 4], "spread": [20, 2], "total": [37, 3]}, "2022-23": {"ml": [20, 7], "spread": [21, 0], "total": [30, 7]}, "2023-24": {"ml": [28, 0], "spread": [22, 2], "total": [33, 0]}, "2024-25": {"ml": [30, 0], "spread": [25, 2], "total": [37, 2]}, "2025-26": {"ml": [20, 3], "spread": [26, 0], "total": [44, 0]}}

## New features chosen by selection (per fold: ml / spread / total)

```
2019-20: 2, 3, 8 -> ml: ['diff_e5_adjdef_assists_5g', 'diff_e5_adjoff_offensiveRebounds_20g'] | spread: ['diff_e5_adjoff_threePointersAttempted_5g', 'diff_e5_adjdef_freeThrowsAttempted_20g', 'diff_e5_pace_20g'] | total: ['e5_pace_5g_away', 'e5_adjoff_blocks_20g_home', 'e5_adjoff_defensiveRebounds_20g_away', 'e5_adjdef_defensiveRebounds_20g_away', 'e5_adjoff_defensiveRebounds_5g_home', 'e5_adjoff_threePointersAttempted_20g_away']
2020-21: 2, 5, 9 -> ml: ['diff_e5_adjdef_fieldGoalsAttempted_5g', 'diff_e5_adjdef_assists_5g'] | spread: ['diff_e5_adjdef_fieldGoalsAttempted_5g', 'diff_e5_adjdef_assists_20g', 'diff_e5_adjoff_turnovers_20g', 'diff_e5_adjdef_defensiveRebounds_20g', 'diff_e5_adjoff_offensiveRebounds_20g'] | total: ['e5_pace_5g_away', 'e5_adjoff_freeThrowsAttempted_5g_home', 'e5_adjoff_threePointersAttempted_20g_away', 'e5_adjoff_turnovers_5g_away', 'e5_adjoff_defensiveRebounds_20g_away', 'e5_adjdef_points_5g_away']
2021-22: 2, 3, 10 -> ml: ['diff_e5_adjdef_fieldGoalsAttempted_5g', 'diff_e5_adjoff_steals_5g'] | spread: ['diff_e5_adjoff_threePointersAttempted_20g', 'diff_e5_adjdef_fieldGoalsAttempted_5g', 'diff_e5_adjoff_turnovers_20g'] | total: ['e5_pace_5g_away', 'e5_adjdef_defensiveRebounds_5g_home', 'e5_adjoff_turnovers_5g_away', 'e5_adjoff_defensiveRebounds_5g_home', 'e5_adjoff_points_20g_away', 'e5_adjdef_assists_20g_home']
2022-23: 4, 4, 8 -> ml: ['diff_e5_adjdef_fieldGoalsAttempted_5g', 'diff_e5_adjdef_assists_5g', 'diff_e5_adjdef_steals_20g', 'diff_e5_adjoff_fieldGoalsAttempted_5g'] | spread: ['diff_e5_adjoff_threePointersAttempted_5g', 'diff_e5_adjdef_blocks_5g', 'diff_e5_adjoff_steals_5g', 'diff_e5_adjoff_turnovers_20g'] | total: ['e5_pace_5g_away', 'e5_adjoff_points_20g_away', 'e5_adjoff_defensiveRebounds_20g_away', 'e5_adjdef_assists_5g_away', 'e5_adjoff_fieldGoalsAttempted_5g_away', 'e5_adjoff_freeThrowsAttempted_20g_home']
2023-24: 5, 8, 7 -> ml: ['diff_e5_adjoff_threePointersAttempted_5g', 'diff_e5_adjdef_assists_20g', 'diff_e5_adjdef_assists_5g', 'diff_e5_adjdef_points_20g', 'diff_e5_pace_20g'] | spread: ['diff_e5_adjoff_threePointersAttempted_5g', 'diff_e5_adjdef_assists_20g', 'diff_e5_adjoff_steals_5g', 'diff_e5_adjoff_turnovers_20g', 'diff_e5_adjoff_offensiveRebounds_20g', 'diff_e5_adjdef_points_20g'] | total: ['e5_pace_5g_away', 'e5_adjdef_points_5g_away', 'e5_adjoff_turnovers_5g_home', 'e5_adjoff_blocks_20g_away', 'e5_adjoff_blocks_5g_away', 'e5_adjoff_defensiveRebounds_20g_home']
2024-25: 5, 7, 9 -> ml: ['diff_e5_adjdef_assists_20g', 'diff_e5_adjdef_fieldGoalsAttempted_5g', 'diff_e5_pace_5g', 'diff_e5_adjdef_fieldGoalsAttempted_20g', 'diff_e5_adjoff_steals_5g'] | spread: ['diff_e5_adjdef_blocks_20g', 'diff_e5_adjoff_turnovers_5g', 'diff_e5_adjdef_assists_20g', 'diff_e5_adjoff_turnovers_20g', 'diff_e5_adjdef_fieldGoalsAttempted_5g', 'diff_e5_adjdef_points_20g'] | total: ['e5_pace_5g_away', 'e5_adjoff_blocks_5g_away', 'e5_adjoff_steals_20g_home', 'e5_adjdef_points_20g_home', 'e5_adjoff_freeThrowsAttempted_20g_home', 'e5_adjoff_points_20g_away']
2025-26: 6, 7, 8 -> ml: ['diff_e5_adjdef_assists_20g', 'diff_e5_adjdef_blocks_20g', 'diff_e5_pace_5g', 'diff_e5_adjoff_steals_5g', 'diff_e5_adjoff_fieldGoalsAttempted_20g', 'diff_e5_adjdef_fieldGoalsAttempted_20g'] | spread: ['diff_e5_adjoff_turnovers_20g', 'diff_e5_adjoff_offensiveRebounds_20g', 'diff_e5_adjoff_threePointersAttempted_5g', 'diff_e5_adjdef_fieldGoalsAttempted_5g', 'diff_e5_adjoff_defensiveRebounds_5g', 'diff_e5_adjoff_freeThrowsAttempted_20g'] | total: ['e5_adjoff_blocks_5g_away', 'e5_adjoff_freeThrowsAttempted_5g_home', 'e5_pace_5g_away', 'e5_adjdef_points_20g_home', 'e5_adjdef_fieldGoalsAttempted_5g_home', 'e5_adjoff_offensiveRebounds_5g_home']
```

## Betting (5% rule), experiment

|              |      ml |   spread |   total |
|:-------------|--------:|---------:|--------:|
| bets         | 1931    |   636    | 2122    |
| hit_rate     |   47.64 |    52.83 |   52.97 |
| roi          |    0.31 |     0.86 |    1.12 |
| max_drawdown |   78.39 |    31    |   24    |

CLV points (mean): {'spread': {'n': 599, 'mean': 0.4661936560934892}, 'total': {'n': 2036, 'mean': 0.693516699410609}}

## Decision (pre-registered)

- **REJECT**  (main criterion per metric: {'ML log loss': False, 'Spread MAE': False, 'Total MAE': False}; no-harm per metric: {'ML log loss': False, 'Spread MAE': True, 'Total MAE': True})
- ML log loss: diff +0.00383 (base 0.6199), 95% CI [-0.00221, +0.01084], 3/7 seasons improved
- Spread MAE: diff -0.00887 (base 10.7011), 95% CI [-0.02353, +0.00622], 5/7 seasons improved
- Total MAE: diff -0.01006 (base 14.7721), 95% CI [-0.04245, +0.01923], 3/7 seasons improved