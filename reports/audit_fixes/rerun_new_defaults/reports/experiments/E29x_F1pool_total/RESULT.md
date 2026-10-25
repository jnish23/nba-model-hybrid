# E29x_F1pool_total

Arm columns (15): ['trk_dist_per_min_last_home', 'trk_dist_per_min_last_away', 'diff_trk_dist_per_min_last', 'trk_dist_per_poss_last_home', 'trk_dist_per_poss_last_away', 'diff_trk_dist_per_poss_last', 'trk_xload_last_home', 'trk_xload_last_away', 'diff_trk_xload_last', 'trk_xload_4n_home', 'trk_xload_4n_away', 'diff_trk_xload_4n', 'trk_rot_xload_last_home', 'trk_rot_xload_last_away', 'diff_trk_rot_xload_last']
Registered decision-bearing arm: False. Reference: `E25_with_odds`.

## Validity checks

{"same_games_same_order": true}

## Paired comparison vs the reference (exp - ref; negative = better)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |          -0.00042 | [-0.00131, 0.00056] | 4/7                | False                    |
| ML Brier    | 8916 |          -0.0002  | [-0.00060, 0.00022] | 4/7                | False                    |
| Spread MAE  | 8916 |          -0.00502 | [-0.01699, 0.00741] | 5/7                | False                    |
| Spread MSE  | 8916 |          -0.18099 | [-0.54416, 0.19839] | 5/7                | False                    |
| Total MAE   | 8916 |           0.03567 | [0.01011, 0.06076]  | 2/7                | False                    |
| Total MSE   | 8916 |           1.82268 | [0.82554, 2.78269]  | 2/7                | False                    |

## Per season, total MAE difference

{"2019-20": 0.1578, "2020-21": 0.0574, "2021-22": 0.0422, "2022-23": -0.0152, "2023-24": -0.0038, "2024-25": 0.0192, "2025-26": 0.011}

Noise floor for total MAE (largest seed-only difference): 0.005. Pooled difference +0.03567: exceeds the floor.

## Share of total-model gain taken by the arm's columns, per fold

- 2019-20: 0.005  (top: [('odds_total', np.float64(0.8037)), ('def_OPP_FGM_20g_home', np.float64(0.0762)), ('off_netRating_20g_home', np.float64(0.0144))])
- 2020-21: 0.003  (top: [('odds_total', np.float64(0.8212)), ('def_OPP_FGA_5g_home', np.float64(0.0834)), ('off_offensiveRating_5g_away', np.float64(0.0256))])
- 2021-22: 0.000  (top: [('odds_total', np.float64(0.7099)), ('def_OPP_FG3M_20g_home', np.float64(0.0999)), ('def_OPP_FGA_5g_home', np.float64(0.0421))])
- 2022-23: 0.001  (top: [('odds_total', np.float64(0.7594)), ('def_OPP_FG3M_20g_home', np.float64(0.0922)), ('def_OPP_PTS_20g_home', np.float64(0.071))])
- 2023-24: 0.000  (top: [('odds_total', np.float64(0.7762)), ('def_OPP_PTS_20g_home', np.float64(0.1282)), ('inj_lost_new_home', np.float64(0.0217))])
- 2024-25: 0.000  (top: [('odds_total', np.float64(0.7531)), ('def_OPP_FG3A_20g_home', np.float64(0.1122)), ('def_OPP_PTS_20g_home', np.float64(0.0715))])
- 2025-26: 0.002  (top: [('odds_total', np.float64(0.7423)), ('def_OPP_PTS_20g_home', np.float64(0.11)), ('inj_lost_new_home', np.float64(0.0305))])
Folds with at least 5%: 0/7

## Verdict (registered rule; EXPLORATORY arm, decides nothing): **REJECT**
- decide.mc on total MAE: False; no-harm: {'total MSE': False, 'ML log loss': True, 'spread MAE': True}; candidate rule (beats the floor and >= 4/7 seasons): False