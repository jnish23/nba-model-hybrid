# A6 descriptive: why a tree model of the total trails its own input line

E13 test frame (8,899 games, 7 seasons); excess = raw model minus opening line; positive means the raw model is worse.

## Pooled MAE and paired differences vs the raw model, fix (c) and the line

| forecast | MAE | RMSE | bias |
|---|---|---|---|
| raw | 14.763 | 18.697 | +1.268 |
| fix_c | 14.610 | 18.471 | +0.507 |
| line | 14.618 | 18.480 | +0.440 |
| line_plus_mean_actual_minus_line (L1) | 14.620 | 18.483 | +0.062 |
| line_plus_mean_actual_minus_pred (L2) | 14.628 | 18.489 | -0.386 |
| raw_plus_mean_actual_minus_pred (L3) | 14.753 | 18.671 | +0.442 |

| comparison (first minus second, MAE) | diff [95% CI] | seasons first better |
|---|---|---|
| line_plus_mean_actual_minus_line (L1) minus raw | -0.143 [-0.222, -0.072] | 7/7 |
| line_plus_mean_actual_minus_line (L1) minus fix_c | +0.010 [-0.013, +0.034] | 2/7 |
| line_plus_mean_actual_minus_line (L1) minus line | +0.003 [-0.007, +0.013] | 3/7 |
| line_plus_mean_actual_minus_pred (L2) minus raw | -0.135 [-0.214, -0.061] | 7/7 |
| line_plus_mean_actual_minus_pred (L2) minus fix_c | +0.018 [-0.011, +0.049] | 2/7 |
| line_plus_mean_actual_minus_pred (L2) minus line | +0.011 [-0.009, +0.032] | 2/7 |
| raw_plus_mean_actual_minus_pred (L3) minus raw | -0.010 [-0.032, +0.011] | 5/7 |
| raw_plus_mean_actual_minus_pred (L3) minus fix_c | +0.143 [+0.091, +0.204] | 0/7 |
| raw_plus_mean_actual_minus_pred (L3) minus line | +0.135 [+0.071, +0.213] | 1/7 |

Per-season MAE: raw: 15.28, 14.95, 14.68, 14.51, 14.54, 14.59, 14.88; fix_c: 15.13, 14.66, 14.59, 14.37, 14.29, 14.54, 14.77; line: 15.13, 14.66, 14.55, 14.39, 14.31, 14.58, 14.78; line_plus_mean_actual_minus_line (L1): 15.12, 14.69, 14.55, 14.39, 14.32, 14.56, 14.77; line_plus_mean_actual_minus_pred (L2): 15.12, 14.71, 14.56, 14.39, 14.35, 14.55, 14.78; raw_plus_mean_actual_minus_pred (L3): 15.23, 14.94, 14.70, 14.53, 14.53, 14.55, 14.88

## Excess error by opening-total bucket

| bucket | games | MAE raw | MAE line | MAE fix (c) | excess MAE per game | contribution to pooled excess MAE | excess MSE contribution | mean gap (pred - line) | mean actual - line |
|---|---|---|---|---|---|---|---|---|---|
| <215 | 995 | 13.85 | 13.82 | 13.82 | +0.04 | +0.004 | +0.11 | +1.36 | +0.98 |
| 215-225 | 3093 | 14.56 | 14.51 | 14.51 | +0.05 | +0.016 | +0.27 | +0.56 | +0.36 |
| 225-235 | 3526 | 14.81 | 14.74 | 14.73 | +0.07 | +0.029 | +1.47 | -0.81 | +0.60 |
| 235-242 | 1049 | 15.55 | 15.13 | 15.09 | +0.43 | +0.050 | +3.39 | -4.90 | -0.21 |
| >242 | 236 | 17.03 | 15.29 | 15.28 | +1.74 | +0.046 | +2.81 | -10.49 | -0.29 |

Pooled excess: MAE +0.146, MSE +8.05. Lines below 215 or at/above 242 (extremes) are 13.8% of games and carry +0.050 of the MAE excess (35%) and +2.92 of the MSE excess (36%). Lines at/above 242 alone: 2.7% of games, 35% of the MSE excess.

Lines at/above 235: 14.4% of games, 66% of the MAE excess and 77% of the MSE excess.

By season, games with a line at/above 235: 2019-20: share 6.6%, mean gap -12.2, mean actual-line -1.6; 2020-21: share 8.4%, mean gap -11.1, mean actual-line -0.8; 2021-22: share 4.0%, mean gap -6.9, mean actual-line -4.1; 2022-23: share 16.9%, mean gap -5.6, mean actual-line -1.3; 2023-24: share 22.9%, mean gap -6.4, mean actual-line +0.6; 2024-25: share 15.8%, mean gap -3.9, mean actual-line -0.1; 2025-26: share 24.9%, mean gap -3.8, mean actual-line +0.8

## Decile view (line deciles)

| decile | line range | n | excess MAE/game | mean gap | mean actual - line | mean actual - raw |
|---|---|---|---|---|---|---|
| 1 | 195.5-214.5 | 995 | +0.04 | +1.36 | +0.98 | -0.39 |
| 2 | 215.0-218.0 | 877 | +0.00 | +0.53 | +0.78 | +0.25 |
| 3 | 218.5-221.0 | 897 | +0.09 | +0.45 | +0.75 | +0.30 |
| 4 | 221.5-223.5 | 895 | +0.03 | +0.44 | -0.40 | -0.83 |
| 5 | 224.0-225.5 | 827 | +0.01 | +1.17 | +0.93 | -0.24 |
| 6 | 226.0-227.5 | 878 | +0.10 | +0.26 | +0.59 | +0.33 |
| 7 | 228.0-230.0 | 869 | +0.05 | -0.82 | +0.50 | +1.31 |
| 8 | 230.5-233.0 | 931 | +0.14 | -1.50 | +1.27 | +2.77 |
| 9 | 233.5-236.5 | 859 | +0.17 | -3.72 | -1.13 | +2.60 |
| 10 | 237.0-261.5 | 871 | +0.84 | -6.75 | +0.00 | +6.75 |

## Does the forecast follow the line?

- slope of the raw forecast on the line 0.753 versus slope of the actual total on the line 0.962; sd line 8.74, sd forecast 7.12, sd actual 20.29
- the forecast's gap from the line: sd 3.46; correlation with (actual - line) +0.042; slope of (actual - line) on the gap +0.223; correlation of the gap with the line level -0.623, slope of the gap on the line -0.247
- raw MSE minus line MSE = E[gap^2] - 2 E[gap x (actual - line)] = 12.65 - 4.60 = +8.05
- a quadratic in the line level explains 52.7% of the variance of the gap
