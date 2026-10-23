# Phase 4 evaluation: 2024-25 (v3_seen_2024-25 + R2b)

84711 played player-game rows (three decision times each, pooled), 1320 games. Intervals: 95% bootstrap over whole games (2000 draws); the date-clustered interval is the sensitivity check. Differences are paired (model minus best baseline; negative = model better).

## Pre-registered go/no-go criteria

- 1. CRPS beats the best baseline with the 95% interval below 0 in all 8 stats: **PASS** (8/8; failing: none)
- 2. Mean bias within +/-2% for points, rebounds, assists: **PASS** (points -0.5%, rebounds -0.6%, assists -1.5%)
- 3. ECE of P(over) at ladder lines at most 0.02 in every stat: **PASS** (worst 0.0123 in ra)
- 4. CRPS at T-5 not significantly worse than at T-240 (interval not entirely above 0) in every stat: **PASS** (failing: none)

## 1. Accuracy against the baselines (CRPS, lower is better)

| stat | n | crps_model | crps_poisson | crps_normal | crps_empirical20 | best_baseline | diff_vs_best | diff_lo_game | diff_hi_game | diff_lo_date | diff_hi_date | rel_improvement_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| points | 84399 | 3.1563 | 3.5505 | 3.3224 | 3.3951 | normal | -0.1661 | -0.1802 | -0.1523 | -0.1887 | -0.1478 | 4.9999 |
| rebounds | 84399 | 1.2996 | 1.3704 | 1.3729 | 1.4026 | poisson | -0.0707 | -0.0778 | -0.0642 | -0.0827 | -0.0605 | 5.1629 |
| assists | 84399 | 0.8875 | 0.9307 | 0.9504 | 0.9528 | poisson | -0.0432 | -0.0472 | -0.039 | -0.0498 | -0.0375 | 4.6373 |
| threes | 84399 | 0.5825 | 0.6006 | 0.6249 | 0.615 | poisson | -0.018 | -0.0204 | -0.0156 | -0.021 | -0.0153 | 2.9994 |
| pra | 84399 | 4.1572 | 4.781 | 4.4279 | 4.5998 | normal | -0.2706 | -0.2941 | -0.2476 | -0.3135 | -0.2356 | 6.1123 |
| pr | 84399 | 3.8228 | 4.349 | 4.0424 | 4.1786 | normal | -0.2196 | -0.2386 | -0.2002 | -0.2545 | -0.1916 | 5.4334 |
| pa | 84399 | 3.4748 | 3.9471 | 3.6905 | 3.7898 | normal | -0.2157 | -0.2337 | -0.1978 | -0.2458 | -0.1909 | 5.8456 |
| ra | 84399 | 1.7883 | 1.9287 | 1.9006 | 1.9549 | normal | -0.1123 | -0.121 | -0.1035 | -0.1273 | -0.0994 | 5.9085 |

The empirical baseline fell back to Poisson for 1.5% of rows (fewer than 5 prior games). The Normal baseline's one sd per stat is fitted on this season itself, which favours the baseline.

## 2. Bias of the mean (observed / predicted - 1)

### Overall

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | all | 84711 | -0.5 | -1.1 | 0.12 |
| rebounds | all | 84711 | -0.65 | -1.33 | 0.04 |
| assists | all | 84711 | -1.51 | -2.3 | -0.67 |
| threes | all | 84711 | -1.75 | -2.88 | -0.65 |
| pra | all | 84711 | -0.61 | -1.13 | -0.07 |
| pr | all | 84711 | -0.51 | -1.02 | 0.02 |
| pa | all | 84711 | -0.63 | -1.22 | -0.01 |
| ra | all | 84711 | -0.87 | -1.42 | -0.31 |

### By projected minutes

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 15-28 | 36740 | 0.89 | -0.21 | 2.04 |
| points | 28+ | 29232 | -0.95 | -1.75 | -0.12 |
| points | under 15 | 18739 | -3.64 | -6.83 | -0.46 |
| rebounds | 15-28 | 36740 | -1.05 | -2.09 | 0.04 |
| rebounds | 28+ | 29232 | 0.09 | -0.87 | 1.08 |
| rebounds | under 15 | 18739 | -2.49 | -5.29 | 0.4 |
| assists | 15-28 | 36740 | 0.06 | -1.33 | 1.47 |
| assists | 28+ | 29232 | -2.25 | -3.36 | -1.2 |
| assists | under 15 | 18739 | -3.08 | -6.8 | 0.68 |
| threes | 15-28 | 36740 | 0.52 | -1.29 | 2.41 |
| threes | 28+ | 29232 | -3.28 | -4.68 | -1.86 |
| threes | under 15 | 18739 | -2.57 | -7.17 | 1.95 |
| pra | 15-28 | 36740 | 0.36 | -0.53 | 1.28 |
| pra | 28+ | 29232 | -0.89 | -1.61 | -0.18 |
| pra | under 15 | 18739 | -3.09 | -5.72 | -0.38 |

### By month

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 2024-10 | 4776 | -0.23 | -3.2 | 2.93 |
| points | 2024-11 | 14370 | -0.22 | -1.61 | 1.3 |
| points | 2024-12 | 12159 | -0.72 | -2.29 | 0.9 |
| points | 2025-01 | 14679 | -0.83 | -2.06 | 0.5 |
| points | 2025-02 | 11214 | -0.3 | -2.03 | 1.31 |
| points | 2025-03 | 15087 | -0.25 | -1.64 | 1.09 |
| points | 2025-04 | 9423 | -0.17 | -2.11 | 1.8 |
| points | 2025-05 | 2526 | -1.68 | -4.78 | 1.64 |
| pra | 2024-10 | 4776 | -0.56 | -3.03 | 2.01 |
| pra | 2024-11 | 14370 | -0.13 | -1.37 | 1.16 |
| pra | 2024-12 | 12159 | -0.79 | -2.1 | 0.53 |
| pra | 2025-01 | 14679 | -0.97 | -2.04 | 0.2 |
| pra | 2025-02 | 11214 | -0.15 | -1.62 | 1.23 |
| pra | 2025-03 | 15087 | -0.56 | -1.7 | 0.59 |
| pra | 2025-04 | 9423 | -0.39 | -2.15 | 1.34 |
| pra | 2025-05 | 2526 | -2.17 | -4.53 | 0.49 |

### By position group

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | C | 8229 | -0.31 | -2.48 | 1.72 |
| points | F | 24369 | -0.82 | -2.05 | 0.44 |
| points | FC | 8217 | -1.63 | -3.58 | 0.22 |
| points | G | 35127 | -1.24 | -2.15 | -0.27 |
| points | GF | 8769 | 3.88 | 2.05 | 5.79 |
| rebounds | C | 8229 | -2.96 | -4.82 | -1.2 |
| rebounds | F | 24369 | -2.16 | -3.53 | -0.83 |
| rebounds | FC | 8217 | -1.06 | -2.9 | 0.77 |
| rebounds | G | 35127 | 1.27 | 0.1 | 2.51 |
| rebounds | GF | 8769 | 2.53 | 0.56 | 4.67 |
| assists | C | 8229 | 3.8 | 0.76 | 6.9 |
| assists | F | 24369 | -2.75 | -4.4 | -0.96 |
| assists | FC | 8217 | -0.41 | -3.44 | 2.67 |
| assists | G | 35127 | -2.05 | -3.12 | -0.89 |
| assists | GF | 8769 | -0.65 | -3.24 | 1.81 |

## 3. Calibration of P(X > line) at the ladder lines (raw, no recalibration)

| stat | logloss | brier | ece |
|---|---|---|---|
| points | 0.6635 | 0.2356 | 0.0055 |
| rebounds | 0.5833 | 0.1993 | 0.0073 |
| assists | 0.5369 | 0.1802 | 0.0088 |
| threes | 0.4572 | 0.1509 | 0.0094 |
| pra | 0.6744 | 0.2408 | 0.0062 |
| pr | 0.6715 | 0.2394 | 0.0065 |
| pa | 0.6684 | 0.238 | 0.005 |
| ra | 0.6265 | 0.2183 | 0.0123 |

## 4. Decision time (CRPS and the paired T-5 minus T-240 difference)

| stat | crps_T240 | crps_T60 | crps_T5 | T5_minus_T240 | lo | hi |
|---|---|---|---|---|---|---|
| points | 3.1526 | 3.153 | 3.148 | -0.0046 | -0.0079 | -0.0015 |
| rebounds | 1.299 | 1.2988 | 1.2979 | -0.0011 | -0.0023 | 0.0002 |
| assists | 0.8861 | 0.886 | 0.8852 | -0.001 | -0.0018 | -0.0001 |
| threes | 0.5814 | 0.5814 | 0.5809 | -0.0005 | -0.0011 | -0 |
| pra | 4.1555 | 4.1549 | 4.1464 | -0.0091 | -0.0143 | -0.0039 |
| pr | 3.8203 | 3.8203 | 3.8138 | -0.0065 | -0.0108 | -0.002 |
| pa | 3.4718 | 3.4719 | 3.4648 | -0.007 | -0.0113 | -0.0029 |
| ra | 1.7877 | 1.7871 | 1.7847 | -0.003 | -0.005 | -0.001 |
