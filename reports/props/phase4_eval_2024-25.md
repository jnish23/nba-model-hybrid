# Phase 4 evaluation: 2024-25 (v2_val_2024-25)

84711 played player-game rows (three decision times each, pooled), 1320 games. Intervals: 95% bootstrap over whole games (2000 draws); the date-clustered interval is the sensitivity check. Differences are paired (model minus best baseline; negative = model better).

## Pre-registered go/no-go criteria

- 1. CRPS beats the best baseline with the 95% interval below 0 in all 8 stats: **PASS** (8/8; failing: none)
- 2. Mean bias within +/-2% for points, rebounds, assists: **FAIL** (points -4.2%, rebounds -2.9%, assists -5.0%)
- 3. ECE of P(over) at ladder lines at most 0.02 in every stat: **FAIL** (worst 0.0370 in pra)
- 4. CRPS at T-5 not significantly worse than at T-240 (interval not entirely above 0) in every stat: **PASS** (failing: none)

## 1. Accuracy against the baselines (CRPS, lower is better)

| stat | n | crps_model | crps_poisson | crps_normal | crps_empirical20 | best_baseline | diff_vs_best | diff_lo_game | diff_hi_game | diff_lo_date | diff_hi_date | rel_improvement_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| points | 84399 | 3.1677 | 3.5505 | 3.3224 | 3.3951 | normal | -0.1548 | -0.1693 | -0.1401 | -0.1776 | -0.1361 | 4.6579 |
| rebounds | 84399 | 1.3011 | 1.3704 | 1.3729 | 1.4026 | poisson | -0.0693 | -0.0766 | -0.0625 | -0.0813 | -0.059 | 5.0565 |
| assists | 84399 | 0.8923 | 0.9307 | 0.9504 | 0.9528 | poisson | -0.0384 | -0.0428 | -0.0337 | -0.045 | -0.0325 | 4.1209 |
| threes | 84399 | 0.5851 | 0.6006 | 0.6249 | 0.615 | poisson | -0.0155 | -0.0182 | -0.0127 | -0.0183 | -0.0127 | 2.5748 |
| pra | 84399 | 4.1765 | 4.781 | 4.4279 | 4.5998 | normal | -0.2514 | -0.2752 | -0.227 | -0.2944 | -0.2162 | 5.6767 |
| pr | 84399 | 3.8361 | 4.349 | 4.0424 | 4.1786 | normal | -0.2063 | -0.2265 | -0.1869 | -0.242 | -0.1777 | 5.1041 |
| pa | 84399 | 3.4921 | 3.9471 | 3.6905 | 3.7898 | normal | -0.1984 | -0.2172 | -0.1793 | -0.2283 | -0.1731 | 5.3752 |
| ra | 84399 | 1.7948 | 1.9287 | 1.9006 | 1.9549 | normal | -0.1058 | -0.1149 | -0.0967 | -0.1207 | -0.0929 | 5.5657 |

The empirical baseline fell back to Poisson for 1.5% of rows (fewer than 5 prior games). The Normal baseline's one sd per stat is fitted on this season itself, which favours the baseline.

## 2. Bias of the mean (observed / predicted - 1)

### Overall

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | all | 84711 | -4.24 | -4.81 | -3.63 |
| rebounds | all | 84711 | -2.91 | -3.57 | -2.24 |
| assists | all | 84711 | -5.01 | -5.78 | -4.22 |
| threes | all | 84711 | -5.92 | -6.97 | -4.87 |
| pra | all | 84711 | -4.03 | -4.51 | -3.52 |
| pr | all | 84711 | -3.87 | -4.36 | -3.36 |
| pa | all | 84711 | -4.38 | -4.94 | -3.8 |
| ra | all | 84711 | -3.69 | -4.21 | -3.13 |

### By projected minutes

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 15-28 | 37087 | -3.48 | -4.58 | -2.37 |
| points | 28+ | 29993 | -3.97 | -4.74 | -3.19 |
| points | under 15 | 17631 | -10.29 | -13.31 | -7.22 |
| rebounds | 15-28 | 37087 | -3.64 | -4.66 | -2.54 |
| rebounds | 28+ | 29993 | -1.47 | -2.42 | -0.52 |
| rebounds | under 15 | 17631 | -7.12 | -9.89 | -4.27 |
| assists | 15-28 | 37087 | -4.34 | -5.7 | -2.96 |
| assists | 28+ | 29993 | -5.21 | -6.24 | -4.19 |
| assists | under 15 | 17631 | -6.74 | -10.45 | -3.07 |
| threes | 15-28 | 37087 | -4.57 | -6.31 | -2.77 |
| threes | 28+ | 29993 | -6.36 | -7.75 | -5 |
| threes | under 15 | 17631 | -9.99 | -14.34 | -5.79 |
| pra | 15-28 | 37087 | -3.62 | -4.53 | -2.71 |
| pra | 28+ | 29993 | -3.65 | -4.32 | -3 |
| pra | under 15 | 17631 | -8.83 | -11.53 | -6.09 |

### By month

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 2024-10 | 4776 | -2.04 | -4.9 | 1.03 |
| points | 2024-11 | 14370 | -2.7 | -4.04 | -1.22 |
| points | 2024-12 | 12159 | -4.14 | -5.62 | -2.58 |
| points | 2025-01 | 14679 | -5.37 | -6.54 | -4.08 |
| points | 2025-02 | 11214 | -4.27 | -5.91 | -2.67 |
| points | 2025-03 | 15087 | -4.45 | -5.74 | -3.14 |
| points | 2025-04 | 9423 | -5.05 | -6.83 | -3.18 |
| points | 2025-05 | 2526 | -4.83 | -7.81 | -1.74 |
| pra | 2024-10 | 4776 | -1.97 | -4.42 | 0.54 |
| pra | 2024-11 | 14370 | -2.35 | -3.56 | -1.11 |
| pra | 2024-12 | 12159 | -3.7 | -4.96 | -2.4 |
| pra | 2025-01 | 14679 | -4.95 | -5.97 | -3.86 |
| pra | 2025-02 | 11214 | -3.68 | -5.08 | -2.35 |
| pra | 2025-03 | 15087 | -4.56 | -5.63 | -3.45 |
| pra | 2025-04 | 9423 | -5.37 | -6.97 | -3.78 |
| pra | 2025-05 | 2526 | -5.25 | -7.49 | -2.73 |

### By position group

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | C | 8229 | -3.95 | -6.03 | -2.04 |
| points | F | 24369 | -4.58 | -5.78 | -3.38 |
| points | FC | 8217 | -4.97 | -6.85 | -3.22 |
| points | G | 35127 | -5.01 | -5.91 | -4.08 |
| points | GF | 8769 | -0.1 | -1.85 | 1.71 |
| rebounds | C | 8229 | -5.19 | -7 | -3.47 |
| rebounds | F | 24369 | -4.55 | -5.88 | -3.26 |
| rebounds | FC | 8217 | -3.38 | -5.15 | -1.62 |
| rebounds | G | 35127 | -0.9 | -2.06 | 0.31 |
| rebounds | GF | 8769 | 0.39 | -1.5 | 2.5 |
| assists | C | 8229 | 1.15 | -1.88 | 4.2 |
| assists | F | 24369 | -7.26 | -8.85 | -5.54 |
| assists | FC | 8217 | -1.81 | -4.82 | 1.25 |
| assists | G | 35127 | -5.5 | -6.53 | -4.38 |
| assists | GF | 8769 | -4.39 | -6.89 | -2.07 |

## 3. Calibration of P(X > line) at the ladder lines (raw, no recalibration)

| stat | logloss | brier | ece |
|---|---|---|---|
| points | 0.66 | 0.234 | 0.0315 |
| rebounds | 0.5828 | 0.1991 | 0.0205 |
| assists | 0.5363 | 0.18 | 0.0233 |
| threes | 0.4562 | 0.1505 | 0.0214 |
| pra | 0.6712 | 0.2392 | 0.037 |
| pr | 0.668 | 0.2377 | 0.0345 |
| pa | 0.665 | 0.2363 | 0.0349 |
| ra | 0.6249 | 0.2176 | 0.0326 |

## 4. Decision time (CRPS and the paired T-5 minus T-240 difference)

| stat | crps_T240 | crps_T60 | crps_T5 | T5_minus_T240 | lo | hi |
|---|---|---|---|---|---|---|
| points | 3.1635 | 3.1641 | 3.1607 | -0.0028 | -0.0066 | 0.0008 |
| rebounds | 1.3 | 1.3002 | 1.3 | -0 | -0.0014 | 0.0015 |
| assists | 0.8905 | 0.8908 | 0.8905 | 0 | -0.001 | 0.001 |
| threes | 0.5839 | 0.5839 | 0.5836 | -0.0003 | -0.0008 | 0.0003 |
| pra | 4.1728 | 4.174 | 4.1685 | -0.0043 | -0.0102 | 0.0015 |
| pr | 3.8323 | 3.8335 | 3.829 | -0.0033 | -0.0083 | 0.0019 |
| pa | 3.4879 | 3.4888 | 3.4844 | -0.0035 | -0.0083 | 0.0009 |
| ra | 1.7932 | 1.7933 | 1.7927 | -0.0006 | -0.0027 | 0.0018 |
