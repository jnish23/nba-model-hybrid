# Phase 4 evaluation: 2024-25 (v3_seen_2024-25)

84711 played player-game rows (three decision times each, pooled), 1320 games. Intervals: 95% bootstrap over whole games (2000 draws); the date-clustered interval is the sensitivity check. Differences are paired (model minus best baseline; negative = model better).

## Pre-registered go/no-go criteria

- 1. CRPS beats the best baseline with the 95% interval below 0 in all 8 stats: **PASS** (8/8; failing: none)
- 2. Mean bias within +/-2% for points, rebounds, assists: **FAIL** (points -1.4%, rebounds -1.6%, assists -3.7%)
- 3. ECE of P(over) at ladder lines at most 0.02 in every stat: **FAIL** (worst 0.0226 in ra)
- 4. CRPS at T-5 not significantly worse than at T-240 (interval not entirely above 0) in every stat: **PASS** (failing: none)

## 1. Accuracy against the baselines (CRPS, lower is better)

| stat | n | crps_model | crps_poisson | crps_normal | crps_empirical20 | best_baseline | diff_vs_best | diff_lo_game | diff_hi_game | diff_lo_date | diff_hi_date | rel_improvement_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| points | 84399 | 3.1568 | 3.5505 | 3.3224 | 3.3951 | normal | -0.1657 | -0.1798 | -0.1516 | -0.1884 | -0.1474 | 4.9861 |
| rebounds | 84399 | 1.3001 | 1.3704 | 1.3729 | 1.4026 | poisson | -0.0703 | -0.0774 | -0.0636 | -0.0822 | -0.06 | 5.128 |
| assists | 84399 | 0.889 | 0.9307 | 0.9504 | 0.9528 | poisson | -0.0417 | -0.046 | -0.0374 | -0.0484 | -0.036 | 4.4837 |
| threes | 84399 | 0.5834 | 0.6006 | 0.6249 | 0.615 | poisson | -0.0172 | -0.0198 | -0.0146 | -0.0201 | -0.0144 | 2.8604 |
| pra | 84399 | 4.1598 | 4.781 | 4.4279 | 4.5998 | normal | -0.268 | -0.2918 | -0.2444 | -0.3106 | -0.2326 | 6.0532 |
| pr | 84399 | 3.8241 | 4.349 | 4.0424 | 4.1786 | normal | -0.2184 | -0.2377 | -0.1989 | -0.253 | -0.1902 | 5.4022 |
| pa | 84399 | 3.4764 | 3.9471 | 3.6905 | 3.7898 | normal | -0.2142 | -0.2324 | -0.1959 | -0.2442 | -0.189 | 5.8028 |
| ra | 84399 | 1.7908 | 1.9287 | 1.9006 | 1.9549 | normal | -0.1098 | -0.1187 | -0.1008 | -0.1251 | -0.0969 | 5.7785 |

The empirical baseline fell back to Poisson for 1.5% of rows (fewer than 5 prior games). The Normal baseline's one sd per stat is fitted on this season itself, which favours the baseline.

## 2. Bias of the mean (observed / predicted - 1)

### Overall

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | all | 84711 | -1.42 | -2.02 | -0.8 |
| rebounds | all | 84711 | -1.61 | -2.29 | -0.93 |
| assists | all | 84711 | -3.66 | -4.44 | -2.84 |
| threes | all | 84711 | -3.69 | -4.78 | -2.62 |
| pra | all | 84711 | -1.79 | -2.29 | -1.26 |
| pr | all | 84711 | -1.47 | -1.97 | -0.95 |
| pa | all | 84711 | -1.84 | -2.43 | -1.24 |
| ra | all | 84711 | -2.38 | -2.93 | -1.81 |

### By projected minutes

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 15-28 | 36740 | -0.02 | -1.13 | 1.12 |
| points | 28+ | 29232 | -1.88 | -2.68 | -1.07 |
| points | under 15 | 18739 | -4.45 | -7.59 | -1.3 |
| rebounds | 15-28 | 36740 | -2.01 | -3.05 | -0.95 |
| rebounds | 28+ | 29232 | -0.9 | -1.86 | 0.07 |
| rebounds | under 15 | 18739 | -3.32 | -6.09 | -0.43 |
| assists | 15-28 | 36740 | -2.01 | -3.37 | -0.64 |
| assists | 28+ | 29232 | -4.5 | -5.59 | -3.47 |
| assists | under 15 | 18739 | -4.83 | -8.48 | -1.12 |
| threes | 15-28 | 36740 | -1.46 | -3.22 | 0.39 |
| threes | 28+ | 29232 | -5.28 | -6.67 | -3.89 |
| threes | under 15 | 18739 | -3.91 | -8.42 | 0.57 |
| pra | 15-28 | 36740 | -0.82 | -1.71 | 0.08 |
| pra | 28+ | 29232 | -2.08 | -2.79 | -1.38 |
| pra | under 15 | 18739 | -4.16 | -6.77 | -1.47 |

### By month

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 2024-10 | 4776 | 0.1 | -2.83 | 3.22 |
| points | 2024-11 | 14370 | -0.47 | -1.85 | 1.04 |
| points | 2024-12 | 12159 | -1.62 | -3.16 | -0.01 |
| points | 2025-01 | 14679 | -2.48 | -3.7 | -1.16 |
| points | 2025-02 | 11214 | -1.2 | -2.91 | 0.41 |
| points | 2025-03 | 15087 | -1.34 | -2.7 | -0.01 |
| points | 2025-04 | 9423 | -0.99 | -2.9 | 0.95 |
| points | 2025-05 | 2526 | -3.62 | -6.66 | -0.29 |
| pra | 2024-10 | 4776 | -0.46 | -2.94 | 2.1 |
| pra | 2024-11 | 14370 | -0.6 | -1.83 | 0.68 |
| pra | 2024-12 | 12159 | -1.79 | -3.08 | -0.49 |
| pra | 2025-01 | 14679 | -2.73 | -3.78 | -1.57 |
| pra | 2025-02 | 11214 | -1.25 | -2.68 | 0.12 |
| pra | 2025-03 | 15087 | -2.06 | -3.2 | -0.92 |
| pra | 2025-04 | 9423 | -1.86 | -3.58 | -0.14 |
| pra | 2025-05 | 2526 | -4.27 | -6.61 | -1.65 |

### By position group

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | C | 8229 | -1.23 | -3.37 | 0.76 |
| points | F | 24369 | -1.71 | -2.94 | -0.46 |
| points | FC | 8217 | -2.54 | -4.48 | -0.7 |
| points | G | 35127 | -2.15 | -3.06 | -1.19 |
| points | GF | 8769 | 2.88 | 1.08 | 4.77 |
| rebounds | C | 8229 | -3.94 | -5.79 | -2.22 |
| rebounds | F | 24369 | -3.11 | -4.46 | -1.8 |
| rebounds | FC | 8217 | -2.05 | -3.86 | -0.24 |
| rebounds | G | 35127 | 0.33 | -0.84 | 1.55 |
| rebounds | GF | 8769 | 1.5 | -0.46 | 3.63 |
| assists | C | 8229 | 1.59 | -1.4 | 4.62 |
| assists | F | 24369 | -4.77 | -6.38 | -3.02 |
| assists | FC | 8217 | -2.49 | -5.48 | 0.53 |
| assists | G | 35127 | -4.25 | -5.29 | -3.13 |
| assists | GF | 8769 | -2.86 | -5.38 | -0.46 |

## 3. Calibration of P(X > line) at the ladder lines (raw, no recalibration)

| stat | logloss | brier | ece |
|---|---|---|---|
| points | 0.6626 | 0.2352 | 0.0104 |
| rebounds | 0.583 | 0.1991 | 0.0124 |
| assists | 0.5365 | 0.1801 | 0.017 |
| threes | 0.4566 | 0.1507 | 0.0142 |
| pra | 0.6736 | 0.2404 | 0.0162 |
| pr | 0.6706 | 0.239 | 0.0137 |
| pa | 0.6674 | 0.2374 | 0.0135 |
| ra | 0.626 | 0.218 | 0.0226 |

## 4. Decision time (CRPS and the paired T-5 minus T-240 difference)

| stat | crps_T240 | crps_T60 | crps_T5 | T5_minus_T240 | lo | hi |
|---|---|---|---|---|---|---|
| points | 3.1528 | 3.1535 | 3.1488 | -0.004 | -0.0073 | -0.0008 |
| rebounds | 1.2993 | 1.2992 | 1.2986 | -0.0007 | -0.002 | 0.0006 |
| assists | 0.8874 | 0.8874 | 0.8867 | -0.0007 | -0.0015 | 0.0001 |
| threes | 0.5822 | 0.5823 | 0.5818 | -0.0005 | -0.001 | 0 |
| pra | 4.1573 | 4.1573 | 4.15 | -0.0073 | -0.0125 | -0.0021 |
| pr | 3.821 | 3.8214 | 3.8157 | -0.0053 | -0.0096 | -0.0008 |
| pa | 3.4729 | 3.4734 | 3.467 | -0.0059 | -0.0102 | -0.0019 |
| ra | 1.7897 | 1.7894 | 1.7876 | -0.0021 | -0.0041 | 0 |
