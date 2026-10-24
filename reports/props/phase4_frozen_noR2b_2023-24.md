# Phase 4 evaluation: 2023-24 (v3_dev)

84609 played player-game rows (three decision times each, pooled), 1318 games. Intervals: 95% bootstrap over whole games (2000 draws); the date-clustered interval is the sensitivity check. Differences are paired (model minus best baseline; negative = model better).

## Pre-registered go/no-go criteria

- 1. CRPS beats the best baseline with the 95% interval below 0 in all 8 stats: **PASS** (8/8; failing: none)
- 2. Mean bias within +/-2% for points, rebounds, assists: **FAIL** (points -5.0%, rebounds -2.3%, assists -2.4%)
- 3. ECE of P(over) at ladder lines at most 0.02 in every stat: **FAIL** (worst 0.0377 in pra)
- 4. CRPS at T-5 not significantly worse than at T-240 (interval not entirely above 0) in every stat: **PASS** (failing: none)

## 1. Accuracy against the baselines (CRPS, lower is better)

| stat | n | crps_model | crps_poisson | crps_normal | crps_empirical20 | best_baseline | diff_vs_best | diff_lo_game | diff_hi_game | diff_lo_date | diff_hi_date | rel_improvement_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| points | 84306 | 3.1141 | 3.4948 | 3.2799 | 3.3342 | normal | -0.1657 | -0.1781 | -0.1528 | -0.1808 | -0.1516 | 5.0525 |
| rebounds | 84306 | 1.2795 | 1.3448 | 1.3513 | 1.3688 | poisson | -0.0653 | -0.071 | -0.0596 | -0.0729 | -0.0582 | 4.8547 |
| assists | 84306 | 0.8882 | 0.9317 | 0.9522 | 0.9508 | poisson | -0.0435 | -0.0474 | -0.0397 | -0.0474 | -0.0399 | 4.6692 |
| threes | 84306 | 0.5563 | 0.572 | 0.6011 | 0.5864 | poisson | -0.0156 | -0.0178 | -0.0133 | -0.0178 | -0.0134 | 2.7309 |
| pra | 84306 | 4.144 | 4.7543 | 4.4066 | 4.5488 | normal | -0.2626 | -0.2819 | -0.2423 | -0.2875 | -0.2394 | 5.9601 |
| pr | 84306 | 3.8042 | 4.3176 | 4.0161 | 4.1269 | normal | -0.2119 | -0.2281 | -0.1951 | -0.2333 | -0.1922 | 5.277 |
| pa | 84306 | 3.4415 | 3.8999 | 3.6564 | 3.7341 | normal | -0.215 | -0.2308 | -0.1984 | -0.2339 | -0.1968 | 5.8791 |
| ra | 84306 | 1.7727 | 1.906 | 1.8805 | 1.926 | normal | -0.1078 | -0.1154 | -0.1006 | -0.1172 | -0.0991 | 5.7346 |

The empirical baseline fell back to Poisson for 1.4% of rows (fewer than 5 prior games). The Normal baseline's one sd per stat is fitted on this season itself, which favours the baseline.

## 2. Bias of the mean (observed / predicted - 1)

### Overall

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | all | 84609 | -4.97 | -5.51 | -4.43 |
| rebounds | all | 84609 | -2.25 | -2.92 | -1.58 |
| assists | all | 84609 | -2.44 | -3.21 | -1.64 |
| threes | all | 84609 | -4.22 | -5.33 | -3.09 |
| pra | all | 84609 | -3.98 | -4.44 | -3.5 |
| pr | all | 84609 | -4.23 | -4.7 | -3.76 |
| pa | all | 84609 | -4.5 | -5.03 | -3.94 |
| ra | all | 84609 | -2.31 | -2.82 | -1.76 |

### By projected minutes

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 15-28 | 35775 | -4.91 | -5.96 | -3.79 |
| points | 28+ | 30461 | -3.85 | -4.65 | -3.06 |
| points | under 15 | 18373 | -14.33 | -17.31 | -11.54 |
| rebounds | 15-28 | 35775 | -2.57 | -3.57 | -1.48 |
| rebounds | 28+ | 30461 | -0.43 | -1.41 | 0.62 |
| rebounds | under 15 | 18373 | -9.77 | -12.39 | -7.09 |
| assists | 15-28 | 35775 | -2.88 | -4.37 | -1.3 |
| assists | 28+ | 30461 | -1.74 | -2.8 | -0.77 |
| assists | under 15 | 18373 | -6.51 | -10.4 | -2.65 |
| threes | 15-28 | 35775 | -5.51 | -7.37 | -3.66 |
| threes | 28+ | 30461 | -1.82 | -3.24 | -0.29 |
| threes | under 15 | 18373 | -15.58 | -19.53 | -11.23 |
| pra | 15-28 | 35775 | -4 | -4.82 | -3.09 |
| pra | 28+ | 30461 | -2.85 | -3.52 | -2.19 |
| pra | under 15 | 18373 | -12 | -14.57 | -9.54 |

### By month

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 2023-10 | 3570 | -2.95 | -5.31 | -0.62 |
| points | 2023-11 | 14121 | -3.43 | -4.67 | -2.3 |
| points | 2023-12 | 13263 | -3.21 | -4.58 | -1.88 |
| points | 2024-01 | 14973 | -5.23 | -6.65 | -3.75 |
| points | 2024-02 | 11190 | -6.13 | -7.53 | -4.72 |
| points | 2024-03 | 14643 | -6.61 | -7.87 | -5.35 |
| points | 2024-04 | 9969 | -6.83 | -8.49 | -5.08 |
| points | 2024-05 | 2538 | -1.1 | -3.79 | 1.95 |
| pra | 2023-10 | 3570 | -1.2 | -3.64 | 1.16 |
| pra | 2023-11 | 14121 | -3.06 | -4.1 | -2.04 |
| pra | 2023-12 | 13263 | -2.69 | -3.85 | -1.51 |
| pra | 2024-01 | 14973 | -4.15 | -5.34 | -2.87 |
| pra | 2024-02 | 11190 | -4.32 | -5.59 | -3.04 |
| pra | 2024-03 | 14643 | -5.33 | -6.35 | -4.21 |
| pra | 2024-04 | 9969 | -6.04 | -7.44 | -4.47 |
| pra | 2024-05 | 2538 | -0.19 | -2.4 | 2.29 |

### By position group

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | C | 6894 | -5.49 | -7.39 | -3.52 |
| points | F | 25305 | -4.76 | -5.89 | -3.64 |
| points | FC | 9522 | -4.83 | -6.48 | -3.27 |
| points | G | 33099 | -5.51 | -6.43 | -4.6 |
| points | GF | 9789 | -3.43 | -5.1 | -1.78 |
| rebounds | C | 6894 | -3.64 | -5.44 | -1.92 |
| rebounds | F | 25305 | -3.88 | -5.09 | -2.58 |
| rebounds | FC | 9522 | -3 | -4.51 | -1.39 |
| rebounds | G | 33099 | -0.7 | -1.9 | 0.55 |
| rebounds | GF | 9789 | 1.37 | -0.76 | 3.44 |
| assists | C | 6894 | 2.53 | -0.91 | 6.1 |
| assists | F | 25305 | -4.51 | -6.13 | -2.83 |
| assists | FC | 9522 | 2.38 | -0.42 | 5.11 |
| assists | G | 33099 | -3.36 | -4.51 | -2.19 |
| assists | GF | 9789 | -0.26 | -2.63 | 2.05 |

## 3. Calibration of P(X > line) at the ladder lines (raw, no recalibration)

| stat | logloss | brier | ece |
|---|---|---|---|
| points | 0.6581 | 0.2331 | 0.0343 |
| rebounds | 0.5803 | 0.1979 | 0.0177 |
| assists | 0.539 | 0.1813 | 0.0109 |
| threes | 0.4481 | 0.1475 | 0.0153 |
| pra | 0.6693 | 0.2383 | 0.0377 |
| pr | 0.6662 | 0.2369 | 0.0374 |
| pa | 0.6627 | 0.2352 | 0.0347 |
| ra | 0.624 | 0.2172 | 0.0225 |

## 4. Decision time (CRPS and the paired T-5 minus T-240 difference)

| stat | crps_T240 | crps_T60 | crps_T5 | T5_minus_T240 | lo | hi |
|---|---|---|---|---|---|---|
| points | 3.1118 | 3.1103 | 3.1065 | -0.0052 | -0.0086 | -0.002 |
| rebounds | 1.2785 | 1.2783 | 1.2772 | -0.0013 | -0.0023 | -0.0003 |
| assists | 0.887 | 0.8866 | 0.8858 | -0.0013 | -0.0021 | -0.0003 |
| threes | 0.5557 | 0.5554 | 0.5551 | -0.0005 | -0.001 | -0.0001 |
| pra | 4.1438 | 4.1407 | 4.1337 | -0.0102 | -0.0154 | -0.0052 |
| pr | 3.8029 | 3.8004 | 3.7953 | -0.0076 | -0.0119 | -0.0035 |
| pa | 3.4402 | 3.4383 | 3.4325 | -0.0077 | -0.012 | -0.0035 |
| ra | 1.772 | 1.7709 | 1.7689 | -0.0031 | -0.0048 | -0.0013 |
