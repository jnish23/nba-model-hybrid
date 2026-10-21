# Phase 4 evaluation: 2025-26 (v3_final_2025-26 + R2b)

86088 played player-game rows (three decision times each, pooled), 1321 games. Intervals: 95% bootstrap over whole games (2000 draws); the date-clustered interval is the sensitivity check. Differences are paired (model minus best baseline; negative = model better).

## Pre-registered go/no-go criteria

- 1. CRPS beats the best baseline with the 95% interval below 0 in all 8 stats: **PASS** (8/8; failing: none)
- 2. Mean bias within +/-2% for points, rebounds, assists: **PASS** (points -0.1%, rebounds -0.2%, assists +0.4%)
- 3. ECE of P(over) at ladder lines at most 0.02 in every stat: **PASS** (worst 0.0084 in threes)
- 4. CRPS at T-5 not significantly worse than at T-240 (interval not entirely above 0) in every stat: **PASS** (failing: none)

## 1. Accuracy against the baselines (CRPS, lower is better)

| stat | n | crps_model | crps_poisson | crps_normal | crps_empirical20 | best_baseline | diff_vs_best | diff_lo_game | diff_hi_game | diff_lo_date | diff_hi_date | rel_improvement_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| points | 85779 | 3.1292 | 3.5525 | 3.3195 | 3.3956 | normal | -0.1904 | -0.2046 | -0.1763 | -0.2151 | -0.1693 | 5.7343 |
| rebounds | 85779 | 1.2765 | 1.3546 | 1.3572 | 1.3813 | poisson | -0.0781 | -0.0847 | -0.0718 | -0.0897 | -0.069 | 5.7665 |
| assists | 85779 | 0.8896 | 0.9418 | 0.9584 | 0.9606 | poisson | -0.0522 | -0.0567 | -0.0478 | -0.0593 | -0.0459 | 5.5374 |
| threes | 85779 | 0.5741 | 0.5937 | 0.6184 | 0.6065 | poisson | -0.0196 | -0.0217 | -0.0174 | -0.0229 | -0.0169 | 3.2985 |
| pra | 85779 | 4.12 | 4.7969 | 4.4348 | 4.6017 | normal | -0.3148 | -0.34 | -0.2912 | -0.3641 | -0.2752 | 7.0994 |
| pr | 85779 | 3.7789 | 4.3448 | 4.0335 | 4.1711 | normal | -0.2545 | -0.275 | -0.2354 | -0.2948 | -0.2237 | 6.3105 |
| pa | 85779 | 3.4548 | 3.9709 | 3.7046 | 3.8021 | normal | -0.2498 | -0.2692 | -0.2316 | -0.2843 | -0.2202 | 6.7433 |
| ra | 85779 | 1.7695 | 1.9276 | 1.8969 | 1.9453 | normal | -0.1273 | -0.1371 | -0.1179 | -0.1442 | -0.1132 | 6.712 |

The empirical baseline fell back to Poisson for 1.4% of rows (fewer than 5 prior games). The Normal baseline's one sd per stat is fitted on this season itself, which favours the baseline.

## 2. Bias of the mean (observed / predicted - 1)

### Overall

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | all | 86088 | -0.09 | -0.72 | 0.48 |
| rebounds | all | 86088 | -0.22 | -0.93 | 0.51 |
| assists | all | 86088 | 0.42 | -0.38 | 1.24 |
| threes | all | 86088 | -1.54 | -2.65 | -0.46 |
| pra | all | 86088 | 0.02 | -0.53 | 0.53 |
| pr | all | 86088 | -0.1 | -0.64 | 0.39 |
| pa | all | 86088 | 0.06 | -0.57 | 0.65 |
| ra | all | 86088 | 0.14 | -0.45 | 0.69 |

### By projected minutes

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 15-28 | 40635 | 0.47 | -0.63 | 1.49 |
| points | 28+ | 26161 | -0.2 | -1.08 | 0.67 |
| points | under 15 | 19292 | -2.28 | -5.29 | 0.64 |
| rebounds | 15-28 | 40635 | -0.18 | -1.28 | 0.92 |
| rebounds | 28+ | 26161 | -0.41 | -1.48 | 0.58 |
| rebounds | under 15 | 19292 | 0.36 | -2.4 | 3.16 |
| assists | 15-28 | 40635 | 0.98 | -0.35 | 2.31 |
| assists | 28+ | 26161 | 0.29 | -0.83 | 1.32 |
| assists | under 15 | 19292 | -1.68 | -5.27 | 1.89 |
| threes | 15-28 | 40635 | -0.07 | -1.67 | 1.61 |
| threes | 28+ | 26161 | -2.38 | -3.93 | -0.86 |
| threes | under 15 | 19292 | -4.54 | -8.61 | -0.41 |
| pra | 15-28 | 40635 | 0.43 | -0.45 | 1.28 |
| pra | 28+ | 26161 | -0.12 | -0.86 | 0.58 |
| pra | under 15 | 19292 | -1.22 | -3.84 | 1.3 |

### By month

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 2025-10 | 5373 | 4.61 | 1.65 | 7.38 |
| points | 2025-11 | 14352 | -0.64 | -1.95 | 0.74 |
| points | 2025-12 | 12795 | -0.95 | -2.49 | 0.57 |
| points | 2026-01 | 14964 | -1.06 | -2.31 | 0.2 |
| points | 2026-02 | 10770 | 1.22 | -0.46 | 2.92 |
| points | 2026-03 | 15510 | -0.47 | -1.78 | 0.89 |
| points | 2026-04 | 9390 | 0.49 | -1.67 | 2.66 |
| points | 2026-05 | 2631 | -0.71 | -3.25 | 2.05 |
| pra | 2025-10 | 5373 | 4.02 | 1.43 | 6.48 |
| pra | 2025-11 | 14352 | -0.72 | -1.83 | 0.48 |
| pra | 2025-12 | 12795 | -0.76 | -2.01 | 0.57 |
| pra | 2026-01 | 14964 | -0.47 | -1.54 | 0.61 |
| pra | 2026-02 | 10770 | 1.08 | -0.34 | 2.51 |
| pra | 2026-03 | 15510 | -0.46 | -1.59 | 0.69 |
| pra | 2026-04 | 9390 | 0.65 | -1.21 | 2.57 |
| pra | 2026-05 | 2631 | -0.21 | -2.42 | 2.1 |

### By position group

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | C | 8976 | -2.24 | -4.09 | -0.37 |
| points | F | 25374 | 1.26 | 0.09 | 2.44 |
| points | FC | 7551 | -1.72 | -3.71 | 0.24 |
| points | G | 35424 | -1.06 | -2.1 | -0.09 |
| points | GF | 8763 | 3.36 | 1.36 | 5.26 |
| rebounds | C | 8976 | -2.94 | -4.76 | -1.22 |
| rebounds | F | 25374 | -0.61 | -1.86 | 0.64 |
| rebounds | FC | 7551 | -0.36 | -2.26 | 1.43 |
| rebounds | G | 35424 | 0.85 | -0.46 | 2.17 |
| rebounds | GF | 8763 | 2.74 | 0.6 | 4.8 |
| assists | C | 8976 | 3.14 | 0.08 | 6.08 |
| assists | F | 25374 | -0.16 | -1.85 | 1.52 |
| assists | FC | 7551 | 0.02 | -3.1 | 3.21 |
| assists | G | 35424 | -0.13 | -1.32 | 1.03 |
| assists | GF | 8763 | 2.55 | -0.05 | 5.12 |

## 3. Calibration of P(X > line) at the ladder lines (raw, no recalibration)

| stat | logloss | brier | ece |
|---|---|---|---|
| points | 0.6641 | 0.2359 | 0.0023 |
| rebounds | 0.5809 | 0.1982 | 0.0033 |
| assists | 0.5399 | 0.1816 | 0.0029 |
| threes | 0.4611 | 0.1524 | 0.0084 |
| pra | 0.6755 | 0.2413 | 0.0027 |
| pr | 0.6722 | 0.2397 | 0.0017 |
| pa | 0.6692 | 0.2383 | 0.0044 |
| ra | 0.6256 | 0.2178 | 0.0036 |

## 4. Decision time (CRPS and the paired T-5 minus T-240 difference)

| stat | crps_T240 | crps_T60 | crps_T5 | T5_minus_T240 | lo | hi |
|---|---|---|---|---|---|---|
| points | 3.1283 | 3.127 | 3.1239 | -0.0044 | -0.0074 | -0.0013 |
| rebounds | 1.2774 | 1.2768 | 1.2755 | -0.0019 | -0.003 | -0.0007 |
| assists | 0.8888 | 0.8888 | 0.8884 | -0.0004 | -0.0012 | 0.0003 |
| threes | 0.5733 | 0.5733 | 0.5732 | -0.0002 | -0.0006 | 0.0003 |
| pra | 4.1221 | 4.1198 | 4.1145 | -0.0076 | -0.0127 | -0.0027 |
| pr | 3.7805 | 3.7782 | 3.7737 | -0.0068 | -0.011 | -0.0028 |
| pa | 3.4545 | 3.4531 | 3.4493 | -0.0052 | -0.0093 | -0.0014 |
| ra | 1.7705 | 1.7697 | 1.7683 | -0.0023 | -0.0041 | -0.0004 |
