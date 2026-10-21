# Phase 4 evaluation: 2022-23 (v3_dev + R2b)

83211 played player-game rows (three decision times each, pooled), 1320 games. Intervals: 95% bootstrap over whole games (2000 draws); the date-clustered interval is the sensitivity check. Differences are paired (model minus best baseline; negative = model better).

## Pre-registered go/no-go criteria

- 1. CRPS beats the best baseline with the 95% interval below 0 in all 8 stats: **PASS** (8/8; failing: none)
- 2. Mean bias within +/-2% for points, rebounds, assists: **PASS** (points -0.1%, rebounds -0.2%, assists +0.0%)
- 3. ECE of P(over) at ladder lines at most 0.02 in every stat: **PASS** (worst 0.0110 in assists)
- 4. CRPS at T-5 not significantly worse than at T-240 (interval not entirely above 0) in every stat: **PASS** (failing: none)

## 1. Accuracy against the baselines (CRPS, lower is better)

| stat | n | crps_model | crps_poisson | crps_normal | crps_empirical20 | best_baseline | diff_vs_best | diff_lo_game | diff_hi_game | diff_lo_date | diff_hi_date | rel_improvement_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| points | 82956 | 3.1827 | 3.5907 | 3.3648 | 3.4201 | normal | -0.1821 | -0.1943 | -0.1693 | -0.1983 | -0.1673 | 5.4131 |
| rebounds | 82956 | 1.3021 | 1.3731 | 1.3786 | 1.3982 | poisson | -0.071 | -0.0766 | -0.0654 | -0.0804 | -0.0634 | 5.1703 |
| assists | 82956 | 0.8636 | 0.9033 | 0.9276 | 0.9217 | poisson | -0.0397 | -0.0431 | -0.0362 | -0.0451 | -0.0348 | 4.3909 |
| threes | 82956 | 0.5575 | 0.5737 | 0.604 | 0.5884 | poisson | -0.0162 | -0.018 | -0.0144 | -0.0184 | -0.0141 | 2.8198 |
| pra | 82956 | 4.185 | 4.8236 | 4.4654 | 4.6199 | normal | -0.2804 | -0.2998 | -0.2606 | -0.3118 | -0.2549 | 6.2783 |
| pr | 82956 | 3.8607 | 4.4068 | 4.0945 | 4.2144 | normal | -0.2338 | -0.2504 | -0.2172 | -0.2591 | -0.2125 | 5.7094 |
| pa | 82956 | 3.4936 | 3.975 | 3.7217 | 3.8002 | normal | -0.228 | -0.243 | -0.2125 | -0.2498 | -0.2091 | 6.1275 |
| ra | 82956 | 1.7703 | 1.9075 | 1.8824 | 1.9293 | normal | -0.112 | -0.1198 | -0.1041 | -0.1237 | -0.102 | 5.9524 |

The empirical baseline fell back to Poisson for 1.2% of rows (fewer than 5 prior games). The Normal baseline's one sd per stat is fitted on this season itself, which favours the baseline.

## 2. Bias of the mean (observed / predicted - 1)

### Overall

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | all | 83211 | -0.13 | -0.75 | 0.45 |
| rebounds | all | 83211 | -0.21 | -0.87 | 0.53 |
| assists | all | 83211 | 0.02 | -0.81 | 0.87 |
| threes | all | 83211 | -0.64 | -1.82 | 0.57 |
| pra | all | 83211 | -0.07 | -0.59 | 0.43 |
| pr | all | 83211 | -0.12 | -0.63 | 0.37 |
| pa | all | 83211 | -0.07 | -0.68 | 0.52 |
| ra | all | 83211 | -0.02 | -0.54 | 0.56 |

### By projected minutes

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 15-28 | 36419 | 1.09 | -0.1 | 2.28 |
| points | 28+ | 29195 | -0.25 | -0.99 | 0.51 |
| points | under 15 | 17597 | -4.78 | -7.81 | -1.72 |
| rebounds | 15-28 | 36419 | 0.92 | -0.18 | 2.06 |
| rebounds | 28+ | 29195 | -0.04 | -1.08 | 1 |
| rebounds | under 15 | 17597 | -5.31 | -7.93 | -2.43 |
| assists | 15-28 | 36419 | 1.7 | 0.08 | 3.33 |
| assists | 28+ | 29195 | -0.57 | -1.52 | 0.36 |
| assists | under 15 | 17597 | -3.18 | -6.98 | 1.03 |
| threes | 15-28 | 36419 | -0.04 | -1.99 | 1.87 |
| threes | 28+ | 29195 | -1.06 | -2.62 | 0.5 |
| threes | under 15 | 17597 | -0.57 | -5.3 | 4.35 |
| pra | 15-28 | 36419 | 1.18 | 0.21 | 2.19 |
| pra | 28+ | 29195 | -0.21 | -0.82 | 0.41 |
| pra | under 15 | 17597 | -4.62 | -7.19 | -1.92 |

### By month

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 2022-10 | 6795 | 0.74 | -1.69 | 3.14 |
| points | 2022-11 | 14097 | 0.99 | -0.36 | 2.37 |
| points | 2022-12 | 13986 | -0.36 | -1.62 | 0.95 |
| points | 2023-01 | 13818 | -0.05 | -1.27 | 1.14 |
| points | 2023-02 | 10227 | -1.62 | -3.18 | 0.06 |
| points | 2023-03 | 14313 | -0.02 | -1.57 | 1.53 |
| points | 2023-04 | 7581 | -0.01 | -2.75 | 2.86 |
| points | 2023-05 | 2094 | -2.37 | -5.7 | 1.08 |
| pra | 2022-10 | 6795 | 0.2 | -1.76 | 2.23 |
| pra | 2022-11 | 14097 | 0.61 | -0.53 | 1.78 |
| pra | 2022-12 | 13986 | -0.54 | -1.6 | 0.58 |
| pra | 2023-01 | 13818 | 0.23 | -0.8 | 1.3 |
| pra | 2023-02 | 10227 | -1.25 | -2.52 | 0.15 |
| pra | 2023-03 | 14313 | 0.13 | -1.17 | 1.44 |
| pra | 2023-04 | 7581 | 0.57 | -1.87 | 3.16 |
| pra | 2023-05 | 2094 | -2.01 | -4.7 | 0.7 |

### By position group

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | C | 6843 | -3.05 | -5.19 | -0.79 |
| points | F | 23814 | 0.9 | -0.33 | 2.2 |
| points | FC | 9981 | -1.6 | -3.25 | 0.21 |
| points | G | 31503 | -0.22 | -1.23 | 0.76 |
| points | GF | 11070 | 1.14 | -0.55 | 2.75 |
| rebounds | C | 6843 | -1.27 | -3.22 | 0.66 |
| rebounds | F | 23814 | -1.12 | -2.4 | 0.23 |
| rebounds | FC | 9981 | -1.22 | -2.92 | 0.53 |
| rebounds | G | 31503 | 1.62 | 0.31 | 3.01 |
| rebounds | GF | 11070 | 0.64 | -1.34 | 2.55 |
| assists | C | 6843 | 0.05 | -3.44 | 3.71 |
| assists | F | 23814 | -0.64 | -2.38 | 1.19 |
| assists | FC | 9981 | -0.37 | -3.12 | 2.33 |
| assists | G | 31503 | -0.1 | -1.33 | 1.08 |
| assists | GF | 11070 | 1.84 | -0.68 | 4.45 |

## 3. Calibration of P(X > line) at the ladder lines (raw, no recalibration)

| stat | logloss | brier | ece |
|---|---|---|---|
| points | 0.6629 | 0.2353 | 0.0045 |
| rebounds | 0.5818 | 0.1985 | 0.0052 |
| assists | 0.5378 | 0.1809 | 0.011 |
| threes | 0.4539 | 0.1498 | 0.0048 |
| pra | 0.674 | 0.2406 | 0.0045 |
| pr | 0.6716 | 0.2395 | 0.0039 |
| pa | 0.6672 | 0.2374 | 0.005 |
| ra | 0.6238 | 0.217 | 0.0044 |

## 4. Decision time (CRPS and the paired T-5 minus T-240 difference)

| stat | crps_T240 | crps_T60 | crps_T5 | T5_minus_T240 | lo | hi |
|---|---|---|---|---|---|---|
| points | 3.1848 | 3.1806 | 3.1768 | -0.0081 | -0.0111 | -0.0052 |
| rebounds | 1.3027 | 1.3014 | 1.3006 | -0.0021 | -0.0032 | -0.001 |
| assists | 0.8637 | 0.8629 | 0.862 | -0.0017 | -0.0024 | -0.001 |
| threes | 0.5572 | 0.5568 | 0.5564 | -0.0008 | -0.0011 | -0.0004 |
| pra | 4.1915 | 4.1843 | 4.1772 | -0.0142 | -0.0193 | -0.0092 |
| pr | 3.8653 | 3.8595 | 3.8541 | -0.0113 | -0.0154 | -0.0073 |
| pa | 3.4976 | 3.4922 | 3.4865 | -0.0111 | -0.0151 | -0.0073 |
| ra | 1.7721 | 1.7696 | 1.768 | -0.0042 | -0.006 | -0.0023 |
