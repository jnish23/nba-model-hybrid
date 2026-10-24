# Phase 4 evaluation: 2022-23 (v3_dev)

83211 played player-game rows (three decision times each, pooled), 1320 games. Intervals: 95% bootstrap over whole games (2000 draws); the date-clustered interval is the sensitivity check. Differences are paired (model minus best baseline; negative = model better).

## Pre-registered go/no-go criteria

- 1. CRPS beats the best baseline with the 95% interval below 0 in all 8 stats: **PASS** (8/8; failing: none)
- 2. Mean bias within +/-2% for points, rebounds, assists: **PASS** (points -0.5%, rebounds -0.8%, assists +0.1%)
- 3. ECE of P(over) at ladder lines at most 0.02 in every stat: **PASS** (worst 0.0106 in assists)
- 4. CRPS at T-5 not significantly worse than at T-240 (interval not entirely above 0) in every stat: **PASS** (failing: none)

## 1. Accuracy against the baselines (CRPS, lower is better)

| stat | n | crps_model | crps_poisson | crps_normal | crps_empirical20 | best_baseline | diff_vs_best | diff_lo_game | diff_hi_game | diff_lo_date | diff_hi_date | rel_improvement_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| points | 82956 | 3.1827 | 3.5907 | 3.3648 | 3.4201 | normal | -0.1821 | -0.1942 | -0.1694 | -0.1984 | -0.1671 | 5.413 |
| rebounds | 82956 | 1.302 | 1.3731 | 1.3786 | 1.3982 | poisson | -0.071 | -0.0766 | -0.0655 | -0.0805 | -0.0634 | 5.173 |
| assists | 82956 | 0.8633 | 0.9033 | 0.9276 | 0.9217 | poisson | -0.04 | -0.0435 | -0.0365 | -0.0452 | -0.0351 | 4.4241 |
| threes | 82956 | 0.5575 | 0.5737 | 0.604 | 0.5884 | poisson | -0.0161 | -0.018 | -0.0143 | -0.0184 | -0.014 | 2.8151 |
| pra | 82956 | 4.184 | 4.8236 | 4.4654 | 4.6199 | normal | -0.2814 | -0.3008 | -0.2617 | -0.3129 | -0.2561 | 6.3021 |
| pr | 82956 | 3.8603 | 4.4068 | 4.0945 | 4.2144 | normal | -0.2341 | -0.2508 | -0.2174 | -0.2594 | -0.2126 | 5.7185 |
| pa | 82956 | 3.4929 | 3.975 | 3.7217 | 3.8002 | normal | -0.2288 | -0.2437 | -0.213 | -0.2502 | -0.2099 | 6.1473 |
| ra | 82956 | 1.7701 | 1.9075 | 1.8824 | 1.9293 | normal | -0.1123 | -0.12 | -0.1044 | -0.1239 | -0.1021 | 5.9658 |

The empirical baseline fell back to Poisson for 1.2% of rows (fewer than 5 prior games). The Normal baseline's one sd per stat is fitted on this season itself, which favours the baseline.

## 2. Bias of the mean (observed / predicted - 1)

### Overall

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | all | 83211 | -0.46 | -1.08 | 0.12 |
| rebounds | all | 83211 | -0.83 | -1.49 | -0.1 |
| assists | all | 83211 | 0.07 | -0.78 | 0.9 |
| threes | all | 83211 | -1.58 | -2.73 | -0.39 |
| pra | all | 83211 | -0.47 | -0.99 | 0.02 |
| pr | all | 83211 | -0.56 | -1.07 | -0.07 |
| pa | all | 83211 | -0.37 | -0.98 | 0.21 |
| ra | all | 83211 | -0.48 | -0.99 | 0.08 |

### By projected minutes

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 15-28 | 36419 | 0.77 | -0.41 | 1.95 |
| points | 28+ | 29195 | -0.59 | -1.33 | 0.17 |
| points | under 15 | 17597 | -5.06 | -8.1 | -1.97 |
| rebounds | 15-28 | 36419 | 0.27 | -0.82 | 1.39 |
| rebounds | 28+ | 29195 | -0.66 | -1.7 | 0.37 |
| rebounds | under 15 | 17597 | -5.84 | -8.45 | -2.99 |
| assists | 15-28 | 36419 | 1.83 | 0.2 | 3.44 |
| assists | 28+ | 29195 | -0.61 | -1.54 | 0.33 |
| assists | under 15 | 17597 | -2.8 | -6.6 | 1.46 |
| threes | 15-28 | 36419 | -0.95 | -2.88 | 0.94 |
| threes | 28+ | 29195 | -2.03 | -3.58 | -0.48 |
| threes | under 15 | 17597 | -1.35 | -6.06 | 3.54 |
| pra | 15-28 | 36419 | 0.78 | -0.2 | 1.79 |
| pra | 28+ | 29195 | -0.61 | -1.22 | 0 |
| pra | under 15 | 17597 | -4.99 | -7.56 | -2.32 |

### By month

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 2022-10 | 6795 | -0.46 | -2.8 | 1.87 |
| points | 2022-11 | 14097 | 0.93 | -0.42 | 2.31 |
| points | 2022-12 | 13986 | 0.24 | -1.02 | 1.57 |
| points | 2023-01 | 13818 | 0.51 | -0.72 | 1.69 |
| points | 2023-02 | 10227 | -2.17 | -3.72 | -0.52 |
| points | 2023-03 | 14313 | -1.28 | -2.81 | 0.23 |
| points | 2023-04 | 7581 | -0.63 | -3.33 | 2.21 |
| points | 2023-05 | 2094 | -4.81 | -8.12 | -1.39 |
| pra | 2022-10 | 6795 | -0.8 | -2.7 | 1.17 |
| pra | 2022-11 | 14097 | 0.28 | -0.86 | 1.44 |
| pra | 2022-12 | 13986 | -0.61 | -1.66 | 0.52 |
| pra | 2023-01 | 13818 | 0.29 | -0.75 | 1.33 |
| pra | 2023-02 | 10227 | -1.64 | -2.91 | -0.24 |
| pra | 2023-03 | 14313 | -0.78 | -2.04 | 0.5 |
| pra | 2023-04 | 7581 | 0.5 | -1.93 | 3.08 |
| pra | 2023-05 | 2094 | -3.82 | -6.49 | -1.13 |

### By position group

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | C | 6843 | -3.36 | -5.5 | -1.12 |
| points | F | 23814 | 0.56 | -0.66 | 1.85 |
| points | FC | 9981 | -1.94 | -3.59 | -0.14 |
| points | G | 31503 | -0.54 | -1.56 | 0.43 |
| points | GF | 11070 | 0.83 | -0.84 | 2.44 |
| rebounds | C | 6843 | -1.94 | -3.86 | -0.06 |
| rebounds | F | 23814 | -1.74 | -3.04 | -0.4 |
| rebounds | FC | 9981 | -1.84 | -3.54 | -0.1 |
| rebounds | G | 31503 | 1.03 | -0.27 | 2.4 |
| rebounds | GF | 11070 | -0.01 | -1.97 | 1.87 |
| assists | C | 6843 | 0.16 | -3.3 | 3.79 |
| assists | F | 23814 | -0.51 | -2.24 | 1.31 |
| assists | FC | 9981 | -0.23 | -2.95 | 2.46 |
| assists | G | 31503 | -0.11 | -1.35 | 1.04 |
| assists | GF | 11070 | 1.86 | -0.68 | 4.44 |

## 3. Calibration of P(X > line) at the ladder lines (raw, no recalibration)

| stat | logloss | brier | ece |
|---|---|---|---|
| points | 0.6625 | 0.2351 | 0.0046 |
| rebounds | 0.5819 | 0.1986 | 0.0085 |
| assists | 0.5378 | 0.1809 | 0.0106 |
| threes | 0.4534 | 0.1496 | 0.007 |
| pra | 0.6736 | 0.2404 | 0.0067 |
| pr | 0.6714 | 0.2394 | 0.0057 |
| pa | 0.667 | 0.2373 | 0.0045 |
| ra | 0.6234 | 0.2169 | 0.0078 |

## 4. Decision time (CRPS and the paired T-5 minus T-240 difference)

| stat | crps_T240 | crps_T60 | crps_T5 | T5_minus_T240 | lo | hi |
|---|---|---|---|---|---|---|
| points | 3.1849 | 3.1806 | 3.1768 | -0.0081 | -0.0111 | -0.0052 |
| rebounds | 1.3026 | 1.3014 | 1.3006 | -0.0021 | -0.0031 | -0.001 |
| assists | 0.8634 | 0.8626 | 0.8617 | -0.0017 | -0.0025 | -0.001 |
| threes | 0.5572 | 0.5568 | 0.5564 | -0.0008 | -0.0012 | -0.0004 |
| pra | 4.1904 | 4.1832 | 4.1761 | -0.0143 | -0.0193 | -0.0093 |
| pr | 3.8649 | 3.859 | 3.8537 | -0.0112 | -0.0154 | -0.0073 |
| pa | 3.4969 | 3.4914 | 3.4857 | -0.0112 | -0.0152 | -0.0074 |
| ra | 1.7719 | 1.7694 | 1.7677 | -0.0041 | -0.006 | -0.0023 |
