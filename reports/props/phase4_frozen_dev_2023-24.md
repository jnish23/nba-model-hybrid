# Phase 4 evaluation: 2023-24 (v3_dev + R2b)

84609 played player-game rows (three decision times each, pooled), 1318 games. Intervals: 95% bootstrap over whole games (2000 draws); the date-clustered interval is the sensitivity check. Differences are paired (model minus best baseline; negative = model better).

## Pre-registered go/no-go criteria

- 1. CRPS beats the best baseline with the 95% interval below 0 in all 8 stats: **PASS** (8/8; failing: none)
- 2. Mean bias within +/-2% for points, rebounds, assists: **PASS** (points -1.5%, rebounds -0.8%, assists -1.2%)
- 3. ECE of P(over) at ladder lines at most 0.02 in every stat: **PASS** (worst 0.0136 in pra)
- 4. CRPS at T-5 not significantly worse than at T-240 (interval not entirely above 0) in every stat: **PASS** (failing: none)

## 1. Accuracy against the baselines (CRPS, lower is better)

| stat | n | crps_model | crps_poisson | crps_normal | crps_empirical20 | best_baseline | diff_vs_best | diff_lo_game | diff_hi_game | diff_lo_date | diff_hi_date | rel_improvement_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| points | 84306 | 3.1034 | 3.4948 | 3.2799 | 3.3342 | normal | -0.1764 | -0.1877 | -0.1645 | -0.1908 | -0.1629 | 5.3795 |
| rebounds | 84306 | 1.279 | 1.3448 | 1.3513 | 1.3688 | poisson | -0.0658 | -0.0713 | -0.0603 | -0.0736 | -0.0586 | 4.8936 |
| assists | 84306 | 0.8872 | 0.9317 | 0.9522 | 0.9508 | poisson | -0.0445 | -0.0483 | -0.0408 | -0.0485 | -0.0407 | 4.7764 |
| threes | 84306 | 0.5557 | 0.572 | 0.6011 | 0.5864 | poisson | -0.0163 | -0.0183 | -0.0141 | -0.0183 | -0.0142 | 2.8433 |
| pra | 84306 | 4.1314 | 4.7543 | 4.4066 | 4.5488 | normal | -0.2752 | -0.2927 | -0.256 | -0.3003 | -0.252 | 6.245 |
| pr | 84306 | 3.7928 | 4.3176 | 4.0161 | 4.1269 | normal | -0.2233 | -0.2378 | -0.2074 | -0.2444 | -0.204 | 5.5605 |
| pa | 84306 | 3.4292 | 3.8999 | 3.6564 | 3.7341 | normal | -0.2272 | -0.2417 | -0.2125 | -0.2452 | -0.2094 | 6.2138 |
| ra | 84306 | 1.7714 | 1.906 | 1.8805 | 1.926 | normal | -0.1092 | -0.1167 | -0.102 | -0.1189 | -0.1002 | 5.8049 |

The empirical baseline fell back to Poisson for 1.4% of rows (fewer than 5 prior games). The Normal baseline's one sd per stat is fitted on this season itself, which favours the baseline.

## 2. Bias of the mean (observed / predicted - 1)

### Overall

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | all | 84609 | -1.46 | -2.02 | -0.89 |
| rebounds | all | 84609 | -0.84 | -1.51 | -0.16 |
| assists | all | 84609 | -1.19 | -1.97 | -0.37 |
| threes | all | 84609 | -2.05 | -3.16 | -0.87 |
| pra | all | 84609 | -1.19 | -1.66 | -0.7 |
| pr | all | 84609 | -1.23 | -1.7 | -0.74 |
| pa | all | 84609 | -1.36 | -1.91 | -0.79 |
| ra | all | 84609 | -0.87 | -1.38 | -0.32 |

### By projected minutes

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 15-28 | 35775 | -1.45 | -2.52 | -0.31 |
| points | 28+ | 30461 | -0.23 | -1.06 | 0.59 |
| points | under 15 | 18373 | -11.38 | -14.43 | -8.47 |
| rebounds | 15-28 | 35775 | -1.16 | -2.17 | -0.05 |
| rebounds | 28+ | 30461 | 1.05 | 0.06 | 2.08 |
| rebounds | under 15 | 18373 | -8.64 | -11.29 | -5.94 |
| assists | 15-28 | 35775 | -1.79 | -3.27 | -0.22 |
| assists | 28+ | 30461 | -0.36 | -1.43 | 0.65 |
| assists | under 15 | 18373 | -5.62 | -9.52 | -1.75 |
| threes | 15-28 | 35775 | -3.55 | -5.43 | -1.67 |
| threes | 28+ | 30461 | 0.6 | -0.85 | 2.16 |
| threes | under 15 | 18373 | -14.03 | -18.07 | -9.63 |
| pra | 15-28 | 35775 | -1.24 | -2.09 | -0.32 |
| pra | 28+ | 30461 | 0.01 | -0.67 | 0.69 |
| pra | under 15 | 18373 | -9.6 | -12.23 | -7.08 |

### By month

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 2023-10 | 3570 | -1.83 | -4.23 | 0.55 |
| points | 2023-11 | 14121 | -1.12 | -2.37 | 0.06 |
| points | 2023-12 | 13263 | -0.91 | -2.32 | 0.45 |
| points | 2024-01 | 14973 | -1.37 | -2.85 | 0.19 |
| points | 2024-02 | 11190 | -2.74 | -4.18 | -1.26 |
| points | 2024-03 | 14643 | -1.49 | -2.83 | -0.16 |
| points | 2024-04 | 9969 | -1.67 | -3.44 | 0.16 |
| points | 2024-05 | 2538 | 1.55 | -1.17 | 4.59 |
| pra | 2023-10 | 3570 | -0.83 | -3.34 | 1.57 |
| pra | 2023-11 | 14121 | -1.29 | -2.35 | -0.27 |
| pra | 2023-12 | 13263 | -0.73 | -1.92 | 0.48 |
| pra | 2024-01 | 14973 | -1 | -2.24 | 0.31 |
| pra | 2024-02 | 11190 | -1.77 | -3.09 | -0.47 |
| pra | 2024-03 | 14643 | -1.44 | -2.51 | -0.28 |
| pra | 2024-04 | 9969 | -1.64 | -3.1 | 0.01 |
| pra | 2024-05 | 2538 | 1.94 | -0.27 | 4.5 |

### By position group

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | C | 6894 | -1.97 | -4 | 0.04 |
| points | F | 25305 | -1.18 | -2.35 | -0.02 |
| points | FC | 9522 | -1.43 | -3.14 | 0.18 |
| points | G | 33099 | -2 | -2.95 | -1.06 |
| points | GF | 9789 | 0.04 | -1.68 | 1.73 |
| rebounds | C | 6894 | -2.15 | -3.98 | -0.4 |
| rebounds | F | 25305 | -2.47 | -3.68 | -1.14 |
| rebounds | FC | 9522 | -1.55 | -3.09 | 0.07 |
| rebounds | G | 33099 | 0.68 | -0.55 | 1.92 |
| rebounds | GF | 9789 | 2.74 | 0.61 | 4.85 |
| assists | C | 6894 | 3.76 | 0.29 | 7.34 |
| assists | F | 25305 | -3.32 | -4.96 | -1.65 |
| assists | FC | 9522 | 3.55 | 0.74 | 6.34 |
| assists | G | 33099 | -2.07 | -3.2 | -0.86 |
| assists | GF | 9789 | 0.96 | -1.43 | 3.25 |

## 3. Calibration of P(X > line) at the ladder lines (raw, no recalibration)

| stat | logloss | brier | ece |
|---|---|---|---|
| points | 0.6608 | 0.2344 | 0.0109 |
| rebounds | 0.58 | 0.1978 | 0.01 |
| assists | 0.5393 | 0.1814 | 0.0061 |
| threes | 0.449 | 0.1478 | 0.0109 |
| pra | 0.6717 | 0.2395 | 0.0136 |
| pr | 0.6686 | 0.238 | 0.0134 |
| pa | 0.6655 | 0.2366 | 0.0106 |
| ra | 0.6243 | 0.2173 | 0.0124 |

## 4. Decision time (CRPS and the paired T-5 minus T-240 difference)

| stat | crps_T240 | crps_T60 | crps_T5 | T5_minus_T240 | lo | hi |
|---|---|---|---|---|---|---|
| points | 3.1017 | 3.0998 | 3.0949 | -0.0067 | -0.0101 | -0.0036 |
| rebounds | 1.2782 | 1.2778 | 1.2765 | -0.0017 | -0.0027 | -0.0006 |
| assists | 0.8861 | 0.8856 | 0.8847 | -0.0015 | -0.0023 | -0.0006 |
| threes | 0.555 | 0.5548 | 0.5545 | -0.0006 | -0.001 | -0.0001 |
| pra | 4.1325 | 4.1286 | 4.1194 | -0.0131 | -0.0184 | -0.0081 |
| pr | 3.7925 | 3.7895 | 3.7825 | -0.01 | -0.0144 | -0.0059 |
| pa | 3.4288 | 3.4264 | 3.4191 | -0.0097 | -0.014 | -0.0055 |
| ra | 1.771 | 1.7696 | 1.7672 | -0.0038 | -0.0056 | -0.002 |
