# Phase 4 evaluation: 2025-26 (v3_final_2025-26)

86088 played player-game rows (three decision times each, pooled), 1321 games. Intervals: 95% bootstrap over whole games (2000 draws); the date-clustered interval is the sensitivity check. Differences are paired (model minus best baseline; negative = model better).

## Pre-registered go/no-go criteria

- 1. CRPS beats the best baseline with the 95% interval below 0 in all 8 stats: **PASS** (8/8; failing: none)
- 2. Mean bias within +/-2% for points, rebounds, assists: **PASS** (points -0.1%, rebounds -0.7%, assists +1.5%)
- 3. ECE of P(over) at ladder lines at most 0.02 in every stat: **PASS** (worst 0.0141 in threes)
- 4. CRPS at T-5 not significantly worse than at T-240 (interval not entirely above 0) in every stat: **PASS** (failing: none)

## 1. Accuracy against the baselines (CRPS, lower is better)

| stat | n | crps_model | crps_poisson | crps_normal | crps_empirical20 | best_baseline | diff_vs_best | diff_lo_game | diff_hi_game | diff_lo_date | diff_hi_date | rel_improvement_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| points | 85779 | 3.1305 | 3.5525 | 3.3195 | 3.3956 | normal | -0.189 | -0.2034 | -0.1749 | -0.2142 | -0.1678 | 5.6943 |
| rebounds | 85779 | 1.2764 | 1.3546 | 1.3572 | 1.3813 | poisson | -0.0781 | -0.0848 | -0.0718 | -0.0899 | -0.069 | 5.7694 |
| assists | 85779 | 0.8898 | 0.9418 | 0.9584 | 0.9606 | poisson | -0.052 | -0.0564 | -0.0478 | -0.0592 | -0.0459 | 5.5242 |
| threes | 85779 | 0.5746 | 0.5937 | 0.6184 | 0.6065 | poisson | -0.019 | -0.0213 | -0.0168 | -0.0224 | -0.0162 | 3.2068 |
| pra | 85779 | 4.1208 | 4.7969 | 4.4348 | 4.6017 | normal | -0.314 | -0.3391 | -0.2907 | -0.3642 | -0.275 | 7.0797 |
| pr | 85779 | 3.7799 | 4.3448 | 4.0335 | 4.1711 | normal | -0.2536 | -0.274 | -0.2345 | -0.2944 | -0.2227 | 6.2873 |
| pa | 85779 | 3.4562 | 3.9709 | 3.7046 | 3.8021 | normal | -0.2484 | -0.2676 | -0.2305 | -0.2831 | -0.2194 | 6.7062 |
| ra | 85779 | 1.7696 | 1.9276 | 1.8969 | 1.9453 | normal | -0.1273 | -0.137 | -0.1178 | -0.1442 | -0.1131 | 6.7098 |

The empirical baseline fell back to Poisson for 1.4% of rows (fewer than 5 prior games). The Normal baseline's one sd per stat is fitted on this season itself, which favours the baseline.

## 2. Bias of the mean (observed / predicted - 1)

### Overall

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | all | 86088 | -0.12 | -0.75 | 0.47 |
| rebounds | all | 86088 | -0.73 | -1.45 | -0.01 |
| assists | all | 86088 | 1.46 | 0.63 | 2.27 |
| threes | all | 86088 | -3.81 | -4.89 | -2.77 |
| pra | all | 86088 | -0.03 | -0.57 | 0.47 |
| pr | all | 86088 | -0.28 | -0.81 | 0.21 |
| pa | all | 86088 | 0.18 | -0.45 | 0.77 |
| ra | all | 86088 | 0.11 | -0.48 | 0.66 |

### By projected minutes

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 15-28 | 40635 | 0.5 | -0.58 | 1.54 |
| points | 28+ | 26161 | -0.3 | -1.2 | 0.58 |
| points | under 15 | 19292 | -2.13 | -5.16 | 0.79 |
| rebounds | 15-28 | 40635 | -0.68 | -1.77 | 0.4 |
| rebounds | 28+ | 26161 | -0.94 | -2 | 0.03 |
| rebounds | under 15 | 19292 | -0.11 | -2.85 | 2.69 |
| assists | 15-28 | 40635 | 2.15 | 0.8 | 3.49 |
| assists | 28+ | 26161 | 1.15 | 0.03 | 2.18 |
| assists | under 15 | 19292 | -0.27 | -3.93 | 3.44 |
| threes | 15-28 | 40635 | -2.33 | -3.89 | -0.69 |
| threes | 28+ | 26161 | -4.73 | -6.25 | -3.24 |
| threes | under 15 | 19292 | -6.36 | -10.36 | -2.3 |
| pra | 15-28 | 40635 | 0.42 | -0.45 | 1.25 |
| pra | 28+ | 26161 | -0.22 | -0.97 | 0.49 |
| pra | under 15 | 19292 | -1.19 | -3.82 | 1.33 |

### By month

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | 2025-10 | 5373 | 6.56 | 3.64 | 9.32 |
| points | 2025-11 | 14352 | 1.64 | 0.34 | 3.05 |
| points | 2025-12 | 12795 | -0.66 | -2.2 | 0.88 |
| points | 2026-01 | 14964 | -2.78 | -4 | -1.56 |
| points | 2026-02 | 10770 | 0.56 | -1.07 | 2.24 |
| points | 2026-03 | 15510 | -0.85 | -2.16 | 0.49 |
| points | 2026-04 | 9390 | 0.12 | -2.03 | 2.32 |
| points | 2026-05 | 2631 | -2.48 | -4.94 | 0.18 |
| pra | 2025-10 | 5373 | 4.96 | 2.42 | 7.43 |
| pra | 2025-11 | 14352 | 0.75 | -0.36 | 1.97 |
| pra | 2025-12 | 12795 | -0.97 | -2.22 | 0.34 |
| pra | 2026-01 | 14964 | -1.78 | -2.84 | -0.73 |
| pra | 2026-02 | 10770 | 0.97 | -0.47 | 2.41 |
| pra | 2026-03 | 15510 | -0.48 | -1.6 | 0.64 |
| pra | 2026-04 | 9390 | 0.42 | -1.44 | 2.35 |
| pra | 2026-05 | 2631 | -1.44 | -3.61 | 0.82 |

### By position group

| stat | segment | n | obs_over_pred_minus_1_% | lo_% | hi_% |
|---|---|---|---|---|---|
| points | C | 8976 | -2.22 | -4.08 | -0.31 |
| points | F | 25374 | 1.27 | 0.11 | 2.46 |
| points | FC | 7551 | -1.75 | -3.7 | 0.23 |
| points | G | 35424 | -1.09 | -2.13 | -0.12 |
| points | GF | 8763 | 3.24 | 1.23 | 5.15 |
| rebounds | C | 8976 | -3.47 | -5.27 | -1.76 |
| rebounds | F | 25374 | -1.12 | -2.36 | 0.13 |
| rebounds | FC | 7551 | -0.89 | -2.79 | 0.89 |
| rebounds | G | 35424 | 0.35 | -0.95 | 1.68 |
| rebounds | GF | 8763 | 2.21 | 0.08 | 4.24 |
| assists | C | 8976 | 4.28 | 1.18 | 7.22 |
| assists | F | 25374 | 1.01 | -0.7 | 2.68 |
| assists | FC | 7551 | 0.98 | -2.21 | 4.2 |
| assists | G | 35424 | 0.85 | -0.36 | 2.03 |
| assists | GF | 8763 | 3.53 | 0.92 | 6.17 |

## 3. Calibration of P(X > line) at the ladder lines (raw, no recalibration)

| stat | logloss | brier | ece |
|---|---|---|---|
| points | 0.664 | 0.2359 | 0.0025 |
| rebounds | 0.5807 | 0.1981 | 0.0058 |
| assists | 0.5404 | 0.1818 | 0.0037 |
| threes | 0.46 | 0.152 | 0.0141 |
| pra | 0.6754 | 0.2413 | 0.0028 |
| pr | 0.6721 | 0.2397 | 0.0015 |
| pa | 0.6696 | 0.2385 | 0.005 |
| ra | 0.6258 | 0.218 | 0.0045 |

## 4. Decision time (CRPS and the paired T-5 minus T-240 difference)

| stat | crps_T240 | crps_T60 | crps_T5 | T5_minus_T240 | lo | hi |
|---|---|---|---|---|---|---|
| points | 3.1296 | 3.1283 | 3.1253 | -0.0043 | -0.0074 | -0.0013 |
| rebounds | 1.2772 | 1.2767 | 1.2755 | -0.0017 | -0.0029 | -0.0006 |
| assists | 0.889 | 0.8889 | 0.8884 | -0.0006 | -0.0014 | 0.0002 |
| threes | 0.5738 | 0.5738 | 0.5737 | -0.0001 | -0.0005 | 0.0004 |
| pra | 4.123 | 4.1206 | 4.1154 | -0.0075 | -0.0127 | -0.0027 |
| pr | 3.7813 | 3.7791 | 3.7747 | -0.0066 | -0.0108 | -0.0027 |
| pa | 3.4559 | 3.4544 | 3.4506 | -0.0053 | -0.0094 | -0.0015 |
| ra | 1.7706 | 1.7698 | 1.7683 | -0.0023 | -0.0042 | -0.0005 |
