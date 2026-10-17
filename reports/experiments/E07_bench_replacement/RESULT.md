# E07_bench_replacement: bench-depth-aware replacement level in the injury value

## Paired comparison vs baseline (exp - base)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |           0.00215 | [-0.00198, 0.00649] | 2/7                | False                    |
| ML Brier    | 8916 |           0.00035 | [-0.00012, 0.00079] | 3/7                | False                    |
| Spread MAE  | 8916 |           0.0001  | [-0.00730, 0.00747] | 3/7                | False                    |
| Spread MSE  | 8916 |          -4e-05   | [-0.19677, 0.19008] | 4/7                | False                    |
| Total MAE   | 8916 |           0.01151 | [-0.00228, 0.02565] | 2/7                | False                    |
| Total MSE   | 8916 |           0.14713 | [-0.36586, 0.67615] | 4/7                | False                    |

## Per season (value, difference vs baseline)

|         | ml_logloss       | ml_brier         | spread_mae        | total_mae         |
|:--------|:-----------------|:-----------------|:------------------|:------------------|
| 2019-20 | 0.6320 (+0.0000) | 0.2206 (+0.0000) | 10.0964 (+0.0000) | 15.2925 (+0.0000) |
| 2020-21 | 0.6361 (+0.0000) | 0.2226 (+0.0000) | 10.9308 (+0.0000) | 14.9521 (+0.0000) |
| 2021-22 | 0.6288 (+0.0000) | 0.2154 (+0.0000) | 11.0001 (+0.0000) | 14.6798 (+0.0000) |
| 2022-23 | 0.6509 (+0.0000) | 0.2211 (+0.0000) | 9.9439 (+0.0000)  | 14.5073 (+0.0000) |
| 2023-24 | 0.6093 (+0.0034) | 0.2067 (+0.0011) | 10.7945 (+0.0083) | 14.6343 (+0.0370) |
| 2024-25 | 0.5963 (+0.0003) | 0.2053 (+0.0003) | 10.8626 (-0.0070) | 14.5967 (+0.0147) |
| 2025-26 | 0.6039 (+0.0108) | 0.2053 (+0.0009) | 11.2244 (-0.0006) | 14.9098 (+0.0260) |

## Pooled primary metrics

|                   |   baseline |   E07_bench_replacement |
|:------------------|-----------:|------------------------:|
| games             |  8916      |               8916      |
| ml_logloss        |     0.6199 |                  0.622  |
| ml_brier          |     0.2132 |                  0.2136 |
| ml_ece            |     0.0107 |                  0.0126 |
| ml_acc            |     0.6611 |                  0.6617 |
| ml_distinct_probs |   256      |                244      |
| spread_mae        |    10.7011 |                 10.7012 |
| spread_rmse       |    13.709  |                 13.709  |
| spread_bias       |    -0.154  |                 -0.1308 |
| total_mae         |    14.7721 |                 14.7836 |
| total_rmse        |    18.7114 |                 18.7153 |
| total_bias        |     1.285  |                  1.2065 |

## Selected-feature counts and tuned configs by fold

{"2019-20": {"ml": [21, 4], "spread": [23, 4], "total": [29, 0]}, "2020-21": {"ml": [20, 4], "spread": [20, 4], "total": [33, 4]}, "2021-22": {"ml": [20, 2], "spread": [20, 4], "total": [37, 0]}, "2022-23": {"ml": [20, 4], "spread": [20, 2], "total": [30, 4]}, "2023-24": {"ml": [21, 2], "spread": [20, 2], "total": [30, 4]}, "2024-25": {"ml": [22, 4], "spread": [24, 4], "total": [27, 0]}, "2025-26": {"ml": [22, 0], "spread": [21, 2], "total": [35, 4]}}

## New features chosen by selection (per fold: ml / spread / total)

```
2019-20: 0, 0, 0 -> ml: [] | spread: [] | total: []
2020-21: 0, 0, 0 -> ml: [] | spread: [] | total: []
2021-22: 0, 0, 0 -> ml: [] | spread: [] | total: []
2022-23: 0, 0, 0 -> ml: [] | spread: [] | total: []
2023-24: 0, 0, 0 -> ml: [] | spread: [] | total: []
2024-25: 0, 0, 0 -> ml: [] | spread: [] | total: []
2025-26: 0, 0, 0 -> ml: [] | spread: [] | total: []
```

## Betting (5% rule), experiment

|              |      ml |   spread |   total |
|:-------------|--------:|---------:|--------:|
| bets         | 1878    |   688    | 2041    |
| hit_rate     |   48.72 |    51.74 |   53.6  |
| roi          |    1.5  |    -1.22 |    2.33 |
| max_drawdown |   57.51 |    32.27 |   40.82 |

CLV points (mean): {'spread': {'n': 656, 'mean': 0.43140243902439024}, 'total': {'n': 1975, 'mean': 0.7349367088607595}}

## Games with at least one Out player (7144 games in the training set; subset used for the keep rule)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 7144 |           0.00238 | [-0.00321, 0.00822] | 2/7                | False                    |
| ML Brier    | 7144 |           0.00041 | [-0.00012, 0.00094] | 3/7                | False                    |
| Spread MAE  | 7144 |           0.00241 | [-0.00565, 0.01074] | 3/7                | False                    |
| Spread MSE  | 7144 |           0.05018 | [-0.17983, 0.27919] | 4/7                | False                    |
| Total MAE   | 7144 |           0.01421 | [-0.00196, 0.03014] | 1/7                | False                    |
| Total MSE   | 7144 |           0.20236 | [-0.38269, 0.78076] | 5/7                | False                    |

## Feature-importance change (refit of each fold's chosen config; folds 2023-24..2025-26)

|                       |   ('inj features selected', 'E07_bench_replacement') |   ('inj features selected', 'baseline') |   ('inj share of total gain', 'E07_bench_replacement') |   ('inj share of total gain', 'baseline') |
|:----------------------|-----------------------------------------------------:|----------------------------------------:|-------------------------------------------------------:|------------------------------------------:|
| ('2023-24', 'ml')     |                                                    0 |                                       0 |                                                 0      |                                    0      |
| ('2023-24', 'spread') |                                                    1 |                                       0 |                                                 0.0018 |                                    0      |
| ('2023-24', 'total')  |                                                    0 |                                       1 |                                                 0      |                                    0.0009 |
| ('2024-25', 'ml')     |                                                    2 |                                       1 |                                                 0.0115 |                                    0.0066 |
| ('2024-25', 'spread') |                                                    2 |                                       1 |                                                 0.005  |                                    0.0028 |
| ('2024-25', 'total')  |                                                    0 |                                       1 |                                                 0      |                                    0.0077 |
| ('2025-26', 'ml')     |                                                    2 |                                       1 |                                                 0.0051 |                                    0.0026 |
| ('2025-26', 'spread') |                                                    3 |                                       3 |                                                 0.0036 |                                    0.0008 |
| ('2025-26', 'total')  |                                                    1 |                                       1 |                                                 0.0064 |                                    0.0034 |

## Decision (pre-registered)

- **REJECT**  (main criterion per metric: {'games with >=1 Out: Spread MAE': False, 'games with >=1 Out: Total MAE': False}; no-harm per metric: {'ML log loss': True, 'Spread MAE': True, 'Total MAE': True})
- ML log loss: diff +0.00215 (base 0.6199), 95% CI [-0.00198, +0.00649], 2/7 seasons improved
- Spread MAE: diff +0.00010 (base 10.7011), 95% CI [-0.00730, +0.00747], 3/7 seasons improved
- Total MAE: diff +0.01151 (base 14.7721), 95% CI [-0.00228, +0.02565], 2/7 seasons improved