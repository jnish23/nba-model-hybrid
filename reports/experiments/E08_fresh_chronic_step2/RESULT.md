# E08_fresh_chronic_step2: fresh vs chronic absences (folds 2022-23..2025-26)

## Paired comparison vs baseline (exp - base)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 5279 |           0.00522 | [0.00009, 0.01287]  | 1/4                | False                    |
| ML Brier    | 5279 |           0.00044 | [-0.00035, 0.00121] | 2/4                | False                    |
| Spread MAE  | 5279 |           0.01628 | [0.00370, 0.02910]  | 0/4                | False                    |
| Spread MSE  | 5279 |           0.28908 | [-0.07040, 0.65692] | 1/4                | False                    |
| Total MAE   | 5279 |           0.00734 | [-0.01940, 0.03536] | 0/4                | False                    |
| Total MSE   | 5279 |           0.07446 | [-0.93997, 1.17094] | 1/4                | False                    |

## Per season (value, difference vs baseline)

|         | ml_logloss       | ml_brier         | spread_mae        | total_mae         |
|:--------|:-----------------|:-----------------|:------------------|:------------------|
| 2022-23 | 0.6509 (+0.0000) | 0.2211 (+0.0000) | 9.9439 (+0.0000)  | 14.5073 (+0.0000) |
| 2023-24 | 0.6263 (+0.0204) | 0.2071 (+0.0015) | 10.7933 (+0.0072) | 14.6123 (+0.0150) |
| 2024-25 | 0.5990 (+0.0030) | 0.2061 (+0.0011) | 10.8808 (+0.0112) | 14.5877 (+0.0057) |
| 2025-26 | 0.5906 (-0.0025) | 0.2035 (-0.0008) | 11.2716 (+0.0467) | 14.8925 (+0.0087) |

## Pooled primary metrics

|                   |   baseline |   E08_fresh_chronic_step2 |
|:------------------|-----------:|--------------------------:|
| games             |  5279      |                 5279      |
| ml_logloss        |     0.6114 |                    0.6167 |
| ml_brier          |     0.209  |                    0.2095 |
| ml_ece            |     0.0127 |                    0.0138 |
| ml_acc            |     0.6732 |                    0.6765 |
| ml_distinct_probs |   151      |                  143      |
| spread_mae        |    10.7062 |                   10.7225 |
| spread_rmse       |    13.7697 |                   13.7802 |
| spread_bias       |     0.0571 |                    0.0964 |
| total_mae         |    14.6427 |                   14.65   |
| total_rmse        |    18.5658 |                   18.5678 |
| total_bias        |     1.3871 |                    1.3433 |

## Selected-feature counts and tuned configs by fold

{"2022-23": {"ml": [20, 4], "spread": [20, 2], "total": [30, 4]}, "2023-24": {"ml": [25, 4], "spread": [20, 4], "total": [38, 0]}, "2024-25": {"ml": [26, 2], "spread": [23, 0], "total": [30, 2]}, "2025-26": {"ml": [21, 2], "spread": [20, 4], "total": [32, 2]}}

## New features chosen by selection (per fold: ml / spread / total)

```
2022-23: 0, 0, 0 -> ml: [] | spread: [] | total: []
2023-24: 0, 0, 0 -> ml: [] | spread: [] | total: []
2024-25: 0, 0, 0 -> ml: [] | spread: [] | total: []
2025-26: 0, 0, 0 -> ml: [] | spread: [] | total: []
```

## Betting (5% rule), experiment

|              |     ml |   spread |   total |
|:-------------|-------:|---------:|--------:|
| bets         | 914    |   363    | 1240    |
| hit_rate     |  50.98 |    53.17 |   54.11 |
| roi          |  -4.53 |     1.5  |    3.31 |
| max_drawdown |  50.32 |    26.09 |   35.73 |

CLV points (mean): {'spread': {'n': 363, 'mean': 0.23140495867768596}, 'total': {'n': 1240, 'mean': 0.854233870967742}}

## Decision (pre-registered)

- **REJECT**  (main criterion per metric: {'ML log loss': False, 'Spread MAE': False, 'Total MAE': False}; no-harm per metric: {'ML log loss': False, 'Spread MAE': True, 'Total MAE': True})
- ML log loss: diff +0.00522 (base 0.6114), 95% CI [+0.00009, +0.01287], 1/4 seasons improved
- Spread MAE: diff +0.01628 (base 10.7062), 95% CI [+0.00370, +0.02910], 0/4 seasons improved
- Total MAE: diff +0.00734 (base 14.6427), 95% CI [-0.01940, +0.03536], 0/4 seasons improved