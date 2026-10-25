# E31b_all

Exploratory (E31b), post-audit defaults {'calib_cap': 1300, 'refit_on_all': True, 'ml_calibrator': 'platt'}. Reference `E25_with_odds`. Nesting check passed in every fold and target. Total time 390s.

## Features per fold (reference / stage 1 / all)

```
2019-20: ml 25/113/136, spread 25/113/136, total 25/214/268
2020-21: ml 25/114/136, spread 25/114/136, total 27/215/268
2021-22: ml 25/115/136, spread 25/115/136, total 33/214/268
2022-23: ml 25/119/140, spread 25/119/140, total 33/217/276
2023-24: ml 25/118/140, spread 25/118/140, total 30/216/276
2024-25: ml 25/118/140, spread 25/118/140, total 34/215/276
2025-26: ml 25/119/140, spread 25/119/140, total 33/213/276
```

## Paired comparison (arm - reference; negative = arm better)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |          -0.00104 | [-0.00225, 0.00025] | 5/7                | False                    |
| ML Brier    | 8916 |          -0.00051 | [-0.00103, 0.00005] | 5/7                | False                    |
| Spread MAE  | 8916 |          -0.00541 | [-0.02153, 0.01147] | 3/7                | False                    |
| Spread MSE  | 8916 |          -0.30619 | [-0.81399, 0.15356] | 3/7                | False                    |
| Total MAE   | 8916 |           0.01921 | [-0.02019, 0.05926] | 3/7                | False                    |
| Total MSE   | 8916 |           0.53819 | [-0.87914, 1.95165] | 3/7                | False                    |

## Per season

- ML log loss: {"2019-20": -0.0, "2020-21": 0.0005, "2021-22": -0.0033, "2022-23": 0.0006, "2023-24": -0.0029, "2024-25": -0.0014, "2025-26": -0.0005}  (pooled -0.00104; seed-only noise floor about 0.003)
- Spread MAE: {"2019-20": 0.0212, "2020-21": 0.0008, "2021-22": 0.0137, "2022-23": 0.0183, "2023-24": -0.0209, "2024-25": -0.0328, "2025-26": -0.0338}  (pooled -0.00541; seed-only noise floor about 0.011)
- Total MAE: {"2019-20": 0.1629, "2020-21": -0.0245, "2021-22": -0.0049, "2022-23": 0.0024, "2023-24": 0.0386, "2024-25": -0.0532, "2025-26": 0.0275}  (pooled +0.01921; seed-only noise floor about 0.005)