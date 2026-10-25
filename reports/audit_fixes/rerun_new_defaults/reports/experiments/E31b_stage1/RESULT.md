# E31b_stage1

Exploratory (E31b), post-audit defaults {'calib_cap': 1300, 'refit_on_all': True, 'ml_calibrator': 'platt'}. Reference `E25_with_odds`. Nesting check passed in every fold and target. Total time 297s.

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
| ML log loss | 8916 |          -0.00062 | [-0.00174, 0.00061] | 4/7                | False                    |
| ML Brier    | 8916 |          -0.00037 | [-0.00087, 0.00016] | 4/7                | False                    |
| Spread MAE  | 8916 |          -0.00077 | [-0.01608, 0.01594] | 3/7                | False                    |
| Spread MSE  | 8916 |          -0.13312 | [-0.55874, 0.32178] | 3/7                | False                    |
| Total MAE   | 8916 |           0.01979 | [-0.01738, 0.05676] | 2/7                | False                    |
| Total MSE   | 8916 |           0.80186 | [-0.59016, 2.07258] | 3/7                | False                    |

## Per season

- ML log loss: {"2019-20": 0.0003, "2020-21": 0.0012, "2021-22": -0.003, "2022-23": 0.001, "2023-24": -0.0012, "2024-25": -0.0012, "2025-26": -0.0012}  (pooled -0.00062; seed-only noise floor about 0.003)
- Spread MAE: {"2019-20": 0.028, "2020-21": 0.0076, "2021-22": 0.027, "2022-23": 0.0134, "2023-24": -0.0123, "2024-25": -0.0295, "2025-26": -0.0347}  (pooled -0.00077; seed-only noise floor about 0.011)
- Total MAE: {"2019-20": 0.1757, "2020-21": -0.0047, "2021-22": 0.0168, "2022-23": 0.0013, "2023-24": 0.0126, "2024-25": -0.0628, "2025-26": 0.0178}  (pooled +0.01979; seed-only noise floor about 0.005)