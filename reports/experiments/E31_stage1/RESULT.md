# E31_stage1

Exploratory (E31). Reference `E25_with_odds`. Subset check passed in every fold and target. Total time 392s.

## Features per fold (reference / stage 1 / all)

```
2019-20: ml 21/111/134, spread 23/111/134, total 29/210/266
2020-21: ml 20/111/134, spread 20/111/134, total 33/212/266
2021-22: ml 20/111/134, spread 20/111/134, total 37/212/266
2022-23: ml 20/112/134, spread 20/112/134, total 30/210/266
2023-24: ml 20/116/140, spread 21/116/140, total 31/214/276
2024-25: ml 26/115/140, spread 22/115/140, total 31/213/276
2025-26: ml 20/115/140, spread 27/115/140, total 29/212/276
```

## Paired comparison (arm - reference; negative = arm better)

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |           0.00104 | [-0.00437, 0.00627] | 2/7                | False                    |
| ML Brier    | 8916 |           0.00068 | [-0.00010, 0.00151] | 2/7                | False                    |
| Spread MAE  | 8916 |           0.00084 | [-0.01425, 0.01592] | 4/7                | False                    |
| Spread MSE  | 8916 |           0.01772 | [-0.39462, 0.41291] | 2/7                | False                    |
| Total MAE   | 8916 |           0.02038 | [-0.00789, 0.04892] | 1/7                | False                    |
| Total MSE   | 8916 |           0.50983 | [-0.61233, 1.67968] | 3/7                | False                    |

## Per season

- ML log loss: {"2019-20": -0.0026, "2020-21": 0.0039, "2021-22": 0.0019, "2022-23": 0.0054, "2023-24": -0.0103, "2024-25": 0.0076, "2025-26": 0.0012}  (pooled +0.00104; seed-only noise floor about 0.003)
- Spread MAE: {"2019-20": 0.0355, "2020-21": -0.0398, "2021-22": -0.0189, "2022-23": 0.0228, "2023-24": -0.0113, "2024-25": -0.0009, "2025-26": 0.0185}  (pooled +0.00084; seed-only noise floor about 0.011)
- Total MAE: {"2019-20": -0.0663, "2020-21": 0.0198, "2021-22": 0.0408, "2022-23": 0.0364, "2023-24": 0.0112, "2024-25": 0.0306, "2025-26": 0.0584}  (pooled +0.02038; seed-only noise floor about 0.005)