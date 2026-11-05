# E31_all

Exploratory (E31). Reference `E25_with_odds`. Subset check passed in every fold and target. Total time 439s.

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
| ML log loss | 8916 |           0.00121 | [-0.00706, 0.00910] | 4/7                | False                    |
| ML Brier    | 8916 |           0.00054 | [-0.00026, 0.00145] | 3/7                | False                    |
| Spread MAE  | 8916 |          -0.00141 | [-0.01655, 0.01419] | 3/7                | False                    |
| Spread MSE  | 8916 |          -0.04531 | [-0.49478, 0.37759] | 4/7                | False                    |
| Total MAE   | 8916 |           0.01508 | [-0.01535, 0.04641] | 2/7                | False                    |
| Total MSE   | 8916 |           1.18505 | [-0.08789, 2.43171] | 2/7                | False                    |

## Per season

- ML log loss: {"2019-20": -0.002, "2020-21": 0.0031, "2021-22": 0.0175, "2022-23": -0.0028, "2023-24": -0.0117, "2024-25": -0.0004, "2025-26": 0.0044}  (pooled +0.00121; seed-only noise floor about 0.003)
- Spread MAE: {"2019-20": 0.0157, "2020-21": -0.0277, "2021-22": -0.0344, "2022-23": 0.016, "2023-24": -0.0096, "2024-25": 0.0103, "2025-26": 0.0192}  (pooled -0.00141; seed-only noise floor about 0.011)
- Total MAE: {"2019-20": -0.051, "2020-21": 0.0638, "2021-22": 0.0235, "2022-23": -0.0037, "2023-24": 0.0328, "2024-25": 0.0163, "2025-26": 0.0205}  (pooled +0.01508; seed-only noise floor about 0.005)