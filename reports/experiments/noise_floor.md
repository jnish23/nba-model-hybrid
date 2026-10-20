# Selection-noise floor (baseline with other selection seeds vs the seed-42 baseline)

## seed 43

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |           0.0001  | [-0.00674, 0.00713] | 3/7                | False                    |
| ML Brier    | 8916 |          -0.0005  | [-0.00125, 0.00033] | 3/7                | False                    |
| Spread MAE  | 8916 |           0.01016 | [-0.00250, 0.02212] | 3/7                | False                    |
| Spread MSE  | 8916 |           0.16688 | [-0.19287, 0.52335] | 4/7                | False                    |
| Total MAE   | 8916 |           0.00499 | [-0.01877, 0.02793] | 2/7                | False                    |
| Total MSE   | 8916 |          -0.16094 | [-1.08407, 0.74942] | 4/7                | False                    |

## seed 44

| metric      |    n |   diff (exp-base) | 95% CI              | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:--------------------|:-------------------|:-------------------------|
| ML log loss | 8916 |           0.00321 | [-0.00161, 0.00915] | 3/7                | False                    |
| ML Brier    | 8916 |           0.00035 | [-0.00033, 0.00102] | 3/7                | False                    |
| Spread MAE  | 8916 |           0.011   | [-0.00215, 0.02463] | 2/7                | False                    |
| Spread MSE  | 8916 |           0.22711 | [-0.15443, 0.61061] | 2/7                | False                    |
| Total MAE   | 8916 |           0.00489 | [-0.01795, 0.02752] | 3/7                | False                    |
| Total MSE   | 8916 |           0.22749 | [-0.71605, 1.09510] | 2/7                | False                    |
