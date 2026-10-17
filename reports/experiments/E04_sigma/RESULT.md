# E04_sigma

- spread per-season loss diff (registered): {'2019-20': 0.00038, '2020-21': 0.00032, '2021-22': 0.00077, '2022-23': 0.00031, '2023-24': 7e-05, '2024-25': 0.00044, '2025-26': 0.0002}
- spread: corr(sigma_i, |residual|) = 0.010
- total per-season loss diff (registered): {'2019-20': -0.00065, '2020-21': 0.00148, '2021-22': -0.00021, '2022-23': 0.0002, '2023-24': -0.0043, '2024-25': -0.00091, '2025-26': -0.0006}
- total: corr(sigma_i, |residual|) = 0.028

| market                                                                |   global-sigma loglik |   hetero loglik |   diff (loss, exp-base) | CI                   | seasons better   |   ECE global |   ECE hetero |   KEEP | mean sigma global/hetero   |
|:----------------------------------------------------------------------|----------------------:|----------------:|------------------------:|:---------------------|:-----------------|-------------:|-------------:|-------:|:---------------------------|
| spread                                                                |              -0.6952  |        -0.69556 |                 0.00036 | [+0.00020, +0.00052] | 0/7              |       0.0242 |       0.0256 |      0 | 13.41/12.89                |
| spread (diagnostic: rescaled to global mean, NOT the registered rule) |             nan       |        -0.69525 |                 5e-05   | [-0.00007, +0.00017] | 3/7              |     nan      |       0.0242 |    nan | nan                        |
| total                                                                 |              -0.69955 |        -0.6988  |                -0.00075 | [-0.00122, -0.00031] | 5/7              |       0.0369 |       0.0355 |      1 | 18.67/18.98                |
| total (diagnostic: rescaled to global mean, NOT the registered rule)  |             nan       |        -0.69875 |                -0.0008  | [-0.00124, -0.00040] | 4/7              |     nan      |       0.0357 |    nan | nan                        |

## Decision

- spread: **REJECT**
- total: **KEEP**