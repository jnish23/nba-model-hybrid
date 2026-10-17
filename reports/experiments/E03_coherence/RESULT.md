# E03_coherence

## (a) Contradictions in the baseline

- ML vs spread (ml_prob > .5 vs predicted margin > 0): **6.56%** of 8916 games
- ML vs opening line (ml_prob > .5 vs opening home spread < 0): 7.29%
- predicted margin vs opening line direction: 2.66%
- by season: {'2019-20': 0.112, '2020-21': 0.0888, '2021-22': 0.096, '2022-23': 0.0545, '2023-24': 0.0417, '2024-25': 0.0379, '2025-26': 0.0371}
- baseline log loss 0.61989; blend weights on the classifier by fold: {'2019-20': 0.2, '2020-21': 0.1, '2021-22': 0.5, '2022-23': 1.0, '2023-24': 0.5, '2024-25': 0.4, '2025-26': 0.2}

## (b), (c) vs baseline

| variant                |   logloss diff | CI                   | seasons better   |   Brier diff |    ECE |   contradiction rate | KEEP   |
|:-----------------------|---------------:|:---------------------|:-----------------|-------------:|-------:|---------------------:|:-------|
| blend_with_spread_prob |        0.00599 | [-0.00048, +0.01339] | 2/7              |     -0.00026 | 0.013  |               0.0483 | False  |
| stack_logit_margin     |       -0.00765 | [-0.01335, -0.00245] | 7/7              |     -0.00123 | 0.0071 |               0.0266 | True   |

## Decision

- blend_with_spread_prob: **REJECT**
- stack_logit_margin: **KEEP**

### blend per-season log loss diff
{'2019-20': -0.00464, '2020-21': 0.00839, '2021-22': 0.00101, '2022-23': 0.0, '2023-24': 0.03632, '2024-25': 0.00149, '2025-26': -0.00173}

### stack per-season log loss diff
{'2019-20': -0.00704, '2020-21': -0.00559, '2021-22': -0.00952, '2022-23': -0.01865, '2023-24': -0.0102, '2024-25': -0.00118, '2025-26': -0.00108}