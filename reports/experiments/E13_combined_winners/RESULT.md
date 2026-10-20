# E13_combined_winners

Common games with the baseline: 8899 of 8916 (E1 needs both opening lines).

## Combined vs baseline (paired, day-bootstrap)

| metric      |    n |   diff (exp-base) | 95% CI               | seasons improved   | CI excludes 0 (better)   |
|:------------|-----:|------------------:|:---------------------|:-------------------|:-------------------------|
| ML log loss | 8899 |          -0.0085  | [-0.01447, -0.00328] | 7/7                | True                     |
| ML Brier    | 8899 |          -0.0015  | [-0.00227, -0.00071] | 7/7                | True                     |
| Spread MAE  | 8899 |          -0.02385 | [-0.04776, -0.00072] | 7/7                | True                     |
| Spread MSE  | 8899 |          -0.7653  | [-1.44485, -0.16341] | 7/7                | True                     |
| Total MAE   | 8899 |           0       | [0.00000, 0.00000]   | 0/7                | False                    |
| Total MSE   | 8899 |           0       | [0.00000, 0.00000]   | 0/7                | False                    |

## Additivity

- ML log loss: E3 alone -0.00763; E1-spread's own ML effect 0 (ML classifier untouched); expected additive -0.00763; combined -0.00850; combined - E3 alone = -0.00087 (95% CI -0.00157..-0.00015)
- ML seasons improved vs baseline: combined 7/7 (E3 alone 7/7)
- ECE: {"baseline": 0.0104, "E3 alone": 0.0068, "combined": 0.007}; contradiction rate (ML vs spread): {"baseline": 0.0657, "E3 alone": 0.0266, "combined": 0.0233}
- spread MAE: E1 alone -0.02385, combined -0.02385; combined - E1 alone = +0.00000 (identical by construction: the spread model is untouched by the other winners)
- total cover log loss (E4): combined -0.00075 (95% CI -0.00122..-0.00031), seasons better 5/7; ECE 0.0369 -> 0.0355; E4 alone is the same number (the point forecast is unchanged)

## Betting, unified 5%-edge rule, opening prices (descriptive; ROI differences are not tested against a keep rule)

| market   | run      |   bets |   hit % |   ROI % | ROI 95% CI    |   max DD ($100 units) | ROI diff vs baseline (95% CI)   |
|:---------|:---------|-------:|--------:|--------:|:--------------|----------------------:|:--------------------------------|
| ml       | baseline |   1771 |   48.45 |    0.74 | [-5.7, 7.4]   |                  57.5 | nan                             |
| ml       | combined |    447 |   42.28 |    1.51 | [-12.0, 15.3] |                  38.4 | +0.77 [-10.78, +13.28]          |
| spread   | baseline |    644 |   50.47 |   -3.66 | [-11.3, 3.7]  |                  34.7 | nan                             |
| spread   | combined |    141 |   47.52 |   -9.28 | [-25.5, 7.4]  |                  22.5 | -5.63 [-22.68, +9.40]           |
| total    | baseline |   1973 |   52.97 |    1.12 | [-2.9, 5.2]   |                  34.2 | nan                             |
| total    | combined |   1915 |   53.47 |    2.08 | [-2.2, 6.1]   |                  35.3 | +0.97 [-0.44, +2.45]            |

## Per-season ML log loss difference (combined - baseline)

{'2019-20': -0.00851, '2020-21': -0.00753, '2021-22': -0.00997, '2022-23': -0.01936, '2023-24': -0.00995, '2024-25': -0.00142, '2025-26': -0.00263}

## Per-season spread MAE difference (combined - baseline)

{'2019-20': -0.03702, '2020-21': -0.00362, '2021-22': -0.0211, '2022-23': -0.03404, '2023-24': -0.01186, '2024-25': -0.02685, '2025-26': -0.03184}

## Per-season total cover log-loss difference (E4)

{'2019-20': -0.00065, '2020-21': 0.00148, '2021-22': -0.00021, '2022-23': 0.0002, '2023-24': -0.0043, '2024-25': -0.00091, '2025-26': -0.0006}
