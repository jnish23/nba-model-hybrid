# E11_exposure

Bets: 4388 (5% edge rule) on 3614 games, 706 games with more than one bet.
Latent ML/spread dependence rho by fold: {"2019-20": 0.869, "2020-21": 0.874, "2021-22": 0.906, "2022-23": 0.904, "2023-24": 0.903, "2024-25": 0.9, "2025-26": 0.884}

| strategy                             |   bets |   avg stake % of bankroll |   actual log-growth (sum over days) |   actual max DD |   boot median log-growth | log-growth 95% CI   |   daily vol (median) |   max DD median |   max DD p95 |   paired growth diff vs (i) | diff 95% CI     |   KEEP |
|:-------------------------------------|-------:|--------------------------:|------------------------------------:|----------------:|-------------------------:|:--------------------|---------------------:|----------------:|-------------:|----------------------------:|:----------------|-------:|
| (i) independent 1/4 Kelly [baseline] |   4388 |                      4.89 |                             -5.1349 |           0.999 |                  -5.1428 | [-12.587, 2.383]    |               0.1114 |           0.999 |            1 |                    nan      | nan             |    nan |
| (ii) best-EV market per game         |   3614 |                      5.02 |                             -4.1641 |           0.996 |                  -4.0781 | [-10.791, 2.313]    |               0.1003 |           0.998 |            1 |                      0.9963 | [-1.231, 3.338] |      0 |
| (iii) joint Kelly (1/4)              |   4319 |                      4.31 |                             -3.7932 |           0.997 |                  -3.6692 | [-9.858, 2.501]     |               0.0958 |           0.997 |            1 |                      1.3157 | [-1.090, 3.456] |      0 |

## Decision (pre-registered)

- (ii) best-EV market per game: **REJECT**
- (iii) joint Kelly (1/4): **REJECT**

Bets per game: {1: 2908, 2: 638, 3: 68}