# 7c / 7d on rule C3 (k_walkforward.json (['2023-24'] fit)), model a11_r0_p5

Median-price ROI (primary), settlement `stands`, date-clustered bootstrap (2000 resamples, seed 20261112), paired on the same resamples; Bonferroni level 0.05/23. 2023-24 is in sample for the rule parameters, 2024-25 out of sample.

## 7c: caps and quarter-Kelly (f* = EV/(d-1) at the best price; stake = min(0.25 f*, 0.02) B, B = 100; per-game total at most 0.05 B, lower-EV bets cut first)

| variant | season | bets | roi | n_stakes | stake_units | n_dates | sharpe | max_drawdown | dd_over_stake | roi_best | sharpe_best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flat | 2023-24 | 25488 | 0.0135 | 25488 | 25488 | 207 | 0.0792 | 101.8135 | 0.004 | 0.0501 | 0.3794 |
| cap1 | 2023-24 | 13505 | 0.0165 | 13505 | 13505 | 207 | 0.1333 | 58.5527 | 0.0043 | 0.0575 | 0.3968 |
| cap2 | 2023-24 | 21020 | 0.0186 | 21020 | 21020 | 207 | 0.1299 | 59.1836 | 0.0028 | 0.0571 | 0.4402 |
| kelly | 2023-24 | 25488 | 0.0452 | 4258 | 6583.7466 | 207 | 0.119 | 55.8876 | 0.0085 | 0.0975 | 0.2672 |
| flat | 2024-25 | 25095 | -0.0106 | 25095 | 25095 | 213 | -0.0243 | 324.0818 | 0.0129 | 0.0348 | 0.2808 |
| cap1 | 2024-25 | 13834 | -0.0007 | 13834 | 13834 | 213 | 0.0194 | 94.7305 | 0.0068 | 0.0495 | 0.2941 |
| cap2 | 2024-25 | 21148 | -0.0059 | 21148 | 21148 | 213 | 0.0084 | 187.7622 | 0.0089 | 0.0414 | 0.3139 |
| kelly | 2024-25 | 25095 | 0.0085 | 4241 | 6540.1641 | 213 | 0.0668 | 69.7565 | 0.0107 | 0.0712 | 0.2473 |

### Paired median-ROI differences against flat

| season | comparison | point | lo | hi | se | lo_bonf | hi_bonf |
|---|---|---|---|---|---|---|---|
| 2023-24 | cap1 - flat (median ROI) | 0.003 | -0.0071 | 0.0141 | 0.0054 | -0.0128 | 0.0203 |
| 2023-24 | cap2 - flat (median ROI) | 0.005 | 0.0001 | 0.0102 | 0.0026 | -0.0024 | 0.0128 |
| 2023-24 | kelly - flat (median ROI) | 0.0317 | 0.004 | 0.0587 | 0.0139 | -0.0116 | 0.0757 |
| 2024-25 | cap1 - flat (median ROI) | 0.01 | 0.0006 | 0.0194 | 0.0045 | -0.0045 | 0.0218 |
| 2024-25 | cap2 - flat (median ROI) | 0.0048 | 0.0005 | 0.0092 | 0.0022 | -0.0012 | 0.0123 |
| 2024-25 | kelly - flat (median ROI) | 0.0191 | -0.0094 | 0.0488 | 0.0149 | -0.0227 | 0.0639 |

### By stat

| variant | season | stat | bets | stakes | roi_med | roi_best |
|---|---|---|---|---|---|---|
| flat | 2023-24 | assists | 6195 | 6195 | 0.0059 | 0.0466 |
| flat | 2023-24 | ra | 4441 | 4441 | 0.02 | 0.0572 |
| flat | 2023-24 | rebounds | 8827 | 8827 | 0.0167 | 0.0512 |
| flat | 2023-24 | threes | 6025 | 6025 | 0.012 | 0.0468 |
| cap1 | 2023-24 | assists | 3623 | 3623 | 0.0128 | 0.056 |
| cap1 | 2023-24 | ra | 1053 | 1053 | 0.0295 | 0.0872 |
| cap1 | 2023-24 | rebounds | 5200 | 5200 | 0.0166 | 0.0554 |
| cap1 | 2023-24 | threes | 3629 | 3629 | 0.0164 | 0.0534 |
| cap2 | 2023-24 | assists | 5272 | 5272 | 0.0144 | 0.0564 |
| cap2 | 2023-24 | ra | 2969 | 2969 | 0.0249 | 0.0684 |
| cap2 | 2023-24 | rebounds | 7648 | 7648 | 0.0234 | 0.0593 |
| cap2 | 2023-24 | threes | 5131 | 5131 | 0.0121 | 0.0478 |
| kelly | 2023-24 | assists | 6195 | 6195 | 0.0858 | 0.1367 |
| kelly | 2023-24 | ra | 4441 | 4441 | -0.0266 | 0.0413 |
| kelly | 2023-24 | rebounds | 8827 | 8827 | 0.0328 | 0.0903 |
| kelly | 2023-24 | threes | 6025 | 6025 | 0.0361 | 0.0804 |
| flat | 2024-25 | assists | 6332 | 6332 | -0.0063 | 0.0425 |
| flat | 2024-25 | ra | 3832 | 3832 | -0.0115 | 0.036 |
| flat | 2024-25 | rebounds | 9217 | 9217 | 0.003 | 0.0457 |
| flat | 2024-25 | threes | 5714 | 5714 | -0.0368 | 0.0081 |
| cap1 | 2024-25 | assists | 3637 | 3637 | 0.0102 | 0.0638 |
| cap1 | 2024-25 | ra | 1188 | 1188 | 0.0002 | 0.0623 |
| cap1 | 2024-25 | rebounds | 5814 | 5814 | 0.0072 | 0.0537 |
| cap1 | 2024-25 | threes | 3195 | 3195 | -0.0278 | 0.021 |
| cap2 | 2024-25 | assists | 5409 | 5409 | 0.0022 | 0.0527 |
| cap2 | 2024-25 | ra | 2699 | 2699 | 0.0034 | 0.0571 |
| cap2 | 2024-25 | rebounds | 8190 | 8190 | 0.0059 | 0.0499 |
| cap2 | 2024-25 | threes | 4850 | 4850 | -0.0399 | 0.0058 |
| kelly | 2024-25 | assists | 6332 | 6332 | 0.0006 | 0.0612 |
| kelly | 2024-25 | ra | 3832 | 3832 | 0.0366 | 0.1107 |
| kelly | 2024-25 | rebounds | 9217 | 9217 | 0.046 | 0.1138 |
| kelly | 2024-25 | threes | 5714 | 5714 | -0.0391 | 0.0157 |

### Bets sharing a player-game: {'2023-24': {'bets': 25488, 'share_of_bets_sharing_a_player_game': 0.7649874450721909, 'distinct_player_games': 13505, 'mean_bets_per_player_game': 1.8873009996297667, 'max_bets_per_player_game': 4}, '2024-25': {'bets': 25095, 'share_of_bets_sharing_a_player_game': 0.7401872883044431, 'distinct_player_games': 13834, 'mean_bets_per_player_game': 1.8140089634234495, 'max_bets_per_player_game': 4}}

### Keep rule, cap1: **ADOPT**

| item | clause | value | threshold | result |
|---|---|---|---|---|
| 7c | cap1: 2024-25 Sharpe at least uncapped (-0.0243) | 0.0194 | -0.0243 | PASS |
| 7c | cap1: 2024-25 median ROI no more than 0.5 points below uncapped (-0.0106) | -0.0007 | -0.0156 | PASS |
| 7c | cap1: 2023-24 Sharpe no lower than uncapped (0.0792) | 0.1333 | 0.0792 | PASS |

### Keep rule, cap2: **ADOPT**

| item | clause | value | threshold | result |
|---|---|---|---|---|
| 7c | cap2: 2024-25 Sharpe at least uncapped (-0.0243) | 0.0084 | -0.0243 | PASS |
| 7c | cap2: 2024-25 median ROI no more than 0.5 points below uncapped (-0.0106) | -0.0059 | -0.0156 | PASS |
| 7c | cap2: 2023-24 Sharpe no lower than uncapped (0.0792) | 0.1299 | 0.0792 | PASS |

### Keep rule, kelly: **ADOPT** (Kelly: evaluation only, no adoption before the fill study, owner decision G9)

| item | clause | value | threshold | result |
|---|---|---|---|---|
| 7c | Kelly: 2024-25 Sharpe at least 1.10 x flat (-0.0243) | 0.0668 | -0.0219 | PASS |
| 7c | Kelly: 2024-25 max drawdown / total stake at most flat's | 0.0107 | 0.0129 | PASS |
| 7c | Kelly: 2024-25 median ROI no more than 0.5 points below flat (-0.0106) | 0.0085 | -0.0156 | PASS |
| 7c | Kelly: 2023-24 Sharpe not below flat (0.0792) (ordering not reversed) | 0.119 | 0.0792 | PASS |

## 7d: under bets, expected minutes < 20 against >= 20 (the single test; unadjusted 95% bound as amended 2026-11-12; Bonferroni interval printed beside it)

Keep (sub-hypothesis of the forward test): **DROP**

| item | clause | value | threshold | result |
|---|---|---|---|---|
| 7d | pooled under <20 minus >=20: unadjusted 95% lower bound above 0 | -0.0532 | 0 | FAIL |
| 7d | 2023-24: point estimate positive | -0.0277 | 0 | FAIL |
| 7d | 2024-25: point estimate positive | -0.0162 | 0 | FAIL |

### Splits (unders: the test; overs: descriptive)

| side | sample | stakes_lt20 | roi_lt20 | lt20_lo | lt20_hi | stakes_ge20 | roi_ge20 | ge20_lo | ge20_hi | diff | diff_lo | diff_hi | diff_lo_bonf | diff_hi_bonf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| under | pooled | 5063 | -0.0124 | -0.0409 | 0.0195 | 36793 | 0.01 | -0.0012 | 0.021 | -0.0224 | -0.0532 | 0.0109 | -0.0706 | 0.0266 |
| under | 2023-24 | 2600 | -0.0066 | -0.0484 | 0.0349 | 19371 | 0.0212 | 0.0054 | 0.0372 | -0.0277 | -0.0691 | 0.0149 | -0.0936 | 0.0365 |
| under | 2024-25 | 2463 | -0.0185 | -0.0597 | 0.0242 | 17422 | -0.0024 | -0.0178 | 0.013 | -0.0162 | -0.0626 | 0.0308 | -0.0865 | 0.0557 |
| over | pooled | 133 | -0.1922 | -0.3622 | -0.0218 | 8594 | -0.0235 | -0.0466 | -0.0022 | -0.1687 | -0.3383 | 0.0075 | -0.4355 | 0.0875 |
| over | 2023-24 | 58 | -0.1662 | -0.4186 | 0.114 | 3459 | -0.0111 | -0.0457 | 0.0249 | -0.1551 | -0.4093 | 0.123 | -0.5659 | 0.3283 |
| over | 2024-25 | 75 | -0.2123 | -0.4252 | 0.0208 | 5135 | -0.0319 | -0.0583 | -0.0049 | -0.1804 | -0.3942 | 0.0501 | -0.5147 | 0.1905 |

### No-model version: every under at T-60, median price (descriptive)

| season | expected_minutes | stakes | roi_med |
|---|---|---|---|
| 2023-24 | <20 | 4369 | -0.0402 |
| 2023-24 | >=20 | 48950 | -0.035 |
| 2024-25 | <20 | 3164 | -0.0217 |
| 2024-25 | >=20 | 45741 | -0.0397 |


`expected_minutes` = props.betting.evaluate.quote_table: idx["minutes_mean"] = MinutesModel.mean_of over the minutes distribution of the prediction rows.
