# 7c / 7d on rule C3 (candidate rule file (fit on ['2023-24', '2024-25']; descriptive only)), model a11_pg2_p5

Median-price ROI (primary), settlement `stands`, date-clustered bootstrap (2000 resamples, seed 20261112), paired on the same resamples; Bonferroni level 0.05/23. 2023-24 is in sample for the rule parameters, 2024-25 out of sample.

## 7c: caps and quarter-Kelly (f* = EV/(d-1) at the best price; stake = min(0.25 f*, 0.02) B, B = 100; per-game total at most 0.05 B, lower-EV bets cut first)

| variant | season | bets | roi | n_stakes | stake_units | n_dates | sharpe | max_drawdown | dd_over_stake | roi_best | sharpe_best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flat | 2023-24 | 21564 | 0.023 | 21564 | 21564 | 207 | 0.1668 | 73.8003 | 0.0034 | 0.0619 | 0.4563 |
| cap1 | 2023-24 | 12042 | 0.028 | 12042 | 12042 | 207 | 0.2006 | 60.9985 | 0.0051 | 0.0721 | 0.4749 |
| cap2 | 2023-24 | 18193 | 0.0294 | 18193 | 18193 | 207 | 0.2379 | 48.0124 | 0.0026 | 0.0703 | 0.5381 |
| kelly | 2023-24 | 21564 | 0.0486 | 4397 | 6582.203 | 207 | 0.1653 | 40.0003 | 0.0061 | 0.1002 | 0.305 |
| cap1_causal | 2023-24 | 12042 | 0.0305 | 12042 | 12042 | 207 | 0.1911 | 51.4505 | 0.0043 | 0.0724 | 0.4406 |
| cap2_causal | 2023-24 | 18193 | 0.0279 | 18193 | 18193 | 207 | 0.2039 | 55.0987 | 0.003 | 0.0678 | 0.4961 |
| flat | 2024-25 | 19715 | -0.003 | 19715 | 19715 | 213 | 0.055 | 160.547 | 0.0081 | 0.0469 | 0.3708 |
| cap1 | 2024-25 | 11881 | 0.0013 | 11881 | 11881 | 213 | 0.0602 | 101.5092 | 0.0085 | 0.0556 | 0.3437 |
| cap2 | 2024-25 | 17237 | -0.0001 | 17237 | 17237 | 213 | 0.0694 | 116.5858 | 0.0068 | 0.0516 | 0.3869 |
| kelly | 2024-25 | 19715 | 0.0172 | 4441 | 6517.7003 | 213 | 0.1721 | 52.9851 | 0.0081 | 0.0812 | 0.3658 |
| cap1_causal | 2024-25 | 11881 | 0.0071 | 11881 | 11881 | 213 | 0.0958 | 69.285 | 0.0058 | 0.0595 | 0.3762 |
| cap2_causal | 2024-25 | 17237 | -0.0014 | 17237 | 17237 | 213 | 0.0638 | 136.5241 | 0.0079 | 0.0492 | 0.3743 |

### Paired median-ROI differences against flat

| season | comparison | point | lo | hi | se | lo_bonf | hi_bonf |
|---|---|---|---|---|---|---|---|
| 2023-24 | cap1 - flat (median ROI) | 0.005 | -0.0053 | 0.0152 | 0.0053 | -0.0108 | 0.0216 |
| 2023-24 | cap2 - flat (median ROI) | 0.0063 | 0.001 | 0.0118 | 0.0027 | -0.0018 | 0.0152 |
| 2023-24 | kelly - flat (median ROI) | 0.0256 | -0.0016 | 0.0529 | 0.0141 | -0.0166 | 0.0685 |
| 2023-24 | cap1_causal - flat (median ROI) | 0.0075 | -0.0026 | 0.0179 | 0.0053 | -0.0088 | 0.0251 |
| 2023-24 | cap2_causal - flat (median ROI) | 0.0049 | -0.0005 | 0.0103 | 0.0027 | -0.0029 | 0.013 |
| 2023-24 | cap1_causal - cap1 (median ROI) | 0.0025 | -0.004 | 0.0088 | 0.0032 | -0.0076 | 0.0124 |
| 2023-24 | cap2_causal - cap2 (median ROI) | -0.0014 | -0.0048 | 0.002 | 0.0017 | -0.007 | 0.0033 |
| 2024-25 | cap1 - flat (median ROI) | 0.0043 | -0.0063 | 0.0152 | 0.0055 | -0.0103 | 0.021 |
| 2024-25 | cap2 - flat (median ROI) | 0.0029 | -0.0013 | 0.0075 | 0.0022 | -0.0032 | 0.0097 |
| 2024-25 | kelly - flat (median ROI) | 0.0202 | -0.0063 | 0.0476 | 0.0141 | -0.0232 | 0.0622 |
| 2024-25 | cap1_causal - flat (median ROI) | 0.0102 | 0 | 0.0205 | 0.0052 | -0.0044 | 0.025 |
| 2024-25 | cap2_causal - flat (median ROI) | 0.0017 | -0.0026 | 0.0062 | 0.0023 | -0.0048 | 0.0089 |
| 2024-25 | cap1_causal - cap1 (median ROI) | 0.0059 | -0.001 | 0.0126 | 0.0034 | -0.0039 | 0.0154 |
| 2024-25 | cap2_causal - cap2 (median ROI) | -0.0013 | -0.0047 | 0.0025 | 0.0018 | -0.0064 | 0.0039 |

### By stat

| variant | season | stat | bets | stakes | roi_med | roi_best |
|---|---|---|---|---|---|---|
| flat | 2023-24 | assists | 5244 | 5244 | 0.0253 | 0.067 |
| flat | 2023-24 | ra | 4273 | 4273 | 0.0251 | 0.0605 |
| flat | 2023-24 | rebounds | 7805 | 7805 | 0.0176 | 0.0561 |
| flat | 2023-24 | threes | 4242 | 4242 | 0.0281 | 0.0675 |
| cap1 | 2023-24 | assists | 3296 | 3296 | 0.0374 | 0.0821 |
| cap1 | 2023-24 | ra | 1346 | 1346 | -0.0031 | 0.0467 |
| cap1 | 2023-24 | rebounds | 4816 | 4816 | 0.0276 | 0.0702 |
| cap1 | 2023-24 | threes | 2584 | 2584 | 0.033 | 0.0762 |
| cap2 | 2023-24 | assists | 4572 | 4572 | 0.0327 | 0.0759 |
| cap2 | 2023-24 | ra | 3366 | 3366 | 0.0282 | 0.067 |
| cap2 | 2023-24 | rebounds | 6711 | 6711 | 0.0233 | 0.0637 |
| cap2 | 2023-24 | threes | 3544 | 3544 | 0.0375 | 0.0788 |
| kelly | 2023-24 | assists | 5244 | 5244 | 0.0638 | 0.1113 |
| kelly | 2023-24 | ra | 4273 | 4273 | 0.0289 | 0.0788 |
| kelly | 2023-24 | rebounds | 7805 | 7805 | 0.0465 | 0.1054 |
| kelly | 2023-24 | threes | 4242 | 4242 | 0.044 | 0.0931 |
| cap1_causal | 2023-24 | assists | 3285 | 3285 | 0.0412 | 0.0838 |
| cap1_causal | 2023-24 | ra | 1231 | 1231 | -0.0006 | 0.0473 |
| cap1_causal | 2023-24 | rebounds | 4939 | 4939 | 0.0303 | 0.0713 |
| cap1_causal | 2023-24 | threes | 2587 | 2587 | 0.0319 | 0.0721 |
| cap2_causal | 2023-24 | assists | 4564 | 4564 | 0.0336 | 0.0758 |
| cap2_causal | 2023-24 | ra | 3383 | 3383 | 0.0296 | 0.0673 |
| cap2_causal | 2023-24 | rebounds | 6721 | 6721 | 0.021 | 0.0605 |
| cap2_causal | 2023-24 | threes | 3525 | 3525 | 0.032 | 0.072 |
| flat | 2024-25 | assists | 4962 | 4962 | 0.0128 | 0.0641 |
| flat | 2024-25 | ra | 3305 | 3305 | 0.0059 | 0.0558 |
| flat | 2024-25 | rebounds | 7773 | 7773 | -0.0052 | 0.043 |
| flat | 2024-25 | threes | 3675 | 3675 | -0.0278 | 0.024 |
| cap1 | 2024-25 | assists | 3131 | 3131 | 0.0305 | 0.0858 |
| cap1 | 2024-25 | ra | 1305 | 1305 | -0.0208 | 0.0395 |
| cap1 | 2024-25 | rebounds | 5260 | 5260 | -0.001 | 0.0503 |
| cap1 | 2024-25 | threes | 2185 | 2185 | -0.0218 | 0.0346 |
| cap2 | 2024-25 | assists | 4414 | 4414 | 0.0209 | 0.0738 |
| cap2 | 2024-25 | ra | 2672 | 2672 | 0.0041 | 0.0577 |
| cap2 | 2024-25 | rebounds | 7010 | 7010 | 0.0002 | 0.0497 |
| cap2 | 2024-25 | threes | 3141 | 3141 | -0.0339 | 0.0193 |
| kelly | 2024-25 | assists | 4962 | 4962 | 0.0048 | 0.0641 |
| kelly | 2024-25 | ra | 3305 | 3305 | 0.0348 | 0.1017 |
| kelly | 2024-25 | rebounds | 7773 | 7773 | 0.0146 | 0.081 |
| kelly | 2024-25 | threes | 3675 | 3675 | 0.0287 | 0.0937 |
| cap1_causal | 2024-25 | assists | 3088 | 3088 | 0.039 | 0.0925 |
| cap1_causal | 2024-25 | ra | 1221 | 1221 | -0.0052 | 0.0537 |
| cap1_causal | 2024-25 | rebounds | 5415 | 5415 | 0.0034 | 0.0534 |
| cap1_causal | 2024-25 | threes | 2157 | 2157 | -0.0223 | 0.0307 |
| cap2_causal | 2024-25 | assists | 4366 | 4366 | 0.0213 | 0.0731 |
| cap2_causal | 2024-25 | ra | 2709 | 2709 | -0.0005 | 0.0516 |
| cap2_causal | 2024-25 | rebounds | 7063 | 7063 | -0.0021 | 0.0464 |
| cap2_causal | 2024-25 | threes | 3099 | 3099 | -0.0322 | 0.0199 |

### Bets sharing a player-game: {'2023-24': {'bets': 21564, 'share_of_bets_sharing_a_player_game': 0.7268132071971805, 'distinct_player_games': 12042, 'mean_bets_per_player_game': 1.7907324364723467, 'max_bets_per_player_game': 4}, '2024-25': {'bets': 19715, 'share_of_bets_sharing_a_player_game': 0.6690337306619325, 'distinct_player_games': 11881, 'mean_bets_per_player_game': 1.659372106725023, 'max_bets_per_player_game': 4}}

### Keep rule, cap1: **ADOPT**

| item | clause | value | threshold | result |
|---|---|---|---|---|
| 7c | cap1: 2024-25 Sharpe at least uncapped (0.0550) | 0.0602 | 0.055 | PASS |
| 7c | cap1: 2024-25 median ROI no more than 0.5 points below uncapped (-0.0030) | 0.0013 | -0.008 | PASS |
| 7c | cap1: 2023-24 Sharpe no lower than uncapped (0.1668) | 0.2006 | 0.1668 | PASS |

### Keep rule, cap2: **ADOPT**

| item | clause | value | threshold | result |
|---|---|---|---|---|
| 7c | cap2: 2024-25 Sharpe at least uncapped (0.0550) | 0.0694 | 0.055 | PASS |
| 7c | cap2: 2024-25 median ROI no more than 0.5 points below uncapped (-0.0030) | -0.0001 | -0.008 | PASS |
| 7c | cap2: 2023-24 Sharpe no lower than uncapped (0.1668) | 0.2379 | 0.1668 | PASS |

### Keep rule, cap1_causal: **ADOPT**

| item | clause | value | threshold | result |
|---|---|---|---|---|
| 7c | cap1_causal: 2024-25 Sharpe at least uncapped (0.0550) | 0.0958 | 0.055 | PASS |
| 7c | cap1_causal: 2024-25 median ROI no more than 0.5 points below uncapped (-0.0030) | 0.0071 | -0.008 | PASS |
| 7c | cap1_causal: 2023-24 Sharpe no lower than uncapped (0.1668) | 0.1911 | 0.1668 | PASS |

### Keep rule, cap2_causal: **ADOPT**

| item | clause | value | threshold | result |
|---|---|---|---|---|
| 7c | cap2_causal: 2024-25 Sharpe at least uncapped (0.0550) | 0.0638 | 0.055 | PASS |
| 7c | cap2_causal: 2024-25 median ROI no more than 0.5 points below uncapped (-0.0030) | -0.0014 | -0.008 | PASS |
| 7c | cap2_causal: 2023-24 Sharpe no lower than uncapped (0.1668) | 0.2039 | 0.1668 | PASS |

### Keep rule, kelly: **not adopted** (Kelly: evaluation only, no adoption before the fill study, owner decision G9)

| item | clause | value | threshold | result |
|---|---|---|---|---|
| 7c | Kelly: 2024-25 Sharpe at least 1.10 x flat (0.0550) | 0.1721 | 0.0605 | PASS |
| 7c | Kelly: 2024-25 max drawdown / total stake at most flat's | 0.0081 | 0.0081 | PASS |
| 7c | Kelly: 2024-25 median ROI no more than 0.5 points below flat (-0.0030) | 0.0172 | -0.008 | PASS |
| 7c | Kelly: 2023-24 Sharpe not below flat (0.1668) (ordering not reversed) | 0.1653 | 0.1668 | FAIL |
