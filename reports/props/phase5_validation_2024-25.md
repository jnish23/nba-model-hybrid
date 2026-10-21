# Phase 5 VALIDATION stage: 2024-25, run once with the frozen rule file phase5_dev_2023-24_noR2b.frozen_rule.json; k and the rule were NOT re-estimated (no recalibration, 20,000-draw simulations)

In the development stage everything is in-sample for k and for the choice among the tried rules, so results are optimistic by construction; in the validation stage k and the rule come from the development season. Bettable books: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Unibet (Fanatics is absent before 2025). Prices are quotes stamped at or before each decision time, main line on the consensus line, no stale quotes; pushes refund; the market probability is the de-vigged consensus (at least 3 agreeing books, all books counted).

## 1. Information test (model vs market at the book line)

| stat | n | logloss_model | logloss_market | diff | lo | hi | brier_diff | brier_lo | brier_hi | mean_p_model | mean_p_market | hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 303205 | 0.704 | 0.6882 | 0.0159 | 0.0138 | 0.0178 | 0.0072 | 0.0062 | 0.0081 | 0.4865 | 0.4961 | 0.4859 |
| assists | 32750 | 0.6949 | 0.6849 | 0.01 | 0.0069 | 0.0134 | 0.0044 | 0.003 | 0.006 | 0.4951 | 0.4927 | 0.4805 |
| pa | 25576 | 0.7155 | 0.6928 | 0.0227 | 0.0186 | 0.0268 | 0.0105 | 0.0085 | 0.0124 | 0.4885 | 0.5005 | 0.5003 |
| points | 48871 | 0.7114 | 0.692 | 0.0195 | 0.0164 | 0.0229 | 0.0089 | 0.0075 | 0.0105 | 0.4834 | 0.4995 | 0.4915 |
| pr | 37658 | 0.7154 | 0.6929 | 0.0225 | 0.019 | 0.0261 | 0.0102 | 0.0086 | 0.0119 | 0.4832 | 0.5004 | 0.4937 |
| pra | 43534 | 0.7176 | 0.6921 | 0.0255 | 0.0218 | 0.029 | 0.0114 | 0.0098 | 0.0131 | 0.4866 | 0.4981 | 0.4914 |
| ra | 25198 | 0.7039 | 0.6922 | 0.0117 | 0.0079 | 0.0157 | 0.0054 | 0.0035 | 0.0073 | 0.498 | 0.4998 | 0.4848 |
| rebounds | 47384 | 0.695 | 0.6861 | 0.0089 | 0.0064 | 0.0115 | 0.004 | 0.0028 | 0.0052 | 0.4797 | 0.495 | 0.4765 |
| threes | 42234 | 0.6817 | 0.6751 | 0.0065 | 0.0042 | 0.0089 | 0.003 | 0.0019 | 0.0041 | 0.4855 | 0.485 | 0.4737 |

### Information slope and k

| stat | n | slope | slope_lo | slope_hi | mean hit - market | lo | hi |
|---|---|---|---|---|---|---|---|
| all | 303205 | 0.0968 | 0.0472 | 0.15 | -0.0101 | -0.0158 | -0.0048 |
| assists | 32750 | 0.2506 | 0.1579 | 0.3334 | -0.0122 | -0.0214 | -0.003 |
| pa | 25576 | -0.0037 | -0.0979 | 0.0899 | -0.0002 | -0.0116 | 0.0095 |
| points | 48871 | 0.0536 | -0.0219 | 0.1236 | -0.008 | -0.0145 | -0.0016 |
| pr | 37658 | -0.0038 | -0.0835 | 0.0763 | -0.0068 | -0.015 | 0.0007 |
| pra | 43534 | 0.0138 | -0.0551 | 0.0808 | -0.0067 | -0.0138 | 0.0013 |
| ra | 25198 | 0.1784 | 0.0657 | 0.2847 | -0.015 | -0.0259 | -0.0051 |
| rebounds | 47384 | 0.2299 | 0.1495 | 0.3094 | -0.0185 | -0.026 | -0.0111 |
| threes | 42234 | 0.1954 | 0.0873 | 0.3029 | -0.0113 | -0.0188 | -0.0036 |

k per stat (slope floored at 0, capped at 0.5): {'assists': 0.31742071974827707, 'pa': 0.12968939755013928, 'points': 0.1243057526197666, 'pr': 0.17459432951234113, 'pra': 0.10959136481676977, 'ra': 0.25767004346604383, 'rebounds': 0.2888962514894001, 'threes': 0.4117239046218011}

### Calibration by model-minus-market gap

| model minus market | n | mean p_market | mean p_model | hit rate | hit - market | lo | hi | hit - model | mean gap |
|---|---|---|---|---|---|---|---|---|---|
| (-1.0, -0.1] | 46763 | 0.4959 | 0.3345 | 0.4753 | -0.0206 | -0.0315 | -0.0092 | 0.1408 | -0.1614 |
| (-0.1, -0.05] | 45788 | 0.4939 | 0.4211 | 0.48 | -0.014 | -0.0238 | -0.0043 | 0.0589 | -0.0728 |
| (-0.05, -0.02] | 37750 | 0.4946 | 0.46 | 0.4803 | -0.0143 | -0.0245 | -0.0043 | 0.0202 | -0.0346 |
| (-0.02, 0.02] | 56066 | 0.4961 | 0.4962 | 0.4821 | -0.0141 | -0.0232 | -0.0051 | -0.0142 | 0.0001 |
| (0.02, 0.05] | 39308 | 0.4974 | 0.532 | 0.4869 | -0.0105 | -0.02 | -0.0013 | -0.0451 | 0.0346 |
| (0.05, 0.1] | 45978 | 0.4977 | 0.5701 | 0.4996 | 0.0019 | -0.0084 | 0.0121 | -0.0705 | 0.0724 |
| (0.1, 1.0] | 31552 | 0.4969 | 0.6421 | 0.5027 | 0.0059 | -0.0075 | 0.019 | -0.1394 | 0.1453 |

## 2. Primary rule (e = 0.03, earliest time with positive EV, at least 3 books, pooled over stats), all settlement treatments

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | stands | 22059 | 11317 | 0.513 | 727.5605 | 0.033 | 0.0169 | 0.0497 | 2.0412 |
| PRIMARY | early_exit_flagged_books | 21979 | 11303 | 0.5143 | 780.0115 | 0.0355 | 0.0195 | 0.0519 | 2.041 |
| PRIMARY | early_exit_all_books | 21789 | 11291 | 0.5182 | 945.124 | 0.0434 | 0.027 | 0.06 | 2.0405 |

CLV to T-5 (bets placed at T-240 or T-60): {'bets_before_T5': 16107, 'with_T5_quote': 15472, 'line_unchanged': 13831, 'line_moved': 1641}

| measure | n | mean | lo | hi |
|---|---|---|---|---|
| consensus fair probability change to T-5, line unchanged | 13831 | 0.0044 | 0.004 | 0.0048 |
| line moved by T-5: points moved in the bettor's favour | 1641 | 0.1804 | 0.104 | 0.2535 |

### Primary rule by side, stat and book

| best_side | bets | mean_ev |
|---|---|---|
| over | 10230 | 0.0555 |
| under | 11829 | 0.0604 |

| stat | bets | mean_ev | profit |
|---|---|---|---|
| assists | 4978 | 0.0622 | 209.2956 |
| pa | 388 | 0.0436 | -16.6356 |
| points | 1129 | 0.0432 | 15.7556 |
| pr | 1337 | 0.0485 | 19.6361 |
| pra | 431 | 0.0441 | 32.3162 |
| ra | 2534 | 0.0544 | 126.1811 |
| rebounds | 5643 | 0.0587 | 232.0088 |
| threes | 5619 | 0.0631 | 109.0027 |

| book | bets | profit |
|---|---|---|
| betmgm | 5886 | 287.2022 |
| betrivers | 2538 | 105.741 |
| draftkings | 5933 | 167.5267 |
| fanatics | 55 | -20.1757 |
| fanduel | 5502 | 158.282 |
| williamhill_us | 2145 | 28.9843 |

### Decomposition: where the primary rule's ROI comes from (settlement: bet stands)

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|
| ALL UNDERS at T-60, median price (no model) | 100213 | 51520 | 0.5141 | -4580.1951 | -0.0457 | -0.0556 | -0.0347 | 1.8709 |
| ALL UNDERS at T-60, best price (no model) | 100213 | 51520 | 0.5141 | -2037.9147 | -0.0203 | -0.0306 | -0.009 | 1.9208 |
| ALL OVERS at T-60, median price (no model) | 100213 | 48693 | 0.4859 | -8558.9414 | -0.0854 | -0.0966 | -0.0752 | 1.8998 |
| ALL OVERS at T-60, best price (no model) | 100213 | 48693 | 0.4859 | -5867.5068 | -0.0586 | -0.07 | -0.0481 | 1.9562 |
| PRIMARY bets repriced at the MEDIAN bettable price | 22059 | 11317 | 0.513 | -376.61 | -0.0171 | -0.0323 | -0.001 | 1.9415 |
| MARKET-ONLY rule (k = 0): best price vs consensus fair probability, no model | 2266 | 1047 | 0.462 | 51.1568 | 0.0226 | -0.0209 | 0.07 | 2.2307 |
| PRIMARY bets whose quote was 0-5 minutes old | 14501 | 7419 | 0.5116 | 431.7831 | 0.0298 | 0.0105 | 0.0491 | 2.0424 |
| PRIMARY bets whose quote was 5-15 minutes old | 548 | 280 | 0.5109 | 16.4371 | 0.03 | -0.0578 | 0.1126 | 2.0328 |
| PRIMARY bets, overs only | 10230 | 5026 | 0.4913 | 156.882 | 0.0153 | -0.0072 | 0.0382 | 2.0992 |
| PRIMARY bets, unders only | 11829 | 6291 | 0.5318 | 570.6784 | 0.0482 | 0.0251 | 0.0721 | 1.991 |
| PRIMARY rule using only quotes at most 15 minutes old | 22059 | 11317 | 0.513 | 727.5605 | 0.033 | 0.0169 | 0.0497 | 2.0412 |

### Placebo model (same k, selection and prices; probabilities swapped with a random other market of the same stat)

| run | bets | roi | roi_lo | roi_hi | clv_prob_pts | clv_lo | clv_hi | line_move_pts |
|---|---|---|---|---|---|---|---|---|
| placebo 11 | 43351 | -0.0338 | -0.0436 | -0.0244 | 0.2307 | 0.2065 | 0.2543 | -0.5884 |
| placebo 12 | 43479 | -0.0327 | -0.0421 | -0.0237 | 0.25 | 0.2284 | 0.2736 | -0.609 |
| placebo 13 | 43117 | -0.0364 | -0.0463 | -0.0271 | 0.26 | 0.2365 | 0.2832 | -0.5744 |
| placebo 14 | 43374 | -0.038 | -0.0476 | -0.0283 | 0.2505 | 0.2287 | 0.2733 | -0.5797 |

### CLV: real minus placebo

| measure | real | real_lo | placebo_mean | placebo_max_hi | excess | excess_conservative_lower |
|---|---|---|---|---|---|---|
| CLV, consensus fair probability to T-5, line unchanged (points) | 0.4405 | 0.3985 | 0.2478 | 0.2832 | 0.1928 | 0.1153 |
| line move in the bettor's favour by T-5 (points) | 0.1804 | 0.104 | -0.5879 |  | 0.7682 |  |

### Exposure: {'bets': 22059, 'distinct_player_games': 11909, 'mean_bets_per_player_game': 1.8522965824166597, 'max_bets_per_player_game': 8}

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj | distinct_player_games |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary, max 1 bet per player-game | stands | 11909 | 6071 | 0.5098 | 546.8962 | 0.0459 | 0.0267 | 0.0654 | 2.0831 | 0.017 | 0.0738 | 11909 |
| primary, max 2 bets per player-game | stands | 17690 | 9056 | 0.5119 | 702.692 | 0.0397 | 0.0244 | 0.0556 | 2.0606 | 0.0148 | 0.0627 | 11909 |
| primary, max 2 bets per game | stands | 2619 | 1300 | 0.4964 | 120.8226 | 0.0461 | 0.004 | 0.0889 | 2.1386 | -0.0183 | 0.1122 | 2403 |

## 3. 9 tried rules (+3 exposure caps above) (settlement: bet stands). `lo_adj/hi_adj` = interval at the Bonferroni level for 35 rules

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj |
|---|---|---|---|---|---|---|---|---|---|---|
| e=0.03, time=earliest, min_books=3 | 22059 | 11317 | 0.513 | 727.5605 | 0.033 | 0.0169 | 0.0497 | 2.0412 | 0.0076 | 0.0574 |
| primary, assists only | 4978 | 2591 | 0.5205 | 209.2956 | 0.042 | 0.0153 | 0.0698 | 2.0298 | 0.0019 | 0.0832 |
| primary, pa only | 388 | 189 | 0.4871 | -16.6356 | -0.0429 | -0.1477 | 0.0537 | 1.968 | -0.2058 | 0.1104 |
| primary, points only | 1129 | 576 | 0.5102 | 15.7556 | 0.014 | -0.0432 | 0.0718 | 1.9921 | -0.0857 | 0.0974 |
| primary, pr only | 1337 | 693 | 0.5183 | 19.6361 | 0.0147 | -0.0421 | 0.0708 | 1.9568 | -0.075 | 0.1003 |
| primary, pra only | 431 | 234 | 0.5429 | 32.3162 | 0.075 | -0.023 | 0.171 | 1.9887 | -0.0836 | 0.219 |
| primary, ra only | 2534 | 1343 | 0.53 | 126.1811 | 0.0498 | 0.0132 | 0.0867 | 1.985 | -0.0104 | 0.1065 |
| primary, rebounds only | 5643 | 2928 | 0.5189 | 232.0088 | 0.0411 | 0.0157 | 0.0664 | 2.0238 | 0.0017 | 0.0813 |
| primary, threes only | 5619 | 2763 | 0.4917 | 109.0027 | 0.0194 | -0.0097 | 0.047 | 2.133 | -0.0288 | 0.0709 |
