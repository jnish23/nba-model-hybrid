# Phase 5 VALIDATION stage: 2025-26, run once with the frozen rule file phase5_dev_2023-24_noR2b.frozen_rule.json; k and the rule were NOT re-estimated (no recalibration, 20,000-draw simulations)

In the development stage everything is in-sample for k and for the choice among the tried rules, so results are optimistic by construction; in the validation stage k and the rule come from the development season. Bettable books: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Unibet (Fanatics is absent before 2025). Prices are quotes stamped at or before each decision time, main line on the consensus line, no stale quotes; pushes refund; the market probability is the de-vigged consensus (at least 3 agreeing books, all books counted).

## 1. Information test (model vs market at the book line)

| stat | n | logloss_model | logloss_market | diff | lo | hi | brier_diff | brier_lo | brier_hi | mean_p_model | mean_p_market | hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 303997 | 0.7067 | 0.6891 | 0.0176 | 0.0152 | 0.02 | 0.0076 | 0.0066 | 0.0086 | 0.4678 | 0.4957 | 0.4816 |
| assists | 35993 | 0.7013 | 0.6854 | 0.0159 | 0.012 | 0.0195 | 0.007 | 0.0051 | 0.0086 | 0.461 | 0.4949 | 0.4808 |
| pa | 24474 | 0.7201 | 0.6929 | 0.0272 | 0.0224 | 0.0325 | 0.0117 | 0.0095 | 0.0139 | 0.4595 | 0.4988 | 0.4892 |
| points | 53362 | 0.7132 | 0.6922 | 0.0209 | 0.0176 | 0.024 | 0.0092 | 0.0077 | 0.0107 | 0.469 | 0.4992 | 0.4868 |
| pr | 32620 | 0.7159 | 0.6927 | 0.0232 | 0.0189 | 0.0282 | 0.0101 | 0.0082 | 0.0122 | 0.4647 | 0.4987 | 0.4842 |
| pra | 39487 | 0.7208 | 0.6929 | 0.0279 | 0.0239 | 0.0325 | 0.012 | 0.0103 | 0.014 | 0.4629 | 0.4982 | 0.4848 |
| ra | 25820 | 0.7095 | 0.6914 | 0.0181 | 0.0137 | 0.0226 | 0.008 | 0.0059 | 0.01 | 0.4693 | 0.4996 | 0.4869 |
| rebounds | 50448 | 0.6989 | 0.6889 | 0.01 | 0.0072 | 0.013 | 0.0042 | 0.0029 | 0.0055 | 0.4738 | 0.4944 | 0.4812 |
| threes | 41793 | 0.6825 | 0.6787 | 0.0038 | 0.0013 | 0.0061 | 0.0016 | 0.0005 | 0.0027 | 0.4758 | 0.4847 | 0.4631 |

### Information slope and k

| stat | n | slope | slope_lo | slope_hi | mean hit - market | lo | hi |
|---|---|---|---|---|---|---|---|
| all | 303997 | 0.121 | 0.0717 | 0.1711 | -0.0141 | -0.019 | -0.0092 |
| assists | 35993 | 0.1726 | 0.0989 | 0.2567 | -0.0141 | -0.0224 | -0.0054 |
| pa | 24474 | 0.0449 | -0.0383 | 0.1293 | -0.0097 | -0.0196 | 0.0005 |
| points | 53362 | 0.0802 | 0.022 | 0.1451 | -0.0123 | -0.0188 | -0.0062 |
| pr | 32620 | 0.0724 | -0.0136 | 0.152 | -0.0145 | -0.0227 | -0.0063 |
| pra | 39487 | 0.0522 | -0.016 | 0.1159 | -0.0134 | -0.0212 | -0.0052 |
| ra | 25820 | 0.0888 | -0.0156 | 0.1964 | -0.0127 | -0.0218 | -0.0035 |
| rebounds | 50448 | 0.2394 | 0.1576 | 0.3164 | -0.0131 | -0.0204 | -0.0065 |
| threes | 41793 | 0.336 | 0.2253 | 0.4521 | -0.0216 | -0.0293 | -0.0138 |

k per stat (slope floored at 0, capped at 0.5): {'assists': 0.31742071974827707, 'pa': 0.12968939755013928, 'points': 0.1243057526197666, 'pr': 0.17459432951234113, 'pra': 0.10959136481676977, 'ra': 0.25767004346604383, 'rebounds': 0.2888962514894001, 'threes': 0.4117239046218011}

### Calibration by model-minus-market gap

| model minus market | n | mean p_market | mean p_model | hit rate | hit - market | lo | hi | hit - model | mean gap |
|---|---|---|---|---|---|---|---|---|---|
| (-1.0, -0.1] | 61198 | 0.497 | 0.3282 | 0.4755 | -0.0215 | -0.0314 | -0.0104 | 0.1473 | -0.1688 |
| (-0.1, -0.05] | 51712 | 0.4954 | 0.4222 | 0.4652 | -0.0302 | -0.039 | -0.0212 | 0.043 | -0.0732 |
| (-0.05, -0.02] | 40635 | 0.4949 | 0.4603 | 0.4771 | -0.0179 | -0.027 | -0.008 | 0.0168 | -0.0347 |
| (-0.02, 0.02] | 55784 | 0.4957 | 0.4955 | 0.4857 | -0.01 | -0.0183 | -0.0015 | -0.0098 | -0.0002 |
| (0.02, 0.05] | 35280 | 0.4955 | 0.5297 | 0.4855 | -0.0099 | -0.0194 | 0.0003 | -0.0442 | 0.0343 |
| (0.05, 0.1] | 37504 | 0.4954 | 0.5673 | 0.497 | 0.0016 | -0.0095 | 0.0126 | -0.0704 | 0.072 |
| (0.1, 1.0] | 21884 | 0.4953 | 0.6386 | 0.5025 | 0.0072 | -0.0088 | 0.0225 | -0.1362 | 0.1434 |

## 2. Primary rule (e = 0.03, earliest time with positive EV, at least 3 books, pooled over stats), all settlement treatments

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | stands | 21855 | 11233 | 0.514 | 806.2664 | 0.0369 | 0.0205 | 0.0537 | 2.0417 |
| PRIMARY | early_exit_flagged_books | 21809 | 11232 | 0.515 | 850.5469 | 0.039 | 0.0227 | 0.0558 | 2.0417 |
| PRIMARY | early_exit_all_books | 21651 | 11218 | 0.5181 | 977.7655 | 0.0452 | 0.0286 | 0.0621 | 2.041 |

CLV to T-5 (bets placed at T-240 or T-60): {'bets_before_T5': 16946, 'with_T5_quote': 16228, 'line_unchanged': 14384, 'line_moved': 1844}

| measure | n | mean | lo | hi |
|---|---|---|---|---|
| consensus fair probability change to T-5, line unchanged | 14384 | 0.0045 | 0.004 | 0.0049 |
| line moved by T-5: points moved in the bettor's favour | 1844 | 0.1193 | 0.047 | 0.1872 |

### Primary rule by side, stat and book

| best_side | bets | mean_ev |
|---|---|---|
| over | 7409 | 0.0545 |
| under | 14446 | 0.0608 |

| stat | bets | mean_ev | profit |
|---|---|---|---|
| assists | 5224 | 0.0655 | 140.8181 |
| pa | 395 | 0.0433 | -5.312 |
| points | 1114 | 0.042 | 38.5851 |
| pr | 1307 | 0.0463 | -17.0009 |
| pra | 397 | 0.0404 | 4.39 |
| ra | 2497 | 0.0565 | 27.8484 |
| rebounds | 5680 | 0.059 | 323.6827 |
| threes | 5241 | 0.0617 | 293.255 |

| book | bets | profit |
|---|---|---|
| betmgm | 6580 | 193.2061 |
| betrivers | 2106 | 104.2457 |
| draftkings | 5016 | 226.5276 |
| fanatics | 1038 | -32.2929 |
| fanduel | 6056 | 261.3697 |
| williamhill_us | 1059 | 53.2101 |

### Decomposition: where the primary rule's ROI comes from (settlement: bet stands)

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|
| ALL UNDERS at T-60, median price (no model) | 101367 | 52612 | 0.519 | -3633.0726 | -0.0358 | -0.046 | -0.0256 | 1.8712 |
| ALL UNDERS at T-60, best price (no model) | 101367 | 52612 | 0.519 | -1182.4259 | -0.0117 | -0.0221 | -0.0012 | 1.9184 |
| ALL OVERS at T-60, median price (no model) | 101367 | 48755 | 0.481 | -9336.3629 | -0.0921 | -0.1025 | -0.0818 | 1.9038 |
| ALL OVERS at T-60, best price (no model) | 101367 | 48755 | 0.481 | -6835.1046 | -0.0674 | -0.0781 | -0.057 | 1.956 |
| PRIMARY bets repriced at the MEDIAN bettable price | 21855 | 11233 | 0.514 | -212.3305 | -0.0097 | -0.0252 | 0.0063 | 1.949 |
| MARKET-ONLY rule (k = 0): best price vs consensus fair probability, no model | 2030 | 985 | 0.4852 | 141.5964 | 0.0698 | 0.0173 | 0.1223 | 2.2261 |
| PRIMARY bets whose quote was 0-5 minutes old | 13956 | 7284 | 0.5219 | 729.0636 | 0.0522 | 0.0334 | 0.072 | 2.0414 |
| PRIMARY bets, overs only | 7409 | 3571 | 0.482 | 132.5394 | 0.0179 | -0.0098 | 0.0444 | 2.14 |
| PRIMARY bets, unders only | 14446 | 7662 | 0.5304 | 673.727 | 0.0466 | 0.0259 | 0.0662 | 1.9914 |
| PRIMARY rule using only quotes at most 15 minutes old | 21855 | 11233 | 0.514 | 806.2664 | 0.0369 | 0.0205 | 0.0537 | 2.0417 |

### Placebo model (same k, selection and prices; probabilities swapped with a random other market of the same stat)

| run | bets | roi | roi_lo | roi_hi | clv_prob_pts | clv_lo | clv_hi | line_move_pts |
|---|---|---|---|---|---|---|---|---|
| placebo 11 | 44128 | -0.0186 | -0.0285 | -0.0082 | 0.3016 | 0.2754 | 0.3276 | -0.6067 |
| placebo 12 | 44025 | -0.0143 | -0.0244 | -0.0043 | 0.2927 | 0.2694 | 0.3185 | -0.6093 |
| placebo 13 | 43886 | -0.0239 | -0.0332 | -0.0143 | 0.2961 | 0.2725 | 0.3212 | -0.6277 |
| placebo 14 | 44017 | -0.0235 | -0.0339 | -0.0129 | 0.2948 | 0.2711 | 0.3199 | -0.6192 |

### CLV: real minus placebo

| measure | real | real_lo | placebo_mean | placebo_max_hi | excess | excess_conservative_lower |
|---|---|---|---|---|---|---|
| CLV, consensus fair probability to T-5, line unchanged (points) | 0.4463 | 0.4015 | 0.2963 | 0.3276 | 0.1499 | 0.0739 |
| line move in the bettor's favour by T-5 (points) | 0.1193 | 0.047 | -0.6157 |  | 0.735 |  |

### Exposure: {'bets': 21855, 'distinct_player_games': 11812, 'mean_bets_per_player_game': 1.8502370470707754, 'max_bets_per_player_game': 8}

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj | distinct_player_games |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary, max 1 bet per player-game | stands | 11812 | 6039 | 0.5113 | 652.6118 | 0.0552 | 0.0354 | 0.0734 | 2.093 | 0.0247 | 0.0817 | 11812 |
| primary, max 2 bets per player-game | stands | 17302 | 8893 | 0.514 | 827.7802 | 0.0478 | 0.0315 | 0.0644 | 2.0664 | 0.02 | 0.0719 | 11812 |
| primary, max 2 bets per game | stands | 2581 | 1273 | 0.4932 | 108.2283 | 0.0419 | -0.0012 | 0.0856 | 2.1455 | -0.0192 | 0.1068 | 2398 |

## 3. 9 tried rules (+3 exposure caps above) (settlement: bet stands). `lo_adj/hi_adj` = interval at the Bonferroni level for 35 rules

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj |
|---|---|---|---|---|---|---|---|---|---|---|
| e=0.03, time=earliest, min_books=3 | 21855 | 11233 | 0.514 | 806.2664 | 0.0369 | 0.0205 | 0.0537 | 2.0417 | 0.0104 | 0.0629 |
| primary, assists only | 5224 | 2677 | 0.5124 | 140.8181 | 0.027 | -0.0007 | 0.0532 | 2.0267 | -0.0136 | 0.0761 |
| primary, pa only | 395 | 199 | 0.5038 | -5.312 | -0.0134 | -0.0986 | 0.0766 | 1.9595 | -0.1641 | 0.128 |
| primary, points only | 1114 | 581 | 0.5215 | 38.5851 | 0.0346 | -0.0217 | 0.0887 | 1.9835 | -0.0517 | 0.1145 |
| primary, pr only | 1307 | 665 | 0.5088 | -17.0009 | -0.013 | -0.0676 | 0.039 | 1.9426 | -0.1073 | 0.0637 |
| primary, pra only | 397 | 204 | 0.5139 | 4.39 | 0.0111 | -0.0916 | 0.106 | 1.9635 | -0.1431 | 0.1468 |
| primary, ra only | 2497 | 1281 | 0.513 | 27.8484 | 0.0112 | -0.0262 | 0.0514 | 1.9732 | -0.0459 | 0.0834 |
| primary, rebounds only | 5680 | 2992 | 0.5268 | 323.6827 | 0.057 | 0.0324 | 0.0818 | 2.0229 | 0.0157 | 0.1012 |
| primary, threes only | 5241 | 2634 | 0.5026 | 293.255 | 0.056 | 0.0279 | 0.0863 | 2.159 | 0.0105 | 0.1066 |
