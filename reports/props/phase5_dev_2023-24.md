# Phase 5 development stage: 2023-24 (frozen configuration + R2b, 20,000-draw simulations)

Everything here is in-sample for k and for the choice among the tried rules: results are optimistic by construction. The validation season (2024-25) is the check, run once with the frozen file `phase5_dev_2023-24.frozen_rule.json`. Bettable books: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Unibet (Fanatics is absent before 2025). Prices are quotes stamped at or before each decision time, main line on the consensus line, no stale quotes; pushes refund; the market probability is the de-vigged consensus (at least 3 agreeing books, all books counted).

## 1. Information test (model vs market at the book line)

| stat | n | logloss_model | logloss_market | diff | lo | hi | brier_diff | brier_lo | brier_hi | mean_p_model | mean_p_market | hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 333777 | 0.7015 | 0.6876 | 0.0139 | 0.0116 | 0.0161 | 0.006 | 0.0049 | 0.007 | 0.4661 | 0.4971 | 0.4792 |
| assists | 34118 | 0.694 | 0.6851 | 0.0089 | 0.0052 | 0.0127 | 0.0037 | 0.002 | 0.0055 | 0.4816 | 0.4942 | 0.49 |
| pa | 38208 | 0.7124 | 0.6919 | 0.0205 | 0.0164 | 0.0245 | 0.009 | 0.0072 | 0.0109 | 0.4656 | 0.4998 | 0.4836 |
| points | 47900 | 0.709 | 0.6918 | 0.0172 | 0.0138 | 0.0204 | 0.0077 | 0.0061 | 0.0092 | 0.4707 | 0.4995 | 0.4812 |
| pr | 42990 | 0.7096 | 0.6918 | 0.0179 | 0.014 | 0.0215 | 0.0078 | 0.006 | 0.0095 | 0.4597 | 0.5002 | 0.4782 |
| pra | 41020 | 0.7153 | 0.6922 | 0.0231 | 0.0191 | 0.0267 | 0.0101 | 0.0083 | 0.0118 | 0.4607 | 0.4997 | 0.4785 |
| ra | 35834 | 0.7035 | 0.69 | 0.0135 | 0.0093 | 0.0174 | 0.0056 | 0.0037 | 0.0073 | 0.464 | 0.4982 | 0.4823 |
| rebounds | 48697 | 0.6961 | 0.6874 | 0.0087 | 0.0059 | 0.0116 | 0.0036 | 0.0023 | 0.0049 | 0.4633 | 0.497 | 0.4738 |
| threes | 45010 | 0.674 | 0.6715 | 0.0025 | 0 | 0.0051 | 0.0009 | -0.0002 | 0.0021 | 0.4657 | 0.4884 | 0.4704 |

### Information slope and k

| stat | n | slope | slope_lo | slope_hi | mean hit - market | lo | hi |
|---|---|---|---|---|---|---|---|
| all | 333777 | 0.2268 | 0.1816 | 0.2735 | -0.0178 | -0.0229 | -0.0129 |
| assists | 34118 | 0.3226 | 0.2387 | 0.4029 | -0.0041 | -0.0133 | 0.0038 |
| pa | 38208 | 0.1555 | 0.0849 | 0.2237 | -0.0162 | -0.025 | -0.0077 |
| points | 47900 | 0.1634 | 0.0996 | 0.2331 | -0.0183 | -0.0248 | -0.0116 |
| pr | 42990 | 0.2074 | 0.1451 | 0.2732 | -0.022 | -0.0292 | -0.0148 |
| pra | 41020 | 0.1462 | 0.0862 | 0.2093 | -0.0212 | -0.0294 | -0.0136 |
| ra | 35834 | 0.2558 | 0.181 | 0.3395 | -0.0159 | -0.0238 | -0.0081 |
| rebounds | 48697 | 0.2984 | 0.2256 | 0.3697 | -0.0231 | -0.0302 | -0.016 |
| threes | 45010 | 0.4244 | 0.3223 | 0.5199 | -0.018 | -0.0253 | -0.0107 |

k per stat (slope floored at 0, capped at 0.5): {'assists': 0.3225849963742759, 'pa': 0.1554504008792689, 'points': 0.16343231590406318, 'pr': 0.20742140602658352, 'pra': 0.14624187534558764, 'ra': 0.2558196016920124, 'rebounds': 0.29839961163718504, 'threes': 0.4243896343110713}

### Calibration by model-minus-market gap

| model minus market | n | mean p_market | mean p_model | hit rate | hit - market | lo | hi | hit - model | mean gap |
|---|---|---|---|---|---|---|---|---|---|
| (-1.0, -0.1] | 73714 | 0.4973 | 0.3298 | 0.4558 | -0.0414 | -0.0509 | -0.0318 | 0.126 | -0.1675 |
| (-0.1, -0.05] | 60427 | 0.4953 | 0.4218 | 0.4601 | -0.0351 | -0.0437 | -0.0272 | 0.0384 | -0.0735 |
| (-0.05, -0.02] | 43546 | 0.4958 | 0.4608 | 0.4697 | -0.0261 | -0.0353 | -0.0168 | 0.0089 | -0.0349 |
| (-0.02, 0.02] | 56499 | 0.4978 | 0.4972 | 0.4918 | -0.0059 | -0.0144 | 0.0032 | -0.0054 | -0.0005 |
| (0.02, 0.05] | 34424 | 0.4984 | 0.5328 | 0.4953 | -0.0031 | -0.0136 | 0.007 | -0.0375 | 0.0344 |
| (0.05, 0.1] | 37993 | 0.4993 | 0.5715 | 0.51 | 0.0107 | -0.0007 | 0.0215 | -0.0615 | 0.0722 |
| (0.1, 1.0] | 27174 | 0.4966 | 0.6464 | 0.5111 | 0.0145 | 0.0006 | 0.0286 | -0.1353 | 0.1497 |

## 2. Primary rule (e = 0.03, earliest time with positive EV, at least 3 books, pooled over stats), all settlement treatments

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | stands | 27934 | 14677 | 0.5254 | 1332.8273 | 0.0477 | 0.0303 | 0.0649 | 2.0245 |
| PRIMARY | early_exit_flagged_books | 27777 | 14660 | 0.5278 | 1453.2392 | 0.0523 | 0.0346 | 0.0698 | 2.0232 |
| PRIMARY | early_exit_all_books | 27636 | 14650 | 0.5301 | 1573.9655 | 0.057 | 0.0389 | 0.0745 | 2.0231 |

CLV to T-5 (bets placed at T-240 or T-60): {'bets_before_T5': 19954, 'with_T5_quote': 19674, 'line_unchanged': 16630, 'line_moved': 3044}

| measure | n | mean | lo | hi |
|---|---|---|---|---|
| consensus fair probability change to T-5, line unchanged | 16630 | 0.0041 | 0.0037 | 0.0045 |
| line moved by T-5: points moved in the bettor's favour | 3044 | 0.1265 | 0.0408 | 0.2106 |

### Primary rule by side, stat and book

| best_side | bets | mean_ev |
|---|---|---|
| over | 7554 | 0.0552 |
| under | 20380 | 0.0613 |

| stat | bets | mean_ev | profit |
|---|---|---|---|
| assists | 5076 | 0.0673 | 276.6712 |
| pa | 1356 | 0.0474 | 84.1087 |
| points | 2497 | 0.0478 | 71.8845 |
| pr | 2952 | 0.053 | 122.9188 |
| pra | 1435 | 0.0469 | 59.6856 |
| ra | 3167 | 0.0599 | 167.482 |
| rebounds | 5756 | 0.0606 | 342.5185 |
| threes | 5695 | 0.0666 | 207.558 |

| book | bets | profit |
|---|---|---|
| betmgm | 3124 | 264.9834 |
| betrivers | 1065 | 78.115 |
| draftkings | 10295 | 648.9721 |
| fanduel | 7533 | 225.8405 |
| unibet_us | 1109 | 97.3294 |
| williamhill_us | 4808 | 17.5869 |

### Decomposition: where the primary rule's ROI comes from (settlement: bet stands)

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|
| ALL UNDERS at T-60, median price (no model) | 108721 | 56642 | 0.521 | -3418.8947 | -0.0314 | -0.0428 | -0.0209 | 1.8762 |
| ALL UNDERS at T-60, best price (no model) | 108721 | 56642 | 0.521 | -1010.2392 | -0.0093 | -0.0208 | 0.0015 | 1.9197 |
| ALL OVERS at T-60, median price (no model) | 108721 | 52079 | 0.479 | -10864.9581 | -0.0999 | -0.11 | -0.0888 | 1.8994 |
| ALL OVERS at T-60, best price (no model) | 108721 | 52079 | 0.479 | -8489.3244 | -0.0781 | -0.0883 | -0.0667 | 1.9462 |
| PRIMARY bets repriced at the MEDIAN bettable price | 27934 | 14677 | 0.5254 | 256.7771 | 0.0092 | -0.0077 | 0.0258 | 1.9488 |
| MARKET-ONLY rule (k = 0): best price vs consensus fair probability, no model | 1536 | 731 | 0.4759 | 65.4006 | 0.0426 | -0.0154 | 0.0995 | 2.2559 |
| PRIMARY bets whose quote was 0-5 minutes old | 21596 | 11345 | 0.5253 | 1020.1663 | 0.0472 | 0.0288 | 0.065 | 2.024 |
| PRIMARY bets whose quote was 5-15 minutes old | 1291 | 685 | 0.5306 | 70.0883 | 0.0543 | -0.0069 | 0.1217 | 2.0073 |
| PRIMARY bets, overs only | 7554 | 3743 | 0.4955 | 150.8408 | 0.02 | -0.0082 | 0.0494 | 2.1104 |
| PRIMARY bets, unders only | 20380 | 10934 | 0.5365 | 1181.9865 | 0.058 | 0.0398 | 0.0784 | 1.9926 |
| PRIMARY rule using only quotes at most 15 minutes old | 27934 | 14677 | 0.5254 | 1332.8273 | 0.0477 | 0.0303 | 0.0649 | 2.0245 |

### Placebo model (same k, selection and prices; probabilities swapped with a random other market of the same stat)

| run | bets | roi | roi_lo | roi_hi | clv_prob_pts | clv_lo | clv_hi | line_move_pts |
|---|---|---|---|---|---|---|---|---|
| placebo 11 | 55553 | -0.0347 | -0.0437 | -0.0256 | 0.1833 | 0.1649 | 0.2018 | -0.5095 |
| placebo 12 | 55435 | -0.0335 | -0.0433 | -0.0241 | 0.1784 | 0.1587 | 0.1984 | -0.482 |
| placebo 13 | 55302 | -0.0337 | -0.0435 | -0.0244 | 0.1915 | 0.1736 | 0.2093 | -0.5223 |
| placebo 14 | 55531 | -0.0359 | -0.0453 | -0.0259 | 0.1914 | 0.1731 | 0.2094 | -0.4933 |

### CLV: real minus placebo

| measure | real | real_lo | placebo_mean | placebo_max_hi | excess | excess_conservative_lower |
|---|---|---|---|---|---|---|
| CLV, consensus fair probability to T-5, line unchanged (points) | 0.4104 | 0.3725 | 0.1861 | 0.2094 | 0.2242 | 0.1631 |
| line move in the bettor's favour by T-5 (points) | 0.1265 | 0.0408 | -0.5018 |  | 0.6283 |  |

### Exposure: {'bets': 27934, 'distinct_player_games': 12198, 'mean_bets_per_player_game': 2.2900475487784884, 'max_bets_per_player_game': 8}

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj | distinct_player_games |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary, max 1 bet per player-game | stands | 12198 | 6334 | 0.5193 | 787.7036 | 0.0646 | 0.0457 | 0.0846 | 2.0914 | 0.0358 | 0.0964 | 12198 |
| primary, max 2 bets per player-game | stands | 18877 | 9829 | 0.5207 | 1018.2453 | 0.0539 | 0.0377 | 0.0708 | 2.0597 | 0.0275 | 0.0839 | 12198 |
| primary, max 2 bets per game | stands | 2634 | 1345 | 0.5106 | 200.1673 | 0.076 | 0.0323 | 0.1148 | 2.1662 | -0.0022 | 0.1362 | 2349 |

## 3. 32 tried rules (+3 exposure caps above) (settlement: bet stands). `lo_adj/hi_adj` = interval at the Bonferroni level for 35 rules

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj |
|---|---|---|---|---|---|---|---|---|---|---|
| e=0.02, time=earliest, min_books=3 | 36610 | 19306 | 0.5273 | 1661.4132 | 0.0454 | 0.0296 | 0.0604 | 2.0109 | 0.022 | 0.0722 |
| e=0.02, time=earliest, min_books=5 | 30491 | 16096 | 0.5279 | 1464.4194 | 0.048 | 0.032 | 0.063 | 2.0124 | 0.0218 | 0.0724 |
| e=0.02, time=240, min_books=3 | 17736 | 9405 | 0.5303 | 911.0955 | 0.0514 | 0.0323 | 0.0721 | 2.0095 | 0.021 | 0.0822 |
| e=0.02, time=240, min_books=5 | 15183 | 8085 | 0.5325 | 862.3708 | 0.0568 | 0.0375 | 0.0767 | 2.0111 | 0.0204 | 0.0892 |
| e=0.02, time=60, min_books=3 | 22120 | 11645 | 0.5264 | 931.628 | 0.0421 | 0.0246 | 0.0607 | 2.0093 | 0.0127 | 0.0695 |
| e=0.02, time=60, min_books=5 | 18098 | 9523 | 0.5262 | 785.5424 | 0.0434 | 0.0239 | 0.0622 | 2.0127 | 0.0088 | 0.0711 |
| e=0.02, time=5, min_books=3 | 28105 | 14719 | 0.5237 | 1030.7716 | 0.0367 | 0.0189 | 0.053 | 2.01 | 0.0117 | 0.0643 |
| e=0.02, time=5, min_books=5 | 22410 | 11769 | 0.5252 | 949.0954 | 0.0424 | 0.0248 | 0.0593 | 2.0131 | 0.0188 | 0.0695 |
| e=0.03, time=earliest, min_books=3 | 27934 | 14677 | 0.5254 | 1332.8273 | 0.0477 | 0.0303 | 0.0649 | 2.0245 | 0.0226 | 0.0778 |
| e=0.03, time=earliest, min_books=5 | 23314 | 12254 | 0.5256 | 1153.5154 | 0.0495 | 0.0311 | 0.0675 | 2.025 | 0.0249 | 0.077 |
| e=0.03, time=240, min_books=3 | 13105 | 6980 | 0.5326 | 817.0005 | 0.0623 | 0.041 | 0.0861 | 2.0221 | 0.0264 | 0.0951 |
| e=0.03, time=240, min_books=5 | 11279 | 6013 | 0.5331 | 726.4097 | 0.0644 | 0.0414 | 0.0889 | 2.0226 | 0.0275 | 0.098 |
| e=0.03, time=60, min_books=3 | 16402 | 8625 | 0.5259 | 759.874 | 0.0463 | 0.0271 | 0.0657 | 2.0218 | 0.0168 | 0.0788 |
| e=0.03, time=60, min_books=5 | 13518 | 7094 | 0.5248 | 618.5003 | 0.0458 | 0.0239 | 0.0658 | 2.0243 | 0.0082 | 0.077 |
| e=0.03, time=5, min_books=3 | 21307 | 11133 | 0.5225 | 877.6547 | 0.0412 | 0.0213 | 0.0594 | 2.0242 | 0.0165 | 0.0723 |
| e=0.03, time=5, min_books=5 | 17052 | 8950 | 0.5249 | 827.8188 | 0.0485 | 0.0282 | 0.0675 | 2.0263 | 0.0214 | 0.077 |
| e=0.05, time=earliest, min_books=3 | 15968 | 8339 | 0.5222 | 894.1845 | 0.056 | 0.036 | 0.076 | 2.0559 | 0.0247 | 0.0876 |
| e=0.05, time=earliest, min_books=5 | 13286 | 6964 | 0.5242 | 819.8727 | 0.0617 | 0.0402 | 0.0844 | 2.0554 | 0.0302 | 0.0961 |
| e=0.05, time=240, min_books=3 | 6846 | 3606 | 0.5267 | 453.3324 | 0.0662 | 0.0379 | 0.0972 | 2.0563 | 0.0256 | 0.1102 |
| e=0.05, time=240, min_books=5 | 5939 | 3131 | 0.5272 | 403.5357 | 0.0679 | 0.037 | 0.1001 | 2.0547 | 0.0176 | 0.1181 |
| e=0.05, time=60, min_books=3 | 8878 | 4643 | 0.523 | 492.4362 | 0.0555 | 0.0291 | 0.0831 | 2.0542 | 0.0119 | 0.1053 |
| e=0.05, time=60, min_books=5 | 7365 | 3854 | 0.5233 | 422.9026 | 0.0574 | 0.0308 | 0.0858 | 2.0547 | 0.0159 | 0.0979 |
| e=0.05, time=5, min_books=3 | 12050 | 6259 | 0.5194 | 593.4687 | 0.0493 | 0.0256 | 0.0745 | 2.0561 | 0.0129 | 0.0875 |
| e=0.05, time=5, min_books=5 | 9641 | 5036 | 0.5224 | 562.6004 | 0.0584 | 0.0346 | 0.0823 | 2.0572 | 0.0234 | 0.0964 |
| primary, assists only | 5076 | 2667 | 0.5254 | 276.6712 | 0.0545 | 0.026 | 0.0818 | 2.0336 | 0.0123 | 0.1013 |
| primary, pa only | 1356 | 730 | 0.5383 | 84.1087 | 0.062 | 0.007 | 0.1135 | 1.9807 | -0.0141 | 0.1316 |
| primary, points only | 2497 | 1293 | 0.5178 | 71.8845 | 0.0288 | -0.0122 | 0.0712 | 1.9896 | -0.0245 | 0.0954 |
| primary, pr only | 2952 | 1580 | 0.5352 | 122.9188 | 0.0416 | 0.0078 | 0.0744 | 1.9529 | -0.0144 | 0.1018 |
| primary, pra only | 1435 | 759 | 0.5289 | 59.6856 | 0.0416 | -0.0123 | 0.096 | 1.977 | -0.0328 | 0.1264 |
| primary, ra only | 3167 | 1705 | 0.5384 | 167.482 | 0.0529 | 0.0187 | 0.0861 | 1.9658 | 0.0012 | 0.1031 |
| primary, rebounds only | 5756 | 3076 | 0.5344 | 342.5185 | 0.0595 | 0.0349 | 0.0859 | 1.9993 | 0.0182 | 0.0996 |
| primary, threes only | 5695 | 2867 | 0.5034 | 207.558 | 0.0364 | 0.0117 | 0.063 | 2.1492 | -0.001 | 0.0788 |
