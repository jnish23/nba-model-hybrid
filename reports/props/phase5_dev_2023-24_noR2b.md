# Phase 5 development stage: 2023-24 (frozen configuration + R2b, 20,000-draw simulations)

Everything here is in-sample for k and for the choice among the tried rules: results are optimistic by construction. The validation season (2024-25) is the check, run once with the frozen file `phase5_dev_2023-24_noR2b.frozen_rule.json`. Bettable books: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Unibet (Fanatics is absent before 2025). Prices are quotes stamped at or before each decision time, main line on the consensus line, no stale quotes; pushes refund; the market probability is the de-vigged consensus (at least 3 agreeing books, all books counted).

## 1. Information test (model vs market at the book line)

| stat | n | logloss_model | logloss_market | diff | lo | hi | brier_diff | brier_lo | brier_hi | mean_p_model | mean_p_market | hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 333777 | 0.7011 | 0.6876 | 0.0135 | 0.0113 | 0.0156 | 0.0059 | 0.0049 | 0.0068 | 0.485 | 0.4971 | 0.4792 |
| assists | 34118 | 0.6945 | 0.6851 | 0.0094 | 0.0057 | 0.0134 | 0.0039 | 0.0022 | 0.0057 | 0.4873 | 0.4942 | 0.49 |
| pa | 38208 | 0.7116 | 0.6919 | 0.0197 | 0.016 | 0.0234 | 0.0088 | 0.0071 | 0.0106 | 0.4951 | 0.4998 | 0.4836 |
| points | 47900 | 0.7088 | 0.6918 | 0.0171 | 0.014 | 0.0202 | 0.0077 | 0.0063 | 0.0092 | 0.4978 | 0.4995 | 0.4812 |
| pr | 42990 | 0.7087 | 0.6918 | 0.0169 | 0.0136 | 0.0202 | 0.0075 | 0.006 | 0.009 | 0.488 | 0.5002 | 0.4782 |
| pra | 41020 | 0.7144 | 0.6922 | 0.0222 | 0.0187 | 0.0259 | 0.0099 | 0.0083 | 0.0116 | 0.4893 | 0.4997 | 0.4785 |
| ra | 35834 | 0.7026 | 0.69 | 0.0126 | 0.0084 | 0.0163 | 0.0052 | 0.0033 | 0.0069 | 0.4759 | 0.4982 | 0.4823 |
| rebounds | 48697 | 0.6957 | 0.6874 | 0.0083 | 0.0056 | 0.011 | 0.0035 | 0.0022 | 0.0047 | 0.4728 | 0.497 | 0.4738 |
| threes | 45010 | 0.6741 | 0.6715 | 0.0026 | 0.0003 | 0.0052 | 0.001 | -0.0001 | 0.0021 | 0.4746 | 0.4884 | 0.4704 |

### Information slope and k

| stat | n | slope | slope_lo | slope_hi | mean hit - market | lo | hi |
|---|---|---|---|---|---|---|---|
| all | 333777 | 0.2071 | 0.1595 | 0.2559 | -0.0178 | -0.0229 | -0.0129 |
| assists | 34118 | 0.3174 | 0.233 | 0.3968 | -0.0041 | -0.0133 | 0.0038 |
| pa | 38208 | 0.1297 | 0.0561 | 0.1996 | -0.0162 | -0.025 | -0.0077 |
| points | 47900 | 0.1243 | 0.056 | 0.195 | -0.0183 | -0.0248 | -0.0116 |
| pr | 42990 | 0.1746 | 0.107 | 0.2409 | -0.022 | -0.0292 | -0.0148 |
| pra | 41020 | 0.1096 | 0.0448 | 0.176 | -0.0212 | -0.0294 | -0.0136 |
| ra | 35834 | 0.2577 | 0.1822 | 0.3455 | -0.0159 | -0.0238 | -0.0081 |
| rebounds | 48697 | 0.2889 | 0.214 | 0.3631 | -0.0231 | -0.0302 | -0.016 |
| threes | 45010 | 0.4117 | 0.3101 | 0.51 | -0.018 | -0.0253 | -0.0107 |

k per stat (slope floored at 0, capped at 0.5): {'assists': 0.31742071974827707, 'pa': 0.12968939755013928, 'points': 0.1243057526197666, 'pr': 0.17459432951234113, 'pra': 0.10959136481676977, 'ra': 0.25767004346604383, 'rebounds': 0.2888962514894001, 'threes': 0.4117239046218011}

### Calibration by model-minus-market gap

| model minus market | n | mean p_market | mean p_model | hit rate | hit - market | lo | hi | hit - model | mean gap |
|---|---|---|---|---|---|---|---|---|---|
| (-1.0, -0.1] | 55818 | 0.497 | 0.3298 | 0.4537 | -0.0433 | -0.0548 | -0.033 | 0.1239 | -0.1672 |
| (-0.1, -0.05] | 51202 | 0.4946 | 0.4216 | 0.4592 | -0.0354 | -0.0448 | -0.0257 | 0.0376 | -0.073 |
| (-0.05, -0.02] | 41521 | 0.4955 | 0.4609 | 0.4616 | -0.0339 | -0.0429 | -0.0251 | 0.0006 | -0.0346 |
| (-0.02, 0.02] | 60122 | 0.4972 | 0.497 | 0.4827 | -0.0145 | -0.0229 | -0.0058 | -0.0144 | -0.0002 |
| (0.02, 0.05] | 39746 | 0.4988 | 0.5333 | 0.5009 | 0.0021 | -0.0069 | 0.0116 | -0.0324 | 0.0345 |
| (0.05, 0.1] | 47511 | 0.4993 | 0.572 | 0.5005 | 0.0012 | -0.0086 | 0.0119 | -0.0715 | 0.0727 |
| (0.1, 1.0] | 37857 | 0.4975 | 0.6469 | 0.5086 | 0.0111 | -0.0014 | 0.0249 | -0.1383 | 0.1494 |

## 2. Primary rule (e = 0.03, earliest time with positive EV, at least 3 books, pooled over stats), all settlement treatments

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | stands | 22723 | 11846 | 0.5213 | 1101.3395 | 0.0485 | 0.0305 | 0.0653 | 2.0456 |
| PRIMARY | early_exit_flagged_books | 22563 | 11827 | 0.5242 | 1220.0513 | 0.0541 | 0.036 | 0.0712 | 2.044 |
| PRIMARY | early_exit_all_books | 22429 | 11819 | 0.527 | 1337.5177 | 0.0596 | 0.0418 | 0.0766 | 2.0439 |

CLV to T-5 (bets placed at T-240 or T-60): {'bets_before_T5': 15960, 'with_T5_quote': 15747, 'line_unchanged': 13756, 'line_moved': 1991}

| measure | n | mean | lo | hi |
|---|---|---|---|---|
| consensus fair probability change to T-5, line unchanged | 13756 | 0.0045 | 0.0041 | 0.005 |
| line moved by T-5: points moved in the bettor's favour | 1991 | 0.0773 | -0.0104 | 0.1752 |

### Primary rule by side, stat and book

| best_side | bets | mean_ev |
|---|---|---|
| over | 8062 | 0.0552 |
| under | 14661 | 0.0626 |

| stat | bets | mean_ev | profit |
|---|---|---|---|
| assists | 5009 | 0.067 | 316.4137 |
| pa | 743 | 0.0447 | 3.0466 |
| points | 1199 | 0.0443 | 29.8808 |
| pr | 1709 | 0.0486 | 63.2662 |
| pra | 552 | 0.0444 | 29.7786 |
| ra | 3012 | 0.0595 | 146.5994 |
| rebounds | 5178 | 0.0591 | 302.1754 |
| threes | 5321 | 0.0653 | 210.1787 |

| book | bets | profit |
|---|---|---|
| betmgm | 2682 | 228.9212 |
| betrivers | 915 | 78.8666 |
| draftkings | 8209 | 411.6162 |
| fanduel | 6317 | 278.5025 |
| unibet_us | 984 | 108.7388 |
| williamhill_us | 3616 | -5.3059 |

### Decomposition: where the primary rule's ROI comes from (settlement: bet stands)

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|
| ALL UNDERS at T-60, median price (no model) | 108721 | 56642 | 0.521 | -3418.8947 | -0.0314 | -0.0428 | -0.0209 | 1.8762 |
| ALL UNDERS at T-60, best price (no model) | 108721 | 56642 | 0.521 | -1010.2392 | -0.0093 | -0.0208 | 0.0015 | 1.9197 |
| ALL OVERS at T-60, median price (no model) | 108721 | 52079 | 0.479 | -10864.9581 | -0.0999 | -0.11 | -0.0888 | 1.8994 |
| ALL OVERS at T-60, best price (no model) | 108721 | 52079 | 0.479 | -8489.3244 | -0.0781 | -0.0883 | -0.0667 | 1.9462 |
| PRIMARY bets repriced at the MEDIAN bettable price | 22723 | 11846 | 0.5213 | 185.8722 | 0.0082 | -0.0092 | 0.0243 | 1.9663 |
| MARKET-ONLY rule (k = 0): best price vs consensus fair probability, no model | 1536 | 731 | 0.4759 | 65.4006 | 0.0426 | -0.0154 | 0.0995 | 2.2559 |
| PRIMARY bets whose quote was 0-5 minutes old | 17518 | 9143 | 0.5219 | 866.4051 | 0.0495 | 0.0302 | 0.0679 | 2.0448 |
| PRIMARY bets whose quote was 5-15 minutes old | 1061 | 559 | 0.5269 | 56.4073 | 0.0532 | -0.0154 | 0.1253 | 2.0206 |
| PRIMARY bets, overs only | 8062 | 3974 | 0.4929 | 155.5876 | 0.0193 | -0.0089 | 0.046 | 2.12 |
| PRIMARY bets, unders only | 14661 | 7872 | 0.5369 | 945.7519 | 0.0645 | 0.043 | 0.0864 | 2.0047 |
| PRIMARY rule using only quotes at most 15 minutes old | 22723 | 11846 | 0.5213 | 1101.3395 | 0.0485 | 0.0305 | 0.0653 | 2.0456 |

### Placebo model (same k, selection and prices; probabilities swapped with a random other market of the same stat)

| run | bets | roi | roi_lo | roi_hi | clv_prob_pts | clv_lo | clv_hi | line_move_pts |
|---|---|---|---|---|---|---|---|---|
| placebo 11 | 48420 | -0.0414 | -0.051 | -0.032 | 0.1881 | 0.1669 | 0.2086 | -0.6018 |
| placebo 12 | 48347 | -0.041 | -0.0505 | -0.0317 | 0.1844 | 0.1651 | 0.2055 | -0.566 |
| placebo 13 | 48207 | -0.037 | -0.0469 | -0.0276 | 0.196 | 0.1756 | 0.2154 | -0.5746 |
| placebo 14 | 48226 | -0.0412 | -0.0512 | -0.0316 | 0.1943 | 0.1759 | 0.2126 | -0.5615 |

### CLV: real minus placebo

| measure | real | real_lo | placebo_mean | placebo_max_hi | excess | excess_conservative_lower |
|---|---|---|---|---|---|---|
| CLV, consensus fair probability to T-5, line unchanged (points) | 0.4549 | 0.4103 | 0.1907 | 0.2154 | 0.2642 | 0.1949 |
| line move in the bettor's favour by T-5 (points) | 0.0773 | -0.0104 | -0.576 |  | 0.6533 |  |

### Exposure: {'bets': 22723, 'distinct_player_games': 11526, 'mean_bets_per_player_game': 1.9714558389727572, 'max_bets_per_player_game': 8}

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj | distinct_player_games |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary, max 1 bet per player-game | stands | 11526 | 5966 | 0.5176 | 768.8829 | 0.0667 | 0.0466 | 0.086 | 2.1046 | 0.0383 | 0.1013 | 11526 |
| primary, max 2 bets per player-game | stands | 17244 | 8978 | 0.5206 | 1009.2873 | 0.0585 | 0.041 | 0.0757 | 2.0716 | 0.0334 | 0.0867 | 11526 |
| primary, max 2 bets per game | stands | 2634 | 1347 | 0.5114 | 213.0195 | 0.0809 | 0.0382 | 0.12 | 2.1819 | 0.01 | 0.1424 | 2380 |

## 3. 32 tried rules (+3 exposure caps above) (settlement: bet stands). `lo_adj/hi_adj` = interval at the Bonferroni level for 35 rules

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj |
|---|---|---|---|---|---|---|---|---|---|---|
| e=0.02, time=earliest, min_books=3 | 30464 | 15901 | 0.522 | 1282.1693 | 0.0421 | 0.0262 | 0.0572 | 2.0284 | 0.0151 | 0.0707 |
| e=0.02, time=earliest, min_books=5 | 25532 | 13332 | 0.5222 | 1118.3596 | 0.0438 | 0.0284 | 0.0595 | 2.029 | 0.0157 | 0.0687 |
| e=0.02, time=240, min_books=3 | 14259 | 7483 | 0.5248 | 669.8986 | 0.047 | 0.0258 | 0.0666 | 2.0263 | 0.0109 | 0.0792 |
| e=0.02, time=240, min_books=5 | 12394 | 6520 | 0.5261 | 625.9346 | 0.0505 | 0.0274 | 0.0705 | 2.0271 | 0.0168 | 0.0858 |
| e=0.02, time=60, min_books=3 | 17931 | 9372 | 0.5227 | 750.151 | 0.0418 | 0.0228 | 0.0614 | 2.0268 | 0.0134 | 0.0697 |
| e=0.02, time=60, min_books=5 | 14907 | 7755 | 0.5202 | 577.1042 | 0.0387 | 0.0204 | 0.0577 | 2.029 | 0.0117 | 0.0659 |
| e=0.02, time=5, min_books=3 | 23231 | 12052 | 0.5188 | 803.612 | 0.0346 | 0.0162 | 0.0514 | 2.0283 | 0.0083 | 0.0673 |
| e=0.02, time=5, min_books=5 | 18808 | 9778 | 0.5199 | 739.4202 | 0.0393 | 0.021 | 0.0563 | 2.0305 | 0.0105 | 0.0698 |
| e=0.03, time=earliest, min_books=3 | 22723 | 11846 | 0.5213 | 1101.3395 | 0.0485 | 0.0305 | 0.0653 | 2.0456 | 0.0192 | 0.0781 |
| e=0.03, time=earliest, min_books=5 | 19058 | 9952 | 0.5222 | 984.5751 | 0.0517 | 0.0337 | 0.0694 | 2.046 | 0.0211 | 0.0796 |
| e=0.03, time=240, min_books=3 | 10333 | 5418 | 0.5243 | 552.6519 | 0.0535 | 0.0298 | 0.0767 | 2.0416 | 0.0169 | 0.0938 |
| e=0.03, time=240, min_books=5 | 9006 | 4721 | 0.5242 | 489.182 | 0.0543 | 0.0304 | 0.0786 | 2.0416 | 0.0159 | 0.0919 |
| e=0.03, time=60, min_books=3 | 13126 | 6846 | 0.5216 | 611.0705 | 0.0466 | 0.0259 | 0.0674 | 2.0432 | 0.014 | 0.0739 |
| e=0.03, time=60, min_books=5 | 10984 | 5728 | 0.5215 | 523.8317 | 0.0477 | 0.0258 | 0.071 | 2.0446 | 0.0119 | 0.0795 |
| e=0.03, time=5, min_books=3 | 17285 | 8938 | 0.5171 | 667.4666 | 0.0386 | 0.019 | 0.0578 | 2.0454 | 0.0069 | 0.0666 |
| e=0.03, time=5, min_books=5 | 14022 | 7288 | 0.5198 | 650.2643 | 0.0464 | 0.0263 | 0.0659 | 2.0465 | 0.0167 | 0.0782 |
| e=0.05, time=earliest, min_books=3 | 12742 | 6577 | 0.5162 | 667.3798 | 0.0524 | 0.0316 | 0.0744 | 2.0796 | 0.0214 | 0.0876 |
| e=0.05, time=earliest, min_books=5 | 10642 | 5522 | 0.5189 | 640.5432 | 0.0602 | 0.0375 | 0.0826 | 2.0797 | 0.0241 | 0.0967 |
| e=0.05, time=240, min_books=3 | 5442 | 2866 | 0.5266 | 403.1617 | 0.0741 | 0.0424 | 0.108 | 2.0785 | 0.0227 | 0.1207 |
| e=0.05, time=240, min_books=5 | 4762 | 2506 | 0.5262 | 356.7768 | 0.0749 | 0.0406 | 0.1072 | 2.0776 | 0.0174 | 0.1194 |
| e=0.05, time=60, min_books=3 | 7047 | 3665 | 0.5201 | 404.7728 | 0.0574 | 0.0285 | 0.0849 | 2.0752 | -0.0003 | 0.1042 |
| e=0.05, time=60, min_books=5 | 5912 | 3093 | 0.5232 | 383.2844 | 0.0648 | 0.0385 | 0.0957 | 2.0754 | 0.0159 | 0.1193 |
| e=0.05, time=5, min_books=3 | 9759 | 5003 | 0.5127 | 427.5951 | 0.0438 | 0.021 | 0.0684 | 2.0783 | 0.0104 | 0.085 |
| e=0.05, time=5, min_books=5 | 7888 | 4062 | 0.515 | 406.6297 | 0.0516 | 0.0228 | 0.0774 | 2.0788 | 0.0133 | 0.0902 |
| primary, assists only | 5009 | 2651 | 0.5292 | 316.4137 | 0.0632 | 0.0358 | 0.0913 | 2.0358 | 0.0135 | 0.1081 |
| primary, pa only | 743 | 373 | 0.502 | 3.0466 | 0.0041 | -0.0692 | 0.0733 | 2.0045 | -0.1061 | 0.1089 |
| primary, points only | 1199 | 610 | 0.5088 | 29.8808 | 0.0249 | -0.0314 | 0.0794 | 2.0182 | -0.0513 | 0.1153 |
| primary, pr only | 1709 | 903 | 0.5284 | 63.2662 | 0.037 | -0.0062 | 0.08 | 1.9694 | -0.0301 | 0.1129 |
| primary, pra only | 552 | 289 | 0.5236 | 29.7786 | 0.0539 | -0.0372 | 0.1406 | 2.0197 | -0.0893 | 0.1807 |
| primary, ra only | 3012 | 1613 | 0.5355 | 146.5994 | 0.0487 | 0.0135 | 0.0847 | 1.9677 | -0.0127 | 0.0984 |
| primary, rebounds only | 5178 | 2755 | 0.5321 | 302.1754 | 0.0584 | 0.0345 | 0.0845 | 2.0066 | 0.0205 | 0.0995 |
| primary, threes only | 5321 | 2652 | 0.4984 | 210.1787 | 0.0395 | 0.0105 | 0.0666 | 2.1761 | -0.0064 | 0.0897 |
