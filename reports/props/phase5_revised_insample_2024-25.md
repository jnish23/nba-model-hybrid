# Phase 5 VALIDATION stage: 2024-25, run once with the frozen rule file phase5_revised_rule.frozen_rule.json; k and the rule were NOT re-estimated (no recalibration, 20,000-draw simulations)

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

k per stat (slope floored at 0, capped at 0.5): {'assists': 0.2876616660071663, 'ra': 0.2296613439360755, 'rebounds': 0.2615007750472374, 'threes': 0.31211602398414057}

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
| PRIMARY | stands | 15675 | 7995 | 0.51 | 614.6875 | 0.0392 | 0.0232 | 0.056 | 2.0674 |
| PRIMARY | early_exit_flagged_books | 15617 | 7985 | 0.5113 | 652.8932 | 0.0418 | 0.026 | 0.0587 | 2.0672 |
| PRIMARY | early_exit_all_books | 15490 | 7977 | 0.515 | 762.7158 | 0.0492 | 0.0335 | 0.0666 | 2.0665 |

CLV to T-5 (bets placed at T-240 or T-60): {'bets_before_T5': 11393, 'with_T5_quote': 11007, 'line_unchanged': 10060, 'line_moved': 947}

| measure | n | mean | lo | hi |
|---|---|---|---|---|
| consensus fair probability change to T-5, line unchanged | 10060 | 0.0049 | 0.0044 | 0.0054 |
| line moved by T-5: points moved in the bettor's favour | 947 | 0.1109 | 0.0376 | 0.1795 |

### Primary rule by side, stat and book

| best_side | bets | mean_ev |
|---|---|---|
| over | 7614 | 0.0534 |
| under | 8061 | 0.0594 |

| stat | bets | mean_ev | profit |
|---|---|---|---|
| assists | 4493 | 0.0592 | 210.533 |
| ra | 2136 | 0.0521 | 114.3441 |
| rebounds | 5025 | 0.0563 | 204.1915 |
| threes | 4021 | 0.0559 | 85.6189 |

| book | bets | profit |
|---|---|---|
| betmgm | 4119 | 234.1274 |
| betrivers | 2014 | 138.0655 |
| draftkings | 4222 | 134.8007 |
| fanatics | 4 | -1.65 |
| fanduel | 4023 | 43.058 |
| williamhill_us | 1293 | 66.2859 |

### Decomposition: where the primary rule's ROI comes from (settlement: bet stands)

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|
| ALL UNDERS at T-60, median price (no model) | 48908 | 25518 | 0.5218 | -1884.1948 | -0.0385 | -0.0479 | -0.0279 | 1.8703 |
| ALL UNDERS at T-60, best price (no model) | 48908 | 25518 | 0.5218 | -442.2888 | -0.009 | -0.0188 | 0.002 | 1.9282 |
| ALL OVERS at T-60, median price (no model) | 48908 | 23390 | 0.4782 | -4575.1703 | -0.0935 | -0.1042 | -0.0838 | 1.9288 |
| ALL OVERS at T-60, best price (no model) | 48908 | 23390 | 0.4782 | -3057.07 | -0.0625 | -0.0735 | -0.0524 | 1.9956 |
| PRIMARY bets repriced at the MEDIAN bettable price | 15675 | 7995 | 0.51 | -233.6993 | -0.0149 | -0.03 | 0.0005 | 1.9584 |
| MARKET-ONLY rule (k = 0): best price vs consensus fair probability, no model | 2100 | 966 | 0.46 | 44.7755 | 0.0213 | -0.0255 | 0.0687 | 2.2401 |
| PRIMARY bets whose quote was 0-5 minutes old | 10313 | 5243 | 0.5084 | 372.6598 | 0.0361 | 0.0172 | 0.056 | 2.0692 |
| PRIMARY bets whose quote was 5-15 minutes old | 378 | 192 | 0.5079 | 14.7349 | 0.039 | -0.0729 | 0.1426 | 2.0628 |
| PRIMARY bets, overs only | 7614 | 3722 | 0.4888 | 181.1014 | 0.0238 | -0.0022 | 0.0504 | 2.1303 |
| PRIMARY bets, unders only | 8061 | 4273 | 0.5301 | 433.5861 | 0.0538 | 0.0302 | 0.0771 | 2.008 |
| PRIMARY rule using only quotes at most 15 minutes old | 15675 | 7995 | 0.51 | 614.6875 | 0.0392 | 0.0232 | 0.056 | 2.0674 |

### Placebo model (same k, selection and prices; probabilities swapped with a random other market of the same stat)

| run | bets | roi | roi_lo | roi_hi | clv_prob_pts | clv_lo | clv_hi | line_move_pts |
|---|---|---|---|---|---|---|---|---|
| placebo 11 | 35690 | -0.0403 | -0.0509 | -0.0298 | 0.2905 | 0.2641 | 0.3156 | -0.71 |
| placebo 12 | 35689 | -0.0297 | -0.0395 | -0.0188 | 0.282 | 0.2563 | 0.3062 | -0.7373 |
| placebo 13 | 35630 | -0.0416 | -0.0518 | -0.0309 | 0.2716 | 0.2482 | 0.2966 | -0.7098 |
| placebo 14 | 35696 | -0.0342 | -0.0463 | -0.0227 | 0.2801 | 0.2524 | 0.3046 | -0.7023 |

### CLV: real minus placebo

| measure | real | real_lo | placebo_mean | placebo_max_hi | excess | excess_conservative_lower |
|---|---|---|---|---|---|---|
| CLV, consensus fair probability to T-5, line unchanged (points) | 0.4866 | 0.4361 | 0.2811 | 0.3156 | 0.2056 | 0.1206 |
| line move in the bettor's favour by T-5 (points) | 0.1109 | 0.0376 | -0.7149 |  | 0.8257 |  |

### Exposure: {'bets': 15675, 'distinct_player_games': 10204, 'mean_bets_per_player_game': 1.5361622892983144, 'max_bets_per_player_game': 4}

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj | distinct_player_games |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary, max 1 bet per player-game | stands | 10204 | 5200 | 0.5096 | 522.3322 | 0.0512 | 0.032 | 0.0717 | 2.092 | 0.0234 | 0.081 | 10204 |
| primary, max 2 bets per player-game | stands | 14136 | 7181 | 0.508 | 562.1816 | 0.0398 | 0.0229 | 0.0569 | 2.0766 | 0.0134 | 0.0651 | 10204 |
| primary, max 2 bets per game | stands | 2616 | 1314 | 0.5023 | 138.3844 | 0.0529 | 0.0101 | 0.0973 | 2.1321 | -0.0185 | 0.1244 | 2409 |

## 3. 5 tried rules (+3 exposure caps above) (settlement: bet stands). `lo_adj/hi_adj` = interval at the Bonferroni level for 35 rules

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj |
|---|---|---|---|---|---|---|---|---|---|---|
| e=0.03, time=earliest, min_books=3 | 15675 | 7995 | 0.51 | 614.6875 | 0.0392 | 0.0232 | 0.056 | 2.0674 | 0.0138 | 0.0655 |
| primary, assists only | 4493 | 2337 | 0.5201 | 210.533 | 0.0469 | 0.0178 | 0.0782 | 2.0389 | -0.002 | 0.0885 |
| primary, ra only | 2136 | 1132 | 0.53 | 114.3441 | 0.0535 | 0.0102 | 0.0957 | 1.992 | -0.0064 | 0.1331 |
| primary, rebounds only | 5025 | 2595 | 0.5164 | 204.1915 | 0.0406 | 0.0147 | 0.07 | 2.0316 | -0.0009 | 0.0846 |
| primary, threes only | 4021 | 1931 | 0.4802 | 85.6189 | 0.0213 | -0.0104 | 0.0554 | 2.1841 | -0.0302 | 0.0775 |
