# Phase 5 VALIDATION stage: 2023-24, run once with the frozen rule file phase5_revised_rule.frozen_rule.json; k and the rule were NOT re-estimated (no recalibration, 20,000-draw simulations)

In the development stage everything is in-sample for k and for the choice among the tried rules, so results are optimistic by construction; in the validation stage k and the rule come from the development season. Bettable books: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Unibet (Fanatics is absent before 2025). Prices are quotes stamped at or before each decision time, main line on the consensus line, no stale quotes; pushes refund; the market probability is the de-vigged consensus (at least 3 agreeing books, all books counted).

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

k per stat (slope floored at 0, capped at 0.5): {'assists': 0.2876616660071663, 'ra': 0.2296613439360755, 'rebounds': 0.2615007750472374, 'threes': 0.31211602398414057}

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
| PRIMARY | stands | 15015 | 7801 | 0.5195 | 828.7417 | 0.0552 | 0.0355 | 0.0741 | 2.0701 |
| PRIMARY | early_exit_flagged_books | 14898 | 7787 | 0.5227 | 914.6849 | 0.0614 | 0.0412 | 0.0804 | 2.0682 |
| PRIMARY | early_exit_all_books | 14823 | 7781 | 0.5249 | 977.1447 | 0.0659 | 0.0451 | 0.0845 | 2.0681 |

CLV to T-5 (bets placed at T-240 or T-60): {'bets_before_T5': 10617, 'with_T5_quote': 10507, 'line_unchanged': 9467, 'line_moved': 1040}

| measure | n | mean | lo | hi |
|---|---|---|---|---|
| consensus fair probability change to T-5, line unchanged | 9467 | 0.0051 | 0.0046 | 0.0056 |
| line moved by T-5: points moved in the bettor's favour | 1040 | -0.0019 | -0.0808 | 0.072 |

### Primary rule by side, stat and book

| best_side | bets | mean_ev |
|---|---|---|
| over | 5252 | 0.0539 |
| under | 9763 | 0.0611 |

| stat | bets | mean_ev | profit |
|---|---|---|---|
| assists | 4495 | 0.0634 | 281.0053 |
| ra | 2488 | 0.0562 | 116.9123 |
| rebounds | 4475 | 0.0565 | 262.7098 |
| threes | 3557 | 0.0568 | 168.1143 |

| book | bets | profit |
|---|---|---|
| betmgm | 1930 | 126.1556 |
| betrivers | 757 | 64.4203 |
| draftkings | 5254 | 301.0351 |
| fanduel | 4370 | 256.081 |
| unibet_us | 796 | 69.4831 |
| williamhill_us | 1908 | 11.5666 |

### Decomposition: where the primary rule's ROI comes from (settlement: bet stands)

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|
| ALL UNDERS at T-60, median price (no model) | 53292 | 27816 | 0.522 | -1877.2603 | -0.0352 | -0.0452 | -0.0258 | 1.8794 |
| ALL UNDERS at T-60, best price (no model) | 53292 | 27816 | 0.522 | -610.2906 | -0.0115 | -0.0215 | -0.0017 | 1.9265 |
| ALL OVERS at T-60, median price (no model) | 53292 | 25476 | 0.478 | -5165.8322 | -0.0969 | -0.1065 | -0.0872 | 1.9267 |
| ALL OVERS at T-60, best price (no model) | 53292 | 25476 | 0.478 | -3873.6767 | -0.0727 | -0.0825 | -0.0626 | 1.9792 |
| PRIMARY bets repriced at the MEDIAN bettable price | 15015 | 7801 | 0.5195 | 210.1585 | 0.014 | -0.0053 | 0.0322 | 1.9889 |
| MARKET-ONLY rule (k = 0): best price vs consensus fair probability, no model | 1182 | 559 | 0.4729 | 53.4829 | 0.0452 | -0.0176 | 0.1096 | 2.2928 |
| PRIMARY bets whose quote was 0-5 minutes old | 11492 | 5976 | 0.52 | 647.7101 | 0.0564 | 0.0356 | 0.0769 | 2.0705 |
| PRIMARY bets whose quote was 5-15 minutes old | 739 | 386 | 0.5223 | 37.4419 | 0.0507 | -0.0358 | 0.1392 | 2.0391 |
| PRIMARY bets, overs only | 5252 | 2563 | 0.488 | 126.9869 | 0.0242 | -0.0089 | 0.0548 | 2.1616 |
| PRIMARY bets, unders only | 9763 | 5238 | 0.5365 | 701.7547 | 0.0719 | 0.0493 | 0.0954 | 2.021 |
| PRIMARY rule using only quotes at most 15 minutes old | 15015 | 7801 | 0.5195 | 828.7417 | 0.0552 | 0.0355 | 0.0741 | 2.0701 |

### Placebo model (same k, selection and prices; probabilities swapped with a random other market of the same stat)

| run | bets | roi | roi_lo | roi_hi | clv_prob_pts | clv_lo | clv_hi | line_move_pts |
|---|---|---|---|---|---|---|---|---|
| placebo 11 | 37995 | -0.0472 | -0.0585 | -0.0367 | 0.2126 | 0.1913 | 0.2347 | -0.7133 |
| placebo 12 | 37851 | -0.0398 | -0.0516 | -0.0286 | 0.2094 | 0.1861 | 0.2319 | -0.6992 |
| placebo 13 | 37894 | -0.0391 | -0.0503 | -0.0286 | 0.2098 | 0.1887 | 0.231 | -0.7171 |
| placebo 14 | 37895 | -0.0391 | -0.0492 | -0.0288 | 0.2104 | 0.1885 | 0.2333 | -0.7126 |

### CLV: real minus placebo

| measure | real | real_lo | placebo_mean | placebo_max_hi | excess | excess_conservative_lower |
|---|---|---|---|---|---|---|
| CLV, consensus fair probability to T-5, line unchanged (points) | 0.5107 | 0.4625 | 0.2106 | 0.2347 | 0.3001 | 0.2279 |
| line move in the bettor's favour by T-5 (points) | -0.0019 | -0.0808 | -0.7106 |  | 0.7086 |  |

### Exposure: {'bets': 15015, 'distinct_player_games': 9543, 'mean_bets_per_player_game': 1.5734045897516504, 'max_bets_per_player_game': 4}

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj | distinct_player_games |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary, max 1 bet per player-game | stands | 9543 | 4926 | 0.5162 | 646.6959 | 0.0678 | 0.0463 | 0.09 | 2.1113 | 0.0304 | 0.1041 | 9543 |
| primary, max 2 bets per player-game | stands | 13385 | 6954 | 0.5195 | 829.0774 | 0.0619 | 0.0417 | 0.0819 | 2.083 | 0.032 | 0.0946 | 9543 |
| primary, max 2 bets per game | stands | 2632 | 1364 | 0.5182 | 240.354 | 0.0913 | 0.0478 | 0.1311 | 2.1627 | 0.0248 | 0.1534 | 2369 |

## 3. 5 tried rules (+3 exposure caps above) (settlement: bet stands). `lo_adj/hi_adj` = interval at the Bonferroni level for 35 rules

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj |
|---|---|---|---|---|---|---|---|---|---|---|
| e=0.03, time=earliest, min_books=3 | 15015 | 7801 | 0.5195 | 828.7417 | 0.0552 | 0.0355 | 0.0741 | 2.0701 | 0.0244 | 0.085 |
| primary, assists only | 4495 | 2365 | 0.5261 | 281.0053 | 0.0625 | 0.0327 | 0.0935 | 2.0457 | 0.0143 | 0.1122 |
| primary, ra only | 2488 | 1324 | 0.5322 | 116.9123 | 0.047 | 0.0092 | 0.0837 | 1.9763 | -0.0121 | 0.1116 |
| primary, rebounds only | 4475 | 2375 | 0.5307 | 262.7098 | 0.0587 | 0.0309 | 0.0881 | 2.0115 | 0.0134 | 0.1037 |
| primary, threes only | 3557 | 1737 | 0.4883 | 168.1143 | 0.0473 | 0.0103 | 0.0846 | 2.2405 | -0.0223 | 0.1074 |
