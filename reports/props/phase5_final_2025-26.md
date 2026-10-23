# Phase 5 VALIDATION stage: 2025-26, run once with the frozen rule file phase5_revised_rule.frozen_rule.json; k and the rule were NOT re-estimated (no recalibration, 20,000-draw simulations)

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

k per stat (slope floored at 0, capped at 0.5): {'assists': 0.2876616660071663, 'ra': 0.2296613439360755, 'rebounds': 0.2615007750472374, 'threes': 0.31211602398414057}

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
| PRIMARY | stands | 15593 | 7987 | 0.5122 | 722.2136 | 0.0463 | 0.0296 | 0.0625 | 2.0689 |
| PRIMARY | early_exit_flagged_books | 15569 | 7987 | 0.513 | 746.2136 | 0.0479 | 0.0308 | 0.0644 | 2.0688 |
| PRIMARY | early_exit_all_books | 15452 | 7976 | 0.5162 | 838.718 | 0.0543 | 0.0372 | 0.0706 | 2.068 |

CLV to T-5 (bets placed at T-240 or T-60): {'bets_before_T5': 12007, 'with_T5_quote': 11646, 'line_unchanged': 10585, 'line_moved': 1061}

| measure | n | mean | lo | hi |
|---|---|---|---|---|
| consensus fair probability change to T-5, line unchanged | 10585 | 0.0051 | 0.0045 | 0.0056 |
| line moved by T-5: points moved in the bettor's favour | 1061 | 0.0811 | 0.0087 | 0.1507 |

### Primary rule by side, stat and book

| best_side | bets | mean_ev |
|---|---|---|
| over | 5562 | 0.0526 |
| under | 10031 | 0.0601 |

| stat | bets | mean_ev | profit |
|---|---|---|---|
| assists | 4684 | 0.0624 | 152.572 |
| ra | 2106 | 0.0536 | 25.8083 |
| rebounds | 5053 | 0.0565 | 306.2085 |
| threes | 3750 | 0.0545 | 237.6248 |

| book | bets | profit |
|---|---|---|
| betmgm | 4924 | 210.2802 |
| betrivers | 1694 | 67.4164 |
| draftkings | 3334 | 195.158 |
| fanatics | 655 | -19.8099 |
| fanduel | 4288 | 272.4271 |
| williamhill_us | 698 | -3.2583 |

### Decomposition: where the primary rule's ROI comes from (settlement: bet stands)

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|
| ALL UNDERS at T-60, median price (no model) | 51509 | 26940 | 0.523 | -1785.7839 | -0.0347 | -0.0439 | -0.0256 | 1.8705 |
| ALL UNDERS at T-60, best price (no model) | 51509 | 26940 | 0.523 | -317.6801 | -0.0062 | -0.0157 | 0.0031 | 1.9262 |
| ALL OVERS at T-60, median price (no model) | 51509 | 24569 | 0.477 | -4880.9117 | -0.0948 | -0.104 | -0.0852 | 1.9276 |
| ALL OVERS at T-60, best price (no model) | 51509 | 24569 | 0.477 | -3384.5504 | -0.0657 | -0.0753 | -0.0558 | 1.9902 |
| PRIMARY bets repriced at the MEDIAN bettable price | 15593 | 7987 | 0.5122 | -74.6424 | -0.0048 | -0.0207 | 0.0106 | 1.9669 |
| MARKET-ONLY rule (k = 0): best price vs consensus fair probability, no model | 1907 | 922 | 0.4835 | 133.5148 | 0.07 | 0.0205 | 0.1174 | 2.2354 |
| PRIMARY bets whose quote was 0-5 minutes old | 9910 | 5185 | 0.5232 | 685.752 | 0.0692 | 0.0484 | 0.0897 | 2.0695 |
| PRIMARY bets, overs only | 5562 | 2650 | 0.4764 | 114.7633 | 0.0206 | -0.0085 | 0.0477 | 2.1715 |
| PRIMARY bets, unders only | 10031 | 5337 | 0.5321 | 607.4503 | 0.0606 | 0.0377 | 0.0834 | 2.012 |
| PRIMARY rule using only quotes at most 15 minutes old | 15593 | 7987 | 0.5122 | 722.2136 | 0.0463 | 0.0296 | 0.0625 | 2.0689 |

### Placebo model (same k, selection and prices; probabilities swapped with a random other market of the same stat)

| run | bets | roi | roi_lo | roi_hi | clv_prob_pts | clv_lo | clv_hi | line_move_pts |
|---|---|---|---|---|---|---|---|---|
| placebo 11 | 36250 | -0.0223 | -0.0341 | -0.0109 | 0.3297 | 0.3027 | 0.3548 | -0.7404 |
| placebo 12 | 36367 | -0.0135 | -0.025 | -0.0022 | 0.3346 | 0.3095 | 0.3626 | -0.771 |
| placebo 13 | 36156 | -0.0255 | -0.0362 | -0.0137 | 0.3152 | 0.2877 | 0.3446 | -0.7689 |
| placebo 14 | 36292 | -0.0227 | -0.033 | -0.0106 | 0.3347 | 0.3086 | 0.36 | -0.7346 |

### CLV: real minus placebo

| measure | real | real_lo | placebo_mean | placebo_max_hi | excess | excess_conservative_lower |
|---|---|---|---|---|---|---|
| CLV, consensus fair probability to T-5, line unchanged (points) | 0.5095 | 0.4539 | 0.3285 | 0.3626 | 0.1809 | 0.0913 |
| line move in the bettor's favour by T-5 (points) | 0.0811 | 0.0087 | -0.7537 |  | 0.8348 |  |

### Exposure: {'bets': 15593, 'distinct_player_games': 10145, 'mean_bets_per_player_game': 1.5370133070478067, 'max_bets_per_player_game': 4}

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj | distinct_player_games |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary, max 1 bet per player-game | stands | 10145 | 5201 | 0.5127 | 654.3792 | 0.0645 | 0.0439 | 0.0843 | 2.1044 | 0.0376 | 0.094 | 10145 |
| primary, max 2 bets per player-game | stands | 13961 | 7143 | 0.5116 | 715.508 | 0.0513 | 0.0336 | 0.0694 | 2.0817 | 0.0288 | 0.0792 | 10145 |
| primary, max 2 bets per game | stands | 2575 | 1284 | 0.4986 | 138.294 | 0.0537 | 0.009 | 0.0968 | 2.1387 | -0.015 | 0.1307 | 2396 |

## 3. 5 tried rules (+3 exposure caps above) (settlement: bet stands). `lo_adj/hi_adj` = interval at the Bonferroni level for 35 rules

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj |
|---|---|---|---|---|---|---|---|---|---|---|
| e=0.03, time=earliest, min_books=3 | 15593 | 7987 | 0.5122 | 722.2136 | 0.0463 | 0.0296 | 0.0625 | 2.0689 | 0.0223 | 0.0735 |
| primary, assists only | 4684 | 2398 | 0.512 | 152.572 | 0.0326 | 0.0016 | 0.062 | 2.0384 | -0.0157 | 0.0792 |
| primary, ra only | 2106 | 1079 | 0.5123 | 25.8083 | 0.0123 | -0.0284 | 0.0521 | 1.9782 | -0.0495 | 0.0786 |
| primary, rebounds only | 5053 | 2658 | 0.526 | 306.2085 | 0.0606 | 0.0315 | 0.087 | 2.031 | 0.0175 | 0.1062 |
| primary, threes only | 3750 | 1852 | 0.4939 | 237.6248 | 0.0634 | 0.0298 | 0.0994 | 2.209 | 0.0085 | 0.1154 |
