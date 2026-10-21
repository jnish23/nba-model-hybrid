# Phase 5 VALIDATION stage: 2025-26, run once with the frozen rule file phase5_revised_rule.frozen_rule.json; k and the rule were NOT re-estimated (R2b per game type, 20,000-draw simulations)

In the development stage everything is in-sample for k and for the choice among the tried rules, so results are optimistic by construction; in the validation stage k and the rule come from the development season. Bettable books: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Unibet (Fanatics is absent before 2025). Prices are quotes stamped at or before each decision time, main line on the consensus line, no stale quotes; pushes refund; the market probability is the de-vigged consensus (at least 3 agreeing books, all books counted).

## 1. Information test (model vs market at the book line)

| stat | n | logloss_model | logloss_market | diff | lo | hi | brier_diff | brier_lo | brier_hi | mean_p_model | mean_p_market | hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 303997 | 0.7063 | 0.6891 | 0.0172 | 0.0148 | 0.0195 | 0.0074 | 0.0064 | 0.0085 | 0.4661 | 0.4957 | 0.4816 |
| assists | 35993 | 0.7009 | 0.6854 | 0.0155 | 0.0116 | 0.0191 | 0.0068 | 0.005 | 0.0084 | 0.4653 | 0.4949 | 0.4808 |
| pa | 24474 | 0.7187 | 0.6929 | 0.0258 | 0.0212 | 0.031 | 0.0111 | 0.009 | 0.0133 | 0.4616 | 0.4988 | 0.4892 |
| points | 53362 | 0.7125 | 0.6922 | 0.0203 | 0.0169 | 0.0233 | 0.0089 | 0.0075 | 0.0103 | 0.4687 | 0.4992 | 0.4868 |
| pr | 32620 | 0.7153 | 0.6927 | 0.0227 | 0.0184 | 0.0278 | 0.0098 | 0.008 | 0.012 | 0.464 | 0.4987 | 0.4842 |
| pra | 39487 | 0.7203 | 0.6929 | 0.0274 | 0.0234 | 0.0319 | 0.0118 | 0.01 | 0.0137 | 0.4628 | 0.4982 | 0.4848 |
| ra | 25820 | 0.7096 | 0.6914 | 0.0182 | 0.0138 | 0.0228 | 0.008 | 0.006 | 0.01 | 0.4683 | 0.4996 | 0.4869 |
| rebounds | 50448 | 0.6992 | 0.6889 | 0.0103 | 0.0074 | 0.0134 | 0.0043 | 0.0031 | 0.0057 | 0.47 | 0.4944 | 0.4812 |
| threes | 41793 | 0.6819 | 0.6787 | 0.0032 | 0.0007 | 0.0057 | 0.0013 | 0.0001 | 0.0024 | 0.4648 | 0.4847 | 0.4631 |

### Information slope and k

| stat | n | slope | slope_lo | slope_hi | mean hit - market | lo | hi |
|---|---|---|---|---|---|---|---|
| all | 303997 | 0.1288 | 0.0786 | 0.1794 | -0.0141 | -0.019 | -0.0092 |
| assists | 35993 | 0.1689 | 0.0949 | 0.2525 | -0.0141 | -0.0224 | -0.0054 |
| pa | 24474 | 0.06 | -0.0217 | 0.1436 | -0.0097 | -0.0196 | 0.0005 |
| points | 53362 | 0.0892 | 0.0302 | 0.1544 | -0.0123 | -0.0188 | -0.0062 |
| pr | 32620 | 0.0799 | -0.0084 | 0.1608 | -0.0145 | -0.0227 | -0.0063 |
| pra | 39487 | 0.0592 | -0.0106 | 0.1235 | -0.0134 | -0.0212 | -0.0052 |
| ra | 25820 | 0.0869 | -0.0151 | 0.1923 | -0.0127 | -0.0218 | -0.0035 |
| rebounds | 50448 | 0.2365 | 0.1573 | 0.3147 | -0.0131 | -0.0204 | -0.0065 |
| threes | 41793 | 0.3687 | 0.2642 | 0.4848 | -0.0216 | -0.0293 | -0.0138 |

k per stat (slope floored at 0, capped at 0.5): {'assists': 0.2876616660071663, 'ra': 0.2296613439360755, 'rebounds': 0.2615007750472374, 'threes': 0.31211602398414057}

### Calibration by model-minus-market gap

| model minus market | n | mean p_market | mean p_model | hit rate | hit - market | lo | hi | hit - model | mean gap |
|---|---|---|---|---|---|---|---|---|---|
| (-1.0, -0.1] | 61983 | 0.4968 | 0.3292 | 0.4737 | -0.0231 | -0.0332 | -0.0121 | 0.1444 | -0.1676 |
| (-0.1, -0.05] | 53650 | 0.495 | 0.4217 | 0.4663 | -0.0287 | -0.0372 | -0.0197 | 0.0446 | -0.0733 |
| (-0.05, -0.02] | 41644 | 0.4948 | 0.4601 | 0.4769 | -0.0179 | -0.0267 | -0.0089 | 0.0167 | -0.0347 |
| (-0.02, 0.02] | 56039 | 0.4956 | 0.4952 | 0.4858 | -0.0098 | -0.0176 | -0.0015 | -0.0094 | -0.0004 |
| (0.02, 0.05] | 34300 | 0.4958 | 0.5301 | 0.4886 | -0.0072 | -0.017 | 0.0024 | -0.0415 | 0.0343 |
| (0.05, 0.1] | 35060 | 0.4961 | 0.568 | 0.4981 | 0.002 | -0.0084 | 0.0128 | -0.0699 | 0.072 |
| (0.1, 1.0] | 21321 | 0.4957 | 0.6402 | 0.5026 | 0.007 | -0.01 | 0.023 | -0.1376 | 0.1446 |

## 2. Primary rule (e = 0.03, earliest time with positive EV, at least 3 books, pooled over stats), all settlement treatments

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | stands | 15525 | 7994 | 0.5149 | 755.6773 | 0.0487 | 0.0314 | 0.0654 | 2.0616 |
| PRIMARY | early_exit_flagged_books | 15503 | 7994 | 0.5156 | 777.6773 | 0.0502 | 0.0329 | 0.0669 | 2.0615 |
| PRIMARY | early_exit_all_books | 15399 | 7982 | 0.5183 | 855.2968 | 0.0555 | 0.0382 | 0.0722 | 2.0609 |

CLV to T-5 (bets placed at T-240 or T-60): {'bets_before_T5': 11960, 'with_T5_quote': 11598, 'line_unchanged': 10526, 'line_moved': 1072}

| measure | n | mean | lo | hi |
|---|---|---|---|---|
| consensus fair probability change to T-5, line unchanged | 10526 | 0.0051 | 0.0046 | 0.0057 |
| line moved by T-5: points moved in the bettor's favour | 1072 | 0.0849 | 0.0151 | 0.1572 |

### Primary rule by side, stat and book

| best_side | bets | mean_ev |
|---|---|---|
| over | 5052 | 0.053 |
| under | 10473 | 0.0598 |

| stat | bets | mean_ev | profit |
|---|---|---|---|
| assists | 4601 | 0.0623 | 150.145 |
| ra | 2121 | 0.0535 | 20.7068 |
| rebounds | 5083 | 0.0568 | 293.5859 |
| threes | 3720 | 0.055 | 291.2397 |

| book | bets | profit |
|---|---|---|
| betmgm | 4904 | 238.6437 |
| betrivers | 1699 | 70.9258 |
| draftkings | 3327 | 183.6646 |
| fanatics | 650 | -11.4795 |
| fanduel | 4247 | 268.2993 |
| williamhill_us | 698 | 5.6235 |

### Decomposition: where the primary rule's ROI comes from (settlement: bet stands)

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds |
|---|---|---|---|---|---|---|---|---|
| ALL UNDERS at T-60, median price (no model) | 51509 | 26940 | 0.523 | -1785.7839 | -0.0347 | -0.0439 | -0.0256 | 1.8705 |
| ALL UNDERS at T-60, best price (no model) | 51509 | 26940 | 0.523 | -317.6801 | -0.0062 | -0.0157 | 0.0031 | 1.9262 |
| ALL OVERS at T-60, median price (no model) | 51509 | 24569 | 0.477 | -4880.9117 | -0.0948 | -0.104 | -0.0852 | 1.9276 |
| ALL OVERS at T-60, best price (no model) | 51509 | 24569 | 0.477 | -3384.5504 | -0.0657 | -0.0753 | -0.0558 | 1.9902 |
| PRIMARY bets repriced at the MEDIAN bettable price | 15525 | 7994 | 0.5149 | -39.3844 | -0.0025 | -0.019 | 0.0134 | 1.96 |
| MARKET-ONLY rule (k = 0): best price vs consensus fair probability, no model | 1907 | 922 | 0.4835 | 133.5148 | 0.07 | 0.0205 | 0.1174 | 2.2354 |
| PRIMARY bets whose quote was 0-5 minutes old | 9897 | 5216 | 0.527 | 729.7808 | 0.0737 | 0.0525 | 0.0959 | 2.0618 |
| PRIMARY bets, overs only | 5052 | 2408 | 0.4766 | 92.6433 | 0.0183 | -0.0135 | 0.0479 | 2.1606 |
| PRIMARY bets, unders only | 10473 | 5586 | 0.5334 | 663.034 | 0.0633 | 0.0419 | 0.0837 | 2.0138 |
| PRIMARY rule using only quotes at most 15 minutes old | 15525 | 7994 | 0.5149 | 755.6773 | 0.0487 | 0.0314 | 0.0654 | 2.0616 |

### Placebo model (same k, selection and prices; probabilities swapped with a random other market of the same stat)

| run | bets | roi | roi_lo | roi_hi | clv_prob_pts | clv_lo | clv_hi | line_move_pts |
|---|---|---|---|---|---|---|---|---|
| placebo 11 | 36310 | -0.0208 | -0.0324 | -0.0092 | 0.3257 | 0.3001 | 0.3533 | -0.7395 |
| placebo 12 | 36368 | -0.0141 | -0.0259 | -0.0031 | 0.3299 | 0.3055 | 0.3586 | -0.77 |
| placebo 13 | 36201 | -0.0197 | -0.03 | -0.0082 | 0.3138 | 0.287 | 0.3433 | -0.7597 |
| placebo 14 | 36252 | -0.0209 | -0.0314 | -0.0093 | 0.3357 | 0.3085 | 0.3628 | -0.7388 |

### CLV: real minus placebo

| measure | real | real_lo | placebo_mean | placebo_max_hi | excess | excess_conservative_lower |
|---|---|---|---|---|---|---|
| CLV, consensus fair probability to T-5, line unchanged (points) | 0.5147 | 0.4617 | 0.3263 | 0.3628 | 0.1884 | 0.0989 |
| line move in the bettor's favour by T-5 (points) | 0.0849 | 0.0151 | -0.752 |  | 0.8369 |  |

### Exposure: {'bets': 15525, 'distinct_player_games': 10073, 'mean_bets_per_player_game': 1.5412488831529831, 'max_bets_per_player_game': 4}

| rule | treatment | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj | distinct_player_games |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary, max 1 bet per player-game | stands | 10073 | 5184 | 0.5146 | 658.5525 | 0.0654 | 0.0449 | 0.0856 | 2.096 | 0.0384 | 0.0942 | 10073 |
| primary, max 2 bets per player-game | stands | 13862 | 7131 | 0.5144 | 746.6718 | 0.0539 | 0.0359 | 0.0711 | 2.0742 | 0.0288 | 0.0824 | 10073 |
| primary, max 2 bets per game | stands | 2576 | 1301 | 0.505 | 164.4464 | 0.0638 | 0.0207 | 0.1043 | 2.129 | -0.0021 | 0.1415 | 2393 |

## 3. 5 tried rules (+3 exposure caps above) (settlement: bet stands). `lo_adj/hi_adj` = interval at the Bonferroni level for 35 rules

| rule | bets | wins | win_rate | profit_units | roi | lo | hi | mean_odds | lo_adj | hi_adj |
|---|---|---|---|---|---|---|---|---|---|---|
| e=0.03, time=earliest, min_books=3 | 15525 | 7994 | 0.5149 | 755.6773 | 0.0487 | 0.0314 | 0.0654 | 2.0616 | 0.0224 | 0.0781 |
| primary, assists only | 4601 | 2349 | 0.5105 | 150.145 | 0.0326 | 0.0029 | 0.063 | 2.0438 | -0.0182 | 0.0792 |
| primary, ra only | 2121 | 1084 | 0.5111 | 20.7068 | 0.0098 | -0.0339 | 0.0512 | 1.9782 | -0.0503 | 0.0821 |
| primary, rebounds only | 5083 | 2672 | 0.5257 | 293.5859 | 0.0578 | 0.032 | 0.0833 | 2.0274 | 0.0155 | 0.0972 |
| primary, threes only | 3720 | 1889 | 0.5078 | 291.2397 | 0.0783 | 0.0416 | 0.1149 | 2.1779 | 0.0219 | 0.1295 |
