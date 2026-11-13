# Props forward-test monitor, 2026-11-22

Database `{DB}` (read-only). Bets 84 (2026-11-08 to 2026-11-21), settled 84, signals 0. Model versions: v1 (60), v1b (24). Minutes before tip of the bets: T-240: 28, T-60: 28, T-5: 28.

**Reading guide.** Primary metric: ROI at the median bettable price. The best-price ROI is value only if a limit order at that price fills (see the fill study when present). CLV is the consensus fair-probability move of the bet side from the bet to the T-5 snapshot (about 10 minutes before tip), in points. Intervals are 95% bootstraps over game dates.

## A. Forward test: game_date >= 2026-11-13

### Configuration integrity (bets placed in the forward test)

| model_version | model sha | rule sha | roster rule | bets | first date | last date | slots |
|---|---|---|---|---|---|---|---|
| v1 | aaaaaaaaaaaa | r1 | v2 | 30 | 2026-11-13 | 2026-11-17 | T-240,T-60,T-5 |
| v1b | bbbbbbbbbbbb | r1 | v2 | 24 | 2026-11-18 | 2026-11-21 | T-240,T-60,T-5 |

**More than one configuration: a model, rule or roster-rule change starts a new numbered segment (see the pre-registration).**

### Forward-test record

| measure | value |
|---|---|
| Bets settled (rows in prop_results) | 54 |
| Stakes (win or loss) | 44 |
| Voids / pushes (not stakes) | 8 / 2 |
| Game dates with stakes | 9 |
| ROI at MEDIAN price (primary) [95% date-clustered CI] | -3.7% [-34.8, +22.3]  P(ROI<=0) = 0.592 |
| ROI at BEST price (secondary) [95% CI] | -0.9% [-32.8, +26.0]  P(ROI<=0) = 0.526 |
| ROI flat -110 | -4.5% |
| Mean CLV, points (n with CLV; clustered SE) | +0.44 (n 36; se 0.37) |


**Segments** (n<50 stakes are flagged; intervals are date-clustered)

#### By side

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| over | 20 | n<50 | 0.650 | 0.527 | 0.538 | +27.0% [-31.3, +69.1] | +30.8% [-28.9, +74.1] | +24.1% | +0.09 (n 17, se 0.34) |
| under | 24 | n<50 | 0.375 | 0.493 | 0.479 | -29.2% [-52.8, -6.0] | -27.3% [-51.5, -3.6] | -28.4% | +0.76 (n 19, se 0.66) |


#### By slot (the T-5 slot first ran 2026-11-12)

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| T-240 | 15 | n<50 | 0.467 | 0.508 | 0.505 | -10.6% [-34.0, +10.0] | -7.8% [-32.0, +13.7] | -10.9% | +0.11 (n 9, se 0.84) |
| T-5 | 14 | n<50 | 0.500 | 0.506 | 0.505 | -4.4% [-60.6, +44.8] | -2.0% [-59.5, +48.7] | -4.5% | +0.49 (n 18, se 0.69) |
| T-60 | 15 | n<50 | 0.533 | 0.512 | 0.509 | +3.9% [-48.6, +56.0] | +6.9% [-47.1, +60.3] | +1.8% | +0.69 (n 9, se 0.34) |


#### By stat

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| assists | 10 | n<50 | 0.800 | 0.522 | 0.532 | +58.8% [+1.6, +98.7] | +63.3% [+4.8, +103.8] | +52.7% | +1.24 (n 8, se 0.33) |
| ra | 10 | n<50 | 0.500 | 0.533 | 0.544 | -4.8% [-77.0, +67.1] | -1.8% [-76.3, +72.3] | -4.6% | -0.94 (n 9, se 0.71) |
| rebounds | 12 | n<50 | 0.333 | 0.494 | 0.472 | -37.0% [-80.3, -4.7] | -35.4% [-79.9, -2.2] | -36.4% | +0.72 (n 9, se 0.49) |
| threes | 12 | n<50 | 0.417 | 0.491 | 0.487 | -21.4% [-71.9, +36.1] | -19.2% [-71.1, +39.8] | -20.5% | +0.80 (n 10, se 1.03) |


#### By stat x side

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| assists / over | 10 | n<50 | 0.800 | 0.522 | 0.532 | +58.8% [+1.6, +98.7] | +63.3% [+4.8, +103.8] | +52.7% | +1.24 (n 8, se 0.33) |
| ra / over | 10 | n<50 | 0.500 | 0.533 | 0.544 | -4.8% [-77.0, +67.1] | -1.8% [-76.3, +72.3] | -4.6% | -0.94 (n 9, se 0.71) |
| rebounds / under | 12 | n<50 | 0.333 | 0.494 | 0.472 | -37.0% [-80.3, -4.7] | -35.4% [-79.9, -2.2] | -36.4% | +0.72 (n 9, se 0.49) |
| threes / under | 12 | n<50 | 0.417 | 0.491 | 0.487 | -21.4% [-71.9, +36.1] | -19.2% [-71.1, +39.8] | -20.5% | +0.80 (n 10, se 1.03) |


#### By model version

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 24 | n<50 | 0.500 | 0.517 | 0.511 | -3.3% [-39.7, +27.1] | -0.6% [-37.4, +30.3] | -4.5% | +0.72 (n 20, se 0.63) |
| v1b | 20 | n<50 | 0.500 | 0.498 | 0.501 | -4.1% [-59.1, +40.9] | -1.3% [-58.2, +45.3] | -4.5% | +0.10 (n 16, se 0.16) |


#### By claimed EV at best price (does claimed EV track realised ROI?)

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| 3-4% | 7 | n<50 | 0.571 | 0.507 | 0.509 | +8.3% [-72.1, +66.4] | +11.7% [-71.3, +70.8] | +9.1% | +0.87 (n 5, se 0.42) |
| 4-5% | 7 | n<50 | 0.571 | 0.520 | 0.512 | +8.1% [-38.5, +60.3] | +11.4% [-36.7, +65.5] | +9.1% | +0.25 (n 8, se 0.85) |
| 5-7% | 13 | n<50 | 0.385 | 0.508 | 0.512 | -27.0% [-61.4, +1.9] | -24.9% [-60.1, +5.1] | -26.6% | -0.20 (n 12, se 0.73) |
| 7%+ | 17 | n<50 | 0.529 | 0.505 | 0.498 | +4.4% [-39.5, +34.0] | +7.2% [-38.0, +37.6] | +1.1% | +1.10 (n 11, se 0.81) |


#### By expected minutes

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| <20 | 13 | n<50 | 0.692 | 0.516 | 0.517 | +32.9% [-8.7, +78.9] | +36.7% [-6.1, +83.4] | +32.2% | -0.22 (n 12, se 0.53) |
| 20-28 | 17 | n<50 | 0.529 | 0.510 | 0.511 | +3.4% [-54.7, +40.7] | +6.5% [-53.5, +44.5] | +1.1% | +0.48 (n 14, se 0.70) |
| 28-34 | 5 | n<50 | 0.200 | 0.500 | 0.495 | -60.9% [-100.0, +17.3] | -60.3% [-100.0, +19.0] | -61.8% | +0.39 (n 4, se 0.32) |
| 34+ | 9 | n<50 | 0.333 | 0.499 | 0.488 | -38.1% [-73.9, +8.3] | -36.2% [-73.1, +11.3] | -36.4% | +1.73 (n 6, se 0.72) |


#### By week (Monday)

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | 14 | n<50 | 0.643 | 0.514 | 0.507 | +24.8% [-5.4, +31.4] | +28.1% [-1.2, +34.2] | +22.7% | +1.03 (n 12, se 0.94) |
| 2026-11-16 | 30 | n<50 | 0.433 | 0.506 | 0.506 | -17.0% [-54.4, +18.1] | -14.4% [-53.1, +21.8] | -17.3% | +0.15 (n 24, se 0.21) |


#### Information slope on the bets (hit_over - p_market on p_model - p_market)

| stat | n | dates | mean(hit - p_mkt) | slope through origin | slope with intercept | intercept |
|---|---|---|---|---|---|---|
| assists | 11 | 7 | -0.519 | +0.004 [-7.436, +5.014] | -0.086 [-0.711, +0.527] | -0.519 [-0.542, -0.486] |
| ra | 10 | 8 | -0.533 | -2.259 [-5.523, +5.871] | +0.453 [-0.352, +1.404] | -0.539 [-0.581, -0.505] |
| rebounds | 13 | 9 | -0.498 | -4.160 [-6.869, -1.858] | +0.163 [-0.406, +1.262] | -0.506 [-0.593, -0.486] |
| threes | 12 | 9 | -0.509 | -4.969 [-12.605, +1.138] | -0.777 [-1.191, +0.552] | -0.495 [-0.550, -0.471] |
| pooled | 46 | 9 | -0.513 | -3.050 [-5.301, -1.104] | -0.010 [-0.262, +0.352] | -0.513 [-0.530, -0.496] |


_Market-level slope skipped: needs --nba-db for outcomes of non-bet signals._


#### Calibration of p* on settled stakes

| measure | value |
|---|---|
| Actual wins / expected under p* | 22 / 22.3  (diff -0.3, clustered SE 3.4, z -0.08) |
| Win rate vs mean p* vs mean market fair p (bet side) | 0.500 / 0.506 / 0.508 |
| Brier, bet side: p* / p_model / p_market | 0.2372 / 0.2298 / 0.2406 |


#### CLV against break-even

| price | mean decimal price | break-even CLV (pts) | mean CLV earned (pts) | ROI implied by CLV | realised ROI, same bets |
|---|---|---|---|---|---|
| median | 1.913 | +1.75 | +0.48 | -2.16% | -17.11% |
| best | 1.971 | +0.22 | +0.48 | +0.81% | -14.90% |
| flat | 1.909 | +1.77 | +0.48 | -2.47% |  |

28 stakes with a CLV value. Break-even CLV = mean(1/price - fair probability of the bet side at the bet time), in points.


#### Fill study (prop_fill_checks)

- bets with a fill check: 0


#### Ex-post descriptive splits (never used for decisions)

#### By book of the best price

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| betmgm | 14 | n<50 | 0.500 | 0.506 | 0.505 | -4.4% [-60.6, +44.8] | -2.0% [-59.5, +48.7] | -4.5% | +0.49 (n 18, se 0.69) |
| draftkings | 15 | n<50 | 0.533 | 0.512 | 0.509 | +3.9% [-48.6, +56.0] | +6.9% [-47.1, +60.3] | +1.8% | +0.69 (n 9, se 0.34) |
| fanduel | 15 | n<50 | 0.467 | 0.508 | 0.505 | -10.6% [-34.0, +10.0] | -7.8% [-32.0, +13.7] | -10.9% | +0.11 (n 9, se 0.84) |



### Decision rule (DRAFT pre-registration)

| item | value |
|---|---|
| Status | **CONTINUE** |
| Reason | horizon not reached, no futility trigger |
| Stakes / horizon | 44 / 8800 (to 2027-04-12 if earlier) |
| Game dates | 9 |
| Median-price ROI [95% CI] | -3.67% [-34.51, +23.50] |
| Conditional power at last look |  |


### Sample-size arithmetic on the forward record

From this record's 9 game dates and 44 stakes: median-price ROI -3.67%, half-width now +/-29.0 pts, 4.9 stakes per date, residual SD per date 2.2 units (44 units per 100 stakes).

| target 95% half-width | game dates needed | stakes needed | stakes per date now | resid SD per date, per 100 stakes (units) |
|---|---|---|---|---|
| +/-2.0 pts | 1898 | 9277 | 4.9 | 44.4 |
| +/-1.5 pts | 3373 | 16492 | 4.9 | 44.4 |


### Secondary hypothesis (exploratory): unders with expected minutes < 20

| measure | value |
|---|---|
| Bets settled (rows in prop_results) | 7 |
| Stakes (win or loss) | 7 |
| Voids / pushes (not stakes) | 0 / 0 |
| Game dates with stakes | 6 |
| ROI at MEDIAN price (primary) [95% date-clustered CI] | +35.0% [-19.4, +90.8]  P(ROI<=0) = 0.110 |
| ROI at BEST price (secondary) [95% CI] | +38.7% [-17.0, +96.2]  P(ROI<=0) = 0.110 |
| ROI flat -110 | +36.4% |
| Mean CLV, points (n with CLV; clustered SE) | -0.62 (n 5; se 0.64) |

The pre-registered minimum is 300 stakes; this sample is below it.


## B. Pre-registration record (game_date < 2026-11-13), descriptive only

Reasons it is not part of the test: rows restored from an export after the 2026-10-31 data loss, no T-5 slot before 2026-11-12, and inspected in the audit that wrote the rules.

### Pre-registration record

| measure | value |
|---|---|
| Bets settled (rows in prop_results) | 30 |
| Stakes (win or loss) | 28 |
| Voids / pushes (not stakes) | 1 / 1 |
| Game dates with stakes | 5 |
| ROI at MEDIAN price (primary) [95% date-clustered CI] | -39.3% [-68.2, +0.5]  P(ROI<=0) = 0.964 |
| ROI at BEST price (secondary) [95% CI] | -37.3% [-67.5, +4.5]  P(ROI<=0) = 0.964 |
| ROI flat -110 | -38.6% |
| Mean CLV, points (n with CLV; clustered SE) | +0.97 (n 20; se 0.44) |


**Segments** (n<50 stakes are flagged; intervals are date-clustered)

#### By side

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| over | 14 | n<50 | 0.143 | 0.515 | 0.529 | -73.3% [-100.0, -46.5] | -72.5% [-100.0, -44.9] | -72.7% | +1.50 (n 11, se 0.67) |
| under | 14 | n<50 | 0.500 | 0.499 | 0.499 | -5.3% [-61.0, +60.2] | -2.1% [-59.9, +66.3] | -4.5% | +0.32 (n 9, se 0.33) |


#### By slot (the T-5 slot first ran 2026-11-12)

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| T-240 | 9 | n<50 | 0.333 | 0.514 | 0.515 | -37.6% [-78.4, -5.9] | -36.0% [-77.7, -3.4] | -36.4% | +0.67 (n 5, se 0.58) |
| T-5 | 10 | n<50 | 0.300 | 0.494 | 0.505 | -42.4% [-100.0, +34.0] | -39.9% [-100.0, +40.0] | -42.7% | +0.93 (n 10, se 0.53) |
| T-60 | 9 | n<50 | 0.333 | 0.515 | 0.522 | -37.5% [-81.8, +17.8] | -35.6% [-81.1, +21.1] | -36.4% | +1.33 (n 5, se 0.40) |


#### By stat

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| assists | 8 | n<50 | 0.250 | 0.524 | 0.541 | -53.2% [-100.0, -16.2] | -51.8% [-100.0, -12.9] | -52.3% | +1.07 (n 6, se 0.65) |
| ra | 6 | n<50 | 0.000 | 0.502 | 0.512 | -100.0% [-100.0, -100.0] | -100.0% [-100.0, -100.0] | -100.0% | +2.01 (n 5, se 0.65) |
| rebounds | 8 | n<50 | 0.625 | 0.493 | 0.491 | +18.9% [-35.1, +64.9] | +22.7% [-33.0, +70.8] | +19.3% | +1.06 (n 4, se 0.40) |
| threes | 6 | n<50 | 0.333 | 0.508 | 0.508 | -37.4% [-100.0, +51.9] | -35.0% [-100.0, +58.1] | -36.4% | -0.27 (n 5, se 0.55) |


#### By stat x side

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| assists / over | 8 | n<50 | 0.250 | 0.524 | 0.541 | -53.2% [-100.0, -16.2] | -51.8% [-100.0, -12.9] | -52.3% | +1.07 (n 6, se 0.65) |
| ra / over | 6 | n<50 | 0.000 | 0.502 | 0.512 | -100.0% [-100.0, -100.0] | -100.0% [-100.0, -100.0] | -100.0% | +2.01 (n 5, se 0.65) |
| rebounds / under | 8 | n<50 | 0.625 | 0.493 | 0.491 | +18.9% [-35.1, +64.9] | +22.7% [-33.0, +70.8] | +19.3% | +1.06 (n 4, se 0.40) |
| threes / under | 6 | n<50 | 0.333 | 0.508 | 0.508 | -37.4% [-100.0, +51.9] | -35.0% [-100.0, +58.1] | -36.4% | -0.27 (n 5, se 0.55) |


#### By model version

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 28 | n<50 | 0.321 | 0.507 | 0.514 | -39.3% [-68.2, +0.5] | -37.3% [-67.5, +4.5] | -38.6% | +0.97 (n 20, se 0.44) |


#### By claimed EV at best price (does claimed EV track realised ROI?)

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| 3-4% | 6 | n<50 | 0.333 | 0.481 | 0.485 | -36.8% [-100.0, +47.4] | -35.1% [-100.0, +50.4] | -36.4% | +1.41 (n 5, se 0.42) |
| 4-5% | 1 | n<50 | 0.000 | 0.473 | 0.502 | -100.0% [-100.0, -100.0] | -100.0% [-100.0, -100.0] | -100.0% | -0.28 (n 1, se 0.00) |
| 5-7% | 7 | n<50 | 0.286 | 0.519 | 0.535 | -45.0% [-100.0, -3.8] | -42.9% [-100.0, -0.0] | -45.5% | +1.92 (n 6, se 0.56) |
| 7%+ | 14 | n<50 | 0.357 | 0.514 | 0.516 | -33.1% [-86.8, +29.6] | -30.9% [-86.7, +34.4] | -31.8% | +0.13 (n 8, se 0.50) |


#### By expected minutes

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| <20 | 11 | n<50 | 0.455 | 0.524 | 0.532 | -14.5% [-44.3, +19.1] | -12.2% [-42.6, +22.0] | -13.2% | +1.58 (n 7, se 0.35) |
| 20-28 | 10 | n<50 | 0.200 | 0.487 | 0.489 | -61.3% [-100.0, +10.4] | -59.8% [-100.0, +14.9] | -61.8% | +0.83 (n 9, se 0.58) |
| 28-34 | 5 | n<50 | 0.200 | 0.502 | 0.509 | -63.3% [-100.0, +83.6] | -62.1% [-100.0, +89.3] | -61.8% | -0.45 (n 3, se 0.36) |
| 34+ | 2 | n<50 | 0.500 | 0.524 | 0.545 | -5.0% [-100.0, +89.9] | -0.4% [-100.0, +99.1] | -4.5% | +2.14 (n 1, se 0.00) |


#### By week (Monday)

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-02 | 6 | n<50 | 0.667 | 0.519 | 0.519 | +24.5% [+24.5, +24.5] | +29.6% [+29.6, +29.6] | +27.3% | +1.27 (n 4, se 0.00) |
| 2026-11-09 | 22 | n<50 | 0.227 | 0.503 | 0.512 | -56.6% [-68.8, -26.1] | -55.5% [-68.3, -23.7] | -56.6% | +0.89 (n 16, se 0.55) |


#### Information slope on the bets (hit_over - p_market on p_model - p_market)

| stat | n | dates | mean(hit - p_mkt) | slope through origin | slope with intercept | intercept |
|---|---|---|---|---|---|---|
| assists | 8 | 5 | -0.524 | -5.487 [-10.695, -1.100] | -0.510 [-1.320, +1.761] | -0.507 [-0.581, -0.440] |
| ra | 6 | 5 | -0.502 | -7.584 [-9.181, -6.559] | -0.010 [-1.855, +1.644] | -0.501 [-0.603, -0.389] |
| rebounds | 8 | 5 | -0.507 | -5.660 [-8.613, +3.918] | +0.131 [-2.411, +1.241] | -0.511 [-0.564, -0.382] |
| threes | 7 | 5 | -0.484 | -0.020 [-8.907, +10.821] | -0.107 [-1.448, +1.065] | -0.484 [-0.547, -0.433] |
| pooled | 29 | 5 | -0.505 | -5.336 [-6.932, -2.945] | -0.186 [-0.532, +0.111] | -0.500 [-0.518, -0.475] |


_Market-level slope skipped: needs --nba-db for outcomes of non-bet signals._


#### Calibration of p* on settled stakes

| measure | value |
|---|---|
| Actual wins / expected under p* | 9 / 14.4  (diff -5.4, clustered SE 3.7, z -1.47) |
| Win rate vs mean p* vs mean market fair p (bet side) | 0.321 / 0.514 / 0.507 |
| Brier, bet side: p* / p_model / p_market | 0.2507 / 0.2609 / 0.2432 |


#### CLV against break-even

| price | mean decimal price | break-even CLV (pts) | mean CLV earned (pts) | ROI implied by CLV | realised ROI, same bets |
|---|---|---|---|---|---|
| median | 1.935 | +2.46 | +0.93 | -2.69% | -59.32% |
| best | 1.995 | +0.87 | +0.93 | +0.40% | -57.80% |
| flat | 1.909 | +3.05 | +0.93 | -4.05% |  |

19 stakes with a CLV value. Break-even CLV = mean(1/price - fair probability of the bet side at the bet time), in points.


#### Fill study (prop_fill_checks)

- bets with a fill check: 0


#### Ex-post descriptive splits (never used for decisions)

#### By book of the best price

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| betmgm | 10 | n<50 | 0.300 | 0.494 | 0.505 | -42.4% [-100.0, +34.0] | -39.9% [-100.0, +40.0] | -42.7% | +0.93 (n 10, se 0.53) |
| draftkings | 9 | n<50 | 0.333 | 0.515 | 0.522 | -37.5% [-81.8, +17.8] | -35.6% [-81.1, +21.1] | -36.4% | +1.33 (n 5, se 0.40) |
| fanduel | 9 | n<50 | 0.333 | 0.514 | 0.515 | -37.6% [-78.4, -5.9] | -36.0% [-77.7, -3.4] | -36.4% | +0.67 (n 5, se 0.58) |



### Sample-size arithmetic on the pre-registration record

From this record's 5 game dates and 28 stakes: median-price ROI -39.27%, half-width now +/-35.3 pts, 5.6 stakes per date, residual SD per date 2.3 units (40 units per 100 stakes).

| target 95% half-width | game dates needed | stakes needed | stakes per date now | resid SD per date, per 100 stakes (units) |
|---|---|---|---|---|
| +/-2.0 pts | 1555 | 8707 | 5.6 | 40.2 |
| +/-1.5 pts | 2764 | 15479 | 5.6 | 40.2 |


### Secondary hypothesis candidate on the pre-registration record (descriptive only)

| measure | value |
|---|---|
| Bets settled (rows in prop_results) | 5 |
| Stakes (win or loss) | 5 |
| Voids / pushes (not stakes) | 0 / 0 |
| Game dates with stakes | 3 |
| ROI at MEDIAN price (primary) [95% date-clustered CI] | +51.2% [-1.7, +94.8]  P(ROI<=0) = 0.035 |
| ROI at BEST price (secondary) [95% CI] | +55.9% [+0.3, +100.8]  P(ROI<=0) = 0.000 |
| ROI flat -110 | +52.7% |
| Mean CLV, points (n with CLV; clustered SE) | +1.59 (n 2; se 0.00) |

The pre-registered minimum is 300 stakes; this sample is below it.
