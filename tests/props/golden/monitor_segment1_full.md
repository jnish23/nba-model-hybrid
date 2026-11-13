# Props forward-test monitor, 2026-11-22

Database `{DB}` (read-only). Bets 84 (2026-11-08 to 2026-11-21), settled 84, signals 0. Model versions: v1 (60), v1b (24). Minutes before tip of the bets: T-240: 28, T-60: 28, T-5: 28.

**Reading guide.** Primary metric: ROI at the median bettable price. The best-price ROI is value only if a limit order at that price fills (see the fill study when present). CLV is the consensus fair-probability move of the bet side from the bet to the T-5 snapshot (about 10 minutes before tip), in points. Intervals are 95% bootstraps over game dates.

## Full record (no --since given: descriptive only)

### Configurations in the record

| model_version | model sha | rule sha | roster rule | bets | first date | last date | slots |
|---|---|---|---|---|---|---|---|
| v1 | aaaaaaaaaaaa | r1 | v2 | 60 | 2026-11-08 | 2026-11-17 | T-240,T-60,T-5 |
| v1b | bbbbbbbbbbbb | r1 | v2 | 24 | 2026-11-18 | 2026-11-21 | T-240,T-60,T-5 |

**More than one configuration: a model, rule or roster-rule change starts a new numbered segment (see the pre-registration).**

### All settled bets

| measure | value |
|---|---|
| Bets settled (rows in prop_results) | 84 |
| Stakes (win or loss) | 72 |
| Voids / pushes (not stakes) | 9 / 3 |
| Game dates with stakes | 14 |
| ROI at MEDIAN price (primary) [95% date-clustered CI] | -17.5% [-42.1, +7.1]  P(ROI<=0) = 0.921 |
| ROI at BEST price (secondary) [95% CI] | -15.0% [-40.5, +10.2]  P(ROI<=0) = 0.884 |
| ROI flat -110 | -17.8% |
| Mean CLV, points (n with CLV; clustered SE) | +0.63 (n 56; se 0.29) |


**Segments** (n<50 stakes are flagged; intervals are date-clustered)

#### By side

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| over | 34 | n<50 | 0.441 | 0.522 | 0.534 | -14.3% [-57.5, +28.8] | -11.7% [-56.1, +32.2] | -15.8% | +0.64 (n 28, se 0.39) |
| under | 38 | n<50 | 0.421 | 0.495 | 0.487 | -20.4% [-46.1, +10.1] | -18.0% [-44.6, +13.6] | -19.6% | +0.62 (n 28, se 0.46) |


#### By slot (the T-5 slot first ran 2026-11-12)

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| T-240 | 24 | n<50 | 0.417 | 0.510 | 0.509 | -20.7% [-42.9, -0.8] | -18.3% [-41.2, +2.4] | -20.5% | +0.31 (n 14, se 0.58) |
| T-5 | 24 | n<50 | 0.417 | 0.501 | 0.505 | -20.2% [-62.5, +24.4] | -17.8% [-61.7, +28.6] | -20.5% | +0.64 (n 28, se 0.48) |
| T-60 | 24 | n<50 | 0.458 | 0.513 | 0.514 | -11.6% [-51.5, +30.5] | -9.0% [-49.9, +34.2] | -12.5% | +0.92 (n 14, se 0.27) |


#### By stat

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| assists | 18 | n<50 | 0.556 | 0.523 | 0.536 | +9.0% [-36.3, +54.2] | +12.2% [-34.4, +58.3] | +6.1% | +1.17 (n 14, se 0.34) |
| ra | 16 | n<50 | 0.312 | 0.521 | 0.532 | -40.5% [-87.7, +12.6] | -38.6% [-87.3, +16.2] | -40.3% | +0.12 (n 14, se 0.69) |
| rebounds | 20 | n<50 | 0.450 | 0.494 | 0.480 | -14.7% [-47.9, +17.3] | -12.2% [-46.4, +21.2] | -14.1% | +0.82 (n 13, se 0.37) |
| threes | 18 | n<50 | 0.389 | 0.497 | 0.494 | -26.7% [-65.2, +21.9] | -24.5% [-64.1, +25.0] | -25.8% | +0.44 (n 15, se 0.73) |


#### By stat x side

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| assists / over | 18 | n<50 | 0.556 | 0.523 | 0.536 | +9.0% [-36.3, +54.2] | +12.2% [-34.4, +58.3] | +6.1% | +1.17 (n 14, se 0.34) |
| ra / over | 16 | n<50 | 0.312 | 0.521 | 0.532 | -40.5% [-87.7, +12.6] | -38.6% [-87.3, +16.2] | -40.3% | +0.12 (n 14, se 0.69) |
| rebounds / under | 20 | n<50 | 0.450 | 0.494 | 0.480 | -14.7% [-47.9, +17.3] | -12.2% [-46.4, +21.2] | -14.1% | +0.82 (n 13, se 0.37) |
| threes / under | 18 | n<50 | 0.389 | 0.497 | 0.494 | -26.7% [-65.2, +21.9] | -24.5% [-64.1, +25.0] | -25.8% | +0.44 (n 15, se 0.73) |


#### By model version

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 52 |  | 0.404 | 0.512 | 0.512 | -22.7% [-49.5, +4.7] | -20.3% [-48.0, +7.9] | -22.9% | +0.84 (n 40, se 0.39) |
| v1b | 20 | n<50 | 0.500 | 0.498 | 0.501 | -4.1% [-59.1, +40.9] | -1.3% [-58.2, +45.3] | -4.5% | +0.10 (n 16, se 0.16) |


#### By claimed EV at best price (does claimed EV track realised ROI?)

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| 3-4% | 13 | n<50 | 0.462 | 0.495 | 0.498 | -12.5% [-61.7, +41.2] | -9.9% [-60.7, +45.5] | -11.9% | +1.14 (n 10, se 0.32) |
| 4-5% | 8 | n<50 | 0.500 | 0.514 | 0.510 | -5.4% [-53.9, +43.1] | -2.5% [-52.8, +47.7] | -4.5% | +0.19 (n 9, se 0.76) |
| 5-7% | 20 | n<50 | 0.350 | 0.512 | 0.520 | -33.3% [-60.1, -9.8] | -31.2% [-58.7, -6.7] | -33.2% | +0.51 (n 18, se 0.60) |
| 7%+ | 31 | n<50 | 0.452 | 0.509 | 0.506 | -12.6% [-51.9, +20.4] | -10.0% [-50.5, +23.8] | -13.8% | +0.69 (n 19, se 0.51) |


#### By expected minutes

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| <20 | 24 | n<50 | 0.583 | 0.520 | 0.524 | +11.2% [-14.0, +44.6] | +14.3% [-11.3, +48.3] | +11.4% | +0.44 (n 19, se 0.48) |
| 20-28 | 27 | n<50 | 0.407 | 0.501 | 0.503 | -20.6% [-64.6, +18.4] | -18.1% [-63.4, +22.1] | -22.2% | +0.62 (n 23, se 0.49) |
| 28-34 | 10 | n<50 | 0.200 | 0.501 | 0.502 | -62.1% [-100.0, -3.7] | -61.2% [-100.0, -1.9] | -61.8% | +0.03 (n 7, se 0.33) |
| 34+ | 11 | n<50 | 0.364 | 0.504 | 0.498 | -32.1% [-72.6, +9.4] | -29.7% [-71.6, +13.4] | -30.6% | +1.79 (n 7, se 0.63) |


#### By week (Monday)

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-02 | 6 | n<50 | 0.667 | 0.519 | 0.519 | +24.5% [+24.5, +24.5] | +29.6% [+29.6, +29.6] | +27.3% | +1.27 (n 4, se 0.00) |
| 2026-11-09 | 36 | n<50 | 0.389 | 0.508 | 0.510 | -25.0% [-58.0, +12.0] | -23.0% [-57.0, +14.7] | -25.8% | +0.95 (n 28, se 0.51) |
| 2026-11-16 | 30 | n<50 | 0.433 | 0.506 | 0.506 | -17.0% [-54.4, +18.1] | -14.4% [-53.1, +21.8] | -17.3% | +0.15 (n 24, se 0.21) |


#### Information slope on the bets (hit_over - p_market on p_model - p_market)

| stat | n | dates | mean(hit - p_mkt) | slope through origin | slope with intercept | intercept |
|---|---|---|---|---|---|---|
| assists | 19 | 12 | -0.521 | -2.420 [-6.766, +2.135] | -0.225 [-0.636, +0.284] | -0.518 [-0.537, -0.497] |
| ra | 16 | 13 | -0.521 | -4.650 [-6.641, -0.708] | +0.456 [-0.163, +1.090] | -0.534 [-0.562, -0.504] |
| rebounds | 21 | 14 | -0.502 | -4.493 [-6.659, -2.442] | +0.165 [-0.393, +0.754] | -0.508 [-0.551, -0.484] |
| threes | 19 | 14 | -0.500 | -3.433 [-8.644, +1.849] | -0.582 [-1.027, +0.036] | -0.493 [-0.526, -0.468] |
| pooled | 75 | 14 | -0.510 | -3.823 [-5.433, -2.118] | -0.039 [-0.233, +0.175] | -0.509 [-0.524, -0.495] |


_Market-level slope skipped: needs --nba-db for outcomes of non-bet signals._


#### Calibration of p* on settled stakes

| measure | value |
|---|---|
| Actual wins / expected under p* | 31 / 36.7  (diff -5.7, clustered SE 5.0, z -1.13) |
| Win rate vs mean p* vs mean market fair p (bet side) | 0.431 / 0.509 / 0.508 |
| Brier, bet side: p* / p_model / p_market | 0.2424 / 0.2419 / 0.2416 |


#### CLV against break-even

| price | mean decimal price | break-even CLV (pts) | mean CLV earned (pts) | ROI implied by CLV | realised ROI, same bets |
|---|---|---|---|---|---|
| median | 1.922 | +2.04 | +0.66 | -2.37% | -34.18% |
| best | 1.981 | +0.49 | +0.66 | +0.64% | -32.24% |
| flat | 1.909 | +2.29 | +0.66 | -3.11% |  |

47 stakes with a CLV value. Break-even CLV = mean(1/price - fair probability of the bet side at the bet time), in points.


#### Fill study (prop_fill_checks)

- bets with a fill check: 0


#### Ex-post descriptive splits (never used for decisions)

#### By book of the best price

| segment | stakes | flag | win rate | mean p mkt | mean p* | ROI median [95% CI] | ROI best [95% CI] | ROI flat | CLV pts |
|---|---|---|---|---|---|---|---|---|---|
| betmgm | 24 | n<50 | 0.417 | 0.501 | 0.505 | -20.2% [-62.5, +24.4] | -17.8% [-61.7, +28.6] | -20.5% | +0.64 (n 28, se 0.48) |
| draftkings | 24 | n<50 | 0.458 | 0.513 | 0.514 | -11.6% [-51.5, +30.5] | -9.0% [-49.9, +34.2] | -12.5% | +0.92 (n 14, se 0.27) |
| fanduel | 24 | n<50 | 0.417 | 0.510 | 0.509 | -20.7% [-42.9, -0.8] | -18.3% [-41.2, +2.4] | -20.5% | +0.31 (n 14, se 0.58) |



### Sample-size arithmetic on the full record

From this record's 14 game dates and 72 stakes: median-price ROI -17.52%, half-width now +/-24.8 pts, 5.1 stakes per date, residual SD per date 2.4 units (47 units per 100 stakes).

| target 95% half-width | game dates needed | stakes needed | stakes per date now | resid SD per date, per 100 stakes (units) |
|---|---|---|---|---|
| +/-2.0 pts | 2160 | 11110 | 5.1 | 47.4 |
| +/-1.5 pts | 3841 | 19752 | 5.1 | 47.4 |


### Secondary hypothesis candidate: unders with expected minutes < 20 (post-hoc, descriptive here)

| measure | value |
|---|---|
| Bets settled (rows in prop_results) | 12 |
| Stakes (win or loss) | 12 |
| Voids / pushes (not stakes) | 0 / 0 |
| Game dates with stakes | 9 |
| ROI at MEDIAN price (primary) [95% date-clustered CI] | +41.8% [+2.9, +85.8]  P(ROI<=0) = 0.015 |
| ROI at BEST price (secondary) [95% CI] | +45.9% [+5.5, +92.1]  P(ROI<=0) = 0.015 |
| ROI flat -110 | +43.2% |
| Mean CLV, points (n with CLV; clustered SE) | +0.01 (n 7; se 0.73) |

The pre-registered minimum is 300 stakes; this sample is below it.
