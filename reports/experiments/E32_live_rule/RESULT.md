# E32 result: the live rule's historical analogue (R0) and the "own view" rules (R1, R2)

Registered in `REGISTRATION.md` (sha256 of the brief's E32 text `6a3fe070...9355a1`) before any outcome was computed. Script `experiments/e32_live_rule.py`;
run from the worktree root with `C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe -m experiments.e32_live_rule` (15 s). Every number below is in
`results.json`, `decision.json`, `tables.md` or one of the CSVs in this folder. No retraining, no network, nothing outside `experiments/e32_live_rule.py` and
`reports/experiments/E32_live_rule/` and `reports/audit_pregame_2026-11/A5_E32_live_rule/` was written.

## Bottom line

- H1 confirmed for spread and total, not for ML. Priced at the close, the live-rule analogue R0 has a hit rate of 50.1% (spread, 2,366 bets) and 50.0% (total, 3,546 bets): ROI at -110 is -4.4% [-8.4, -0.6] and -4.6% [-7.6, -1.3]. It breaks even before the vig and pays the vig. ML R0 is -1.2% [-5.5, +3.1] at zero vig (not below 0; -5.2% [-9.4, -1.1] at the median real closing prices).
- H2 fails everywhere: no own-view arm beats R0. Largest paired gain is total R1 +2.8 points ROI [-1.3, +6.8], 4 of 7 seasons. All six (market, arm) cells have a CI containing 0 and at most 4 of 7 seasons higher.
- H3 fails everywhere: no arm has a zero-vig hit-rate CI with a lower bound above 50% (best: total R1 51.4% [49.1, 53.8]). Any forward test of R1 or R2 is a measurement, not an expected edge.
- What R1 and R2 do change: they stop fading the market (about 45% of their bets fade the move versus 94% / 86% for R0 on spread / total) and they cut volume: spread ~370 -> ~85 bets a season, total ~555 -> ~250-280, ML ~620 -> ~420-450.
- The model's stated win probability is far above what happens: 60-63% on spread and total bets against a 50-51% hit rate.
- Forward season (160 games, real 11 am lines): nothing can be decided; R0 17 spread bets 10-7, 32 totals 13-19, all intervals contain the break-even.

## 1. Arms, pooled over 7 seasons (games with an opening and a closing line)

Day-bootstrap 95% CI (resample game dates, 1,000 draws, seed 42). Games: spread 8,416, total 8,419, ML 8,366 (of 8,916 in the file; the rest have no closing line,
mostly 2019-20 and 2021-22 where closing lines are contaminated or missing). ROI counts pushes as bets placed (stake returned); hit rate = W / (W + L).
Model P(win) is measured at the settled line (R1: model versus close, R2: the transferred edge). REF = the old backtest basis (edge and settlement at the open),
shown for context, not an arm of H1-H3.

| market | arm | bets | W-L-P | hit rate | ROI -110 | ROI zero vig | fade the move | no move | mean model P(win) |
|---|---|---|---|---|---|---|---|---|---|
| spread | R0 | 2366 | 1174-1171-21 | 50.1% [48.0, 52.1] | -4.4% [-8.4, -0.6] | +0.1% [-4.1, 4.1] | 94% | 3% | 61.8% |
| spread | R1 | 595 | 295-296-4 | 49.9% [45.7, 54.0] | -4.7% [-12.8, 3.1] | -0.2% [-8.6, 8.0] | 47% | 12% | 60.7% |
| spread | R2 | 644 | 321-318-5 | 50.2% [46.0, 54.2] | -4.1% [-12.0, 3.4] | +0.5% [-7.9, 8.3] | 44% | 11% | 60.0% |
| spread | REF | 649 | 317-322-10 | 49.6% [45.4, 53.4] | -5.2% [-13.2, 2.0] | -0.8% [-9.1, 6.8] | 43% | 11% | 60.0% |
| total | R0 | 3546 | 1764-1766-16 | 50.0% [48.4, 51.7] | -4.6% [-7.6, -1.3] | -0.1% [-3.2, 3.4] | 86% | 3% | 63.1% |
| total | R1 | 1774 | 904-854-16 | 51.4% [49.1, 53.8] | -1.8% [-6.3, 2.8] | +2.8% [-1.8, 7.6] | 45% | 6% | 62.6% |
| total | R2 | 1942 | 987-937-18 | 51.3% [49.1, 53.7] | -2.0% [-6.2, 2.5] | +2.6% [-1.8, 7.3] | 41% | 5% | 62.5% |
| total | REF | 1946 | 1034-905-7 | 53.3% [51.1, 55.6] | +1.8% [-2.5, 6.1] | +6.6% [2.2, 11.1] | 41% | 5% | 62.5% |

Moneyline (probabilities; fair closing price q of the side bet; ROI at zero vig = outcome / q - 1; the registration gives no -110 ROI for ML, so the table has none):

| arm | bets | W-L | hit rate | mean fair price | ROI zero vig | ROI at real closing prices | fade the move | mean model P(win) |
|---|---|---|---|---|---|---|---|---|
| R0 | 4006 | 1742-2264 | 43.5% [41.8, 45.1] | 43.6% | -1.2% [-5.5, 3.1] | -5.2% [-9.4, -1.1] | 81% | 53.4% |
| R1 | 2790 | 1342-1448 | 48.1% [46.2, 50.1] | 48.1% | -1.3% [-6.2, 3.4] | -5.3% [-10.0, -0.9] | 49% | 56.9% |
| R2 | 2975 | 1459-1516 | 49.0% [47.1, 51.1] | 48.8% | +0.4% [-4.2, 5.0] | -3.7% [-8.2, 0.7] | 44% | 57.1% |
| REF | 3048 | 1492-1556 | 49.0% [47.1, 50.8] | 48.4% | +0.5% [-3.9, 4.9] | -5.1% [-9.4, -1.0] | n/a | 56.7% |

ML bets win at the rate their fair price implies (hit rate equals mean q) whatever the arm; the model's own probability is 5-10 points too high. R0 bets underdogs
(mean price 43.6%), so the registered "hit rate above 50%" test of H3 is a poor yardstick for ML; the zero-vig ROI CI is given beside it and also contains 0 for every arm.
Real closing prices (the cross-book median from `core.odds_clean.load_clean_odds`, not a price at one book) exist for all 8,366 ML-sample games, so they add no
coverage restriction; the gap to the zero-vig ROI is the overround (about 4 points).

## 2. Hypotheses (all in `decision.json`)

**H1: "the live analogue loses" if R0's ROI at -110 has a CI entirely below 0.**
- Spread: -4.4% [-8.4, -0.6]. Confirmed (the upper bound is close to 0).
- Total: -4.6% [-7.6, -1.3]. Confirmed.
- ML: the registration has no -110 ROI; read on zero vig, -1.2% [-5.5, +3.1]. Not confirmed. (At real closing prices -5.2% [-9.4, -1.1], descriptive.)
- The same bets at zero vig are 0.1% / -0.1% / -1.2%: the loss is the vig. R0 does not lose money in a way a zero-fee venue would show.
- Per season R0 spread is negative in 5 of 7 seasons (-12.5, -9.4, -14.2, +3.9, +6.2, -4.2, -5.3), total negative in 6 of 7.

**H2: R1 or R2 better than R0 only if the paired ROI difference at -110 (days on the union of bet days) excludes 0 and the arm's ROI is higher in at least 5 of 7 seasons.**

| cell | difference vs R0 (points of ROI) | 95% CI | union bet days | seasons higher | verdict |
|---|---|---|---|---|---|
| spread R1 | -0.3 | [-7.7, +6.7] | 1039 | 4 of 7 | not better |
| spread R2 | +0.3 | [-6.3, +6.9] | 1041 | 4 of 7 | not better |
| total R1 | +2.8 | [-1.3, +6.8] | 1165 | 4 of 7 | not better |
| total R2 | +2.5 | [-1.5, +6.2] | 1167 | 4 of 7 | not better |
| ML R1 (zero vig) | -0.1 | [-3.9, +3.9] | 1245 | 3 of 7 | not better |
| ML R2 (zero vig) | +1.6 | [-2.1, +5.8] | 1248 | 3 of 7 | not better |

Per-season ROI (spread and total at -110, ML at zero vig; bets in brackets):

| market | arm | 2019-20 | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|---|---|---|---|
| spread | R0 | -12.5% (261) | -9.4% (310) | -14.2% (305) | +3.9% (373) | +6.2% (364) | -4.2% (376) | -5.3% (377) |
| spread | R1 | -20.9% (93) | +5.1% (109) | -9.9% (71) | -8.9% (87) | -3.1% (65) | +3.7% (82) | -0.2% (88) |
| spread | R2 | -20.1% (104) | +5.1% (119) | -11.1% (72) | -12.3% (86) | -4.5% (72) | +12.1% (98) | -1.5% (93) |
| total | R0 | -3.8% (388) | +0.6% (505) | -5.0% (430) | -0.7% (555) | -5.0% (606) | -9.4% (553) | -8.5% (509) |
| total | R1 | -6.6% (276) | +0.1% (311) | +3.6% (153) | +0.4% (314) | -5.4% (331) | -4.5% (195) | +2.9% (194) |
| total | R2 | -7.5% (283) | +0.2% (325) | +5.5% (163) | +0.5% (346) | -7.3% (370) | -1.3% (235) | +0.2% (220) |
| ML | R0 | -3.5% (484) | -3.5% (571) | +1.6% (527) | +8.6% (625) | -6.9% (628) | +1.7% (565) | -6.4% (606) |
| ML | R1 | -10.9% (391) | -9.4% (406) | +12.4% (365) | +6.4% (396) | -9.8% (462) | +8.2% (360) | -2.3% (410) |
| ML | R2 | -11.1% (406) | -8.9% (424) | +16.5% (392) | +7.6% (435) | -6.9% (488) | +8.8% (404) | -1.0% (426) |

Season-level confidence intervals are in `tables.md` (they are wide: +-10 to +-25 points of ROI per season).

**H3: an arm is "expected to pay on a zero-fee venue" only if its pooled zero-vig hit-rate CI has a lower bound above 50%.** None is.
Spread R0/R1/R2 49.9-50.2% (lower bounds 48.0 / 45.7 / 46.0), total R0 50.0% [48.4, 51.7], R1 51.4% [49.1, 53.8], R2 51.3% [49.1, 53.7],
ML 43.5 / 48.1 / 49.0% (lower bounds 41.8 / 46.2 / 47.1). Zero-vig ROI CIs all contain 0 (table in section 1).
Total R1 and R2 are the only arms with a point estimate above 51%; at their volume (about 250-280 bets a season) a season's hit rate has a standard error of about 3 points.

## 3. Exploratory readouts (descriptive; they decide nothing)

**(a) Edge-slope regressions** (`edge_slopes.csv`; OLS with intercept, ML logistic with offset, day-bootstrap 95% CI; own = model - open in the market's units, points or logit).

| market | against | term | coefficient [95% CI] |
|---|---|---|---|
| spread | close (joint) | own | 0.094 [-0.103, 0.280] |
| spread | close (joint) | open - close | -0.050 [-0.168, 0.069] |
| spread | close (single) | model - close | -0.008 [-0.114, 0.100] |
| spread | open (joint) | own | 0.094 [-0.103, 0.280] |
| spread | open (joint) | close - open | 1.050 [0.931, 1.168] |
| spread | open (single) | model - open | 0.205 [0.006, 0.390] |
| total | close (joint) | own | 0.131 [0.010, 0.251] |
| total | close (joint) | open - close | -0.133 [-0.238, -0.023] |
| total | close (single) | model - close | 0.001 [-0.080, 0.083] |
| total | open (joint) | own | 0.131 [0.010, 0.251] |
| total | open (joint) | close - open | 1.133 [1.023, 1.238] |
| total | open (single) | model - open | 0.237 [0.118, 0.358] |
| ML (logit) | close (joint) | own | -0.004 [-0.067, 0.090] |
| ML (logit) | close (joint) | open - close | -0.006 [-0.184, 0.179] |
| ML (logit) | close (single) | model - close | -0.004 [-0.063, 0.082] |
| ML (logit) | open (joint) | own | -0.004 [-0.067, 0.090] |
| ML (logit) | open (joint) | close - open | 1.006 [0.821, 1.184] |
| ML (logit) | open (single) | model - open | 0.005 [-0.066, 0.110] |

Reading: result minus close regressed on own gives 0.09 (spread) and 0.13 (total), the same as the lead's P2 (+0.09 and +0.13). Against the open, the market's own move
carries a coefficient of about 1.0-1.1 (the close really is the better number) and the model's own view adds 0.1-0.2 at most (total) or nothing (ML). The negative total coefficient on (open - close), -0.13 [-0.24, -0.02], means that after the total closed below its open, games landed lower still (about 0.13 points per point of move): the open-to-close move kept being right, so an edge built from the market having moved away from the model is worse than the model's own view. Several intervals exclude 0 (total own, total open-close, the single-regressor model-open slopes for spread and total); 18 coefficients are reported without a multiplicity correction. The total regressions have a positive intercept (+0.7 points: games go over the close line by 0.7 on average, a known level bias).

**(b) R2 threshold sweep** (`sweep_R2.csv`; edge threshold in probability points; no winner picked). Zero-vig hit rate and CI; the 5% row is the registered arm.

| threshold | spread bets | spread hit | total bets | total hit | ML bets | ML hit (zero vig ROI) |
|---|---|---|---|---|---|---|
| 0% | 4523 | 50.1% [48.7, 51.6] | 5729 | 50.6% [49.3, 52.0] | 8363 | 48.7% (-0.6%) |
| 1% | 3264 | 50.5% [48.8, 52.2] | 4736 | 50.9% [49.4, 52.3] | 7062 | 48.7% (-0.7%) |
| 2% | 2254 | 50.3% [48.2, 52.3] | 3862 | 51.1% [49.6, 52.7] | 5912 | 48.8% (-0.6%) |
| 3% | 1556 | 49.6% [47.0, 52.0] | 3128 | 51.1% [49.3, 52.9] | 4766 | 48.8% (-0.3%) |
| 4% | 987 | 51.5% [48.2, 54.7] | 2467 | 51.3% [49.3, 53.4] | 3803 | 48.7% (-1.0%) |
| 5% | 644 | 50.2% [46.0, 54.2] | 1942 | 51.3% [49.1, 53.7] | 2975 | 49.0% (+0.4%) |
| 6% | 425 | 50.9% [45.7, 56.0] | 1527 | 51.1% [48.6, 53.8] | 2295 | 49.9% (+2.0%) |
| 7% | 282 | 49.1% [43.0, 54.9] | 1232 | 51.4% [48.7, 54.2] | 1754 | 50.7% (+4.3%) |
| 8% | 200 | 48.2% [41.4, 55.3] | 1016 | 51.7% [48.5, 54.8] | 1298 | 50.2% (+2.5%) |
| 9% | 139 | 48.9% [41.2, 57.1] | 818 | 50.2% [46.8, 53.8] | 967 | 50.6% (+2.5%) |
| 10% | 96 | 52.1% [42.9, 62.6] | 673 | 50.7% [46.6, 54.8] | 688 | 52.5% (+7.4%) |

The curves are flat; no threshold has a hit-rate CI clearly above 50% in spread or total. The ML zero-vig ROI rises with the threshold (+7.4% at 10%, CI [-2.2, +16.7], 688 bets) but every interval includes 0.

**(c) How often the close crosses the model's number** (crossed = sign(model - close) differs from sign(own)). Share of all games with a non-zero own edge: spread 30.0%, total 28.8%, ML 21.9%.
Among the games the R1 selection picks (open edge above 5%) it is 8.3% (54 of 649), 8.8% (172 of 1,946), 8.5% (258 of 3,048). Result on those crossed games, settled at the close:

| market | crossed games | bet the open side (hit, ROI -110 / zero vig) | bet the close side (hit, ROI -110 / zero vig) |
|---|---|---|---|
| spread | 54 | 49.1% [35.6, 64.2], -6.2% [-31.6, 22.5] / -1.9% [-28.3, 28.3] | 50.9% [35.8, 64.4], -2.7% [-31.4, 22.6] / +1.9% [-28.3, 28.3] |
| total | 172 | 50.6% [42.9, 57.9], -3.4% [-18.1, 10.5] / +1.2% [-14.2, 15.7] | 49.4% [42.1, 57.1], -5.6% [-19.4, 9.0] / -1.2% [-15.7, 14.2] |
| ML | 258 | 58.1% [51.9, 64.3], zero vig +0.8% [-10.6, 13.4] | 41.9% [35.7, 48.1], zero vig +4.5% [-12.6, 22.8] |

Too few games to say anything. For comparison the not-crossed (R1) bets are 49.9% / 51.4% / 48.1% (section 1). The ML hit rates differ because the crossed games are priced at a close that has already moved past the model.

**(d) Forward season with a real 11 am line** (`forward_*` bet CSVs, `results.json['forward']`). 160 completed 2026-27 games in `predictions_log_v2` (model_version v1.0.0_hybrid)
joined with `core.odds_clean.load_clean_odds` (open: `nba_odds_opening` consensus; close: last pre-game snapshot, source "snapshot" for all 160; 11 am: the line `predictions_log_v2` logged).
Games with all three lines: spread 159, total 158, ML 159, over 21 game days. Sigma = one global value from `calibration_meta_with_odds_cand_20261019.json` (14.29, 18.59); the per-game total sigma is not logged.
R0, R1, R2 with the 11 am line as the later line, settled at it. CLV = how much better the 11 am price is than the closing price (points; ML in probability points on fair probabilities, positive = the market moved toward the bet).

| market | arm | bets | W-L-P | hit rate [95% CI] | ROI -110 [95% CI] | fade of open-to-11am move | CLV to close [95% CI] | bets with CLV > 0 |
|---|---|---|---|---|---|---|---|---|
| spread | R0 | 17 | 10-7-0 | 58.8% [35.7, 75.0] | +12.3% [-31.9, 43.2] | 94% | +0.29 pts [-0.32, +1.25] | 24% |
| spread | R1 | 8 | 5-3-0 | 62.5% [33.3, 100] | +19.3% [-36.4, 90.9] | 0% | -0.38 pts [-1.33, +0.33] | 12% |
| spread | R2 | 9 | 4-5-0 | 44.4% [25.0, 100] | -15.2% [-52.3, 90.9] | 0% | -0.50 pts [-1.38, +0.33] | 11% |
| total | R0 | 32 | 13-19-0 | 40.6% [25.0, 56.0] | -22.4% [-52.3, +6.9] | 75% | +0.27 pts [-0.25, +0.85] | 31% |
| total | R1 | 21 | 9-12-0 | 42.9% [20.0, 60.6] | -18.2% [-61.8, +15.7] | 24% | -0.33 pts [-1.10, +0.50] | 19% |
| total | R2 | 22 | 10-12-0 | 45.5% [20.0, 63.2] | -13.2% [-61.8, +20.6] | 23% | -0.36 pts [-1.10, +0.32] | 18% |
| ML | R0 | 35 | 19-16 | 54.3% [39.3, 66.7] | zero vig +6.8% [-26.3, +33.2] | 80% | +1.0 pt [-0.1, +2.4] | 54% |
| ML | R1 | 23 | 17-6 | 73.9% [61.3, 88.2] | zero vig +22.7% [-3.8, +55.0] | 26% | -0.3 pt [-2.6, +1.6] | 48% |
| ML | R2 | 25 | 17-8 | 68.0% [54.8, 84.6] | zero vig +11.6% [-16.6, +47.2] | 24% | -0.6 pt [-3.1, +2.0] | 48% |

About 160 games with 8 to 35 bets per cell cannot decide anything: every interval contains the break-even and 0 CLV, the day-bootstrap runs over only 21 days, and the CIs on small cells are
irregular (a few reach 100%). R0 counts agree with the lead's reconstruction (spread 17 bets 10-7, total 32 bets 13-19). The lead's ML count of 18 differs from my 35, probably because the ML arms here use fair probabilities (5-point edge over the de-vigged line), not
the 5% edge over a vig-inclusive price. The only pattern worth noting is the sign of CLV: R0 (which fades the open-to-11am move) has positive average CLV on spread and total (+0.29, +0.27 pts) and R1/R2 negative (-0.4 pts), consistent with the market drifting back toward its open between 11 am and the close; none of it is distinguishable from 0.
Sanity checks on the forward join: the logged line equals the first consensus snapshot at or after 11:00 ET in 72.5% (spread) and 53.8% (total) of 142 comparable games (mean absolute difference 0.21 and 0.38 points);
`odds_snapshots` opening lines equal `nba_odds_opening` in 82.5% of 150 games (spread). The "open" is therefore not exactly the SBR opening the model read.

**(e) Bets a year if R1 or R2 replaced the live rule.** Historical counts scaled to the full file (all games of each season; `bets_per_season.csv`), mean of the 5 full seasons 2021-22..2025-26 (2019-20 and 2020-21 were shortened):

| market | R0 (close-priced live analogue) | R1 | R2 | REF (open-priced backtest basis) | R0 live rule on forward data, scaled to 1,320 games |
|---|---|---|---|---|---|
| spread | 376 | 83 | 88 | 90 | 141 (17 bets in 159 games) |
| total | 555 | 246 | 277 | 278 | 266 (32 in 158) |
| ML (fair-prob edge 5 pts) | 623 | 421 | 453 | 465 | 290 (35 in 159) |

Forward R1 / R2 scaled: spread 66 / 75, total 174 / 183, ML 191 / 208 a year. So R1 or R2 would cut spread volume to about a quarter of the close-priced analogue and to roughly half of what the live rule is placing this season
(the live rule against the 11 am line fires far less often than the close analogue: 11% of games on spread versus 28%). Expect about 70-90 spread bets and 170-280 totals a year, ML 190-450 (the ML range depends on whether the later line is the 11 am or the close).
Per-season counts: spread R1 66-113, total R1 192-336, ML R1 362-477 (full table in `bets_per_season.csv`).

Sensitivities (descriptive, `results.json['sensitivity']`): excluding 2019-20, R0 spread -3.4% [-7.6, +0.6] at -110 (CI now includes 0, 2,105 bets) and total -4.7% [-8.0, -1.2]; R1 total -0.9% [-5.6, +3.9], R2 total -1.1% [-5.9, +3.4].
Clipping the ML probability at 0.02/0.98 instead of 1e-6 (12 games have ml_prob <= 0.001 and 43 >= 0.999) changes nothing (R2 hit 49.0%, zero-vig ROI +0.4%).

## 4. Method checks

- `unit_checks.txt`: 20 hand-made checks pass (push on whole lines, half lines, total over/under, -110 and zero-vig profit, ML fair-price profit, the push-aware cover probability against a hand formula with `math.erf`, and R0/R1/R2 on hand-made crossed and not-crossed games).
- The R0 selection and side equal `team.betting.get_spread_bet` / `get_total_bet` applied game by game at the close line with the fold sigma: 0 mismatches on 8,416 and 8,419 games.
- The REF row (open-priced) gives 649 spread and 1,946 total bets against the lead's 629 and 1,924; R0 gives 2,366 and 3,546 against 2,338 and 3,526; hit rates and ROIs are within a few tenths of the lead's (-4.4% vs about -4.5%, -4.6% vs about -4.6%).
  A rerun with the global sigmas 13.4 / 18.2 gives 652 / 2,024 REF and 2,353 / 3,636 R0 bets, so the lead's sample differs by 1-3% of bets from mine for a reason I did not find (sigma and push handling do not explain it).
- The regression of result minus close on own reproduces the lead's P2 coefficients (0.09 and 0.13).

## 5. Not verified / limits

- The close is the extreme later line. Live entry is about 11 am, when the market has moved only part of the way. R0 against the close fires 28% / 42% / 48% of games (spread / total / ML); against this season's 11 am line it fires 11% / 20% / 22%. So the close analogue overstates both volume and fading, and the historical R0 loss is an upper bound on how badly the live rule's fade behaves. The forward test (d) is the only evidence with a real 11 am line and cannot decide.
- -110 is an assumption for the price of spread and total bets (the file has no spread or total prices; lines are consensus medians, `core.odds_clean`). ML real-price scoring uses the cross-book median closing prices, which no single book offered; the 4-point gap between zero-vig and real-price ROI is the median overround.
- The "open" in `predictions.csv` (`odds_*`) differs from the cleaned consensus opening in about 40% of games (mean absolute difference 0.3 points for spread, 0.4 for total): it is the SBR opening the model was trained on, and the ML arms use the cleaned consensus opening probability (`open_home_prob`). `own` for ML therefore uses a slightly different open than the one the model saw.
- 2019-20 is the bubble season with shortened and contaminated odds history (261 R0 spread bets, ROI -12.5%); 2020-21 had short, empty-arena seasons. Excluding 2019-20 moves spread R0's CI to include 0 (sensitivity above); the H1 verdict for spread is therefore not robust to that choice (it is robust for total).
- The walk-forward predictions come from `E25_with_odds` (trained on a 2026-10-19 training set that has since been lost; the walk-forward test seasons were generated on that set and not regenerated here). They are not the deployed model's predictions.
- The model reads the game-day noon injury report, which the opening line usually predates (E25 README, section 5): an edge against the open that vanishes against the close is what injury news looks like. R1 and R2 therefore cannot be separated from "the model saw news first".
- ML H1/H2 are read on zero-vig ROI because the registration defines no -110 ROI for ML; the H3 hit-rate rule is applied literally to ML although ML bets are mostly underdogs.
- The day bootstrap resamples all game dates of the market sample for single-arm statistics and the union of the two arms' bet days for H2, as registered. It does not account for correlation between R1 and R2 across seasons or for the seven models being trained on overlapping data; 18 slope coefficients and 33 sweep rows are uncorrected for multiplicity.
- `E25_with_odds/predictions.csv` is read from the production checkout (read-only) because the worktree contains only `fold_info.json` of that folder. I did not verify the file's hash against anything.
- Not checked: bets on pushes in real settlement (pushes refund by convention), alternative lines, line shopping, and any price other than the consensus median.
