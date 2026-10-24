# E22 pre-registration: do games between fast teams go OVER the opening total? (forward test)

Written 2026-10-23, after E21 and **before any E22 number exists**. Nothing is computed by this document except the frozen cutoff and the power table below, which use only pace values and in-sample standard errors, never outcomes. Criteria are not edited after the first forward game is scored; a badly specified criterion is reported as a limitation, not changed.

## Why the old data cannot confirm this

E21 was registered to test a pace *gap* effect, which was absent. In the same regressions the **average pace level** of the two teams came out positive for `r = actual total - opening total` (confirmation +0.83 per pace point, interval +0.37 to +1.28; discovery +0.17, interval -0.25 to +0.60; all registered seasons with season dummies +0.51, interval +0.16 to +0.85)
and for the deployed model's residual (confirmation +0.56, interval +0.08 to +1.06). Nobody registered that term, and it was found after one hypothesis had already failed, so all 2019-20..2025-26 games are descriptive only here and carry no weight in any criterion. The only untouched data is the future: this is a forward test on the same footing as E15 and E20, on the same forward games.

## Definitions (identical to E21 so the two can be compared)

- **Regular-season games only** (game id prefix `002`); playoff, play-in and NBA Cup knockout and final games are excluded. `odds_total`, `target_total_pts` and the plausibility filter as in E14-E21. `r = target_total_pts - odds_total`; push = `r == 0`.
- Possessions `poss = FGA + 0.44 FTA - OREB + TOV` from `team_game_full`; **pace** = `poss x 240 / minutes` (per 48 minutes). A team's **rolling pace** = the mean pace of its previous 20 regular-season games, strictly before the game's date, within the same season, requiring at least 10 prior games (built by `build_frame` in `experiments/e21_pace.py`, unchanged).
- **mean_pace** = (home rolling pace + away rolling pace) / 2. **pace_rel** = `mean_pace` minus the league mean pace of all earlier-dated regular-season games in the same season (requires at least 100 such games); used only for the betting readout and the quartile table.
- **Forward sample F**: 2026-27 and later regular-season games with `gameDate >= 2026-10-24`, a valid opening total, and valid `mean_pace` (i.e. at least 10 prior games for both teams). Games enter as the training set is regenerated; nothing is selected by outcome.
- Intervals: day-cluster bootstrap, 1000 draws, seed 20261017 (`experiments/metrics.py` conventions, as in E14-E21).

## Hypothesis H_PACELEVEL (sign fixed now)

In F, the coefficient on `mean_pace` in the OLS regression
`r ~ 1 + mean_pace + (odds_total - mean odds_total in F) + season dummies` is **positive**.
Season dummies (one per forward season, first dropped; none while only one season is available) are in the primary specification because the league pace level drifts between seasons; the coefficient is therefore identified from variation within a season.

**H_PACELEVEL is SUPPORTED iff, at the single decisive evaluation, the 5th percentile of the bootstrap distribution of the `mean_pace` coefficient is above 0** (one-sided 5%, since the sign is fixed in advance). The two-sided 95% interval is reported beside it but does not decide.
If the bound is not above 0, H_PACELEVEL is NOT SUPPORTED, including when the point estimate is positive.

## Sample size and looks (fixed now, so there is no stopping on a lucky run)

- **Decisive evaluation: once, when F reaches 3,200 games** (about three regular seasons at about 1,070 games per season with valid 20-game pace features, i.e. after the 2028-29 regular season). It uses all F games then available.
- Non-decisional looks at 1,070 and 2,140 games: same statistics, no criterion changed, nothing stopped, nothing bet because of them.
- Power (one-sided 5%; the standard error is the in-sample 0.177 for the season-dummies specification at n = 7,183 games scaled by the square root of the sample ratio; day clustering will make the true standard error slightly larger):

| true coefficient (points of total per pace point) | n = 1,070 | n = 2,140 | n = 3,200 | n = 4,300 | n = 5,400 |
|---:|---:|---:|---:|---:|---:|
| 0.8 (the confirmation estimate) | 54% | 79% | 91% | 97% | 99% |
| 0.5 (about the pooled in-sample estimate) | 29% | 46% | 59% | 70% | 79% |
| 0.25 (half of it, allowing for selection after the fact) | 14% | 19% | 24% | 29% | 34% |

  Even a real coefficient of 0.5 fails to show about four times in ten at 3,200 games, and a half-size one almost always fails. NOT SUPPORTED will mean "not shown at this sample size", not "absent".

## Secondary readings (descriptive; none changes the verdict)

1. The same regression without season dummies, on regulation-only games, and with 10- and 30-game windows; the coefficient on the pace gap (E21 found none); each forward season separately.
2. Mean `r`, over rate and under rate (pushes dropped) by quartile of `pace_rel` in F, with quartile edges taken from F itself.
3. The deployed model's pre-game logged total prediction for the same games (`predictions_log_v2` rows written before tip-off, never refit): the coefficient on `mean_pace` in `actual - total_pred ~ 1 + mean_pace + odds_total` (one-sided positive reported), and the model's lean.
4. Open-to-close movement (`close_total - odds_total`) by quartile of `pace_rel`, and the same regression on `actual - close_total`.

## Betting readout (reported, not a decision rule for stakes)

Flat over bet at -110 when `pace_rel >= 1.32`, where 1.32 is the **frozen** 80th percentile of `pace_rel` over the 6,448 regular-season games of 2019-20..2025-26 that have a valid value (computed on pace only, before any outcome was linked to it; the 50th, 60th, 70th and 90th percentiles were -0.07, 0.35, 0.80 and 1.98). At the opening total, the closing total and the best-book line and quoted price (E12 method, `experiments/totals_bets.py`), return per bet with a day-bootstrap interval.
At 3,200 games about 640 bets qualify, so the return interval will be about +/-9 points: it can show a gross failure and cannot confirm a small edge. No stake conclusion follows from E22 unless H_PACELEVEL is SUPPORTED **and** the best-book return point estimate is positive; even then the interval is the evidence. The under side is not reported as a strategy.

## If H_PACELEVEL is supported: model and feature step

Only then: (a) the model residual in secondary reading 3 must also have a positive `mean_pace` coefficient with one-sided p < 0.05; (b) if so, run the standard feature experiment (`experiments/feature_exp.py`) adding the pre-game feature `pace_mean` (the 20-game average above, one-game shift; not `pace_gap`, which E21 found to be null) against the baseline and `noise_floor.md` under the common protocol in `PREREGISTRATION.md`, with the forward seasons added as extra test folds, keeping it iff total MAE
improves in at least 5 of the folds with a pooled paired interval below 0, the forward folds are not worse, and the no-harm rule holds for ML log loss and spread MAE. E5's rejection of pace-normalised team statistics is a separate question and does not bear on a pace-level feature. No model change on E22 alone.

## Implementation

The scoring script `experiments/e22_forward_pace.py` will reuse `build_frame` from `experiments/e21_pace.py` and the E19/E20 helpers, and will be written and committed **before** the first look (1,070 games). It takes the forward regular-season games as they appear and applies the definitions above with no free parameters.

## Choices made here that you can change before the first scoring

1. Decisive size 3,200 games (three seasons). 4,300 (four seasons) raises power at a coefficient of 0.5 from 59% to 70%; 5,400 to 79%.
2. One-sided 5% instead of the two-sided 95% used in E14. Two-sided would cut power at 3,200 games from 59% to 47% for a coefficient of 0.5.
3. Season dummies in the primary specification (the alternative without them is a secondary reading); the 20-game window with at least 10 prior games.
4. The frozen betting cutoff 1.32 and the forward start 2026-10-24.

## Out of scope

No claim about the spread, moneyline or player-prop markets; no use of any 2013-14..2025-26 game as evidence; no tuning of the window, cutoff or specification on F.
