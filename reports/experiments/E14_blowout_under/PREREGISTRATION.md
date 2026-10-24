# E14 pre-registration: do big-spread (blowout-prone) games go under the opening total?

Written 2026-10-23 **before** any number from this experiment was computed. Nothing here is edited after the run; if a
criterion is badly specified that is reported as a limitation. Same protocol family as E08 (`experiments/e08_hypothesis.py`):
fix the hypothesis, find on some seasons, confirm on others, resample game days for error bars.

## Definitions

- Opening total `odds_total` and opening home spread `odds_home_spread` come from the latest training set
  (`data/processed/training_set_v1_reduced_2026-10-19.csv`, columns as used by every other experiment). `|spread|` = absolute value
  of `odds_home_spread`. (Negative = home favoured; the sign does not matter here.)
- Market residual `r = target_total_pts - odds_total` (points). Under = `r < 0`, over = `r > 0`, push = `r == 0` (excluded from rates,
  profit 0 in betting).
- Discovery seasons: 2019-20, 2020-21, 2021-22. Confirmation seasons: 2022-23, 2023-24, 2024-25, 2025-26.
- Error bars everywhere: 95% percentile interval of 1000 resamples of whole game days (seed `metrics.RNG_SEED`), as in `metrics.py`.

## Hypothesis (H1)

The slope of `r` on `|spread|` (OLS, intercept included, no controls) is negative.

**H1 is SUPPORTED iff all three hold:** (i) discovery slope < 0, (ii) confirmation slope < 0, (iii) the confirmation slope's
bootstrap CI upper bound is < 0. Otherwise NOT SUPPORTED. (This is the user's registered rule.)

**Secondary criterion (not in the verdict, reported as "survives the total confound"):** the same regression with `odds_total` and
season dummies added; the confirmation coefficient on `|spread|` is negative with CI upper bound < 0.

## Steps and how each is read

2. **Descriptive.** Under rate and mean `r` by `|spread|` bucket 0-3, 3-6, 6-9, 9-12, 12+ (lower edge inclusive), per period and pooled,
   day-bootstrap CIs. Break-even at -110 is 52.38% under rate.
3. **Does the model absorb it?** Slope of `actual - total_pred` on `|spread|`. Primary data: `reports/oos_predictions_*.csv`
   (deployed OOS, 2021-22..2025-26, joined on game_id; the `odds_*` columns there are used only after being checked to equal the
   training-set opening lines). Secondary data: `reports/experiments/baseline/predictions.csv` (7-fold walk-forward, 2019-20..2025-26).
   Reading: CI of the slope includes 0 -> "flat, absorbed"; CI entirely below 0 -> "model leaves it on the table".
4. **Mechanism.** Overtime = either team's `minutes` in `team_game_full` > 240. Rerun H1 and the controlled regression on regulation-only
   games; report OT rate by bucket and the OT-minus-regulation slope difference. Reading: a regulation-only confirmation slope that is
   still negative (CI < 0) means the effect is not an OT artifact.
5. **Open vs close.** `close_total` from `core.odds_clean.load_clean_odds` (bad closing windows are NaN there and dropped from this step only).
   Report mean `close_total - odds_total` by bucket and its slope on `|spread|`, and the slope of `actual - close_total` on `|spread|`.
6. **Betting.** Flat unit bet on the under at -110 whenever `|spread| >= X`, X in {6,7,8,9,10,11,12,13,14}. X is chosen on
   the **discovery seasons only** as the X with the highest ROI among those with at least 200 discovery bets. That X is then applied unchanged
   to the confirmation seasons. Report ROI per bet (units of 100) with day-bootstrap CI for: opening line at -110 (primary), and
   best-book settlement (highest total across cleaned books in `nba_odds_opening`, settled at that book's quoted under price, -110 where unquoted; the E12 method).
   Control: under on every game. Opening-line ROI includes line movement the model/bettor could not necessarily get (`metrics`' OPEN_BASIS_CAVEAT);
   ROI at the closing total is reported alongside.
7. **Features.** Run only if the gate passes. **Gate:** H1 SUPPORTED, and the controlled confirmation coefficient CI < 0, and the
   regulation-only confirmation slope CI < 0, and the deployed-OOS residual slope (step 3, 2022-23..2025-26) has CI < 0. If the
   gate fails, no feature is built and the report says which condition failed.

## Known limits stated in advance

- Only ~15-20% of games have `|spread| >= 10`; the minimum detectable under-rate shift is about 3 points (see the user's power note).
- 2019-20 (COVID stoppage) and 2020-21 (reduced crowds / compressed schedule) are in the discovery window; 2019-20/2020-21 have
  fewer games with odds (1140 / 1171).
- One primary test; everything else is descriptive and is not corrected for multiplicity.
- Seasons before 2019-20 exist in the training set but are **outside the registered design**; if used they are labelled supplementary.
