# Experiment pre-registration (written before any experiment was run)

Committed before the first experiment result existed. Keep criteria below are **not changed after seeing results**;
if a criterion turns out to be badly specified, that is reported as a limitation, not edited.

## Common protocol

- **Baseline** (`reports/experiments/baseline`): the post-audit deployed configuration. Opening odds as inputs, two-stage
  feature selection on the first 80% of model-train rows, LightGBM hyperparameters chosen **inside each fold** from a fixed
  8-config grid on a time-ordered validation block (last 20% of model-train), isotonic calibration and residual sigma from the
  last 20% of the training window. Seeds fixed (42), LightGBM deterministic.
- **Folds**: test seasons 2019-20 ... 2025-26 (7 folds), training = all earlier seasons. Identical for every experiment.
  Exception: E8 (see there).
- **One change at a time** against the baseline. Experiment code lives in separate modules and the baseline code path is not
  edited; the baseline is re-run at the end and must reproduce bit-for-bit.
- **Lookahead**: any new feature gets a test that corrupts all rows dated on/after a game's date and asserts the feature for that
  game is unchanged (`tests/test_experiment_lookahead.py`).
- **Primary metrics**: ML log loss, Brier; spread and total MAE, RMSE, bias (actual - pred); calibration (reliability by decile, ECE);
  closing line value (points for spread/total bets) where closing lines exist. **Secondary**: ROI, bet count, hit rate, max drawdown
  (units of a $100 stake) with day-bootstrap 95% CIs.
- **Bootstrap**: resample game days with replacement, 1000 draws, fixed seed. Paired comparisons use per-game losses on the
  common games of baseline and experiment.
- **Main criterion (MC) for a metric**: the metric improves in **at least 5 of 7** test seasons (per-season mean paired difference
  has the favorable sign) **and** the pooled paired day-bootstrap 95% CI of the difference excludes zero in the favorable direction.
- **No-harm rule (NH)**: no non-targeted primary metric (ML log loss, Brier, spread MAE, total MAE) is worse than baseline by more than
  0.5% (relative, pooled point estimate).
- A decision is KEEP only if stated criteria hold; otherwise REJECT. "Inconclusive" is reported as REJECT (not adopted).

## A. Model structure

**E1 market-residual targets.** Spread target = margin + opening spread, total target = total - opening total, ML via init_score =
logit(de-vigged opening home probability; where the ML price is missing, an in-fold logistic of home win on the opening spread).
Same selected features as the baseline. Keep **per market**: KEEP-SPREAD if MC holds for spread MAE; KEEP-TOTAL if MC holds for total MAE;
KEEP-ML if MC holds for ML log loss; each also requires NH. Tail analysis (|spread| > 12, opening total > 242) is descriptive:
bets under the 5% rule, bet count and ROI baseline vs E1.

**E2 calibration.** Variants: Platt on log-odds (fit on the last 20% split), cross-fitted isotonic (4 time blocks over the last 2
training seasons, out-of-fold raw predictions). Keep a variant if (i) ECE is lower in at least 5/7 seasons, (ii) the pooled paired
log-loss difference has CI upper bound < +0.0005 (not worse), and (iii) either distinct output probabilities >= 3x the baseline's or
MC holds for log loss. If both qualify, the lower pooled ECE wins.

**E3 coherence.** (a) descriptive: share of games where ML and spread disagree on the winner (ml_prob > .5 vs predicted margin > 0), and
ML vs total-implied-by-nothing is not defined (only ML/spread pair is tested). (b) blend = w * ml_prob + (1-w) * Phi(margin_pred / sigma_spread),
w from a 0..1 grid (step 0.1) by log loss on the calibration set inside each fold. (c) stack: logistic regression on
[logit(ml_prob_raw), margin_pred/sigma_spread] fitted on the calibration-set predictions (out-of-sample for the base models), then isotonic-free.
Keep (b) or (c) if MC holds for ML log loss, NH holds, and the contradiction rate does not increase.

**E4 heteroskedastic sigma.** Second LightGBM (small) on out-of-fold |residual| (4 forward-chaining blocks inside model-train), features:
FGA-EWMA pace proxies, 3PA share, injury counts/missing flags, |line|, total line. sigma_i = E|e|_i * sqrt(pi/2).
Metric: mean log-likelihood of cover outcomes (spread: home covers; total: over; pushes dropped), p = Phi((pred +/- line)/sigma_i).
Keep per market if the log-likelihood improves in >= 5/7 seasons with a pooled CI excluding zero **and** the ECE of cover probabilities
(10 bins) is not worse than the global-sigma ECE by more than 0.005.

## B. Features (extra candidate columns added to the baseline pool; selection decides)

**E5 pace-normalised, opponent-adjusted team stats.** Per-100-possession versions of 10 box-score stats, minus the opponent's entering
EWMA allowance (span 20, strictly prior games), EWMA spans 5/20, home-away diffs. Keep if MC holds for at least 2 of {ML log loss,
spread MAE, total MAE} and NH holds for the third.

**E6 offseason regression to the mean.** kappa in {1.0, 0.75, 0.5} selected per fold by the validation block (mean of the three
targets' validation ranks), applied to team EWMAs. Keep if MC holds for at least 2 of the 3 targets on all games, NH holds, and the
October-November subset point estimate is not worse on any of the three metrics. October-November results are reported separately.

**E7 bench-depth replacement level.** Replace constant REPLACEMENT_BPM = -2.0 in the player value with the team's own healthy bench BPM
(minutes-weighted mean BPM of its rotation players ranked 6-10 by expected minutes who are not on the report as Out/Doubtful), strictly
prior information. Keep if on games with >= 1 Out player (either team) MC holds for spread MAE or total MAE, and NH holds on all games.

**E8 fresh vs chronic absences.** Step 1 (hypothesis test on discovery seasons 2019-20..2021-22 only): regress the market residual
(margin + opening spread) on home-minus-away fresh and chronic lost value (fresh = played in at least one of the team's last 3 games; chronic =
missed >= 5 consecutive team games) with day-cluster bootstrap CIs. Supported iff the fresh coefficient is negative with CI excluding zero
and |chronic| < 0.5 |fresh| (or its CI includes zero). Only if supported, step 2: add the features (plus decayed expected minutes for
chronic absentees: e_minutes * 0.5^((gap-1)/5), a fixed constant, not tuned) and evaluate on folds 2022-23..2025-26 (4 folds, to keep the
discovery window out of the evaluation). Keep if improves in >= 3/4 folds with a pooled CI excluding zero for >= 1 of the three
primary losses and NH.

**E9 official PDF injuries in live inference.** Not a backtest. Done when: the ESPN scrape no longer exists on any live path, a
noon-to-latest downgrade flag exists with a lookahead test, both sources are logged side by side for new games, and the disagreement rate on
the historical overlap is reported. No performance claim.

## C. Betting strategy (on baseline predictions; no retraining)

**E10 selection rule.** Baseline rule = 5% probability edge. Alternatives: (a) flat |pred - line| > 1.0 pt for spread/total (ML keeps the 5%
edge rule), (c) per-fold edge threshold from {0, .01, .02, .03, .04, .05, .075, .10} maximising ROI on the calibration-set predictions
(fallback .05 if < 100 calibration bets). Keep an alternative if its pooled ROI exceeds the baseline rule's by >= 1.0 point with a day-bootstrap
CI of the ROI difference excluding zero, in each market separately; hit rate of selected bets must be within 3 pp of the model's average P(win).

**E11 same-game exposure.** Quarter-Kelly stakes from the model's win probability and the actual price. Strategies: independent (baseline),
one best-EV market per game, joint Kelly via a Gaussian copula (rho from in-fold outcome indicators of ML vs spread bets; total independent).
Bootstrap (1000 resamples of game days) bankroll simulation. Keep a strategy over independent Kelly if median log-growth is higher with a
paired CI excluding zero and its 95th-percentile max drawdown is not higher.

**E12 line shopping and consensus.** `odds_snapshots` has no rows in this database, so multi-book data = the per-book columns of
`nba_odds_opening` / `nba_odds` (six books, cleaned by `odds_clean.py`); coverage reported. (a) consensus de-vigged median probability as the
market reference; (b) bets settled at the best available price/line across books. Decision: adopt best-price settlement as a *reporting*
option only if it improves ROI with a CI excluding zero **and** the control "bet both sides of every game at best price" has ROI <= 0.
Price improvement is measured per bet by key number (3, 7, other), not assumed.

## Winners combined

After all experiments, only KEEP decisions are re-run together (one combined run) and compared with the sum of individual effects.


---

# Totals angles (E16 onward): market residual `r = actual total - opening total`

Added 2026-10-23, after E14/E15 and **before any E16 number was computed**. Shared framework supplied by the user for a family of 8 angles
(the E14 spread angle is #1 of the 8, so E16 = angle 2 of 8; later angles are appended below this section before they are run).

## Shared rules for the family

- Target `r`; under = `r < 0`; pushes dropped from rates (profit 0 in betting). Data and filters exactly as E14 (`experiments/e14_blowout_under.py`).
- Discovery 2019-20..2021-22; confirmation 2022-23..2025-26. Intervals: day-cluster bootstrap, 1000 draws, seed 20261017.
- Three layers in order: (a) market test, (b) deployed-model test on `reports/oos_predictions_*.csv` (2021-22..2025-26), (c) feature test with
  `experiments/feature_exp.py` against the baseline and `noise_floor.md`, run only if (a) and (b) both support the angle. Only (b) and (c) can change the model.
- Betting test last, thresholds picked on discovery only, return at -110 and at best-book line and price (E12 method), break-even 52.38%.
- Sensitivity: rerun without 2019-20 and 2020-21. Because that leaves one discovery season (2021-22), the sensitivity is reported for discovery = 2021-22 alone and for the
  pooled sample 2021-22..2025-26; it is not used to decide anything.
- Multiple testing: Holm correction across the family of 8 using the **one-sided bootstrap p-value of each angle's primary confirmation test** (p = share of draws on the
  wrong side of 0, floored at 1/1001). The family is not complete yet, so E16 reports its raw p and marks Holm as pending; a "supported" label needs raw p < 0.05
  now and Holm-adjusted p < 0.05 once all 8 are in.
- Look-ahead: only pre-tip information. For E16 the inputs are the opening spread, opening total and the overtime outcome of the game itself, the last of which is used
  only as a *target of a fitted probability*, never as a feature for the same game.

## E16 - overtime and close finishes (angle 2 of 8)

**Hypothesis H_OT (sign fixed):** games likely to be close go over and lopsided games go under, because close games reach overtime and end with fouling.
In terms of the fitted close-game probabilities below, the coefficient on them in the regression of `r` is **positive**.

**Disclosure that weakens this registration.** E14 already printed, for all registered seasons including confirmation: the overtime rate by spread bucket, the mean `r` of
overtime versus regulation games by bucket, and the slope of `r` on |spread| for all games and regulation-only games. The confirmation seasons are therefore not untouched for
this angle, and the expected answer (overtime raises `r` in close games by about 20 points per game; the |spread| slope gets more positive without overtime) is known. E16 is a
formal re-expression with the user's probability-based design, not an independent confirmation. Its verdict is labelled exploratory-on-seen-data unless a forward sample (E15's)
is later scored for it.

**Known identification problem, registered up front.** The fitted probabilities below are functions of |spread| (and, for the variants, the opening total) only. A regression of `r` on
such a probability with |spread| and the opening total as controls is identified only through the curvature of the link function, not through independent variation. Its coefficient will
be imprecise and is not by itself evidence about overtime. The decomposition in step 5 (my addition, requested by no one) answers the mechanism question without that problem.

Steps (all cut points fixed now):

1. **Overtime model.** Logistic regression of `ot` (either team's minutes > 240) on |spread|, fitted on the discovery seasons only, then frozen. Calibration: ten bins of predicted
   probability by quantile (bins with duplicate edges are merged), predicted versus observed overtime rate with day-bootstrap intervals, on discovery (in-sample) and confirmation (out-of-sample).
   Calibration passes if no confirmation bin differs from its prediction by more than 2 percentage points beyond its interval; this is reported, not a gate.
2. **Close-finish probability.** `sigma` = standard deviation of `margin + opening home spread` over the discovery games (`margin` = `target_point_diff`). For each game
   `P_close = Phi((3.5 + spread)/sigma) - Phi((-3.5 + spread)/sigma)` (probability that |margin| <= 3 with the continuity correction, expected home margin = -spread). `sigma` is fixed from discovery.
3. **Primary test (market, layer a).** OLS `r ~ P_OT + |spread| + odds_total` (intercept, no season dummies), once on discovery and once on confirmation, with `P_OT` from step 1.
   Coefficient on `P_OT`: H_OT needs it positive in both periods. Primary confirmation statistic = that coefficient with its one-sided bootstrap p. Then the same regression on
   regulation-only games, and the same two regressions with `P_close` in place of `P_OT` (secondary). Mechanism reading: **overtime is the mechanism** iff the all-games `P_OT` coefficient is positive with
   confirmation one-sided p < 0.05 and the regulation-only coefficient's two-sided 95% interval includes 0 (or is smaller than half the all-games coefficient with the difference interval above 0).
   Otherwise the mechanism is not overtime.
4. **Size sanity check.** Report overtime share by spread bucket (0-3, 3-6, 6-9, 9-12, 12+), mean `r` of overtime and of regulation games, and the implied gap
   (overtime share difference between the 0-3 and 12+ buckets times the overtime-minus-regulation mean `r`), compared with the |spread| slope times the bucket distance. The user's expected gap
   of 0.5-1 point and "about 11 points per overtime game" are checked against these.
5. **Decomposition (added).** Write `E[r | s] = (1-p(s)) mu_reg(s) + p(s) mu_ot(s)`. Report the overtime contribution to the |spread| slope as
   `(mu_ot - mu_reg) * dp/ds` using the discovery logistic slope at the mean overtime rate and the pooled overtime-minus-regulation mean `r`, and the remainder (the regulation slope), each with
   a day-bootstrap interval (logistic refitted on each draw by weighting games by how many times their day was drawn). Compare to the directly estimated all-games and regulation-only |spread| slopes.
6. **Model test (layer b).** Slope of the deployed model residual `actual - total_pred` on `P_OT` controlling for |spread| and the opening total, 2021-22..2025-26, and the model's predicted minus
   opening total on the same regressors, to show what the model already leans on. Support requires the residual coefficient positive with one-sided p < 0.05; otherwise the model already absorbs it.
7. **Feature test (layer c).** Not run unless step 3 (confirmation) and step 6 are both supported. The candidate would be `P_OT` and `P_close` computed from pre-game lines.
8. **Betting test.** Over bet at -110 when `P_OT >= c`, where c is the discovery quantile q of `P_OT` for q in {0.5, 0.6, 0.7, 0.8, 0.9}, chosen as the q with the highest discovery
   return among those with at least 200 discovery bets; the numerical cutoff c is then applied unchanged to confirmation. Report return with a day-bootstrap interval at the opening total, at the closing
   total and at the best-book line and quoted price. The mirror (under when `P_OT <= c'`, symmetric cut points on the low side) is reported as exploratory only.

**Power (stated now).** `r` has standard deviation about 18. Two groups of 2,000 games differ with a standard error of about 0.57 points, so an effect under about 1.1 points is not
detectable at 95% two-sided. Non-significant is the expected outcome unless the overtime effect is larger than the overtime arithmetic gives.

**Reading rule for the report.** Each layer is stated as supported / not supported against the rules above; no criterion is changed after the run; anything outside these steps is labelled
exploratory.


## E17 - spread x total interaction (angle 3 of 8)

Added 2026-10-23 before any E17 number was computed. Shared rules above apply (discovery 2019-20..2021-22, confirmation 2022-23..2025-26, day-cluster bootstrap 1000 draws seed 20261017, Holm across the 8
angles pending, layers a/b/c, betting last, sensitivity without 2019-20 and 2020-21, E14 frame and filters).

**Hypothesis H_INT (two-sided):** the effect of |spread| on `r` depends on the opening total. The motivating example (a leading team's bench keeps scoring in a fast game and not in a slow one) implies the interaction
coefficient is **positive** (|spread| pushes `r` up more when the total is high); the test is two-sided because the claim is "behaves differently", and the sign of the example is reported against the result.

**What has already been seen.** E14 printed the |spread| main effect and the coefficient on the opening total (about -0.05 per point), never a cell table or an interaction term. The interaction is therefore untouched in both periods. The |spread| main
effect is positive and known (+0.18 in the registered seasons), so a significant interaction means only that the slope differs by total, not that it appears from nothing.

Steps (cut points fixed now):

1. **Grid.** Terciles of `odds_total` with cutoffs set once from the discovery games (33.3rd and 66.7th percentiles) and applied unchanged to every period; |spread| buckets [0,5), [5,10), [10,inf). For each of the 9 cells and each
   period: games, mean `r`, under rate (pushes excluded), with day-bootstrap intervals. The opening total drifts upward over the seasons (about 222 in discovery to about 229 in 2025-26), so the share of confirmation games in each tercile
   will not be a third; those shares are reported, and a within-season tercile version of the grid and of the interaction regression is reported as a sensitivity (added by me; the user-specified, discovery-cutoff version is the primary).
2. **Primary regression.** OLS `r ~ |spread| + T + |spread| x T + season dummies`, where `T = odds_total - (discovery mean of odds_total)`, fitted separately on discovery and on confirmation (dummies for the seasons present, first dropped).
   The statistic is the coefficient on the product. Two-sided bootstrap p = 2 x min(share of draws <= 0, share >= 0), floored at 2/1001. **H_INT is supported** iff the two periods' product coefficients have the same sign and the confirmation p < 0.05; Holm
   across 8 angles is applied once the family is complete. Also reported: the spread slope at the 16th, 50th and 84th percentile of `T`, and the contrast "mean `r` in the (10+, top tercile) cell minus the (10+, bottom tercile) cell" in confirmation.
3. **Model test (layer b).** The same regression with `actual - total_pred` as the target on the 2021-22..2025-26 out-of-sample predictions, and with `total_pred - odds_total` (the model's lean), season dummies included.
   Layer b supports the angle iff the residual's product coefficient has the same sign as step 2's confirmation coefficient and p < 0.05 two-sided; otherwise the model already absorbs it (or the effect is absent).
4. **Feature test (layer c).** Only if step 2 is supported and layer b is supported: add `abs_spread x total` and the tercile cell indicator as candidates in `experiments/feature_exp.py` against the baseline and `noise_floor.md`. Not run otherwise.
5. **Betting test.** For each of the 9 cells and each side (over, under) at -110 at the opening total, compute the discovery return. Choose the one (cell, side) with the highest discovery return among cells with at least 150 discovery bets; apply that same cell definition to
   confirmation. Report return with a day-bootstrap interval at the opening total, the closing total and the best-book line and quoted price (E12 method), plus the all-games over and under controls. Break-even is 52.38%.
6. **Sensitivity (no 2019-20 / 2020-21).** The step 2 product coefficient with the sample = 2021-22 only (in place of discovery) and pooled 2021-22..2025-26. Reported, not decisive.

**Power (stated now).** Cells hold about 400 (discovery) to 700 (confirmation) games, so a single cell mean has a standard error of about 0.7-0.9 points (sd of `r` about 18), and a difference between two cells about 1.0-1.3. The interaction is estimated from the whole sample but is expected to be weak; the report states its bootstrap standard error and the
implied smallest detectable change in the spread slope across the interquartile range of the total (2.8 x SE x IQR of T). A non-significant result means "not shown".

**Reading rule.** Supported / not supported against the rules above; no criterion changes after the run; anything else is exploratory.
