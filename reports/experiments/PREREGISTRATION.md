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


## E18 - calibration of the totals model by total level (angle 4 of 8)

Added 2026-10-23 before any E18 residual was computed (only file structure and row counts were inspected). Shared rules above apply where they fit; this angle is a **model** test (the user: "a model problem, not a market edge"),
so it follows the E1-E13 walk-forward protocol (`PREREGISTRATION.md` common protocol) instead of the discovery/confirmation split.

**Hypothesis H_CAL:** the deployed-configuration totals model's error depends on the level of the total (known from FINAL_REPORT E1: a large bias for opening totals above 242), and a correction fitted on the calibration block
reduces total MAE out of sample.

**Disclosure that weakens this registration.** E1's tail analysis already reported the bias above 242 on these same seven folds, and E1-E13 have used all seven test seasons, so there is no unseen holdout. What is new is evaluating three fixes forward;
which fixes to try was chosen knowing the bias exists. A fix that wins here is a candidate, and its forward evidence would come from the 2026-27 season (E15's forward sample), not from this run.

**Data.** `reports/experiments/baseline/predictions.csv` (7 folds, 2019-20..2025-26, 8,916 games) and `calibration_set_predictions.csv` (the last 20% of each fold's training window, rows labelled by fold with `experiments.lib.label_calib_folds`).
No model is retrained. `bias = actual - predicted total` (positive = the model is too low). All intervals: day-cluster bootstrap, 1000 draws, seed 20261017, paired where two predictions are compared (`experiments/metrics.py`).

**Step 1 (descriptive; nothing passes or fails).**
- Mean bias by decile of predicted total and of opening total, per season and pooled. Deciles use edges from the pooled test set ("pooled", shows era shifts) and a second version with edges computed inside each season ("within-season").
- Opening total > 242 and predicted total > 242: games per season, mean bias per season and pooled with intervals.
- Level regression `bias ~ (opening total - pooled mean)` once without and once with season dummies; the slope's interval decides the reading **"the level effect survives a season control"** iff it excludes 0 with the dummies.

**Step 2 (candidate fixes, each fitted only on that fold's calibration block, applied to that fold's test games).**
- (a) Isotonic: `IsotonicRegression(out_of_bounds='clip')` of actual total on predicted total.
- (b) Linear in the level: OLS of `actual - pred` on `(opening total - calibration-block mean total)`, intercept included; corrected = pred + fitted value.
- (c) Level-dependent shrinkage toward the line: OLS without intercept of `actual - line` on `(pred - line)` and `(pred - line) x (opening total - calibration mean)`; weight `w = b0 + b1 (L - mean)` clipped to [0, 1]; corrected = line + w (pred - line).
- Robustness R1 ("recent only"): refit (a), (b), (c) on the later 50% of the fold's calibration block by date. R2 ("era-adjusted level"): (b) and (c) with `L` = opening total minus the mean opening total of all games in the previous 365 days (past lines only).

**Step 3 (metrics and the keep rule).**
- Total MAE (primary), per season and pooled. Cover log loss as in E4: `p(over) = Phi((pred - line)/sigma)`, outcome over versus under with pushes dropped, `sigma` = standard deviation of the calibration-block residuals after the same fix (baseline: its saved `sigma_total`).
  5%-edge total bets from `metrics.make_bets(thr=0.05)`: count, hit rate and return at -110 as secondary.
- **KEEP a fix iff** (i) its total MAE is lower than the baseline's in at least 5 of 7 seasons, **and** (ii) the pooled paired MAE difference has a day-bootstrap interval entirely below 0 (the E1-E13 main criterion), **and** (iii) in both R1 and R2 the pooled paired MAE difference
  is also negative (point estimate); otherwise "not kept", and a fix that meets (i) and (ii) but fails (iii) is labelled "era-confounded".
  The user's stricter wording ("error bars excluding zero in at least 5 of 7 seasons") is reported alongside as the count of seasons whose own paired interval excludes 0; it is not the keep rule.
- Descriptive only: MAE, bias and bet count restricted to opening total > 242; discovery (2019-20..2021-22) and confirmation (2022-23..2025-26) subsets.
- Holm input: this angle's p-value = 3 x the smallest one-sided pooled bootstrap p among (a), (b), (c) (capped at 1).

**Power note.** Games above 242 are few before 2023-24; per-season intervals there will be wide, and a fix that changes only the extreme tail moves total MAE little. A pooled MAE gain of about 0.05 points is at the edge of what 8,916 games resolve (E1 used 0.2 points as a visible effect).

**Reading rule.** Keep / not kept / era-confounded per fix; no criterion changes after the run; anything else is exploratory.


## E19 - late-season tanking and locked seeds (angle 5 of 8)

Added 2026-10-23 before any E19 number was computed (only table layouts, row counts and game-id prefixes were inspected). Shared rules above apply (discovery 2019-20..2021-22, confirmation 2022-23..2025-26, day-cluster bootstrap 1000 draws seed 20261017,
Holm across 8 pending, layers a/b/c, betting last, sensitivity without 2019-20 and 2020-21). **Exploratory** by the user's own instruction: flagged games are few and the direction is unknown.

**Scope correction for E14-E18 (found while preparing this angle).** The training-set frame used by E14-E18 contains playoff (game id prefix 004) and play-in (005) games: 627 of the 8,897 registered-season games (590 + 37, 7%).
E19 uses **regular-season games only** (prefix 002) because game numbers and standings exist only there. A check of E14's headline on regular-season games only, run while preparing this registration, gives slope +0.161 (interval +0.061 to +0.268) against +0.180 with
everything; discovery +0.165 (-0.009 to +0.339), confirmation +0.155 (+0.031 to +0.288). The sign and the E14 conclusion do not change, but the discovery interval now includes 0. The E14-E18 figures are left as published and reported with this caveat; the forward test E15 is already defined on regular-season games.

**Rules context (researched 2026-10-23; sources: NBA Communications release https://pr.nba.com/nba-board-of-governors-approves-new-draft-lottery-system-to-address-tanking/ and ESPN https://www.espn.com/nba/story/_/id/48903380/sources-nba-bog-votes-expand-lottery-disincentivizes-tanking; the participation-policy item comes from search snippets of 2023 coverage, not re-read).**
- On 2026-05-28 the Board of Governors approved (29-1) a "3-2-1" lottery effective with the **2027 Draft** (so it governs the 2026-27 season's records), in force for the 2027-2029 drafts. The lottery has 16 teams. Non-playoff teams get 3 balls, but the three worst records lose one (2 balls) and cannot pick lower than 12th ("draft relegation");
  the 9 and 10 play-in seeds get 2 balls and the losers of the 7-8 games 1 ball. A team's own pick cannot be No. 1 in two consecutive drafts or a top-five pick in three consecutive drafts. Newly traded picks cannot carry top-12 to top-15 protections. The league gets expanded power to cut lottery odds, change draft position and fine (ESPN reports up to $10 million).
- Resting stars: the 2023 player participation policy (no more than one star unavailable per game, fines for violations, no shutting down a star) is, as far as one search shows, still the framework; no 2026-27 change was found. Not exhaustive.
- Implication for this angle: the old incentive to be as bad as possible is removed (the worst three are penalised and records 4th through 10th worst are equal), so a bottom-5 win% flag will mix teams with little incentive to lose and teams that may now try to avoid the bottom three. Contenders locked into a seed keep their old incentive to rest (limited by the participation policy and the play-in).
  The 2019-20..2025-26 sample covers the pre-2026 lottery and, from 2023-24, the participation policy; **no historical season has the 3-2-1 rules**. A result here is evidence about the old regime; its transfer to 2026-27 is untested and needs forward data from 2026-27 regular-season games.

**Hypothesis H_TANK (two-sided):** r depends on the late-season, tanking and locked flags and on their interactions with |spread|.

Definitions (fixed now; regular-season games only; standings from games dated strictly before the game's date, built from `team_game_full` results; conferences hard-coded from team ids):
- `game number` of a team = its regular-season games played before the date + 1 (checked against `season_game_number_*`). **Late (L)** = either team's game number >= 70. (Note: 2020-21 had 72 games, so its late window is 3 games per team; 2019-20 was shortened.)
- **Tanker (T)** = either team is among the 5 lowest win% of all 30 teams at that date (rank method 'min', so ties at the boundary are included).
- **Locked (K)** = either team has a conference rank (win%, then wins) from 1 to 10 and is more than 6 games clear of the next-lower rank and, unless it is rank 1, more than 6 games behind the next-higher rank. Games clear = ((W_a - L_a) - (W_b - L_b)) / 2.
  This reads "more than 6 games clear of the next seed" as clear in both directions, i.e. the seed cannot realistically change.

Steps:
1. **Counts and descriptives.** Games in each flag and in L&T, L&K, L-only, not-L, per period. Mean `r`, under rate (pushes dropped) and n for those groups with day-bootstrap intervals.
2. **Primary regression (user's design).** OLS `r ~ |spread| + (odds_total - discovery mean) + L + T + K + L x |spread| + T x |spread| + K x |spread| + season dummies`, separately on discovery and confirmation.
   **Primary statistic: the joint Wald test that the six flag coefficients are zero**, `W = b' S^-1 b` with `S` the bootstrap covariance of those six coefficients, p from chi-square(6). **H_TANK is supported** iff discovery p < 0.10 and confirmation p < 0.05 (a loose find step, a strict confirm step);
   Holm across the 8 angles uses the confirmation p. Individual coefficients with two-sided bootstrap p-values are reported as descriptive.
   Secondary model B (my addition): the same with the late-only groups `LT = L & T`, `LK = L & K`, `LO = L & not T & not K` in place of L, T, K, again with their |spread| interactions.
3. **Line movement.** `close_total - odds_total` (open to close, `odds_clean`, bad closing windows excluded): mean and mean absolute movement for LT, LK, LO and not-L, with intervals and the difference of each flagged group against not-L (paired draws); also mean `actual - close_total`. If the open-to-close movement for flagged games is larger and `actual - open` is not, the market is correcting it before tip.
4. **Model test (layer b).** The step 2 regression on the out-of-sample model residual `actual - total_pred` and on the model's lean, 2021-22..2025-26, plus the mean residual for LT and LK games with intervals.
5. **Feature test (layer c).** Only if step 2 is supported and step 4 shows a residual dependence (joint Wald p < 0.05 on the residual in the confirmation seasons). Not run otherwise.
6. **Betting test.** Groups LT, LK, L (late, any), sides over and under at -110 at the opening total; choose the (group, side) with the highest discovery return among those with at least 100 discovery bets; apply unchanged to confirmation; report at the opening total, the closing total and the best-book line and price, with the all-late over and under controls.
7. **Sensitivity.** Step 2's joint Wald p on (i) 2021-22..2025-26 (without the 2019-20 and 2020-21 seasons) and (ii) 2023-24..2025-26 (the participation-policy era), both reported, not decisive.

**Power (stated now).** Counts will be reported before any result; the user expects about 1,500 late flagged games across 7 seasons and far fewer tanker-only games. With sd(`r`) about 18, a group of 300 games has a standard error of about 1.0 point, so only a group effect of several points is detectable, as the user noted. A joint test with six coefficients also has less power per coefficient than a single test.

**Reading rule.** Supported / not supported against the rules above; no criterion changes after the run; anything else is exploratory.


## E21 - pace mismatch (angle 6 of 8)

Added 2026-10-23 before any E21 number was computed (only column names, null shares and per-season mean paces of the box-score table were inspected). Shared rules above apply (discovery 2019-20..2021-22, confirmation 2022-23..2025-26, day-cluster bootstrap 1000 draws seed 20261017,
Holm across 8 pending, layers a/b/c, betting last, sensitivity without 2019-20 and 2020-21). Regular-season games only (game id prefix 002), as in E19, because rolling windows need regular-season sequences.

**Hypothesis H_PACE (sign fixed):** when the two teams' paces differ, actual game pace falls below the average of the two paces, and totals built from that average go under: the coefficient on the pace gap is **negative** for actual possessions (mechanism) and negative for `r = actual total - opening total` (market).

**What has been seen.** E5 tested pace-normalised team statistics, never a pace gap. For this angle no gap, mechanism or `r`-by-gap number has been seen in either period; confirmation is untouched. The training set holds no pace column; `team_game_full` has a box-score `pace` column, which is only used for a sanity check of the possession formula.

Definitions (fixed now):
- Possessions per team-game `poss = FGA + 0.44 FTA - OREB + TOV` from `team_game_full`; **pace** = `poss x 240 / minutes` (per 48 minutes, so overtime does not inflate it; `minutes` is team minutes, 240 in regulation).
- **Team rolling pace** = the mean pace of the team's previous 20 regular-season games, strictly before the game's date, within the same season, requiring at least 10 prior games (so each team's first 10 games of a season, and any game where either team has fewer, are excluded; counts reported).
  Window robustness: 10 and 30 games, mechanism step only.
- `mean_pace = (home rolling + away rolling) / 2`, `gap = |home rolling - away rolling|`. **Actual game pace** = the mean of the two teams' pace in that game.
- `odds_total`, `r`, plausibility filter and the day bootstrap as in E14-E19. Home and away teams come from the training set's `teamId_home` and `teamId_away`.

Steps:
1. **Mechanism (layer a0, no line involved).** OLS `actual game pace ~ 1 + mean_pace + gap`, separately on discovery and confirmation (and variants with season dummies, regulation-only games, and window 10 and 30). **Supported** iff the `gap` coefficient is negative in discovery and negative in confirmation with one-sided bootstrap p < 0.05 in confirmation.
   Also reported: the `mean_pace` coefficient (1 would mean "the average rule is right"). Supplementary (not registered as decisive): the same regression on 2013-14..2018-19 and on all 13 seasons, which needs no odds.
2. **Market test (layer a), gated on step 1.** OLS `r ~ 1 + gap + mean_pace + (odds_total - discovery mean)` (variants: with season dummies; regulation-only), separately in each period. **Supported** iff the `gap` coefficient is negative in both periods and the confirmation one-sided p < 0.05.
   This p-value is the angle's Holm input if step 1 is supported; if step 1 is not supported the angle's Holm input is 1.0 and step 2 is still run, labelled **exploratory and ungated**.
3. **Model test (layer b).** On the deployed out-of-sample predictions (`reports/oos_predictions_*.csv`, 2021-22..2025-26 and the confirmation seasons), OLS `actual - total_pred ~ 1 + gap + mean_pace + (odds_total - mean)`, one-sided negative for `gap`, two-sided also reported;
   plus the mean residual and mean `r` by quintile of `gap` (edges from the pooled OOS sample). Supported iff the confirmation `gap` coefficient is negative with one-sided p < 0.05.
4. **Feature test (layer c), gated:** only if steps 1, 2 and 3 are all supported. Then add `pace_gap` and `pace_mean` (known before tip-off, one-game shift) to a variant of the training set and run `experiments/feature_exp.py` (seven walk-forward folds) against the baseline and `noise_floor.md`; keep iff total MAE improves in at least 5 of 7 seasons with a pooled paired interval below 0
   and the no-harm rule holds for ML log loss, spread MAE and total MAE. Not run otherwise.
5. **Betting test.** Under at -110 at the opening total when `gap >= c`, where `c` is the discovery quantile q of `gap` for q in {0.5, 0.6, 0.7, 0.8, 0.9}; choose the q with the highest discovery return among those with at least 200 discovery bets; apply that numeric cutoff to confirmation.
   Report return with intervals at the opening total, the closing total and the best-book line and quoted price (E12 method), plus the all-games under control. The mirror (over when `gap >= c`) is exploratory.
6. **Sensitivity.** Steps 1 and 2 without 2019-20 and 2020-21 (pooled 2021-22..2025-26), and with season dummies.

**Power (stated now).** The report states the standard error of the `gap` coefficient in each step and the smallest effect detectable at 80% power (2.8 x SE). Each possession adds about 2.2 points to a game total, so a mechanism coefficient of -0.1 possessions per pace point of gap would be worth about -0.2 points of total per pace point.

**Reading rule.** Supported / not supported against the rules above; no criterion changes after the run; anything else is exploratory.
