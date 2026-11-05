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


## E23 - favourite on a back-to-back (angle 7 of 8)

Added 2026-10-23 before any E23 number was computed (only null shares and value counts of the back-to-back flags were inspected). Shared rules above apply (discovery 2019-20..2021-22, confirmation 2022-23..2025-26, day-cluster bootstrap 1000 draws seed 20261017,
Holm across 8 pending, layers a/b/c, betting last, sensitivity without 2019-20 and 2020-21). **Exploratory** by the user's instruction: cells are small. **Regular-season games only** (game id prefix 002), as in E19-E22. Nothing about back-to-backs has been seen in either period.

**Hypothesis H_B2B (two-sided; the user says the direction is not obvious):** the total relative to the line, `r = actual total - opening total`, depends on whether the favourite (or the underdog) is on the second night of a back-to-back, and on the size of the spread. A second, mechanism hypothesis (step 3): a favourite on a back-to-back wins by 15 or more less often than its spread implies.

Definitions (fixed now):
- `is_b2b_home` and `is_b2b_away` from the training set (days of rest = 1). The **favourite** is the team with the negative spread (home if `odds_home_spread < 0`, away if `> 0`); games with a spread of 0 have no favourite and are excluded from every favourite/underdog cell.
  **FB** = the favourite is on a back-to-back; **UB** = the underdog is on a back-to-back (both can be 1).
- Spread buckets: `|spread| < 8` and `>= 8` (the user's cut). Sensitivity thresholds fixed now: 6 and 10.
- Blowout = `|target_point_diff| >= 15`; **favourite blowout** = the favourite wins by 15 or more. Implied probabilities come from `margin ~ Normal(-odds_home_spread, sigma)` with a continuity correction at 14.5, `sigma` = standard deviation of `target_point_diff + odds_home_spread` over the discovery games, fixed once.
- Data frame, plausibility filter, `r`, bootstrap as in E14-E21.

Steps:
1. **Cells.** Four cells for the favourite (FB yes/no) x (`|spread|` < 8, >= 8), and the same four for the underdog (UB). For each period: games, mean `r`, under rate (pushes dropped) with intervals; also the same tables at thresholds 6 and 10. Counts are reported first.
2. **Primary regression.** OLS `r ~ |spread| + (odds_total - discovery mean) + FB + UB + FB x |spread| + UB x |spread| + season dummies`, separately on discovery and confirmation.
   **Primary statistic: the joint Wald test of the four flag coefficients** (bootstrap covariance, chi-square 4). **H_B2B is supported** iff discovery p < 0.10 and confirmation p < 0.05 (a loose find step, a strict confirm step); Holm uses the confirmation p. Individual coefficients with two-sided bootstrap p-values are descriptive.
3. **Blowout test (mechanism).** For each favourite cell and each period: observed rate of `|margin| >= 15` and of a favourite blowout, the model-implied rate from the spread, and their difference with a day-bootstrap interval. **The mechanism is supported** iff for the cell (FB, `|spread|` >= 8) the favourite-blowout difference
   (observed - implied) is below 0 in both periods and its confirmation interval's upper end is below 0 (one-sided in the sign the mechanism needs). Also a linear-probability regression of the favourite-blowout indicator on the same design as step 2, reported.
4. **Model test (layer b).** The step 2 regression on the out-of-sample model residual `actual - total_pred` and on the model's lean, 2021-22..2025-26, joint Wald p; mean residual in each cell.
5. **Feature test (layer c).** Only if step 2 is supported and the confirmation joint Wald p of the model residual is < 0.05. Not run otherwise.
6. **Betting test.** Over and under at -110 at the opening total for the cells FB&>=8, FB&<8, UB&>=8, UB&<8; choose the (cell, side) with the highest discovery return among those with at least 100 discovery bets; apply unchanged to confirmation; report at the opening total, the closing total and the best-book line and price, with all-games controls.
   Per the user, nothing is bet on a discovery-season result alone: no stake conclusion follows from this step.
7. **Sensitivity.** Step 2's joint Wald p without 2019-20 and 2020-21 (pooled 2021-22..2025-26), and at the thresholds 6 and 10.

**Power (stated now).** Counts are reported before any result. The user expects about 350 games with a favourite of 8+ on a back-to-back over seven seasons, a standard error of about 1 point on the mean of `r` for such a group scaled to its size (sd of `r` about 18.5: 18.5/sqrt(n)), and about 2.7 points on the under rate. A joint test of four coefficients has little power for a modest effect.
Underpowered; "not shown" is the expected outcome unless an effect is several points.

**Reading rule.** Supported / not supported against the rules above; no criterion changes after the run; anything else is exploratory. An unexplained discovery-only result is not a finding.


## E24 - line movement (angle 8 of 8)

Added 2026-10-23 before any E24 relationship between movement and outcome was computed. Before registering I verified what the odds tables are (see "Data verification"); that check used only table structure, the code that fills the tables, counts, and the open-versus-close mean absolute error by season that `odds_clean.check_closing_contamination` itself uses.
Shared rules above apply (discovery 2019-20..2021-22, confirmation 2022-23..2025-26, day-cluster bootstrap 1000 draws seed 20261017, Holm across 8 pending, betting last, sensitivity without 2019-20 and 2020-21). Regular-season games only (game id prefix 002), as in E19-E23.

### Data verification (done first, as requested)

- `nba_odds` and `nba_odds_opening` have **no timestamp column**: only `gameDate` and `status` (Final, Final OT, a tip time such as "19:00 ET" before the game, or an in-game string such as "02:45 4th"). `core/snapshot_odds.py` and `core/odds_clean.py` both say so.
  `nba_odds` is a scrape of the SBR per-book "current" line whenever the scrape ran; for completed games `odds_clean` describes it as "scraped after the game". Pre-game it holds the latest line, in-game scrapes leaked in-game lines (two windows, 2019-10-22..2019-11-19 and 2021-10-19..2021-11-23, are set to missing; individual books more than 3 points (spread) or 6 points (total) from the cross-book median are dropped).
- Evidence that `nba_odds` is mostly a closing line: over the 8,423 games with both lines, the total's mean absolute error is 1.1-2.9% lower at the "close" than at the open in every season (the documented normal ratio is about 0.97), and for the 16 games of 2026-27 that also have timestamped snapshots, the `nba_odds` consensus total and spread equal the last pre-game snapshot's exactly.
  So "close" = the last pre-game line as SBR displayed it afterwards, at an unknown time just before tip. Residual contamination by in-game lines that stay within the tolerances cannot be ruled out; it would make the close look more accurate than it was and would bias a movement test.
- `nba_odds_opening` is SBR's opening line per book; its time is unknown and may be a day or more before tip. The open-to-close move is therefore **not a fixed-horizon quantity**: its standard deviation is about 3.4 points for totals (median absolute 2.0, 99th percentile 9.0) and 1.0 median for spreads (99th percentile 7.0). 94% of totals and 87% of spreads move at all.
- `odds_snapshots` (hourly collector, `line_type` 'current' and 'opening', `in_game` flag) now holds 6,273 rows over 4 days and 18 games (collection began 2026-10-19). Its 'opening' rows mean "first seen by the collector", not the market opening. The `odds_source_log` shows several sources (sbr, theoddsapi) in use, so the book set can change between fetches. Too few games yet for any test; E24 registers a forward test on it.
- Consequence for meaning: the historical test measures whether the last pre-game line has already priced everything that moved the line since an early opening (efficiency of the close). It does not measure what a bettor could do at a fixed time. The forward test below is the usable version.

### Hypothesis H_MOVE (two-sided; momentum and overreaction both count)

The closing line is not efficient with respect to its own movement since the open: the error of the close depends on the move. Momentum (the move continues past the close) is a positive slope; overreaction (it reverses) is a negative slope.

Definitions (fixed now):
- Consensus lines as in `core/odds_clean.py` (`load_clean_odds`, median across kept books, outlier filter, bad closing windows missing), games 2019-20..2025-26 with both an open and a close total (and spread), regular season only.
- `move_total = close_total - open_total`; `move_home = -(close_home_spread - open_home_spread)` (positive = the line moved toward the home team being a bigger favourite).
- `err_total = actual total - close_total`; `err_spread = home margin + close_home_spread` (positive = the home team did better than the close implied).

Steps:
1. **Market-efficiency test (primary).** OLS `err_total ~ 1 + move_total`, separately on discovery and confirmation. **H_MOVE is supported** iff the slopes have the same sign in both periods and the confirmation two-sided bootstrap p < 0.05 (the sign then names momentum or overreaction);
   Holm uses the confirmation p. Variants reported: with `|spread|` and season dummies; without the largest 1% of absolute moves; with the per-book alternative (the median across books of each book's own close minus open, among books quoting both). Secondary: the same for the spread, `err_spread ~ 1 + move_home`.
2. **Cross-market check.** OLS `err_total ~ 1 + move_total + move_home + |move_home|` (and a version with an indicator for `|move_home|` at or above its discovery 75th percentile). Primary statistic: the coefficient on `|move_home|` (a large spread move, which is mostly injury news), two-sided p; reported for both periods.
3. **Model relevance.** The deployed model prices off the opening line (and noon injury information), so movement is not an input and layers (b) and (c) do not apply to the historical test. Reported only: the model's out-of-sample residual `actual - total_pred` against `move_total` in 2021-22..2025-26.
4. **Usability and betting readout (priced at the close).** The bet is placed at the close, so it must use closing prices. Closing over and under prices come from the per-book quotes of `nba_odds` (cleaned with the `odds_clean` outlier rules and settled at the book quoting the line nearest the consensus, best quoted price on ties; -110 is reported beside it as the naive price).
   Rule: bet **with** the move (over after a rise, under after a fall) or **against** it when `|move_total| >= c`, c in {2, 3, 4, 5, 6} points, 2 directions x 5 cutoffs; choose the single combination with the highest discovery return among those with at least 200 discovery bets; apply it unchanged to confirmation; return per bet with day-bootstrap intervals, at the quoted close price and at -110.
5. **Sensitivity.** Step 1 without 2019-20 and 2020-21; discovery = 2021-22 only; excluding games within 14 days of the two flagged contamination windows.

### Forward test on the timestamped snapshots (the actionable version)

- Source `odds_snapshots` with `line_type = 'current'`, `in_game = 0`; consensus across kept books with the `odds_clean` tolerances. **Close** = the last pre-tip snapshot of a game; **T4** = the last snapshot fetched at or before 4 hours before the scheduled start (`game_start`).
  `move_f = close_total - T4_total`; to be robust to changes in the book set between fetches, the primary `move_f` is the median over books quoting at both times of each book's own change. A game needs both snapshots at least 2 hours apart.
- Forward sample: games on or after 2026-10-24 that have both snapshots and a final result. Test: OLS `actual total - close_total ~ 1 + move_f`, two-sided. **Supported** iff the 95% bootstrap interval for the slope excludes 0 at the single decisive evaluation, when the sample reaches **1,000 games** (about one season if the collector runs every game day); looks at 250 and 500 games are reported and decide nothing.
- Power: with a standard deviation of `actual - close` of about 18 and a standard deviation of `move_f` that I do not yet know (it is smaller than the open-to-close move), the report states the standard error and the smallest slope detectable at 80% power; a slope below about 1 divided by the standard deviation of the move in points is out of reach.
- The same test for the spread. The collector's own reliability (missing fetches, source switches) is reported before any result.

**Power for the historical test (stated now).** With the standard deviation of the move about 3.3 points and of `err_total` about 18, the slope's standard error is about 18/(3.3 x sqrt(n)): roughly 0.08 for the 4,400 confirmation games and 0.12 for the 2,000 discovery games, so a slope of about 0.2 (a fifth of the move continuing) is detectable at the 5% level.
The report states the actual standard errors.

**Reading rule.** Supported / not supported against the rules above; no criterion changes after the run; anything else is exploratory. A "supported" label on the historical test is evidence about the scraped close, subject to the contamination caveat; only the forward test says what could be acted on.

## E25 - decorrelation from the market: edge slope and blend (registered 2026-10-24, before any E25 run)

### Why the existing no-odds artifacts are not used (checked first)

- `models/saved/backtest_predictions_no_odds_v2.csv` (2026-01-21, 5,790 games, 2021-22..2025-26 partial): its attached lines are corrupt - 230 games with a moneyline beyond +-2000,
  355 with both moneylines of the same sign, 553 where the listed spread has the opposite sign of `odds_home_spread_open`. It predates the audit-fixes and team-audit-2 odds pipeline (one-book opening odds, merge guards).
- `models/saved/backtest_predictions_no_odds.csv` (2026-03-17): one season only (2025-26, 1,014 games); its feature config contains `roster_bpm_ewma_x` and `roster_bpm_ewma_y`, a duplicated-merge artifact that team-audit-2's merge guards now prevent.
- Both were made by older code than every other experiment here, so they are not comparable with the E1-E24 baseline. The `_no_odds_2026-03-18` model files are final models (no out-of-sample predictions) and are left untouched.
- The E-series baseline itself was built on `training_set_v1_reduced_2026-10-16.csv`; the current set is `..._2026-10-19.csv`. To compare like with like, all three runs below are regenerated on the 10-19 set with `experiments.lib.run_walk_forward` (same folds, per-fold tuning, seed 42).

### Runs (regenerated, written to reports/experiments/E25_*/)

- `E25_with_odds`: the baseline - opening odds in the feature pools (market as a feature; the deployed total model's form).
- `E25_residual`: E1's hooks on `E25_with_odds`'s feature sets (market as an offset: spread = -line + f(X), total = line + f(X), ML init score = logit of the de-vigged opening probability; the deployed spread model's form).
- `E25_no_odds`: `BaselineModel.load_data(use_odds=False)` pools, features selected afresh per fold (no market information at all).

**No-odds validity checks (must all pass before the diagnostic is read):** (1) no selected feature contains `odds` or `close`; (2) the same games as `E25_with_odds`; (3) in every fold the latest training date precedes the first test date;
(4) per-season spread MAE, total MAE and ML log loss within 0.5 points / 0.5 points / 0.03 of the audit's post-fix no-odds walk-forward (`reports/audit_fixes/walk_forward_no_odds.md`, "new: neutral params") - a larger gap is investigated, not ignored;
(5) reported only: per fold, the largest absolute correlation (on the model-training rows) between a selected feature and the opening total, opening spread and opening log-odds - a value above 0.9 is a market proxy and is named.

### Quantities (all against the OPENING line; the closing line is a secondary readout)

- Market expectation: total `L = odds_total`; spread `L = -odds_home_spread` (home margin); ML `L = logit(de-vigged opening home probability)` (E1's `devig_open_prob`; games without a moneyline dropped from ML).
- Edge `e = model - L` (ML on the log-odds scale, using `ml_prob_raw`, before isotonic calibration). Market residual `r = actual - L` (ML: the outcome).
- **Edge slope beta:** OLS `r ~ 1 + e` (ML: logistic regression of the home win on `e` with offset `L`, plus intercept). beta = 1: the edges are real and full-sized; 0: noise; between: overstated by a factor 1/beta.
- Error correlation `corr(actual - model, actual - L)` (spread, total).
- **Blend:** `B = L + k e`, k fitted by least squares without intercept on the discovery seasons (ML: logistic, offset `L`, no intercept), applied unchanged to confirmation. Loss: squared error (spread, total), log loss (ML).
  Secondary: k walk-forward (k for season s fitted on test seasons before s), 2020-21..2025-26.
- Day-cluster bootstrap, 1,000 draws (`metrics.RNG_SEED`). Discovery 2019-20..2021-22, confirmation 2022-23..2025-26. Regular season and playoffs both kept (as in the baseline); a regular-season-only rerun is reported.

### Decision rules (fixed now), per market

1. **Edges informative:** confirmation beta's 95% lower bound > 0. **Overstated:** confirmation beta's 95% upper bound < 1.
2. **Decorrelation adds information (H_DECOR):** in confirmation, beta(no-odds) - beta(with-odds) has a paired bootstrap interval above 0 **and** the no-odds blend's loss is below the opening line's with the paired interval excluding 0.
   Holm across the three markets on the blend-vs-line p-values.
3. Reported, no decision: blend(no-odds) vs blend(with-odds) and vs blend(residual); the same against the closing line; decile plots of r against e.
4. Anything not listed is exploratory. A supported H_DECOR makes the blend a **candidate** (forward test, or the feature route), not a deployment.

## E26 - forward test of edge shrinkage and the totals no-odds blend against ENTRY lines (registered 2026-10-24, before any forward game is scored)

Follows E25. Nothing in the deployed model or the live pipeline changes; this test only reads what the live pipeline already logs.
Frozen inputs (computed from history only, 2026-10-24T07:58, `experiments/e26_freeze.py`, refuses to overwrite): `reports/experiments/E26_forward_shrink/frozen/params.json` and `no_odds_total.joblib`.

### Frozen quantities

- **Shrink factors beta** (edge slope vs the opening line) for the deployed model's form, estimated on E13's out-of-sample predictions (E1 residual spread, E3 margin-stack ML, baseline total), **all seven seasons 2019-20..2025-26**:
  spread 0.474 [0.201, 0.780], total 0.223 [0.099, 0.347], ML 0.320 [0.061, 0.581] (log-odds scale).
  Why all seasons and not discovery only: for a forward test every historical season is "find" data. The discovery-only values are reported but are unstable: spread 0.044 [-0.425, 0.481], total 0.226 [0.053, 0.400], ML 0.134 [-0.280, 0.504].
- **Totals blend weight** k = 0.166 (E25_no_odds, all seasons; discovery-only 0.133).
- **No-odds totals model:** LightGBM with 37 features, selected by the harness on the first part of 2013-14..2025-26; grid config 7; refit on all 16,795 games through 2026-06-13. Training set and model SHA-256 recorded.
  The SHA-256 of the feature-building code (`team/game_features_v2.py`, `core/travel_builder.py`, `core/injury_report_features.py`, `team/train_baseline.py`, `scripts/generate_training_set.py`, `core/odds_clean.py`) is also recorded.
- **Deployed set:** `_with_odds_cand_20261019`, model_set_hash `0165d3361e4228ad2551a787c66e1cd82ce11bd5b9a4a258b7e79467e7cd02e0`.
- Residual SDs for the betting readout only: spread 13.66, total 18.46 (shrunk), 18.44 (blend).

### Forward sample

- NBA games (regular season, play-in and playoffs) on or after **2026-10-25** with a final result and an entry line from `core.odds_clean.load_entry_odds` (consensus at noon ET on game day, or 30 minutes before tip if earlier, from `odds_snapshots`). The 2026-10-20..10-24 games are excluded: they were visible before registration.
- **(a) uses the deployed model's logged prediction** (`predictions_log_v2`). A game is eligible only if its model_set_hash equals the frozen hash **and** its prediction_date is at or before the entry line's fetched_at time, so the prediction existed when the price was available.
  Games predicted by another model set (after a retrain) are excluded from the primary analysis and reported separately, because beta belongs to this model. The number excluded under each rule is reported.
- **(b) uses the frozen no-odds model** on point-in-time feature rows rebuilt after the fact (`scripts/generate_training_set.py`). The scorer refuses to run if any of the hashed feature-code files changed, unless it is run from a checkout of the registration commit.
  Caveat stated now: the rebuilt rows can differ from what a live build would have produced, because of data revisions or a missed injury snapshot; the number of rows with missing injury data is reported.

### Definitions (as in E25, with the entry line in place of the open)

L = `entry_total`; `-entry_home_spread` (home margin); logit(`entry_home_prob`) for ML. Edge e = model - L. Residual r = actual - L; for ML, the outcome.
Forward edge slope beta_f: OLS `r ~ 1 + e`; for ML, logistic regression with offset L and an intercept. Shrunk prediction L + beta e; blend L + k e_no-odds. Intervals from a day-cluster bootstrap (1,000 draws, `metrics.RNG_SEED`).

### Hypotheses (one-sided) and decisive sample sizes

Power (stated now, from history vs the open; entry lines are sharper, so this is optimistic): comparing the *loss* of shrunk or blended predictions with the line needs 11,000-48,000 games, except shrunk vs unshrunk totals (about 1,400). So the primary tests use the slope, the quantity the loss comparison depends on.

| id | claim | supported iff | decisive at |
|---|---|---|---|
| A-total | deployed total edges are overstated vs entry lines (shrinking helps) | 95% upper bound of beta_f < 1 | 1,300 eligible games |
| A-spread | the same for the deployed spread | 95% upper bound of beta_f < 1 | 2,500 |
| A-ml | the same for the deployed ML (log-odds) | 95% upper bound of beta_f < 1 | 2,500 |
| B-total | no-odds total edges carry information vs entry lines (a blend with k > 0 helps) | 95% lower bound of beta_f(no-odds) > 0 | 2,500 |

- Decision at the single decisive look per hypothesis, with Holm across the four (the one-sided bootstrap p is the share of draws on the wrong side of 1, or of 0).
  Looks at 500 and 1,000 games (and 1,300 for the 2,500-game tests) are reported and **decide nothing**. 1,300 games is about one season; 2,500 is about two.
- Reported, no decision:
  - whether each frozen beta lies inside the forward 95% interval (the frozen factor is about right);
  - per-game loss of shrunk vs unshrunk vs line, and blend vs line vs the shrunk deployed total (paired, day bootstrap);
  - beta_f against the closing line;
  - regular season only;
  - discovery-only factors in place of all-season ones;
  - a betting readout: unified 5% probability-edge rule on shrunk probabilities (Normal with the frozen SDs for spread and total, -110 because entry prices for those markets are not stored; ML at entry moneyline prices), with bets, hit rate and return with day-bootstrap intervals.
- **Reading rule.** Supported A-* means the shrink should be applied to the bet rule as a candidate change, with the forward beta reported next to the frozen one. Supported B-total makes the blend a candidate (the feature route or a deployment proposal, which needs the owner's go-ahead).
  Not supported means not shown at that sample size; with power this low that is not evidence of no effect. No criterion changes after the first look.
- The scorer, `experiments/e26_score.py`, must be committed together with this section and the frozen folder before the first look at any forward game.

**Power of the registered slope tests** (stated before any forward data). Historical bootstrap standard errors are scaled to the decisive n and assume the true forward beta equals the frozen value; entry lines are sharper than the open, so the true power is likely lower:
- A-total: SE 0.17 at 1,300 games, power about 1.00;
- A-spread: SE 0.28 at 2,500, power about 0.60 (the deployed spread's edges are small, SD 0.92 points);
- A-ml: SE 0.25 at 2,500, power about 0.86;
- B-total: SE 0.07 at 2,500, power about 0.77.

**Expected thin betting readout** (from a smoke test on history, with the opening line standing in for the entry line): with the frozen shrink factors the 5% probability-edge rule fires about 35 times on totals, 6 on ML and 0 on spread over 2019-20..2025-26. The forward readout will therefore have very few bets and is descriptive only. The smoke test reproduced every frozen beta exactly.

# Tracking and fatigue program (E28 onward; plan in reports/tracking_fatigue/PLAN.md; branch `tracking-fatigue`, cut from `odds-rerun`)

Question: do running load, speed, physical previous games and schedule density, built from tracking data the database already holds, improve the team models (spread, total, moneyline) or the props model.
Nothing deployed changes in this program before step 6, which needs the owner's go-ahead. E27 is reserved for the E18 totals-calibration test, so this program starts at E28.

## E28-audit - data audit (registered 2026-10-24, before any audit number was computed; descriptive, nothing passes or fails)

Inspecting row counts, column names, coverage and the daily log is allowed before registration; relationships with outcomes are not looked at in the audit. Script: `experiments/e28_audit.py`, output `reports/experiments/E28_tracking/audit.md`.

Seen before registration (not results of the audit, only why the questions below matter): `team_track_stats_v3.speed` is NULL in the rows inspected, and the daily log (`data/raw/daily_inference.log`, 2026-10-20 to 2026-10-24) shows the 9:15 run failing to fetch tracking for a number of 2026-27 games.

Checks:
1. **Coverage.** Rows and distinct games per season for team tracking, player tracking and hustle; games with both teams present; playoff (`004`) and play-in (`005`) shares; the 2024-25 player-tracking fill share; the 2019-20 bubble and 2020-21.
2. **Units and sanity.** Parse `minutes` "MM:SS". Team distance per game, mean and SD by season. Player speed range. Player distances summed per team against team `distance` (share within 1%). Outliers: distance 0 with minutes > 0, speed > 6. Whether `speed` is filled anywhere at team level.
3. **Live availability.** From the daily log: for every game in the log window, the date it first appeared in a "Successfully added" line for each tracking and hustle table, against the game date; the share of 9:15 runs that had the previous night's tracking for all games; the error text. For back-to-back second nights in history: the share whose previous game has tracking in the database at all (the live proxy is only the log window, and that is said).
4. **Opening-line timing on back-to-backs.** `nba_odds_opening` has no timestamps and `odds_snapshots` starts 2026-10-19, so only what those hold is reported, and what could not be measured is stated.
5. **Opening-line coverage** by season in `nba_odds_opening`, including seasons before 2019-20.

Stop rule (from the plan): if the previous night's tracking is usually missing at 9:15 and cannot be fetched earlier, the lagged-load families (F1, F2, F3) use "most recent game with data" and the report says so.

## E28 - residual screen of the tracking / fatigue families (registered 2026-10-24, before any residual or mechanism regression was computed)

Written after the audit (`reports/experiments/E28_tracking/audit.md`) and after the feature library (`experiments/features_tracking.py`, tests in `tests/team/test_tracking_features.py`) existed, but before any feature was related to any outcome.
Only distributions of the features themselves (counts, ranges, missingness) were looked at, to set data-quality rules. No model, deployment or schedule changes.

### Feature definitions as implemented (these are the definitions; deviations from the plan are listed)

All team features for game t use games strictly before t. Variant file: `data/processed/experiments/e28_training_set.csv` = the 10-19 training set plus columns `trk_<name>_home / _away / diff_trk_<name>`; the existing 430 columns and the 16,795 rows are unchanged (checked).
- **Norm / sd.** Norm = EWMA (span 20) of the team's (or player's) prior valid games, available after 5 prior valid games. SD = rolling 20-game SD of prior valid games (after 8), floored at 1% of |norm|. z = (x - norm) / sd, clipped to +-6.
- **Data validity.** Team-game tracking is valid iff distance > 0, touches > 0 and 0.05 <= distance per team minute <= 0.095 (the all-zero rows, 350 of them in 2025-26, and 25 broken rows with under 12 miles are missing, never "low load"). Player-game is valid iff minutes >= 5, speed in [2.5, 6.5] mph and distance / hours played in [1.5, 8.0] mph. Team minutes and possessions: `team_game_full.minutes` (identical to the sum of player minutes in every game, audit §2) and pace x minutes / 240.
- **"Last" features use the most recent game with data**, at most 3 games back (older -> NaN), because tracking was missing for the whole 10-23 slate at the 10-24 run (audit §3). F3 box-score items (`prev_ot`, `prev_close`, `prev_n40`, `prev_opp_pace`) use the previous game itself.
- **F1.** `dist_per_min_last`, `dist_per_poss_last`; `xload` = z of distance per possession (over the team's prior games); `xload_last`; `xload_4n` = sum of `xload` over the team's games dated 1-4 days before t (0 if there were none, NaN if games were in the window but none had data or the team has no valid xload yet); `rot_xload_last` = minutes-weighted mean of the rotation's (>= 10 min in that game) player z of distance per minute, at least 4 players with a z.
- **F2.** `rot_speed_z_last` = the same with player speed z; `rot_speed_trend5` = OLS slope over the team's last 5 games of that series (at least 3 valid); `star_speed_z_last` = minutes-weighted mean of the speed z of the 2 players with most minutes in that game (both must have a z).
- **F3.** `prev_ot` (team minutes > 240), `prev_close` (|plus-minus| <= 5), `prev_n40` (players at 40+ minutes), `prev_hustle_z` = z of contested shots + deflections + loose balls recovered + box-outs (team totals; NaN where the team has no hustle row, i.e. mostly before 2016-17), `prev_opp_pace` = pace of the previous game.
- **F4.** `g3in4` (3+ games in the window [t-3, t] days), `g4in6` (4+ in [t-5, t]), `games_last7` (games in [t-7, t-1]), `trip_last_game` (road game whose next game, within 7 days, is at home; read from the published schedule), `homecoming` (home game after 3+ consecutive road games, no gap over 7 days), `prev_altitude` (previous game within 7 days at an arena >= 4,000 ft, i.e. DEN or UTA, and this one is not).
- **F5.** `xload_4n_x_b2b` = `xload_4n` x `is_b2b`; `xload_last_x_dist` = `xload_last` x `dist_miles` / 1000 (both from the training set's existing columns); `one_side_tired` (game level, home perspective): tired score = `xload_4n` + `prev_ot` + `is_b2b`; +1 if home's score is above the 80th percentile and away's below the median, -1 for the reverse, else 0.
- **Deviation from the plan:** the percentiles of the tired score are expanding (all team-games dated strictly before the game, at least 200) rather than "fitted on each fold's training rows", so the column does not depend on a fold and cannot leak; the plan's wording would need a different column per fold.
- **Other deviations/additions:** the plausibility rules and clip above; the 7-day continuity guards in F4; the 3-game staleness cap. All fixed before the screen.

### Data and targets

- Games: regular season only (`gameId` prefix 002) for the main test; playoffs and play-in added as a sensitivity run. Games with any NaN in the family's columns are dropped (listwise); the numbers dropped are reported.
- **(a) model residual** (2019-20 .. 2025-26, the out-of-sample test seasons of the deployed-form runs on the 10-19 training set): total = `target_total_pts - total_pred` (run `E25_with_odds`); margin = `target_point_diff - spread_pred` (run `E25_residual`, the E1 form); home win: logistic regression with the E13 stack probability as offset (stack = logistic on [logit raw ML probability from `E25_with_odds`, E1 margin / sigma from `E25_residual`], fitted per fold on that fold's calibration set, as in `experiments/e13_combined.py`; games common to both runs).
- **(b) market residual** (opening line): total `target_total_pts - odds_total`; margin `target_point_diff + odds_home_spread`; home win: logistic with offset logit(`open_home_prob`), 2019-20 onward only (the 2013-14..2018-19 archive has closing moneylines only). Lines outside the `core.odds_clean` plausibility bounds (total 170-290, |spread| <= 30) are dropped.
  Because the audit found 2013-14..2018-19 opening totals and spreads in the training set (archive source, different from the later seasons), (b) total and margin use **2013-14 .. 2021-22 as find seasons**, with the earlier six reported separately as a sensitivity check; ML (b) finds on 2019-20 .. 2021-22.
- **(c) mechanism** (outcomes of game t, never features): own team's 3P%, FT%, rim (restricted area) FG%, pace, points per possession, one row per team-game, regressed on the family's own-team features plus the opponent's, a home indicator and season dummies. Find 2013-14 .. 2021-22, confirm 2022-23 .. 2025-26.
- Confirmation seasons everywhere: 2022-23 .. 2025-26.

### Regressors, estimation, tests

- Margin and home-win targets use the family's `diff_trk_*` columns (home minus away; the game-level `trk_one_side_tired` as is). The total target uses the sum home + away of the team-level columns (|`trk_one_side_tired`| for F5). All regressors are standardised with the find-period mean and SD (binary flags too), so a coefficient is "points (or log-odds) per 1 SD".
- Model: OLS on the residual with season dummies; for the home win, logistic regression with the offset and season dummies. Statistic: **joint Wald test** that all family coefficients are zero, with the covariance taken from a **day-cluster bootstrap** (1,000 draws, `metrics.RNG_SEED`, whole game days resampled; logistic draws by Newton steps from the full-sample fit); p from a chi-square with the numerical rank of the covariance as degrees of freedom (pseudo-inverse, rcond 1e-8).
- **Find step passes** iff the find-period Wald p < 0.10.
- **Confirmation.** The find-period coefficient vector is applied unchanged to the confirmation seasons' features to form a signal; the confirmation residual is regressed on that signal (OLS / logistic with offset, season dummies; the signal is standardised on the find period). The slope gamma (1 = the find effect is replicated at full size) has a day-bootstrap interval; **same direction iff gamma > 0**, and its one-sided bootstrap p is the share of draws with gamma <= 0 (add-one).
- **Tiers.** Reject = fails the find step, or gamma <= 0. **Candidate** = passes the find step, gamma > 0 and **Benjamini-Hochberg q < 0.10** over the 15 family x target confirmation p-values of that residual type (families that failed the find step enter with p = 1; (a) and (b) corrected separately). A family that passes the find step with gamma > 0 but q >= 0.10 is reported as "not confirmed" and is not a candidate. Only (a) candidates go to the walk-forward test (E29) on that market; (b) candidates become betting-angle candidates (step 8, forward tests).
- **Mechanism rule for F1 and F2 (stated now).** The fatigue story is called **supported** iff, for at least one of 3P%, FT% and pace, the find Wald p < 0.10, gamma > 0 and the one-sided p < 0.0167 (Bonferroni over the three); otherwise the report says that F1/F2 do not move shooting or pace and the story is weak, whatever (a) and (b) show. Rim FG% and points per possession are reported for comparison (the leg-fatigue question of step 7); they decide nothing.
- Reported but deciding nothing: each coefficient with its bootstrap interval in plain units, the confirmation-period Wald p, the playoffs-included rerun, the six archive seasons alone, and per-season signs of gamma.
- Anything outside these rules is exploratory and labelled so. No rule changes after the first residual is seen.

### Expected (stated before running)

F4 is expected to be mostly priced (E23 found nothing on favourites on a back-to-back). F1 and F2 are the novel part, and the plausible size on totals is under one point; most families failing is the normal outcome. The power of a joint test on about 3,600 (a) find games is modest: it can see roughly a 0.3-0.5 point shift per SD on totals, not smaller.

## E29 - forced-inclusion walk-forward test of the E28 candidates (registered 2026-10-24, before any E29 run)

### What E28 produced (read before registering this)

Under the registered E28 rules the only candidate on the model-residual screen (a) is **F1 on the total** (find Wald p 0.0044, gamma 0.96, one-sided p 0.001, BH q 0.015). The opening-line screen (b) has four candidates: F1, F3 and F4 on the margin and F5 on the total. Under the plan these go to forward betting tests (step 8), not to this step.
The exploratory follow-up `reports/experiments/E28_tracking/explore.md` (written after the screen was read, so not registered) shows that the F1 total signal vanishes once the previous game's pace and the team's pace norm are controlled (gamma 0.96 -> 0.06, 95% CI [-1.8, 1.8]). The plan sends every (a) candidate to this step, so the registered test below is run anyway; the two extra arms registered here are there to say what a pass would mean.

### Design (the plan's step 4, for the one market with a candidate)

- Market: total, deployed form = level model (reference run `E25_with_odds`, 10-19 training set, 7 folds 2019-20 .. 2025-26, per-fold tuning from the fixed 8-config grid inside the fold, seed 42).
- **Forced inclusion.** Each fold's total feature list = the reference run's list for that fold + the family's columns (`features_tracking.family_columns`). The ML and spread lists are the reference's, unchanged, so those two models must reproduce the reference predictions exactly (checked, and a failure invalidates the run). The file `reports/experiments/E29_<arm>/features.json` is written before `run_walk_forward` is called, so the harness reuses it. The data are the 10-19 training set plus the E28 columns (`data/processed/experiments/e28_training_set.csv`); the game ids and row order per fold must equal the reference's (checked).
- Compared with `metrics.paired` on the same games: total MAE (main), total MSE (companion), with ML log loss and spread MAE for no-harm. Per season, pooled, the noise floor, and the share of the fitted model's gain taken by the family's columns in each fold (a fold counts as non-trivial if the family takes at least 5% of the total-model gain).

### Arms

| arm | total features | status |
|---|---|---|
| `E29_F1_total` | reference + F1 (15 columns) | **the registered test** |
| `E29x_pace_total` | reference + `trk_prev_opp_pace_home / _away / diff_` (the previous game's pace; 3 columns) | exploratory: what pace persistence alone does |
| `E29x_F1pace_total` | reference + F1 + those 3 | exploratory: does F1 add anything to the pace columns |
| `E29x_F1pool_total` | fresh feature selection per fold with only the F1 columns added to the pool (the other E28 columns removed from the file) | exploratory: the plan's "secondary" arm; seed 42, so its noise floor applies |

### Decision (fixed now; arm `E29_F1_total` only)

- **Adopt:** `decide.mc` on total MAE (at least 5 of 7 seasons better and the pooled paired day-bootstrap 95% interval excludes 0 in favour) **and** `decide.nh` on total MSE, ML log loss and spread MAE (the last two unchanged by construction).
- **Candidate:** the pooled total-MAE improvement exceeds the largest seed-only difference for total MAE in `reports/experiments/noise_floor.md` (0.0050 points; the seed-43 and seed-44 runs differ from seed 42 by +0.00499 and +0.00489) and at least 4 of 7 seasons are better.
- **Reject** otherwise. A candidate is not a model change; it goes to the owner with the numbers.
- Exploratory arms decide nothing. Anything not listed is exploratory.

### Step 5 (combine), stated now

Only Adopt families (and Candidates, labelled) enter the combined variant. If `E29_F1_total` is a Reject there is nothing to combine: step 5 then reports that no family passed, and no feature list is proposed.


### E28 erratum (2026-10-24, after the screen and E29 had run; wording only, no rule changed)

The E28 section calls `gamma` "the slope on the find-period signal (1 = the find effect is replicated at full size)". The signal is standardised to the find-period SD, so `gamma` is points (log-odds for the home win) per SD of the signal and full replication equals the signal's find-period SD, not 1. The decision rules use only the sign and the one-sided p-value of gamma, so every tier stands; the effect sizes first reported in SUMMARY.md were understated and are corrected there.

## E30 - forward test of the four E28 opening-line candidates against ENTRY lines (registered 2026-10-24, before any forward game is scored)

Step 8 of the tracking / fatigue plan. The E28 screen (b) produced four candidates against the opening line; none is a model feature (E29 rejected the feature route). This test asks whether each signal still predicts `actual - entry line` on games nobody has seen, and records closing-line value. Nothing in the deployed model, the live pipeline or the database changes, and the test does not depend on the deployed model set, so a promotion does not end it (unlike E26 (a)).
Frozen inputs (made from history only on 2026-10-24 by `experiments/e30_freeze.py`, which refuses to overwrite): `reports/experiments/E30_forward_tracking/frozen/params.json`. Scorer: `experiments/e30_score.py`; shared code `experiments/e30_signals.py`; features `experiments/features_tracking.py`. Their SHA-256 are in `params.json` (code) and in `reports/experiments/E28_tracking/registration.sha256`; the scorer refuses to run if the hashed code changed.

### The four tests and their frozen signals

Signal = ((regressors - fit mean) / fit SD) @ coefficients - centre, in points; regressors as in E28 (`diff_trk_*` for the margin, home + away sums for the total). Fitted by OLS with season dummies on `actual - opening line` over the regular seasons 2019-20 .. 2025-26 (the same source as the forward lines; the fit on all 13 seasons is stored for a sensitivity run). Games with any regressor missing have no signal and are not scored.

| test | signal | fit games | signal SD (points) | residual SD | 75th pct of abs signal | slope in the 6 archive seasons, not in the fit (SE) |
|---|---|---|---|---|---|---|
| T1 | F1 running load -> home margin | 8132 | 0.580 | 13.50 | 0.641 | 0.55 (0.23) |
| T2 | F3 physical previous game -> home margin | 8262 | 0.595 | 13.53 | 0.563 | 0.62 (0.39) |
| T3 | F4 schedule density -> home margin | 8270 | 0.573 | 13.53 | 0.631 | 0.28 (0.24) |
| T4 | F5 interactions -> game total | 8132 | 0.581 | 18.42 | 0.545 | 1.06 (0.39) |

Coefficients (primary fit; the full lists, with the all-season fit, are in `params.json`):

| test | regressor | fit mean | fit SD | coefficient per SD (95% CI) |
|---|---|---|---|---|
| T1 | dist_per_min_last | -0.0000 | 0.0037 | -0.0476 [-0.468, +0.365] |
| T1 | dist_per_poss_last | 0.0000 | 0.0101 | +0.6006 [+0.066, +1.152] |
| T1 | xload_last | -0.0016 | 1.5193 | -0.6731 [-1.266, -0.093] |
| T1 | xload_4n | 0.0219 | 1.9845 | +0.4308 [-0.004, +0.871] |
| T1 | rot_xload_last | -0.0023 | 0.9362 | +0.3711 [-0.079, +0.781] |
| T2 | prev_ot | -0.0011 | 0.3122 | -0.4033 [-0.824, -0.006] |
| T2 | prev_close | -0.0041 | 0.6020 | -0.2655 [-0.550, +0.037] |
| T2 | prev_n40 | 0.0224 | 0.8904 | -0.0297 [-0.445, +0.361] |
| T2 | prev_hustle_z | 0.0127 | 1.5317 | +0.3411 [+0.024, +0.643] |
| T2 | prev_opp_pace | -0.0667 | 6.4563 | +0.0018 [-0.277, +0.304] |
| T3 | g3in4 | -0.0150 | 0.5871 | -0.2259 [-0.574, +0.158] |
| T3 | g4in6 | -0.0080 | 0.6136 | -0.1531 [-0.558, +0.255] |
| T3 | games_last7 | -0.0111 | 0.8037 | -0.1104 [-0.465, +0.239] |
| T3 | trip_last_game | -0.4271 | 0.4947 | -0.3240 [-0.610, -0.024] |
| T3 | homecoming | 0.1434 | 0.3505 | -0.1546 [-0.426, +0.136] |
| T3 | prev_altitude | -0.0313 | 0.2784 | -0.1650 [-0.473, +0.127] |
| T4 | xload_4n_x_b2b | 0.0034 | 1.0174 | -0.0403 [-0.476, +0.389] |
| T4 | xload_last_x_dist | -0.0065 | 1.2415 | -0.5501 [-0.983, -0.141] |
| T4 | one_side_tired | 0.1725 | 0.3779 | -0.1126 [-0.520, +0.279] |

The in-sample slope of `actual - opening line` on each signal is 1 by construction (checked: 1.00, 0.995, 1.005, 1.03). The right-hand column is the slope of the same frozen signals on the six archive seasons that were not in the fit: **0.55, 0.62, 0.28, 1.06**. Against entry lines, which are later and sharper than the opening line, the expected slope is below the archive values, so anywhere from 0 to 0.6 should be expected, not 1.

### Forward sample

Regular-season games on or after **2026-10-25** with a final result and an entry line from `core.odds_clean.load_entry_odds` (consensus at noon ET on game day, or 30 minutes before tip if earlier, from `odds_snapshots`). The 2026-10-20 .. 10-24 games are excluded (visible before registration; used only to check the pipeline). The signal is rebuilt from `nba_stats.db` after the game with the frozen code; every input to game t is from before t (tested), so the content is point-in-time. Whether the previous night's tracking had arrived by noon is a separate matter (audit §3): the scorer flags each game `live_feasible` from the daily log (both teams' previous-game team tracking in the database after the sync of that morning, or older than the log) and reports a run on that subset; T3 (schedule only) needs no tracking. Games with no signal are counted and listed by reason.

### Primary statistic and decisions

- Per test: OLS slope of `actual - entry line` (home margin: `margin + entry_home_spread`; total: `total - entry_total`) on the frozen signal, with an intercept, day-cluster bootstrap (1,000 draws, `metrics.RNG_SEED`); one-sided p for slope > 0 = (1 + draws <= 0) / (1,001).
- **Two decisive looks per test**, each on the first N eligible games of that test in date order (so later games cannot change a look): **N = 2,460** (about two seasons of regular-season games) and **N = 4,920**. At each look the four one-sided p-values are Holm-adjusted and a test is **supported** iff its adjusted p <= **0.025** (so the two looks together spend 0.05). Looks at 600 and 1,230 games are reported and decide nothing.
- **Power, stated now** (bootstrap standard errors from the historical fits scaled to N: slope SE 0.51 / 0.49 / 0.47 / 0.62 for T1..T4 at 2,460 games and 0.36 / 0.35 / 0.33 / 0.44 at 4,920): if the true forward slope were 1, power at a look is roughly 50% at 2,460 and 80% at 4,920 before Holm; if it were 0.5, roughly 17% and 30%. **A "not supported" result is therefore not evidence of no effect.** The expected value of the test is mostly the slope estimate and its interval.
- Reported, deciding nothing: the slope against the closing line; the **closing-line-value readout** (OLS of the move from entry to close, in points, on the signal; positive = the market moved toward the signal); the all-season-fit signal; the `live_feasible` subset; playoffs included; the forward SD of the signal against the frozen one; whether 1 and the archive slope lie inside the forward interval; and a descriptive betting readout (bet the signal's side when |signal| >= the frozen 75th percentile, -110 on the entry line because entry prices for spreads and totals are not stored, flat one unit, with bets, hit rate excluding pushes, return per bet and mean closing-line value in points).
- **Reading rule.** Supported means the signal is a candidate to enter the bet rule or the model (for the model, only with the owner's go-ahead and a live fix for the tracking gap in audit §3). It is not a deployment. Not supported means not shown at that sample size. No criterion changes after the first look; anything else is exploratory.
- The freeze and scorer files are written and hashed now (before any forward game was scored); committing them is the owner's call, so the hashes are the only timestamp until then.

### E30 amendment 1 (2026-10-24, before any forward game was scored; procedural, no test, signal or decision rule changes)

After the freeze, `experiments/features_tracking.py` was edited in the experiment-harness audit: `trk_prev_opp_pace` (the F3 column) was the previous game's raw pace, which is mostly the team's own pace level and correlated 0.396 with `odds_total` (home + away); it is now the previous game's pace shock, that pace minus the team's own EWMA pace going into that game (correlation 0.025). Commit `f882c7c`. The file's SHA-256 changed from `fa1723c599b71671ec4db5b18cdbed17d362fd1c4cd495daf3b7301512283f93` (registered) to `2b7e1c29c41d6f68ef0ecd90bf8aca0d2b53de26307a8070e6bcb4f2258962a0`; reverting that one block to `g['pace'].shift(1)` reproduces the registered hash exactly. `e30_signals.py` and `core/travel_builder.py` are unchanged, and `frozen/params.json` is untouched (SHA-256 `5d8da6187481498092ea53c99886f4a559f0b2bc74bf7bdfb199c2b8d9d4ca7e`, as registered).
- Only F3 contains the column, so only **T2** can be affected; T1, T3 and T4 do not use it. T2's frozen coefficient on it is +0.0018 per SD (registered table), against 0.34 to -0.40 for its other regressors.
- **Measured effect (12,893 historical games, frozen T2 coefficients, old versus new column):** signal SD 0.5793 versus 0.5792 points; SD of the difference 0.001 points (maximum 0.004); correlation 1.000; the sign differs on 0.06% of games. A scorer dry run on the 18 pre-start games gives identical forward signal SDs for all four tests (T2 0.4372 both ways).
- **Decision:** the frozen parameters, the four tests, the looks and the Holm rule stand as registered. The signal is evaluated with the current code and the registered mean, SD and coefficients; no refit.
- **Scoring:** the scorer stops by default because the hashed file changed; it is run with `--allow-code-change`, which records `code_changed: ['experiments/features_tracking.py']` in every `results_<date>.json`. That is the intended reading of the registration ("pass `--allow-code-change` and report it"), not a deviation.
- Not an amendment, only noted: the re-run of the E28 screen under the corrected column (`reports/audit_fixes/rerun_new_defaults/COMPARISON.md`) no longer lists F3 margin as a candidate against the opening line and lists F4 margin only through the multiplicity correction. The E30 test list was fixed before that re-run and is not changed by it.

## E31 - does the feature selector help? All features versus selected (exploratory; registered 2026-10-24, before any E31 run)

Asked by the owner after the tracking program: the baseline keeps 20-29 of about 400 features (`_select_features_from_subset`: Spearman-cluster dedupe at |rho| > 0.85, then a null-importance noise filter, both on the first 64% of the training rows). Does removing that selection change out-of-sample accuracy? Exploratory: it decides nothing and changes no model.
- Reference: `E25_with_odds` (baseline form on the 10-19 training set: level models for ML, spread and total, opening odds in the pool, 7 folds 2019-20 .. 2025-26, in-fold 8-config grid, seed 42). Same harness, same folds, same games.
- Arms, via a pre-written `features.json` per fold: **E31_all** = the full target-appropriate pool (diff pool for ML and spread, separate home/away pool for the total) after the selector's own zero-variance filter on the selection rows; **E31_stage1** = the pool after the clustering step only (no null-importance filter).
- Check before reading results: the reference's selected list is a subset of the stage-1 list in every fold and target (it must be, by construction); if not, my copy of stage 1 is wrong and the arm is discarded.
- Reported: paired differences (arm - reference) with `metrics.paired` for ML log loss, spread MAE and total MAE (also MSE), per season, the number of features, run time. Noise floor for comparison: seed-only differences of about 0.003 log loss, 0.011 spread MAE, 0.005 total MAE (`noise_floor.md`). A difference inside the floor is "no detectable difference". No adoption rule; any follow-up would be registered separately.

## E31b - the E31 question under the post-audit pipeline (exploratory; registered 2026-10-24, before any E31b run)

E31 asked whether the feature selector helps, but it measured the pre-audit selector inside the pre-audit harness. The experiment-harness audit (`reports/audit_fixes/EXPERIMENT_HARNESS_AUDIT.md`) changed both: the selector (gain importance, bagged proxy fits averaged over 5 seeds, season-stratified shuffles, z-score rule, and for the market-aware form a residual target with the odds columns kept out of the proxy fits and appended afterwards) and the harness (calibration block capped at 1,300 rows, base models refit on model_train + calibration, Platt ML calibration). Whether E31's "no detectable difference" holds for the selector that would now be deployed is a separate question. Exploratory: it decides nothing and changes no model.
- Reference: `E25_with_odds` regenerated under those defaults into `reports/audit_fixes/rerun_new_defaults/reports/experiments/` (the saved `reports/experiments/E25_with_odds` is not touched or reused; `experiments/e31b_nosel.py` refuses a reference whose `split.json` / `selection.json` are not the post-audit ones). Same data (10-19 training set), folds (2019-20 .. 2025-26), games, tuning grid and seed 42.
- Arms, via a pre-written `features.json` per fold that the harness is told to reuse (`features_from`): **E31b_all** = the pool after the selector's zero-variance filter on the selection rows (first 64% of the fold's model_train rows), with the pool's `odds_*` columns; **E31b_stage1** = the clustering output only (no null-importance filter), with the `odds_*` columns. Both are cut from the selector's own intermediate lists (`_select_features_from_subset(stop_after='pool' | 'stage1')` through `select_market_features`), so no second implementation of Stage 1 exists.
- Check before reading results: in every fold and target, reference list is a subset of the stage-1 list, which is a subset of the all-features list. A violation aborts the arm (the lists are then not the selector's).
- Reported: paired differences (arm - reference) with `metrics.paired` for ML log loss, spread MAE and total MAE (also MSE), per season, feature counts, run time. The noise floors quoted by E31 (0.003 log loss, 0.011 spread MAE, 0.005 total MAE) were measured with the pre-audit selector; they are shown for scale only, and the post-audit selector's seed noise is not re-measured here. A difference inside them is "no detectable difference". No adoption rule; the result is compared with E31's only descriptively.
- Not stated in advance: a direction. E31 found nothing; this run is not expected to differ, and a difference would need its own registration before it supported any change.
