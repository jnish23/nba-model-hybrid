# Phase 3 plan: models

Status: plan for your decisions (section 4). Nothing built yet. Phase 3 is offline and uses only the Phase 2 dataset, so the Oct 20/21 live-collector check
(still pending, first games are tomorrow at noon PT) does not block it. Evidence below comes from `scripts/props_phase3_eda.py`, which reads seasons up to 2023-24
only (never the 2024-25 validation season or the 2025-26 holdout) and fits nothing but variance summaries.

## 1. What the data says about the shape of the problem

Played games with expected minutes of at least 10, seasons 2019-20 to 2023-24 (about 112K player-games):

1. **Minutes are the biggest source of uncertainty and have a heavy left tail.** Realized minutes divided by the trailing expectation: for 28+ minute players the 5th percentile is
   0.69 and the 1st is 0.45; for 10-20 minute players the 10th percentile is 0.28. **8.9% of played games have under half the expected minutes**, 4.5% under 30%. Using only the
   trailing mean, the minutes error has a standard deviation of 7.0 minutes.
2. **Most of the overdispersion in counts is the minutes, not the rate.** Residual variance divided by the mean, with rate x expected minutes versus rate x actual minutes:
   points 3.39 vs 2.31; rebounds 1.53 vs 1.10; assists 1.37 vs 1.13; threes 1.23 vs 1.06. Given actual minutes, rebounds, assists and threes are close to Poisson
   (negative-binomial k of 57, 24 and 12 against 10, 10 and 5 without the minutes). Points stay over-dispersed (2.31) because they come in 1s, 2s and 3s.
3. **The joint structure of the combos is mostly shared minutes.** Residual correlations with expected minutes: points-rebounds 0.37, points-assists 0.25, rebounds-assists 0.26.
   With actual minutes they fall to 0.07, 0.006 and 0.06. Points-threes stays at 0.61 (a three is part of the points). So a joint simulation with shared minutes captures nearly all of it.
4. **Many zeros:** 41.5% of played games have no threes, 22% no assists, 9% no rebounds or points (rotation players), so a continuous distribution would misprice the low lines.
5. **Blowouts barely move minutes on average** (about -3% in the highest-probability quartile for starters); the effect is probably role-specific, which the minutes model can learn.

## 2. Architecture (the spec's hierarchy)

| Stage | What it predicts | Training rows |
|---|---|---|
| A. Availability | P(play) at each decision time T, from the status, report age, role, teammates | all listed rows |
| B. Minutes | distribution of minutes given play, including a separate probability of limited minutes or an early exit | played rows |
| C. Rates | per-minute components: points via field-goal, three-point and free-throw attempts and percentages; rebound chances and conversion; assists (creation and conversion) | played rows, exposure = actual minutes |
| D. Outcome | the full discrete distribution of each stat, from B and C (mixture over minutes, then counts); benchmarks: negative-binomial regression, LightGBM distributional | - |
| E. Joint | points, rebounds, assists simulated together with shared minutes and a small residual copula; teammate correlations estimated for sizing | played rows |
| F. Calibration | recalibration of P(X > k) on a time-ordered holdout, by stat and by line region | last part of each training window |

Model inputs are the Phase 2 features only. **Player prop lines are never inputs** (Phase 5 asks whether the model adds anything beyond them). Two feature sets run through the same
code: `base` (no market environment) and `with_env` (adds spread, total and blowout probability); the other ablations the spec lists (teammates, opponent, process-stat rates) are
group switches taken from the feature dictionary, so Phase 4 can run them without new code.

**Void rules** (spec item): I will document, per book (DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Bovada, BetOnline, Fanatics), how a prop is settled when the player does not
play or plays very few minutes, from each book's published rules, with the source and date. The target everywhere is "outcome given the player plays".

## 3. Walk-forward protocol (see decision D8)

Training only on past seasons, testing on the next; no random splits; hyperparameters chosen inside each training window by a time-ordered split (never on the test season);
every configuration tried is logged and counted (for the multiple-testing correction); seeds fixed and recorded.

## 4. Decisions I need from you

**D5. Minutes model.** The spec asks for the early-exit probability to be modelled separately.
 (a) **Two regimes** (recommended): a classifier for "limited minutes" (defined as under 50% of the trailing expectation; 8.9% of played games), and for each regime a LightGBM mean and spread
 with an empirical bank of standardized residuals to sample from. Matches the spec, and the threshold is stated in the open.
 (b) Single LightGBM mean and spread plus the residual bank, no separate regime (simpler; the heavy left tail is carried by the bank). I would build it anyway as an ablation.
 (c) Direct multi-quantile LightGBM with interpolation.

**D6. Rate models.** The spec asks for LightGBM against a hierarchical shrinkage model.
 (a) **Build both and choose per stat by walk-forward performance** (recommended). LightGBM: Poisson objective with log-minutes as the exposure offset. Hierarchical: a regularized log-linear
 model whose backbone is the shrunk player rate already in the features, plus opponent, environment and availability multipliers. Points are modelled as components (attempts and percentages) so their discreteness is right; rebounds and assists directly.
 (b) LightGBM only. (c) Hierarchical only (most interpretable, probably less accurate).

**D7. Outcome distribution.** (a) **Minutes mixture with Poisson/negative-binomial counts as the primary** (recommended): single stats computed as an exact weighted mixture over sampled minutes (no simulation noise), points
 by simulating the components, combos by joint simulation with shared minutes; the negative-binomial regression and a LightGBM distributional model are benchmarks, and Phase 4 decides on CRPS. (b) Direct count model as the primary and the simulation as the benchmark.

**D8. Folds and the holdout.**
 (a) **Expanding windows, one fold per test season** (recommended): test 2021-22 (train 2019-21), 2022-23, 2023-24 for development; 2024-25 as validation; 2025-26 only after the configuration is frozen. Calibration set = the last 20% of each training window by date.
 (b) Rolling 3-season windows (smaller training sets, less drift).
 (c) Also train the minutes and rate models on 2015-16 to 2018-19 using base features only (no injury or market features exist there), for about four more seasons; but the injury and teammate features, the core of H1, would be absent in those rows.
 Default I will apply either way: exclude the 2019-20 bubble games (from 2020-07-30) from training, not from testing; 2020-21 stays (a short season, flagged).

## 5. Deliverables and gate

Code in `props/models/`; artifacts in a props-only registry (`models/props_saved/<version>/`, never mixed with the team registry); walk-forward predictive distributions for every development and validation row; calibration maps;
the void-rule document; tests (a training window always ends before its test window, determinism by seed, probabilities sum to one, joint simulation marginals match the single-stat ones, no label or prop-line column among inputs); a short training report with a smoke
comparison against a rolling-average baseline on the development folds. The formal evaluation (CRPS, log loss, calibration, ablations with confidence intervals) is Phase 4, and I will not look at 2025-26 before the gate that freezes the configuration.
