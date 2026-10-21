# Phase 3 gate report: player-stat models

Date 2026-10-19. Branch `props-phase0-core-refactor` (worktree `nba-model-hybrid-props`). Registry version **`v2_base`** (`data/props_models/v2_base/`, immutable). Details of the run: [phase3_training_report_v2_base.md](phase3_training_report_v2_base.md); plan and decisions: [phase3_plan.md](phase3_plan.md); settlement rules: [void_rules.md](void_rules.md).

**Status: the models are built, trained walk-forward on the three development seasons (2021-22, 2022-23, 2023-24) and pass the smoke checks. This is NOT the Phase 4 evaluation: no confidence intervals, no ablations, no market comparison. 2024-25 and 2025-26 have not been touched.**

## 1. What exists

| Stage | What it does | Module |
|---|---|---|
| A. Availability | LightGBM P(plays) for every listed player at T | `props/models/availability.py` |
| B. Minutes | classifier P(limited minutes), then per regime a LightGBM mean and spread plus an empirical standardized-residual bank; the distribution is 2 x 64 weighted support points (decision D5: two regimes) | `minutes.py` |
| C. Rates | per-minute rates for attempts (2pt, 3pt, FT), rebounds, assists; make probabilities for 2pt, 3pt, FT. Two families per component, LightGBM and hierarchical (Poisson/logistic around a fixed shrunk-rate backbone); the family is chosen per component on the dev folds (D6) | `rates.py` |
| D/E. Outcome | exact minutes-mixture negative-binomial pmfs for rebounds and assists; joint Monte Carlo (4,000 sims, shared minutes) for points, threes and the combos (D7) | `outcome.py` |
| F0. Dispersion | NEW this session: mean-preserving widening of each pmf by a factor fitted on the calibration window, per stat and predicted-minutes bucket | `dispersion.py` |
| F. Isotonic threshold calibration | built, fitted, reported, **not applied** (see 3.3) | `calibrate.py` |
| Folds, registry | expanding windows, fit = first 80% of training dates, calibration = last 20%, bubble games excluded from training only (D8); immutable versions, config log | `folds.py`, `registry.py`, `common.py` |

Features come only from the Phase 2 dataset (`FeatureSpec('base')`: no market-environment columns, no actual starters, nothing after T). The `with_env` variant exists but has not been trained; it belongs to the H1/H2 tests.

Rate families chosen on the dev folds: hierarchical for 2pt attempts, FT attempts, rebounds and all three make probabilities; LightGBM for 3pt attempts and assists. The two families are within 0.1-0.3% of each other everywhere except 3pt attempts, where LightGBM is 3% better.

## 2. Tests

Phase 3 adds 22 tests (18 for data, folds, stages and outcomes, 4 for the dispersion stage). The full suite (team, core and props): **209 passed**. The ones that matter for the rules:

- **No label leakage:** predictions are identical whether or not the test season's outcomes are scrambled (fit, calibrate and predict run twice).
- **Fold integrity:** fit and calibration windows end before the test season; bubble games are excluded from training only; calibration is strictly later than fit.
- **Determinism:** two fits give identical components.
- **Distribution validity:** every pmf sums to 1; the joint simulation reproduces the single-stat marginals; push handling on integer lines (a push is not an over).
- **Recovery on a synthetic league** with known minutes and rates (both families), the two-regime minutes structure, and the calibrator and widening repairing known mis-specification.
- **Registry:** versions are immutable; every configuration tried is appended to `data/props_models/config_log.jsonl`.



## 3. Results on the development folds (smoke check, not a confidence-interval evaluation)

### 3.1 Full-distribution accuracy against the simple baselines

CRPS (lower is better), mean over the three dev folds, all three decision times pooled, played rows. The baselines are rolling-average rate x expected minutes with a Poisson or a discretized Normal; the Normal's single sd is fitted on the test fold itself, which favours the baseline.

| stat | model | best baseline | improvement |
|---|---|---|---|
| points | 3.157 | 3.318 (Normal) | 4.9% |
| rebounds | 1.298 | 1.364 (Poisson) | 4.8% |
| assists | 0.874 | 0.911 (Poisson) | 4.1% |
| threes | 0.5595 | 0.573 (Poisson) | 2.3% |
| pra | 4.174 | 4.428 (Normal) | 5.7% |
| pr / pa / ra | 3.839 / 3.470 / 1.775 | 4.053 / 3.678 / 1.880 | 5.3% / 5.7% / 5.6% |

The model beats the best baseline on every stat in every fold. That clears the Phase 4 entry bar ("do not continue if the model fails to beat simple baselines"), subject to the real evaluation with intervals.

### 3.2 What the diagnostics found, and what was changed

1. **The raw pmfs were slightly too narrow.** Variance ratio (observed squared error / predicted variance) was 1.02-1.10 for single stats and 1.09-1.17 for the combos, with PIT tails of 0.13/0.11 where 0.10 is right. It was concentrated in bench players (minutes under 15: ratios 1.2-1.3) and larger for the combos. Cause: the stats share game-flow noise (the standardized residuals of points, rebounds and assists correlate at 0.21-0.35 even after the shared minutes), which the independent-given-minutes simulation does not carry.
2. **Fix: dispersion calibration (F0).** Fitted on the calibration window only, evaluated out of sample on each test season. Result (mean over folds): combo variance ratio 1.13 to 0.97, PIT tails 0.132/0.108 to 0.101/0.092, CRPS and over-probability log loss better in 3 of 3 folds for points, threes and all combos and in 2 of 3 for rebounds and assists. The gains in score are small (0.1-0.2% CRPS) and the tail fix is the point. The factors are slightly generous (post-fix variance ratios 0.95-0.98, so a touch too wide).
3. **Marginals only, not joints.** The stored pmfs are right per stat. They are NOT a joint model: the correlation between a player's stats (about 0.2-0.35) is not in the stored pmfs. Phase 5 needs it for correlated-bet sizing and will use the empirical residual correlations; same-game-parlay pricing is a stated non-goal.

### 3.3 Calibration of P(X > line)

Measured on a ladder of lines (floor(mean) - 2 + 0.5 up to floor(mean) + 2 + 0.5, like a book ladder) around each row's own mean. After widening, the over-probabilities are close to calibrated without any further map: ECE 0.003-0.020 on 2021-22 and 2022-23. **The isotonic layer made log loss worse in 19 of 24 stat-fold cells** (by up to 0.004) with mixed ECE changes, so it is not applied. 2023-24 is the exception: ECE is 0.03 for points and the combos, which is the bias below, not a shape problem.

### 3.4 Known bias: over-prediction in 2023-24

Mean points are over-predicted by 5.0% in 2023-24 (0.3% in 2022-23, 1.8% in 2021-22); threes by 4.4%, assists by 1.2%, rebounds by 1.6%. By projected minutes (points): under 15 min -11.2%, 15-28 min -3.2%, 28+ min -5.1%. Two different things are mixed:

- **Bench (under 15 min):** a minutes problem. They are given about 0.8 minutes too many (about 8% of their minutes).
- **Starters (28+ min):** minutes are nearly right (0.13 min too many) but points per minute are about 4-5% too high, and threes 3-4%. A scoring-rate gap that the shrunk-rate features did not anticipate.

The bias is similar at T-240, T-60 and T-5, so it is not an information-timing effect. No calibration step fitted on earlier data can see a drift that appears in the test season. I have not diagnosed the cause (candidates: a league-wide scoring or pace change, a rotation change, a feature that drifts) because diagnosing it on 2023-24 and then tuning would be tuning on a dev fold with only one fold showing the effect. Options, to decide with you: measure it again on 2024-25 first (recommended), then consider an in-season rolling recalibration of the mean using only completed games (point-in-time), tested in Phase 4. It matters for betting: a 5% bias in the mean shifts every over/under probability the same way and would show up as a one-sided edge that is not real.

## 4. Things you should know

- **Bug found and fixed in the team pipeline's shared code (needs your call on timing).** `core/odds_clean.py::_closing_from_snapshots` called `DataFrame.setdefault`, which does not exist, so `load_clean_odds` crashes as soon as the `odds_snapshots` table holds any pre-game row. It surfaced because the live snapshot data for 2026-10-20 now exists and a Phase 2 test reads the real database. The bug is in the pre-refactor `models/odds_clean.py` too, so **the main checkout has it**. Fix (4 lines plus a regression test, `tests/core/test_odds_clean_snapshots.py`) is committed in the worktree branch only. It does not alter any output where the table is empty. I did not touch main; say if you want it cherry-picked there now.
- **Configurations tried (multiple-testing count):** `config_log.jsonl` has 125 lines. They are two identical full runs (`v1_base`, then `v2_base`, 61 model configurations each: 5 availability, 8 minutes, 48 hierarchical-rate fits) plus 3 dispersion fits. Distinct choices made on dev folds: rate family per component (8, two options each), widening on/off, isotonic on/off. No hyperparameter search was run; settings were set by judgement and fixed (`DEFAULT_PARAMS`).
- **Pooled decision times:** the headline numbers pool T-240, T-60 and T-5. The diagnostics show the three behave alike (CRPS and bias within a few percent), but the per-T split with intervals belongs to Phase 4.
- **Settlement:** `void_rules.md` records what is and is not verified. Only one book's rule text (Fanatics) was readable; the rest is from secondary sources. Phase 5 settles every bet under two bounding treatments, and I still need you to check each book's in-app NBA prop rules for your state.
- **Not done on purpose:** no `with_env` model (needed for H1/H2), no ablations, no intervals, nothing on 2024-25 or 2025-26, no feature changes.

## 5. Recommendation and what I need from you

Go to Phase 4 (evaluation against outcomes on 2024-25 as the validation season, with bootstrap intervals by game, the per-T split, ablations of the feature groups, and the bench-minutes question). Decisions for you:

1. **Approve the Phase 3 gate**, including keeping the dispersion widening and dropping the isotonic layer (both decided on the dev folds only).
2. **Cherry-pick the `odds_clean` fix to main now,** or leave it with the branch until Phase 1-3 merge? (I recommend now: it is a crash on live data in the team pipeline.)
3. The 2023-24 over-prediction (3.4): measure it on 2024-25 first, then decide on a rolling recalibration (my recommendation), or test the recalibration in Phase 4 straight away.
4. Still outstanding from Phase 1: confirm the live collector on Oct 20/21 (`scripts/check_props_collection.py --date 2026-10-20`).
