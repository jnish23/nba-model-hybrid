# Tracking and fatigue program: summary for the owner (steps 1-5)

Branch `tracking-fatigue` (cut from `odds-rerun`). Nothing deployed was touched: `DEPLOYED_SUFFIX`, `models/saved/`, predictions, scheduled tasks and `scripts/*.bat` are unchanged. Nothing is committed or pushed.
Registrations are in `reports/experiments/PREREGISTRATION.md` (E28-audit, E28, E29); their hashes were recorded before each computation in `registration.sha256` (the file itself is uncommitted, so that is the only timestamp).

## Bottom line

**No family earns a place in the team models.** The one registered candidate on the model residual (F1 on totals) came from pace persistence, not fatigue, and it failed the walk-forward test (E29: total MAE +0.014, MSE +0.91 with the 95% interval above zero). The mechanism check does not show that running load or speed changes shooting once the previous game's pace is controlled. Four opening-line candidates exist, but their edges are 0.4 to 0.75 points per standard deviation of signal, below what a bet needs for a typical game.

## Step 1: data audit (`audit.md`)

- Team and player tracking are complete for 2013-14 to 2025-26 (every game, both teams). Hustle is usable from 2016-17 (2015-16 mostly has one team per game). Team `speed` is NULL everywhere; player speed is filled for over 98% of played rows until 2024-25 (87% in 2025-26).
- **350 team-games in 2025-26 (13.2%, 2025-11-26 to 2026-04-12) have all-zero tracking rows**; they are read as missing. The plan's "2024-25 fill 24,964 of 34,788" is actually 2025-26.
- **Live availability is not reliable.** In the five logged runs, previous-night tracking was in the database for the 10-21, 10-22 and 10-23 runs (lag exactly one day) but **missing for all 12 games of 10-23 on the 10-24 run**, and the endpoint still returned nothing 14 hours later. No back-to-back has happened yet in 2026-27, so the plan's "share of back-to-back second nights with the previous night available at 9:15" could not be measured (0 cases). Hustle arrived for all 28 games. The lagged features therefore use "the most recent game with tracking data, at most 3 games back".
- Opening-line timing on back-to-backs could not be measured (no timestamps in `nba_odds_opening`; snapshots start 2026-10-19). `nba_odds_opening` starts in 2019-20; the training set carries archive opening lines for 2013-14 to 2018-19, used as extra find seasons for the opening-line screen.

## Step 2: feature library

`experiments/features_tracking.py` (19 team features in 5 families, 64 columns, appended to a copy of the training set; the 430 original columns and 16,795 rows are unchanged). `tests/team/test_tracking_features.py`: lookahead corruption, a hand-computed five-game example, the no-data path (NaN, never 0), the staleness cap and the merge check. A deliberate leak makes 5 of the 7 tests fail. Data-quality rules added before the screen (plausibility bounds, z clipped at 6, 7-day continuity guards) and one deviation (expanding rather than per-fold percentiles for `one_side_tired`) are in the E28 registration.

## Step 3: residual screen (`screen.md`, `screen.json`)

Find: (a) 2019-20..2021-22; (b) 2013-14..2021-22 for total and margin; confirmation 2022-23..2025-26. Joint Wald on day-cluster bootstrap, gamma = slope on the find-period signal, BH over 15 tests.

| family | (a) model residual: total / margin / win | (b) opening-line residual: total / margin / win |
|---|---|---|
| F1 running load | **Candidate** / reject / reject | reject / **Candidate** / reject |
| F2 speed | reject / reject / reject | reject / reject / reject |
| F3 physical previous game | reject / not confirmed (q 0.17) / not confirmed (q 0.36) | reject / **Candidate** / reject |
| F4 schedule density | reject / reject / reject | not confirmed / **Candidate** / not confirmed |
| F5 interactions | reject / reject / reject | **Candidate** / reject / reject |

- **Mechanism (c).** By the registered rule the fatigue story is "supported" for F1 and F2 (they move pace and, for F1, FT%). That result is not credible: the per-possession features and the pace outcome both contain the previous game's pace, and teams repeat their pace. With the previous game's pace and the team pace norm as controls (`explore.md` X2, exploratory), the F1 and F2 pace effects vanish (gamma -0.09 and -0.01), F1's FT% result is no longer significant, and neither moves 3P%. **The data do not show running load or speed changing shooting or pace.**
- **F1 total candidate.** With the same controls gamma falls from 0.96 to 0.06 (95% CI -1.8 to 1.8) (`explore.md` X1).
- **Opening-line candidates.** Slope of the confirmation residual on the find-period signal, per SD of the signal: F1 margin 0.49, F3 margin 0.52, F4 margin 0.42 points, F5 total 0.75 points (about 100-140% of the find-period size; F1 total was 66%). Against the closing line they keep about 90%, 90%, 60% and 70% of that (X3). Plain-unit coefficients are in `screen.md`; the clearest are F4 `games_last7` (-0.37 points of margin per SD against the line), F3 `prev_ot` (-0.3) and F5 `xload_last_x_dist` on totals (-0.46). The F4 and F5 signals are strongest in the six archive seasons (Wald p 0.01 and 0.02 there alone), whose lines come from a different source. A one-SD signal is below a bet's break-even (roughly 0.8 points on a spread and 1.1 on a total at -110), so these are not bets for a typical game; only the tails of the signal could approach it.
- Exploratory lead, not a candidate: F3 on totals fails the find step (p 0.25) but the confirmation fit is strong (Wald p < 0.001, mostly the previous game's pace); it needs a fresh registration and fresh data to be anything.

## Step 4: forced-inclusion test (E29, `reports/experiments/E29_*/RESULT.md`)

- `E29_F1_total` (registered): total MAE +0.01395 (95% CI -0.003, +0.032), 2 of 7 seasons better; MSE +0.910 (CI +0.26, +1.61). The F1 columns take under 4% of the total model's gain in every fold. **Reject.**
- Exploratory arms: previous-game pace alone +0.011 MAE (1/7); F1 plus pace +0.018 (1/7); F1 added to the pool with fresh selection -0.014 MAE (3/7, inside the selection-noise range) and the selector never picked an F1 column. ML and spread reproduce the reference exactly in the forced arms.

## Step 5: combine

Nothing to combine: no family is Adopt or Candidate for a model feature, so no feature list is proposed.

## Not done / not tested

- Step 6 (production) needs your go-ahead and now has nothing to promote. Step 7 (props) was not started; it is a separate 2-3 day track and its fatigue premise looks weak after the mechanism check. Step 8 was done afterwards: E30 (registration in `PREREGISTRATION.md`, frozen signals and scorer in `reports/experiments/E30_forward_tracking/`), forward games from 2026-10-25.
- Not tested: ML and spread as walk-forward targets (no (a) candidates there), the 2026-27 season, hustle for 2013-15, playoffs as a main sample (sensitivity only, in `screen.md`), and back-to-back timing of the lines.
- Caveats: the screen's find periods are short for (a) (3 seasons); F3's `prev_hustle_z` has 23% missing (pre-2016), which shortens F3's samples; the `one_side_tired` percentiles deviate from the plan as registered.

## Reproduce

`python -m experiments.e28_audit`, `python -m experiments.e28_screen` (needs `data/processed/experiments/e28_training_set.csv` and `e28_mechanism.csv`, built by `features_tracking.build_variant` and `mechanism_targets`), `python -m experiments.e28_explore`, `python -m experiments.e29_run`.

## Erratum (2026-10-24, found while preparing step 8)

The first version of this summary and the E28 registration described `gamma` as "1 = replicated at full size" and multiplied it by the signal SD to get "expected edge per SD". That was wrong: `gamma` is already points per SD of the signal (full replication equals `signal_sd_find_units`). The tier rules use only the sign and the p-value of gamma, so no tier changes; the effect sizes in the first version were understated by a factor of about 1.1 to 2.3. Corrected above.
