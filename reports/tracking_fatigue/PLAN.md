# Tracking and fatigue features: test plan (for an executing agent)

Written 2026-10-24 for another LLM to follow step by step. **Nothing in this plan has been run.** The owner decided on 2026-10-24 to hold off on edge shrinkage, so this plan does not use it.

**Goal.** Find out whether running load, speed, physical previous games and schedule density, built from tracking data the database already has, improve the team models (spread, total, moneyline) and the props model. Adopt only what passes; record everything else.

## 0. Ground rules (read before anything else)

1. **Python.** Always `.venv\Scripts\python.exe` (3.12). Plain `python` is 3.9 and has no dependencies.
2. **Never change what is deployed.** Do not edit `DEPLOYED_SUFFIX` in `team/model_registry.py`. Do not retrain or overwrite any existing model set in `models/saved/`, regenerate `predictions/predictions_2025-26.csv`, or change scheduled tasks or `scripts/*.bat`. Promotion is the owner's decision (step 6).
3. **Never commit or push** unless the owner asks, and then only with explicit paths. Never run `export_dashboard_data.py --publish`.
4. **Register before you compute.** Each test's hypothesis, features, data and decision rule go in `reports/experiments/PREREGISTRATION.md` as a new section *before* any result for that test is computed.
   - Inspecting row counts, column names and coverage is allowed beforehand; looking at relationships with outcomes is not.
   - Use the next unused E number. E27 is reserved for the E18 totals-calibration test in `reports/upgrade_2026-27/PLAN.md`, so this program starts at **E28**.
5. **No lookahead.** Every feature for game *t* uses only games strictly before *t*. Data published later on game day (injury news, lines) is also off limits unless it's known at the feature cutoff (noon ET).
   - Each new feature module gets a corruption test modelled on `tests/team/test_experiment_lookahead.py`: overwrite all rows from game *t* onward with garbage and assert that features at *t* are unchanged.
6. **Prototype in `experiments/`, not in production code.** New features are built by a variant builder that appends columns to a copy of the training set, as `experiments/features_e05.py` does with `build_variant(csv, out, db)`. Production code changes only in step 6.
7. **Report honestly.** Report nulls and failures with their numbers. Don't change a decision rule after seeing results; anything outside the registered rule is labelled exploratory.

## 1. What already exists (don't rebuild it)

| thing | where |
|---|---|
| Training set (430 columns, 2013-14 onward) | `data/processed/training_set_v1_reduced_2026-10-19.csv`; `team.data_paths.latest_training_set()` |
| Existing schedule columns | `days_rest_*`, `is_b2b_*`, `dist_miles_*`, `tz_delta_*`, `road_trip_count_*`, `game_altitude_diff_*` (home/away + `diff_`); built by `team/game_features_v2.py` and `core/travel_builder.py` |
| Used by the deployed models today | totals: `days_rest_home`, `days_rest_away`, `tz_delta_home`; spread and ML: none of the schedule columns |
| Game results, box score, pace | `team_game_full` in `data/nba_stats.db` (one row per team-game; `minutes`, `pace`, shooting splits) |
| Team tracking | `team_track_stats_v3` (2013-14 to now, about 2,640 rows per season). **`distance` is filled; `speed` is NULL in every row.** Team speed must be built from players. |
| Player tracking | `player_track_stats_v3` (434k rows; `minutes` is a "MM:SS" string; `speed` in mph, `distance` in miles). About 18% of rows are DNPs with 0/NULL; 2024-25 has a lower fill share (24,964 of 34,788), so check it. |
| Hustle | `team_hustle_stats` / `player_hustle_stats` (full from 2015-16; 2013-14 and 2014-15 are fragments): deflections, loose balls, box-outs, contested shots, charges, screen assists |
| Arenas (altitude) | `nba_arenas` |
| Walk-forward harness | `experiments/lib.py` `run_walk_forward(name, csv, features_from, hooks, ...)`. Per-fold feature selection is cached in `reports/experiments/<name>/features.json` and reused when that file exists. Folds: test seasons 2019-20..2025-26. |
| Feature-experiment runner | `experiments/feature_exp.py` `run_feature_experiment` (E5-E8 pattern) |
| Metrics, bootstrap, decisions | `experiments/metrics.py` (`paired`, `core_metrics`, day-cluster bootstrap, `RNG_SEED = 20261017`), `experiments/decide.py` (`mc` = at least 5/7 seasons better and pooled CI excludes 0; `nh` = no-harm) |
| Deployed-form reference runs (10-19 training set) | `reports/experiments/E25_with_odds` (level models: total and ML raw), `E25_residual` (E1 spread form). The ML stack (E3) is recomputed from raw ML and the E1 margin as in `experiments/e13_combined.py`. |
| Selection-noise floor | `reports/experiments/noise_floor.md`: changing only the selection seed moves ML log loss by up to about 0.003 and MAE by about 0.01, with no CI excluding 0 |
| Power and strictness | `reports/experiments/POWER_CHECK/README.md` |
| E23 (favourite on a back-to-back) | found nothing; don't retest it as is |
| Props model | `props/` (pipeline `props/models/pipeline.py`, feature rows `props/features/rows.py`, which already uses touches, passes and rebound chances but not distance or speed). Rules and holdouts in `reports/props/phase4_plan.md`: **2025-26 is untouched; 2024-25 is validation and needs `--families-from`**. Prop lines are in `data/props_odds.db` (`prop_snapshots`), recent only. |

## 2. Decision tiers (use these for every test in this plan)

| tier | rule | what happens |
|---|---|---|
| **Reject** | fails the find step, or the confirmation estimate has the opposite sign | drop it; report it |
| **Candidate** | passes the find step, confirmation in the same direction, and Benjamini-Hochberg q < 0.10 across all families tested in that step | goes on to the next step (as a model feature, or a forward paper test) |
| **Adopt** | the strict rule: `decide.mc` (at least 5/7 seasons better, pooled day-bootstrap CI excludes 0) and `decide.nh` on the other metrics; or a candidate that passes its forward test | eligible for production (step 6), with the owner's go-ahead |

The find step means discovery seasons, and the confirmation step means later seasons; the exact split is given per step. Use 1,000 day-cluster bootstrap draws with `metrics.RNG_SEED`.

## 3. Step-by-step

### Step 1: data audit (registered as E28-audit; descriptive, nothing passes or fails; about half a day)

Write `experiments/e28_audit.py` producing `reports/experiments/E28_tracking/audit.md`. Check:
1. **Coverage.** Rows per season for team and player tracking and hustle; games with both teams present; playoff (`004`) and play-in (`005`) shares; the 2024-25 player fill share (above); the 2019-20 bubble and 2020-21 season.
2. **Units and sanity.**
   - Parse `minutes` "MM:SS" into float.
   - Team distance per game, mean and SD by season, should be about 16-18 miles.
   - Player speed should be about 3.8-4.6 mph.
   - Player distances summed per team should match team `distance`, within 1%.
   - Flag outliers (distance 0 with minutes > 0, speed > 6).
3. **Live availability (decisive for usefulness).** Is last night's tracking data in `nba_stats.db` by the 9:15 ET daily run?
   - Check `data/raw/daily_inference.log` and the row insert times, or compare game dates with what existed at 9:15 over the last several days.
   - If it is not reliably there, every tracking feature must use the previous game *with data*, and lagged features (e.g. a back-to-back's previous night) lose value. Report the share of back-to-back second nights where the previous night's tracking was available by 9:15.
4. **Opening-line timing on back-to-backs.** Report, for second nights of back-to-backs, whether the opening line can predate the end of the previous game (`nba_odds_opening` has no timestamps; `odds_snapshots` exists only since 2026-10-19, so measure what you can and say what you couldn't).
5. **Opening-line coverage before 2019-20** in `nba_odds_opening`. If lines exist for 2013-14..2018-19, the market-residual screen in step 3 can use them as extra discovery seasons.

**Stop rule.** If tracking data for the previous night is usually missing at 9:15 and can't be fetched earlier, record that. The lagged-load families (F1, F2, F3) then use "most recent game with data", and the report must say so.

### Step 2: build the feature library (about 1-2 days)

Write `experiments/features_tracking.py` with `build_variant(csv, out, db)`, which appends the columns below to a copy of the training set. Naming: `trk_<name>_home`, `trk_<name>_away`, `diff_trk_<name>` (home minus away), following the existing convention.

Shared definitions:
- Every rolling statistic is over the team's (or player's) **prior** games only (`groupby(...).shift(1)` after the rolling or EWMA, as in `features_e05._pre_game_ewma`).
- "Norm" means the EWMA with span 20 of prior games; the SD is a 20-game rolling SD of prior games, floored at a small value.
- `z(x) = (x - norm) / sd`.
- Team minutes = the sum of player minutes (about 240; more with overtime). Possessions = `pace x team_minutes / 240`, from `team_game_full`.
- Rotation = players with at least 10 minutes in the team's previous game.

Families (each family is tested as a group):

| family | features (all from games before *t*) |
|---|---|
| **F1 excess running load** | `dist_per_min_last` = last game's team distance / team minutes; `dist_per_poss_last` = distance / possessions; `xload_last` = z of `dist_per_poss_last`; `xload_4n` = sum of `xload` over the team's games in the 4 nights before *t*; `rot_xload_last` = minutes-weighted mean, over the rotation, of each player's z(distance per minute) last game |
| **F2 speed drop** | `rot_speed_z_last` = minutes-weighted mean of player z(speed) last game; `rot_speed_trend5` = slope of that series over the last 5 games; `star_speed_z_last` = the same for the team's top-2 players by minutes |
| **F3 physical previous game** | `prev_ot` (team minutes > 240); `prev_close` (\|margin\| <= 5); `prev_n40` (players at 40+ minutes); `prev_hustle_z` = z of deflections + loose balls + box-outs + contested shots (NaN before 2015-16); `prev_opp_pace` = the previous opponent's pace in that game |
| **F4 schedule density** (no tracking needed) | `g3in4`, `g4in6`, `games_last7`; `trip_last_game` (road game whose next game is at home; schedules are known in advance, so this is allowed); `homecoming` (first home game after a road trip of 3+ games); `prev_altitude` (previous game at DEN or UTA and today's isn't) |
| **F5 interactions and asymmetry** | `xload_4n x is_b2b`, `xload_last x dist_miles` (scaled per 1,000 miles); `one_side_tired` = (a home-tired score above its 80th percentile and away below its 50th) minus the reverse, where the tired score = `xload_4n` + `prev_ot` + `is_b2b`, with the percentiles fitted on each fold's training rows only |

Mechanism targets to attach for step 3 (realised in game *t*; outcomes, never features): team 3P%, FT%, rim FG%, pace, and points per possession, from `team_game_full`.

Tests (`tests/team/test_tracking_features.py`):
- the lookahead corruption test;
- a hand-computed example for one team over 5 games;
- a no-data path (a team with no prior tracking gives NaN, not 0);
- the feature columns don't change the training set's existing columns or row count.

### Step 3: residual screen (registered as E28; cheap, no model retraining; about 1 day)

Purpose: find out quickly which families carry information the deployed models and the market miss, without the feature-selection noise that sank E5-E8.

Write `experiments/e28_screen.py`. For each family F1-F5 and each target:
- **(a) Model residual:** actual minus the deployed-form out-of-sample prediction.
  - total: the `E25_with_odds` total;
  - spread: the `E25_residual` spread;
  - ML: the E13 stack, on the log-odds scale (logistic regression with the prediction as offset).
- **(b) Market residual:** actual minus the opening line, as in E14-E24.
- **(c) Mechanism:** the team 3P%, FT%, pace and points per possession of game *t*.

Method:
- Regression of the residual on the family's features plus season dummies; for ML, logistic regression with offset.
- The test statistic is a joint Wald test, using a day-cluster bootstrap covariance.
- **Find step:** 2019-20..2021-22 (add 2013-14..2018-19 for (b) and (c) if step 1 found lines there).
- **Confirmation:** 2022-23..2025-26.
- BH across the 5 families x 3 targets (total, margin, home win) for (a), separately for (b).
- Regular season only for the main test; playoffs as a sensitivity check.
- Also report the sign and size of each coefficient in plain units (points per 1 SD of load).
- Mechanism first: if F1 and F2 don't move 3P%, FT% or pace in (c), the fatigue story for them is weak, and the report says so whatever (a) shows.

Output: `reports/experiments/E28_tracking/screen.md` and `screen.json`, with a tier per family x target.

### Step 4: feature test in walk-forward, forced inclusion (registered as E29; about 1-2 days of compute)

Only families that are candidates in step 3 (a), on that market. Don't add every family to every market.

Design: fresh feature selection would add noise as large as the effect (`noise_floor.md`). Instead, **force the family in**: each fold's features = the reference run's selected features for that fold + the family's columns.
- Implement by writing `reports/experiments/E29_<family>_<market>/features.json` before calling `run_walk_forward`, so the harness reuses it.
- Rows must be identical to the reference. Build the variant CSV from the same training set and confirm that game IDs per fold match the reference run.
- Run each market in its deployed form:
  - total: level model, no hooks (reference `E25_with_odds`);
  - spread: E1 hooks as in `experiments/e25_regen.py` (reference `E25_residual`);
  - ML: the raw run, then recompute the E3 stack as `e13_combined.py` does, and compare stacked log loss.
- Compare with `metrics.paired` against the reference on the same games. Report per season, pooled, the noise-floor comparison, and the number of folds in which LightGBM gave the family non-trivial gain.
- Decision: **Adopt** if `decide.mc` on the market's main metric plus `decide.nh` on the others. **Candidate** if the pooled improvement exceeds the noise floor (larger than the biggest seed-only difference) and at least 4/7 seasons are better. Otherwise **Reject**.
- Secondary (exploratory): the same family added to the *pool* with fresh selection, to see if the selector picks it up by itself.

### Step 5: combine (about half a day)

Put all Adopt families (and Candidates, labelled as such) in one variant per market, and run them forced in together. Check that gains roughly add up, and that ML log loss, spread MAE and total MAE show no harm (`decide.nh`). The output is the proposed feature list per market. **Stop here and report to the owner.**

### Step 6: production (only with the owner's go-ahead)

1. Move the adopted feature code into the production feature build (`team/game_features_v2.py` or a new module it calls), keeping `experiments/features_tracking.py` as the reference implementation.
2. **Parity test.** For a past date, the live inference path (`scripts/run_inference_v2.py`) must produce exactly the training-set values for the adopted columns.
3. Live fallback: when last night's tracking data is missing at 9:15, use the rule step 1 settled. Training rows must be built with the same rule, so training and live match.
4. Regenerate a training set with `scripts/generate_training_set.py`. Train a candidate set with `scripts/retrain_candidate.py`, which uses a new suffix and gate report and never overwrites.
5. Run a smoke test on today's slate without writing to the live log. `run_inference_v2.py --suffix` currently writes to `predictions_log_v2`, so add a `--no-log` flag first.
6. The owner promotes by changing `DEPLOYED_SUFFIX` in a reviewed commit.
   - **Warn the owner first:** promotion changes `model_set_hash`, which ends the E26 (a) forward tests for the old set (see `reports/experiments/E26_forward_shrink/README.md`).
   - E26's B-total test and E15, E20, E22 and E24 depend only on market lines and are not affected (confirm when doing this).
   - After promoting, freeze the model for the season.

### Step 7: props track (separate, after or alongside steps 3-4)

Read `reports/props/phase4_plan.md` and `props/models/pipeline.py` first, and follow the props rules: 2025-26 is never used, and 2024-25 only with `--families-from`.
- **Screen**, as in step 3 but per player: does (actual - predicted) for minutes, points and threes depend on:
  - the player's `xload_last`, `xload_4n` and speed z (from `player_track_stats_v3`);
  - `prev_min_42plus x is_b2b` (star-minutes management);
  - F3 and F4 at team level?
  Use the out-of-sample predictions from the props development folds.
- **Leg-fatigue check:** does the effect on 3P% and FT% exceed the effect on rim FG%? Use the player's shot-location data (`player_shot_locations_boxscores`) for rim attempts.
- **Feature test:** add candidate features to `props/features/rows.py` in a variant spec, retrain the development folds, and compare CRPS / log loss per stat.
- **Betting:** prop lines exist only recently (`data/props_odds.db`), so any betting claim is a forward test, registered like E26, scored on lines snapshotted after the previous night's box score.

### Step 8: betting angles (optional; only for families that are candidates on the market residual)

A candidate in step 3 (b) is a possible bet against the line, not just a model feature. Register a forward test like E26: score against the noon entry line from `odds_snapshots` (`core.odds_clean.load_entry_odds`), and use only lines posted after the previous night's box score was available (step 1). Report CLV (`metrics.clv_points`) as an early readout.

## 4. Order, effort and stop points

| step | effort | stop or go |
|---|---|---|
| 1 audit | half a day | stop if the data can't be used; report |
| 2 features | 1-2 days | tests must pass |
| 3 screen | 1 day | only candidates go on; **report results to the owner** |
| 4 forced-inclusion test | 1-2 days | |
| 5 combine | half a day | **stop and report; no production change without the owner** |
| 6 production | 1-2 days | owner's go-ahead; warn about E26 |
| 7 props | 2-3 days | can run in parallel after step 2 |
| 8 betting forward tests | registration only | forward data accumulates |

## 5. Expectations, stated before anything runs

- Schedule fatigue is the most publicised factor; E23 found nothing for favourites on back-to-backs. Expect F4 to be mostly priced.
- F1 and F2 are the novel part. A plausible effect is under a point on totals, so it is likely to be a feature-level gain (step 4), not a standalone bet.
- Most families failing is the normal outcome. A clean set of rejections is a useful result, as long as the mechanism check (step 3c) shows whether the fatigue signal exists at all.

## 6. Deliverables checklist

- [ ] PREREGISTRATION sections E28 (audit and screen) and E29 (forced-inclusion test), written before results
- [ ] `experiments/e28_audit.py`, `experiments/features_tracking.py`, `experiments/e28_screen.py`, `experiments/e29_run.py`
- [ ] `tests/team/test_tracking_features.py` (lookahead, hand example, NaN path)
- [ ] `reports/experiments/E28_tracking/{audit.md, screen.md, screen.json}`, `reports/experiments/E29_*/RESULT.md`
- [ ] A short summary for the owner after steps 3 and 5: tier per family x market, sizes in points or log loss, and what was not tested
