# Team-model audit 2: report

Branch `team-audit-2` (cut from `main` at `10c1660`). Scope: moneyline / spread / totals only; `props/` untouched.
Nothing in `models/saved`, `predictions/` or the database was changed or regenerated. Every claim below was checked against the
code and the live data first; commands and numbers are the ones I ran (read-only against `data/`, scratch directories for anything
that trains). Test suite: see the last section.

## 1. Which claims reproduced

| # | Claim | Result |
|---|---|---|
| B1 | Backtest entry price (opening line) predates the injury report the model reads | **Reproduced by code reading.** `make_bets` / `generate_season_predictions` use `odds_*` = opening lines; features use `INJURY_SNAPSHOT = 'game_day_noon'`. The size of the contamination **cannot be measured yet**: `odds_snapshots` holds 660 rows for 2026-10-20..21 only, so no historical game has an at/after-cutoff price. Diagnostic on the 5%-rule bets: CLV is the same whether or not the game had a new absence on the report (spread +0.31 with / +0.27 without, 399 / 213 bets; total +0.72 / +0.72, 1,215 / 692 bets), so the data neither confirms nor clears the claim. |
| B2 | Deployed set ~2.7 seasons stale; 80/20 row split | **Reproduced.** `calibration_meta_with_odds_v6.json`: calibration block = 3,359 games, 2024-01-17..2026-06-13, so the base models stop at 2024-01-16. The saved `_oos_2025-26` set stops at 2023-02-12 (calibration 2023-02-13..2025-06-22). |
| B3 | Totals model leans under | **Symptoms reproduced, cause not.** `_oos_2025-26` on 2025-26: prediction minus actual **-1.65**, minus line **-0.72**, **174 unders / 38 overs** at the 5% rule (audit: 173 / 37). But "trees cannot extrapolate, so model the gap from the opening total" is **not supported**: see section 3. The lean is the stale training window (B2) and a non-stationary market bias. |
| B4 | Saved 2025-26 log does not match its model files | **Reproduced.** Log: 32 distinct `ml_prob` (isotonic step function); calibrator on disk is a `MarginStackCalibrator`; re-scoring the same 1,321 games gives 1,321 distinct values. Max gaps **0.218** probability, **10.56** spread points, **12.35** total points (audit: 0.22 / 10.6 / 12.4). **Log loss: 0.6126 logged vs 0.5896 re-scored. The audit's 0.6463 does not reproduce** (CSV and the `predictions_log_v2` table give the same 0.6126). |
| W3 | Spread std of standardized errors 1.038 | **Reproduced** (1.037: 14.32 observed vs 13.80 saved). Not a bug; the 2025-26 spread error was simply larger than the calibration block's. Not changed (see section 4). |
| W5 | Spread / ML favourite disagreement | Counted **12** games since 2019-20 with the old field-by-field fill (audit: 7). After the fix: **7**. |
| rest | Other warnings | Confirmed from the code as described (see section 2). |

## 2. What changed, by commit

| Item | Change | Tests |
|---|---|---|
| **B1** | `core.odds_clean.load_entry_odds`: consensus of the first pre-game snapshot per game and book at/after game-day noon ET (capped 30 min before tip); no fallback to the opener. `experiments.metrics.make_bets(entry=...)` prices and settles there and skips games without one; default results are labelled `entry_basis='open'`, and `experiments/report.py` and `scripts/evaluate_oos_seasons.py` print the caveat. README caveat added. `generate_season_predictions.py` documents the open-price basis. | `tests/core/test_entry_odds.py`, `tests/team/test_entry_bets.py` |
| **B2** | `train_and_save_all`: calibrators / sigma fit on the **last 183 days** (never splitting a game night), then the base models are **refit on all games** with the same hyperparameters (`refit_on_all=True`). `calib_holdout_days=None` restores the old row split. New `tuner='grid'` (`team/tuning.py`, the grid the backtests validated). `team/retrain.py`, `scripts/retrain_candidate.py`, `scripts/run_weekly_retrain.bat`: weekly candidate set under a dated suffix, gated against the deployed set on the last 120 days (both out-of-sample); never deploys. The `.bat` is **not registered** with Task Scheduler. | `tests/team/test_train_final_models.py`, `test_tuning.py`, `test_retrain.py` |
| **B3** | `ResidualTotalModel` (gap from the opening total) implemented but **opt-in** (`total_target='residual'`); default stays the level model. Reason in section 3. Lean is addressed by B2. | `test_market_models.py`, `test_train_final_models.py` |
| **B4** | `model_manifest<suffix>.json` (hash of the set's files and of the training CSV); `load_bundle` refuses a set changed after its manifest; `train_and_save_all` refuses an existing suffix (`overwrite=True` / `--overwrite`). `predictions_log_v2` gains `model_set_hash`, `training_set_hash` (ALTER TABLE on existing tables, CSV mirror aligned); `run_inference_v2.py` and `generate_season_predictions.py` write them (`--tag`, `--overwrite-models`). `scripts/audit_prediction_log.py` / `team/log_audit.py` check provenance and re-score a log. | `tests/team/test_model_provenance.py` |
| W validated pipeline | `tuner='grid'` is the validated pipeline; the weekly retrain uses it. The deployed set stays FLAML until retrained. | `test_tuning.py` |
| W spread selection | Spread model selects features on the gap `target_spread_gap`. Effect on the 7 walk-forward folds (grid tuner): spread MAE +0.007 on average (-0.022 ... +0.030 by season), i.e. none beyond selection noise (README: ~0.01). | `test_train_final_models.py` |
| W pushes / price | `cover_probability(line=)`: on whole-number lines the win share among bets that settle (pushes refund); this makes such bets slightly *better* than the old formula said, not worse. `get_spread_bet` / `get_total_bet` take the side's own price; `live_odds` carries the current spread/total prices and `run_inference_v2` passes them (at -120 a 5% edge needs ~1 more point). Backtests use the same rule (`push_aware=True`; `False` reproduces old reports). | `test_betting.py`, `test_live_odds.py` |
| W missing opener | A missing opening line stays NaN (no current-line substitute); inference **skips** such games and logs them. | `test_live_odds.py` |
| W training odds | `core/odds_pick.py`: drop books far from the cross-book consensus (the `odds_clean` tolerances), then ONE book supplies every field. Training build and live use the same function. ~130 of 8,990 games change; 2 lose a value. | `tests/core/test_odds_pick.py`, `test_live_odds.py` |
| W era markers | `team/data_checks.missingness_era_report` logged and stored with each model set (`era_marking_features`); `train_from_season` option. **No default change** (needs a decision). | `test_data_checks.py` |
| W merge guards | `checked_merge` (`validate=` plus row-count checks) on the travel, metadata, home/away, roster and odds merges; injury merge validated. | `test_game_features.py` |
| W Brier | `ml_brier_calibrated` is now forward-chained cross-fitted; the in-sample value is kept as `ml_brier_calibrated_in_sample`. | `test_train_final_models.py` |
| W backtest -110 | `team/backtest.py` no longer fills a missing moneyline with -110. | `test_backtester_missing_odds.py` |

## 3. Why the total model was not switched to the gap (B3)

* The owner's own pre-registered experiment E1 already tested it: Total MAE **+0.274** (worse in 7/7 seasons), total bias 1.285 to 1.697
  (`reports/experiments/E01_residual_targets/RESULT.md`).
* Replication in this audit (7 folds, fixed grid, same features): pooled |bias| 0.944 (level) vs 1.348 (gap); MAE 14.74 vs 15.05.
* What drives the lean on 2025-26 (fixed-grid models trained through the date shown, 5% rule, sigma 18.4):

| trained through | model | pred - line | unders / overs |
|---|---|---|---|
| 2023-02-12 (like the saved `_oos_2025-26`) | level | -0.44 | 182 / 69 |
| 2023-02-12 | gap | -0.53 | 154 / 67 |
| 2024-01-16 (like v6's base models) | level | +0.78 | 60 / 174 |
| 2025-06-30 (fresh) | level | +0.40 | 35 / 80 |
| 2025-06-30 | gap | +0.23 | 20 / 11 |

  The sign of the lean flips with the training cut-off and vanishes with fresh data, so it is staleness, not a structural under-bias.
  The market itself was low by 0.93 points in 2025-26 (actual minus line is positive in the last four seasons: +0.60, +0.21, +0.73, +0.93).
* Applying `total_bias` from a half-season holdout does not help either (walk-forward, 7 seasons): mean |bias| 0.944 to 0.867 for the level
  model with offset, but the offsets are noisy (up to +2.5 in 2020-21) and the bets become one-sided the other way (2,149 bets: 827 under /
  1,322 over). I did **not** wire a bias correction into `predict_markets`.

## 4. Not changed, and why

* **Sigma (W3).** The 2025-26 spread error std (14.32) is simply above the calibration block's (13.80). The time-based holdout makes the
  saved sigma more recent; whether to inflate it or use a trailing-season maximum is a policy call.
* **Era markers.** Dropping seasons before 2019-20 cuts the history from ~16,800 to ~9,000 games; there is no cheap walk-forward that
  validates it. The diagnostic and `train_from_season` are in place.
* **Live rows for the deployed set have no `training_set_hash`.** `_with_odds_v6` has no manifest and I did not write one into
  `models/saved` (deployed artifact). Its `model_set_hash` is computed on load.

## 5. Needs the owner's decision (nothing below has been run)

1. **Retrain / redeploy.** Proposed (writes only new files under a new suffix; does not change `DEPLOYED_SUFFIX`):
   `.venv\Scripts\python.exe scripts\retrain_candidate.py --tag 20261020` after `scripts\generate_training_set.py`.
   Cost: see the dry run in section 6. Promotion = a reviewed commit setting `DEPLOYED_SUFFIX` in `team/model_registry.py`.
   Promoting changes what the daily inference publishes (new probabilities / lines, and the two new log columns).
2. **Weekly schedule.** `scripts\run_weekly_retrain.bat` is ready; registering it is a Task Scheduler change I did not make.
3. **Regenerate the 2025-26 out-of-sample log** (B4): `python scripts\generate_season_predictions.py --season 2025-26 --tag v2`
   (trains a new `_oos_2025-26_v2` set with FLAML, 3 x 600 s by default; add `--time-budget` to shorten; replaces the 2025-26 rows of
   `predictions_log_v2` after backing the table up, and replaces `predictions/predictions_2025-26.csv` after copying it to `data/backups/`).
   Not run. Until then the current log should be treated as unlinked to any model files; `scripts/audit_prediction_log.py` shows the gap.
4. **Merging this branch** changes live behaviour on the next scheduled inference: games without an opening line are skipped; spread/total
   edges use the side's own price when SBR supplies it and treat pushes on whole-number lines; two columns are added to
   `predictions_log_v2` and the CSV mirror; opening odds in the training set come from a single book after a consensus filter (~1.4% of games
   change), so regenerate the training set before the next retrain and expect `tests/team/test_parity.py` to keep passing.
5. **Decide the entry-price policy** for ROI / CLV once `odds_snapshots` has a season of history (about the 2026-27 season's end): until
   then report closing-line comparisons only.
6. **Era markers / sigma policy** (section 4).
