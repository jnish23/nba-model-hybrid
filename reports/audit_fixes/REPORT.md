# Audit fixes: report

Branch `audit-fixes` (cut from `multi-season-oos`; `4fe0702` is a checkpoint of your pre-existing uncommitted
source edits). One commit per item, 97 tests (`python -m pytest tests`; 8 need `data/nba_stats.db`), CI runs them.
Raw tables: `walk_forward_with_odds.md`, `walk_forward_no_odds.md`, `oos_flaml_old_vs_new.md` (this folder).

## 1. Item by item

Status = what reading the code and a small check showed. "Not confirmed" items were not changed.

| # | Status | What was found, what changed | Commit |
|---|---|---|---|
| 1 | confirmed, fixed | Live fed the *current* line to `odds_*`. Models now get the opening line (current only as logged fallback), current lines live in `curr_odds_*` and drive edges/EV/the log. Spread differs open vs current in 86% of games; feeding the later line moves a model's spread pred by 1.29 pts on average (p95 4.5), total pred 1.97 pts, P(home) 3.9 pp. | fc8c0ea |
| 2 | confirmed, fixed | Roster/injury merged after the diff step: `roster_bpm_ewma_x/_y`, no `diff_roster_*`, `roster_bpm_home/away` dropped from ML/Spread. Now merged first; one shared `add_derived_features` for batch and live. Values of every shared numeric column unchanged; new `diff_roster_bpm(_ewma)`, `diff_inj_missing`. | 5af4838 |
| 3 | confirmed, fixed | Live `road_trip_count=0` and `body_clock_lag=tz_delta`. Live travel now runs `build_features()` itself on the upcoming games (9/9 road-trip and 18/18 body-clock values differed on 2026-01-15; now equal). | 9d34e00 |
| 4 | confirmed, and worse | `WL`/`MATCHUP` are upper-case in `team_game_full`; the momentum block tested `'wl'`, so **training never had** `win_pct_last10 / win_streak / home_win_pct_20g` either. Fixed the case bug, one shared implementation for batch and live (+9 training columns). Also aligned live `days_rest` (was capped at 7), `games_last_7d` (LIMIT 5) and `is_playoff` (always 0) with training. | 34022b6 |
| 5 | **not confirmed** | Injury features were already wired live (`get_injury_features`). Live vs training: 0 of 144 values differ on 2026-01-15 and 2025-03-12; now also covered by the parity test. | (test in 5cc539f) |
| 6 | confirmed, fixed | Missing features were filled with 0.0 after one WARNING. Now NaN, ERROR log with names, and >10% missing refuses to score. | 6b67436 |
| 7 | confirmed, fixed | Deployed name hardcoded in `run_inference_v2`, `train_baseline` could not produce it, `backfill_inference` loaded `_v3` via the v1 engine. `models/model_registry.py`: `DEPLOYED_SUFFIX`, `load_bundle()` refuses a set whose feature_config differs from the pickles; `train_baseline.py --deploy` writes that set. | 0fac972 |
| 8 | done | Live rows are assembled by the training builders (`models/live_features.py`). `tests/test_parity.py` rebuilds the training set and compares >300 columns for 2020-01-17, 2022-11-23, 2025-03-12, 2026-01-15: identical. Replaying `InferenceEngine.run()` for 2026-01-15 reproduces the model applied to the training rows (max prob diff 5.6e-17). | 5cc539f |
| 9 | confirmed, fixed | `get_ml_bet` priced away as `1 - home implied`, crediting the vig (2.96 pts of probability at -170/+150). Each side now uses its own price; no bet if a price is missing. `calculate_ev_metrics` does not exist in this tree; the home-only logic was in `evaluate_vs_market`, `diagnose_moneyline_bets`, the dashboard. | 79946e1 |
| 10 | confirmed, fixed | Dashboard "last 14 days" ML ROI settled at -110 (no odds column was passed), and an unpriced ML bet fell back to -110. Now real prices (`settle_bet`). On the logged 2025-26 bets (318 at 5%): -8.1% at -110 vs -6.9% at real prices. `train_baseline` had no payout calc; it now reports ROI at actual moneylines. | 00d5517 |
| 11 | confirmed, unified | With residual std 13.4 / 18.2 a flat 1.0 pt is a **+0.6%** win-probability edge over the -110 break-even on spreads and **-0.2%** on totals (negative EV even if the model is right); a 5% edge needs 2.5 pts (spread) / 3.4 pts (total). Live recommended only totals at raw 5/7 pts. One rule (`betting.get_*_bet`: P(win) vs break-even > 5%) now drives live picks, dashboard, `betting_simulation`, `backtest_analysis`, `evaluate_oos_seasons`. Pushes are no longer losses in `backtest_analysis`. | 190b4b5 |
| 12 | confirmed, fixed | 2019-20 injury play rates came from all later seasons (30,503 'Out' rows vs 3,182 of its own). Now a documented prior (Out 0, Doubtful .25, Questionable .5, Probable .75, Available 1). | be6cc91 |
| 13 | confirmed, fixed | `backtest_walk_forward` defaulted to `best_config_*.json` tuned on all seasons. Default is now fixed `NEUTRAL_PARAMS`; `tune_per_fold=True` tunes inside each fold. See 2.4: the leak was not flattering here. | 6d3623b |
| 14 | partly | Feature selection used all model-train rows although FLAML validates on the last 20% of them: fixed (selection stops at the FLAML boundary). The isotonic calibrator was already fit on a separate holdout. Effect in 2.4. | 199c4c5 |
| 15 | confirmed, fixed | `to_sql(if_exists='replace')` had stripped the PRIMARY KEY; the live `predictions_log_v2` had none. Schema defined once, table repaired on start (copy kept as `predictions_log_v2_prepk_20261016_235831`), season rewrite deletes+appends. | 9523a24 |
| 16 | confirmed, fixed | `INSERT OR REPLACE` reset completed games to PENDING/null scores. Now `ON CONFLICT DO UPDATE` of prediction columns only, completed games untouched. Found on the way: the CSV mirror appended 14 fields under a 17-column header. | 7e55a6b |
| 17 | confirmed, fixed | `create_team_full_table.py` dropped the OPP_* join; `TeamFeatureBuilder` summed C/F/G columns (+22.4% FGA, +22.3% PTS on 2023-24+). Script now runs `rebuild_team_game_full()`; builder raises instead of falling back. (The live table already had correct OPP_*.) | b7d991f |
| 18 | **not confirmed as described** | No `load_injury_reports()`; `parse_all_reports()` rescans the folder and `live()` downloads first. The real gap is a silent failure path, so `check_report_freshness` now logs ERROR when the download fails or the newest report is >6 h stale/absent. | cf75e18 |
| 19 | confirmed, fixed | BPM depended on the first loaded season: the same game's `roster_bpm` differs by 2.02 BPM on average in 2019-20 (max 8.6, std 1.46) between a 2013-14 and 2019-20 start; injury features used 2019-20. One `BPM_HISTORY_START = 2013-14`. Static `player_info` positions: dropping position info entirely moves team roster BPM by 0.12 (std 1.5): not material. | e8438e5 |
| 20 | experiment | Offseason regression hook (`season_carryover`), default off. No benefit, see 2.4. | aa899dc |
| 21 | done | UTF-16LE -> UTF-8, `pathlib` backport removed, `pypdf` (and `pytest`) added. | 14344bd |
| 22 | done | Tests for team/game/travel builders, betting, logger, registry, parity, leakage guards; pytest in `checks.yml`. | 59a401c |
| 23 | done | Bare excepts replaced in `team_features_v2` (the `rolling('7D')` fallback to a constant 3 now computes the exact count), `game_features_v2`, `run_inference_v2`, `prediction_logger`. | 4fd7e49 |
| 24 | confirmed, fixed | Column order changed the selected features (test fails on old code). Alphabetical columns, name tie-breaks, deterministic LightGBM. | 9aaf622 |
| 25 | **not confirmed** | No duplicate names in `feature_categories`; nothing to change. | - |
| 26 | partly | Value interpolation was already gone from the builders after the rewrites above and absent in `live_rotation_v2`; `availability_features` now binds its seasons; `update_database` binds all values already, so its table/column names go through `_ident()` validation. | c51a19d |
| 27 | done | Dead `InferenceEngine` methods (-275 lines), `build_nba_odds_table` (now a wrapper), `run_*_backtest` (v2 builder; travel run no longer overwrites the player plot/CSV; ran end to end), unused seaborn import; legacy v0/v1 modules removed after an import search (`live_rotation_v2` kept, now unreferenced). | bda984f 5e25f66 ff8a798 |
| 28 | done | Vectorized; equivalence on the full history: team features (129 cols) bit-identical, BPM / injury features / the whole 430-col training set equal to 1e-14. Training-set build 21.7 -> 13.1 s. (No `_norm_name`/`iterrows` exist; the per-row maps were the equivalent.) | e48e43f |
| 29 | partly | A night's team+travel features: 130 SQL SELECTs / 2.9 s -> 5 / 1.6 s, same values. Loading fewer seasons of player data was **rejected**: it changes the output (roster_bpm by up to 0.58-0.78) and the full history costs 2.8 s. | f8517d7 |

Not asked for but changed: `train_baseline.py` now runs as a script (the documented command failed on imports), deployed set `_with_odds_v5` retrained (`e8b6bcb`), `injury_status_play_rates.json` regenerated (`90a69ff`), README updated.

## 2. Old vs new results

All walk-forward numbers: 8,916 games, seasons 2019-20 to 2025-26, train on earlier seasons only, isotonic
calibration on the last 20% of each training window. "Old" = code at `4fe0702` on the same database, run in a
separate worktree. Bets use the unified rule (5% probability edge, -110 for spread/total, real prices for ML).
One standard error of a ROI is about 1.3% at 5,000 bets, 2.7% at 1,700, 4-5% at 400-500.

### 2.1 Deployed configuration (opening odds as inputs), walk-forward

| | old, tuned on all seasons (as shipped) | old, neutral params | new, neutral params | new, tuned inside each fold |
|---|---|---|---|---|
| ML accuracy | .6535 | .6514 | .6550 | .6642 |
| ML log loss | .6281 | .6299 | .6298 | .6163 |
| ML Brier | .2158 | .2177 | .2164 | .2128 |
| calibration (ECE) | .0168 | .0124 | .0098 | .0112 |
| spread MAE / RMSE | 10.80 / 13.80 | 10.84 / 13.87 | 10.82 / 13.84 | 10.70 / 13.70 |
| total MAE / RMSE | 15.25 / 19.26 | 15.02 / 18.98 | 15.01 / 18.96 | 14.77 / 18.72 |
| ML bets @5% (ROI) | 2,729 (-4.2%) | 3,527 (-3.8%) | 3,496 (-2.3%) | 1,703 (+3.0%) |
| spread bets @5% (ROI) | 1,694 (-4.0%) | 2,616 (-2.7%) | 2,534 (-1.7%) | 499 (-2.4%) |
| total bets @5% (ROI) | 4,309 (-2.1%) | 3,922 (-0.7%) | 3,775 (-1.4%) | 2,019 (+0.3%) |
| old rules, same predictions: ML home-only 5% / spread flat 1pt / total flat 1pt | 4,602 (-4.3%) / 5,251 (-4.8%) / 7,411 (-2.6%) | 5,045 (-4.2%) / 5,862 (-2.6%) / 7,295 (-1.3%) | 5,126 (-0.5%) / 5,829 (-1.7%) / 7,216 (-3.7%) | 3,647 (-1.1%) / 3,485 (-1.6%) / 6,261 (-1.9%) |

### 2.2 No odds as inputs, walk-forward

| | old, tuned on all | old, neutral | new, neutral |
|---|---|---|---|
| ML accuracy / log loss / Brier / ECE | .6443 / .6411 / .2203 / .0119 | .6449 / .6428 / .2213 / .0087 | .6384 / .6526 / .2235 / .0191 |
| spread MAE / total MAE | 10.95 / 15.56 | 11.01 / 15.27 | 11.05 / 15.30 |
| ML bets @5% (ROI) | 4,771 (+1.5%) | 4,992 (+2.7%) | 5,100 (-1.0%) |
| spread / total bets @5% (ROI) | 4,135 (-1.2%) / 5,449 (-1.2%) | 4,541 (-1.9%) / 5,159 (+0.9%) | 4,567 (-2.9%) / 5,111 (-0.7%) |

### 2.3 Out-of-sample seasons, FLAML-tuned deployed pipeline (2021-22 to 2025-26, 6,602 games)

Each season's models trained on earlier seasons only. "Old 600s" is the published run (FLAML 600 s per model);
the other two use 120 s per model, run concurrently in the old and new trees.

| | old (600 s, as published) | old (120 s rerun) | new (120 s) |
|---|---|---|---|
| ML accuracy / log loss | .6664 / .6231 | .6706 / .6252 | .6712 / .6253 |
| Brier / ECE | .2123 / .0124 | .2120 / .0130 | .2109 / .0089 |
| spread MAE / total MAE | 10.76 / 14.74 | 10.77 / 14.71 | 10.77 / 14.66 |
| ML bets @5% (ROI) | 1,574 (-0.4%) | 1,294 (+3.1%) | 1,342 (+6.0%) |
| spread bets @5% (ROI) | 533 (+3.9%) | 500 (+3.9%) | 380 (-0.5%) |
| total bets @5% (ROI) | 1,954 (-0.5%) | 1,740 (-1.5%) | 1,242 (+1.3%) |

These ROI differences (a few points on 400-1,300 bets) are inside one to two standard errors; none of the runs
shows an edge distinguishable from zero.

### 2.4 Effects of individual fixes and experiments

- **Item 13 (tuned on all seasons).** By construction the old configs saw the test folds, but the effect was not
  flattering: with odds, log loss .6281 (tuned on all) vs .6299 (neutral), and tuned-on-all had the *worst* betting
  results. Tuning honestly inside each fold (15 s per model) gives .6163 and spread/total MAE 10.70/14.77, i.e. the
  neutral parameters are the weak point, not the leak.
- **Item 14.** Selecting on all model-train rows made the FLAML validation block look better than it is: validation
  score optimism 0.0023 log loss, 0.068 spread MAE, 0.034 total MAE; held-out 2025-26 scores were not worse with the
  corrected selection (log loss .5972 vs .5982, spread MAE 11.26 vs 11.28, total MAE 14.98 vs 15.04). Feature sets overlap
  only 22-31% (Jaccard) between the two variants, so selection is unstable either way.
- **Item 4 ablation (momentum features).** Removing them: no-odds ML log loss .6495 vs .6526, with odds .6303 vs .6298.
- **Item 20 (offseason regression, `season_carryover`).** 0.75 / 0.5 vs 1.0: no-odds log loss .6529 / .6526 vs .6526,
  with odds .6298 / .6300 vs .6298; MAEs within +-0.1. No benefit; production behaviour unchanged.
- **Selection by gain instead of split counts** (hypothesis for the no-odds log-loss move below): .6522 vs .6526
  without odds, .6286 vs .6298 with odds, total MAE 14.94 vs 15.01. Small, not adopted (option `importance_type`).

### 2.5 Live path

Parity test: >300 feature columns identical to the training set on 4 historical dates. SQL SELECTs for a night's
team + travel features 130 -> 5. Full `run()` replay for 2026-01-15 matches the model applied directly to the
training rows.

## 3. Why the numbers moved

1. **With odds as inputs (the deployed setup) nothing material moved.** Log loss .6299 -> .6298, MAE 10.84 -> 10.82 /
   15.02 -> 15.01, calibration slightly better (ECE .0124 -> .0098 walk-forward, .0130 -> .0089 out of sample). The
   opening line carries most of the signal, so the corrected injury/roster/momentum features change little. The fixes
   matter for *serving* (live inputs now match what the model was trained on, e.g. a later line moves a prediction by
   1-2 points) and for *evaluation honesty*, not for backtest accuracy.
2. **Without odds, ML log loss rose .6428 -> .6526 (and ECE .0087 -> .0191), spread/total MAE +0.04/+0.03.** Controlled
   runs: new code on the old training set .6412; new features without the injury block .6440; without momentum .6495;
   final .6526. But re-running feature selection with other seeds gives a standard deviation of 0.004 in log loss
   (old features: mean .6454; new features: mean .6483), so the shift is about 0.7 standard deviations of selection noise.
   It is not a demonstrated regression. The one structural cause I could identify: `diff_roster_bpm` and
   `diff_roster_bpm_ewma` correlate at 0.89 and collapse into one cluster, and split-count importance keeps the noisier
   one; switching to gain importance did not change the no-odds result (.6522).
3. **Betting numbers moved mostly because the rules changed, not the models.** The legacy ML rule bets about 47% more
   games than the own-price rule on the same predictions (5,126 vs 3,496) because the away side was credited the vig;
   the flat 1-point spread/total rule bets 5,829 / 7,216 games versus 2,534 / 3,775 at a 5% edge.
4. **Injury features changed a lot** (new vs old training set: `inj_lost` differs in 74% of the 8,916 injury-era games,
   `inj_lost_new` in 39%, `inj_n_out` in 1.8%) because of the BPM history start and the 2019-20 prior; this is the
   intended correction. Its effect on the models is within selection noise (see point 2).
5. **No fix produced a betting edge.** Every ROI above is within about two standard errors of zero; the positive ones
   (e.g. new tuned-in-fold ML +3.0% on 1,703 bets) would need replication on fresh seasons.

## 4. Caveats and follow-ups

- README headline tables and `reports/market_eval_*` were generated by the old pipeline and were not regenerated.
- Existing `reports/oos_predictions_*` and the untracked `*_oos_*` models are left as you had them; the retrained
  120 s versions are in `reports/audit_fixes/oos_new_120s/` and `data/backups/oos_after_audit/`; the pre-fix database
  is `data/backups/nba_stats_before_audit_fixes_20261016.db`. The old-code worktree is `C:\Users\jorda\nba-baseline`.
- The tuned-per-fold run used 15 s per model: indicative, not tuned to convergence.
- Still open: `models/backtest.py` fills a missing ML price with -110; one failing game in `run()` stops the whole
  batch (loudly); `models/live_rotation_v2.py` is unreferenced; other copies of the database (e.g. in `data/backups`)
  still hold the key-less `predictions_log_v2`, which the logger repairs the first time it opens them.
