# Experiment harness and feature-selection audit (branch `audit-fixes-experiments`)

Ten audit items against `team/train_baseline.py` and `experiments/`. Each was read against the code first; a fix was made only where the
defect reproduced. One commit per fix (plus two follow-ups, marked *). Tests: `tests/team/test_feature_selection.py`,
`test_experiment_harness.py`, `test_experiment_scripts.py`, `test_tracking_features.py`, `test_train_final_models.py`; the whole `tests/team` suite passes.

## How the impact was measured

* **Reference ("before")** = `reports/experiments/E25_with_odds`, the baseline walk-forward regenerated on the current training set
  (`training_set_v1_reduced_2026-10-19.csv`). The older `reports/experiments/baseline` artifact comes from an earlier CSV and does not
  reproduce on today's data.
* **Harness check:** the new code with the pre-audit switches (`selection='legacy', calib_cap=None, refit_on_all=False`) reproduces
  `E25_with_odds` **bit for bit** (same features in all 7 folds, max |prediction difference| = 0.0), so differences below are due to the fixes.
* Walk-forward variants were run into a scratch folder; nothing under `reports/experiments/` was overwritten. Script-level re-runs
  (E02, E04, E10, E11, E12, E18) used the saved baseline artifacts; both the pre-fix and post-fix script were run on the same artifacts.
* Paired differences are per-game, day-bootstrap 95% CIs, negative = better. ROI figures rest on a few hundred bets per market
  and moved by several points between near-identical models; they are not interpretable and are not used for any conclusion here.

## Verification result per item

| # | Item | Reproduced? | Action |
|---|------|-------------|--------|
| 1 | Feature selection | flaws 1-4 yes. Flaw 5 (`fillna(0)`) is in the code but **0 NaN correlation pairs** occurred in any of the 6 folds checked | fixed 1-4, left 5 |
| 2 | Calibration block = 20% of history; no refit | yes (calibration block 1.6k-3.1k rows, i.e. up to 2.5 seasons old; base models never refit) | fixed |
| 3 | E01/E25 residual runs use raw-target feature lists | yes. "Drops pre-2019 rows" is **not** true: all but 44 of the 16,795 games have lines (19 of the 8,916 test games) | fixed |
| 4 | E11 Gaussian copula with rho < 1 | yes (rho 0.87-0.91, impossible outcomes had positive probability) | fixed |
| 5 | E02 cross-fit uses future blocks | yes. The "`m_full` trained on `train`" confound is **not** in this repo: every arm calibrates the baseline's `ml_prob_raw` | fixed the causal part |
| 6 | E12 totals decision uses the spread control | yes | fixed |
| 7 | Cover probability inconsistent | partly. E04/E10/E11/E18 ignored the push correction: yes. "`make_bets` p_win is unconditional" is **wrong**: `cover_probabilities(..., lines=)` returns W/(W+L), already the no-push probability, so adding `p_win / (1 - p_push)` would condition twice | fixed the inconsistency only |
| 8 | E06 calibration rows sorted by date | yes: **7,042 of 16,218 rows (43%)** get the wrong fold from a positional label | fixed |
| 9 | `trk_prev_opp_pace` is the team's own previous pace | yes | fixed |
| 10 | E17/E19 `np.quantile`; empty locked regime | (a) yes: `E19_tanking/results.json` already holds a NaN CI (2020-21). (b) **not reproduced**: the code differs from the description (it ranks on (W-L)/2 with a 6-game gap) and the regime is sparse, not empty (K = 52 / 82 games, LK = 13 / 26) | fixed (a); (b) left, see below |

## Commits and impact

### 1. Feature selection (`78d52b3`, `7db75f6`*)

`gain` importance with bagged proxy models (colsample 0.5, subsample 0.8); Stage-2 shuffle **within season**; 30 shuffles and a
z-score rule (z > 1.28) instead of the 90th percentile of 10 draws; and, when the lines are inputs, team features are ranked against
the **market residual** with `odds_*` excluded, after which the `odds_*` columns are appended (`select_market_features`; used by
`train_and_save_all`, `backtest_walk_forward` and `experiments/lib.py`). `legacy=True` reproduces the old selection exactly.
The follow-up averages the real importances over 5 bagging seeds: with a single bagged fit the lists became **less** seed-stable than before.

| Mean pairwise Jaccard of the lists across selection seeds (spread / total) | pre-audit | single bagged fit | final (5 fits) |
|---|---|---|---|
| fold 2019-20 | 0.68 / 0.61 | 0.40 / 0.29 | 0.65 / 0.63 |
| fold 2024-25 | 0.51 / 0.53 | 0.36 / 0.27 | 0.67 / 0.67 |

* Stage-2 passers: the new rule passes fewer features (spread 9-16, total 14-21 vs 18-24 / 29-34 before), so the n_min = 20 padding matters more.
* The new lists overlap the old ones little (mean Jaccard 0.2-0.3 for ML / spread, 0.1 for totals), so every re-selected model set changes.
* Era proxies among the selected totals features (|Spearman| > 0.3 with the season index): 7.4 -> 4.4 per fold. (Overlap and era counts were
  measured on the single-fit lists, before the 5-fit follow-up.)
* Accuracy: selection alone moves nothing measurable (V0 -> V1, below). The stratified null is covered by a unit test, not by a visible metric.
* Effect on production: only on the next retrain. Saved models keep their `feature_config`.

### 2. Walk-forward split and refit (`5c5d7cc`)

Calibration block capped at 1,300 rows (one season; was 3,100 rows in the last fold) and the tuned base models are refit on
model_train + calibration after the calibrators and sigma are fitted (`refit_on_all=True`, as in `train_and_save_all`). Each run saves
`split.json`; runs without one are read as the old scheme (`run_split_params`), and `label_calib_folds` detects either layout, so the
existing artifacts and E04/E07/E31 keep working. `fit_tuned` still tunes on 64% of model_train (by design; it refits on all of it).

Cumulative effect on the baseline (7 folds, 8,916 games; difference vs the pre-audit run, CI in brackets):

| Step | ML log loss | Spread MAE | Total MAE |
|---|---|---|---|
| pre-audit (reference) | 0.6225 | 10.694 | 14.768 |
| + selection fixes (V1) | -0.0020 [-0.008, +0.005] | -0.002 [-0.023, +0.018] | -0.013 [-0.053, +0.028] |
| + capped calibration block (V2 vs V1) | **+0.0084 [+0.001, +0.016]** | -0.013 [-0.026, +0.000] | -0.056 [-0.094, -0.015] |
| + refit on all (V3 vs V2) | -0.0002 [-0.008, +0.007] | -0.003 [-0.014, +0.008] | -0.050 [-0.079, -0.022] |
| **all fixes (V3 vs pre-audit)** | +0.0061 [-0.004, +0.017] | -0.018 [-0.040, +0.004] | **-0.118 [-0.170, -0.068]** (6/7 seasons) |

**Totals improve by 0.12 points; ML log loss gets worse.** The ML cost comes from the cap: isotonic regression is now fitted on 1,300
rows instead of 1,600-3,100 (ML ECE 0.012 -> 0.021). Production avoids this by using a logistic stack, not isotonic, for the market-aware ML
model. The harness now defaults to Platt too (`ml_calibrator='platt'`, see the follow-up section); the figures in this section are the
isotonic ones, `ml_calibrator='isotonic'` reproduces them.

### 3. E01 / E25 residual runs (`cf61eac`)

`run_walk_forward(selection=...)` with `'residual'`, `'residual_no_odds'`, `'raw'`, `'legacy'`; E01 and E25_residual use
`residual_no_odds`, E25_no_odds uses `raw`. `hooks['require_columns']` drops rows without a line inside each fold after the
boundaries are set on the full history. A cached `features.json` is reused only if `selection.json` records the same rule.

| Residual-target run vs its baseline | old (E25_residual vs E25_with_odds) | new (E01 vs V3, all fixes) |
|---|---|---|
| Spread MAE | -0.034 [-0.058, -0.012] | -0.024 [-0.048, -0.002] (7/7 seasons) |
| **Total MAE** | **+0.275 [+0.201, +0.346]** (worse, 0/7) | **+0.007 [-0.047, +0.064]** (no difference) |
| ML log loss | -0.000 | -0.002 [-0.012, +0.008] |

The residual **total** model's total MAE falls from 15.035 to 14.666 (-0.369) with the corrected selection. The earlier finding that a
residual total is 0.27-0.30 points worse (quoted in `team/market_models.py` and `train_and_save_all`) came largely from selecting its
features on the raw total. With corrected selection the two are indistinguishable. Nothing was adopted; this is for the owner.

Caution when re-running `python -m experiments.e01_residual`: it compares against the saved `baseline` artifact, which was made by the
old pipeline. Run that way, its pre-registered rule now reports KEEP-TOTAL = True (total MAE -0.116 vs `baseline`), where the saved
result was False (+0.274). That compares a new pipeline with an old one; the like-for-like comparison above (E01 vs V3) is a tie.
Regenerate the baseline under the same defaults before reading the decision.

### 4. E11 joint Kelly (`4590943`)

ML and spread are two thresholds on one margin, so the joint distribution is exact (nested for the same side, `max(0, p1+p2-1)` for
opposite sides) and no copula is fitted. Re-run on the same baseline artifact, essentially unchanged (721 games have more than one bet):

| (iii) joint Kelly vs (i) independent | pre-fix script | post-fix script |
|---|---|---|
| paired log-growth difference [95% CI] | +1.58 [-0.69, +3.86] | +1.58 [-0.74, +3.91] |
| bets staked / average stake | 4,352 / 4.29% | 4,338 / 4.32% |
| decision | REJECT | REJECT |

### 5. E02 cross-fit (`f59e4e7`)

Each block is predicted by a model trained on earlier games only. Cross-fitted isotonic: ECE 0.0146 -> 0.0156, seasons with better ECE
4/7 -> 3/7, log-loss difference vs baseline +0.0051 -> +0.0044. Decision unchanged (not kept).

### 6. E12 totals control (`8950630`)

`both_sides_control(base, mats, market)`; each market is judged against its own control. Totals control ROI = **-1.35%** over 8,898 games
(spread -1.62%). Decisions unchanged (all three still ADOPT as a reporting option); the bug did not flip anything here.

### 7. Cover probability (`134c8c7`)

`make_bets` returns `p_win`, the probability behind the edge; E04, E10, E11, E18 use it (or `cover_probabilities(..., lines=)`)
instead of `Phi(pe / sigma)`. Effect is small: E10 "mean model P(win)" +0.12 pt (spread) / +0.09 pt (total), E04 log-loss shifts <= 0.0004,
E18 cover log loss +0.0001, E11 as above. No decision changed. (E10 still shows hit rate 50.2% against mean model P(win) 59.9%
for spread; that gap is not caused by this item.)

### 8. E06 fold order (`7bbc559`)

`assemble_calibration` keeps fold order and a `fold` column. Positional labelling of the old file mislabels 43% of rows.

### 9. Tracking pace feature (`f882c7c`)

`trk_prev_opp_pace` is now the previous game's pace minus the team's own pace EWMA going into that game (games before it only). On
31,180 team-games: correlation with `odds_total` **+0.296 -> +0.008**, with the team's own pace habit 0.72 -> 0.13; correlation with the
total residual is unchanged (+0.020 vs +0.016), so removing the confound did not reveal signal. `features_tracking.py` and
`test_tracking_features.py` were untracked, so this commit adds them whole.

### 10. Bootstrap NaN (`d0768ce`)

`nanquantile` plus `errstate` in E17 `ratio_diff` and E19 `e14_ratio_diff`; in a test cell a third of the draws produced NaN, and the
saved E19 result has a NaN CI in 2020-21. **Locked-seed regime not redefined:** it is pre-registered, the described code does not match
the repo, and it is sparse (13 / 26 late locked games) rather than empty. A clinch-based definition would be a new, separately registered test.

### Backward compatibility (`50b0c47`)

`e26_freeze.py` now passes `legacy=True`, so re-running the freeze rebuilds the registered E26 model rather than a re-selected one.

## Follow-up: ML calibrator under the capped block (evaluation only, harness unchanged)

Same walk-forward (selection fixes + refit), varying the calibration block and the E02 calibrator arms; ML log loss, 8,916 games,
difference vs the pre-audit run (V0, 0.6225) with day-bootstrap CI:

| Calibration block | isotonic (current) | Platt | cross-fit isotonic | uncalibrated |
|---|---|---|---|---|
| 1,300 rows (current cap) | 0.6286 (+0.0061) | **0.6138 (-0.0087 [-0.017, -0.001], 6/7)** | 0.6219 (-0.0006) | 0.6116 |
| 2,600 rows | 0.6209 (-0.0016) | 0.6123 (-0.0103) | 0.6194 (-0.0031) | 0.6112 |
| old 20% (~2,300 rows) | 0.6219 (-0.0006) | 0.6120 (-0.0106, 7/7) | 0.6201 (-0.0025) | 0.6110 |

* Within the 1,300-row block, Platt beats isotonic by 0.0148 [-0.023, -0.008] in all 7 seasons; ECE 0.021 -> 0.009.
* Platt is nearly independent of block size (0.6138 / 0.6123 / 0.6120), so the cap costs it little. Isotonic loses 0.0067 at 1,300 vs 2,600 rows.
* The raw model probability is slightly better still (Platt vs raw +0.0022 [+0.001, +0.003]): once the odds are inputs, the raw score is
  already close to calibrated and a calibrator adds mostly noise. Recommendation if ML is kept in the harness: Platt on the capped block.
* The base-model recency gain (refit, nearer data) is in all rows; the clean contrast is the column difference within a row.

## Left for the owner

* ~~Whether to switch the harness's ML calibrator from isotonic to Platt~~ - done: `run_walk_forward(..., ml_calibrator='platt')` is the
  default, so the harness matches production's logistic calibration. Only `experiments/lib.py` changed; no production model, saved model
  or `reports/experiments/` artifact. `split.json` records the calibrator and runs without it are read as isotonic
  (`run_split_params`), so `E25_*`, `E01_*` and `baseline` still reproduce with `selection='legacy', calib_cap=None, refit_on_all=False,
  ml_calibrator='isotonic'`. Any experiment that re-runs `run_walk_forward` with defaults now gets Platt ML probabilities, so compare it with
  a reference made the same way (E02's "isotonic baseline" arm only means isotonic on runs made before this change).
* Whether the E1 residual-total result should be re-registered and the "0.27-0.3 points worse" notes updated (see 3).
* Regenerating `E25_*`, `E01_*` and `baseline` under the new defaults was **not** done; existing artifacts are untouched.
* `experiments/e31_nosel.py` (untracked) was edited to follow its reference run's split via `run_split_params`; that edit is in the working
  tree only, not committed.
