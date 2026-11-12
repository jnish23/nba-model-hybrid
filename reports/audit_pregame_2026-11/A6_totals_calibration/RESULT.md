# A6: totals level calibration (E27) and the TotalLevelCalibrator code

Workstream A6, 2026-11-11. Nothing committed or staged; `models/saved/` not written; no network.

## Bottom line

- **E27 PASSES its keep rule.** Fix (c) on the deployed form (E13 run: E1 spread, E3 ML stack, E4 sigma): total MAE 14.763 -> 14.610, better in 7 of 7 seasons, pooled paired difference -0.153 [-0.212, -0.102], R1 -0.146 and R2 -0.147, ML and spread unchanged. The same verdict with E18's seed and on an independent re-run (`E25_with_odds`).
- It brings the total to **level with the opening line, not better**: fix (c) minus line is -0.008 [-0.029, +0.015]. The raw model is worse than the line by +0.146 [+0.080, +0.223].
- A constant shift of the line (`line + mean calibration-block (actual - line)`) is as good as fix (c) (+0.010 [-0.013, +0.034] worse, within noise). Almost all of the gain is "stop following the model's gap", not the level slope.
- The 5%-rule total bets fall from 1,931 to 81 over the 7 seasons (descriptive; neither ROI differs from 0). E4's per-game sigma adds nothing once the forecast is calibrated.
- Code is written and tested, **default off**: the default training output is byte-for-byte what HEAD produces, and the deployed set loads with its `model_set_hash` unchanged. `pytest tests/team tests/core -q -m "not db"`: **417 passed, 8 deselected**.
- Caution for the owner: a scratch retrain on the 2026-11-01 training set with the opt-in fitted a level slope of **+0.025** (more weight on the model at high totals), the opposite sign of 6 of 7 walk-forward folds. See finding 6.

## Findings

### 1. E27 result (full tables: `reports/experiments/E27_candidate_form/RESULT.md`; decision: `decision.json`)

Registration written before any outcome: `reports/experiments/E27_candidate_form/REGISTRATION.md` (brief section verbatim, E18 rule text, rule-text sha256 `9f158f05...a451`, file sha256 `a597b60a447e1dcebe50a85b69a21048640a2c514a48f1b1368c9c405e708231`).
Command: `C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe -m experiments.e27_candidate_form` (about 20 s).

Which saved run: `E13_combined_winners/predictions.csv` (8,899 games). It is the only run that carries all three pieces (E1 spread, E3 logit-margin ML stack, E4 per-game total sigma `sigma_total_h`). Its `total_pred` equals the baseline run's on every common game (checked), and no saved run changes the total point forecast. It has no calibration file, so the fold blocks come from `baseline/calibration_set_predictions.csv`. The per-game sigma exists, so the sigma question is answered (with a caveat, finding 3).
Reproduction check on the baseline run (8,916 games): baseline 14.7721, fix (c) 14.6193, R1 14.6266 match E18's `results.json` to the last digit. R2 does not: 14.5509 here against E18's 14.6201, because the trailing-365-day mean line uses all games of the rebuilt 2026-11-01 training set rather than the lost original. Its sign agrees with E18's, and the rule only needs the sign.

| (8,899 games) | MAE [95% CI] | RMSE | bias (actual - forecast) |
|---|---|---|---|
| raw | 14.763 [14.527, 15.000] | 18.697 | +1.268 [+0.88, +1.67] |
| fix (c) | 14.610 [14.380, 14.847] | 18.471 | +0.507 [+0.11, +0.87] |
| opening line | 14.618 [14.385, 14.859] | 18.480 | +0.440 [+0.06, +0.80] |

Per-season MAE (raw / c / line): 2019-20 15.28 / 15.13 / 15.13; 2020-21 14.95 / 14.66 / 14.66; 2021-22 14.68 / 14.59 / 14.55; 2022-23 14.51 / 14.37 / 14.39; 2023-24 14.54 / 14.29 / 14.31; 2024-25 14.59 / 14.54 / 14.58; 2025-26 14.88 / 14.77 / 14.78.
Paired MAE, day bootstrap (1,000 draws, seed 42): c - raw -0.153 [-0.212, -0.102], 7/7 seasons, 4 seasons with their own CI below 0; c - line -0.008 [-0.029, +0.015], 5/7; raw - line +0.146 [+0.080, +0.223], 0/7. Paired MSE: c - raw -8.42 [-10.87, -6.21]; raw - line +8.05 [+5.21, +11.26].
Keep rule: (i) 7/7 >= 5, (ii) CI below 0, (iii) R1 -0.146 and R2 -0.147 both negative -> KEEP. Seed 20261017 (E18's): -0.153 [-0.207, -0.100], KEEP. Replication on `E25_with_odds` (independent re-run, global sigma): -0.153 [-0.212, -0.105], 7/7, R1 -0.145, R2 -0.146, KEEP.
ML and spread unchanged: the script asserts the `ml_prob` and `spread_pred` columns are bit-identical before and after, and `predict_markets` leaves them identical in the tests (fix (c) only touches `total_pred`).
Edge slope (actual - open on forecast - open): raw +0.223 [+0.096, +0.339]; fix (c) +0.697 [+0.303, +1.056].
Bias by opening-total bucket (raw / fix (c) / line): <215 -0.39 / +0.22 / +0.98; 215-225 -0.20 / +0.14 / +0.36; 225-235 +1.41 / +0.78 / +0.60; 235-242 +4.69 / +0.70 / -0.21; >242 +10.21 [+7.6, +12.9] / +1.50 [-1.2, +4.2] / -0.29 (236 games).
Fitted weights per fold: w0 0.17-0.57, w1 -0.021 to +0.004 per total point (mean level 220-227): typical weight on the model's departure from the line is about 0.3 at the mean level and falls with the level.

### 2. The 5%-rule total bets before and after (descriptive, priced at the open at -110)

| forecast | bets | over / under | hit rate [95% CI] | ROI per 100 [95% CI] |
|---|---|---|---|---|
| raw, E4 per-game sigma | 1,931 | 509 / 1,422 | 53.4% [51.3, 55.6] | +2.03 [-2.1, +6.2] |
| fix (c), calibrated global sigma | 81 | 47 / 34 | 54.3% [43.1, 64.6] | +3.70 [-17.7, +23.2] |

Neither ROI is distinguishable from 0 and the after-sample is tiny. The raw rule was 74% unders. Cover log loss is 0.6929 after calibration (a coin is 0.6931): the calibrated forecast is nearly uninformative about cover. Per the brief's P2 these opening-priced numbers say nothing about the live basis (current line); E32 covers that.

### 3. Single sigma or per-game sigma (cover log loss, lower is better)

| | log loss | paired difference [95% CI] |
|---|---|---|
| raw, E4 per-game sigma | 0.69893 | vs raw global sigma 0.69968: -0.00075 [-0.00122, -0.00030], 5/7 (E4's own result) |
| fix (c), calibrated global sigma | 0.69288 | vs raw + E4: -0.00605 [-0.00843, -0.00368], 7/7 |
| fix (c), E4 sigma as saved | 0.69290 | vs c global: +0.00002 [-0.00010, +0.00013] |
| fix (c), E4 sigma refit on calibrated out-of-fold errors | 0.69285 | vs c global: -0.00003 [-0.00013, +0.00006], 3/7 |

On the calibrated residuals the per-game sigma is no better than a single one. Caveat: the refit uses the rebuilt 2026-11-01 training set, and the same refit on RAW errors correlates only 0.61 with the saved E4 sigma (mean 19.9 vs 19.0), so it is a close cousin, not the saved model; the "as saved" row (E4's sigma fitted on raw residuals, applied to the calibrated forecast) is the clean one, and it agrees. For the code I kept the per-game sigma recomputed on calibrated errors when the opt-in is on (as the plan said); the owner may prefer to drop it (the evidence says it neither helps nor hurts).

### 4. Why a tree model of the total trails its own input line (descriptive; `descriptive.md`, `a6_descriptive.py`)

- The model's departure from the line is mostly noise: sd of the gap 3.46 points, correlation with (actual - line) +0.04, slope +0.22. MSE excess = E[gap^2] - 2E[gap x (actual - line)] = 12.65 - 4.60 = +8.05.
- The excess is one-sided and concentrated at HIGH lines: lines >= 235 are 14.4% of games and carry 66% of the MAE excess (0.096 of 0.146) and 77% of the MSE excess; lines >= 242 are 2.7% of games and 35% of the MSE excess. Top decile (237-262): mean gap -6.75 while actual - line is 0.00; above 242 the gap is -10.5. Low lines (<215) show almost no excess (+0.04 per game) and the model is not too low there. So it is not "both extremes".
- It does not follow the line smoothly: slope of the forecast on the line 0.753 against 0.962 for the actual total; correlation of the gap with the line level -0.62, and a quadratic in the line level alone explains 52.7% of the gap's variance. Half of what the model adds to the line is a smooth function of the line itself, which a tree built on history with lower totals cannot extrapolate.
- Consistent with that, the high-line problem shrinks as high totals enter the history: share of games with a line >= 235 goes from 4-8% (2019-22) to 23-25% (2023-24, 2025-26), and the mean gap on those games from -12.2 (2019-20) to -3.8 (2025-26). Descriptive; not tested as a cause.
- Constant shifts, fitted on each fold's calibration block (pooled MAE, paired vs raw / vs fix (c) / vs line):
  - L1 `line + mean(actual - line)`: 14.620; -0.143 [-0.222, -0.072] (7/7) / +0.010 [-0.013, +0.034] / +0.003 [-0.007, +0.013].
  - L2 `line + mean(actual - raw forecast)`: 14.628; -0.135 [-0.214, -0.061] / +0.018 [-0.011, +0.049] / +0.011 [-0.009, +0.032].
  - L3 `raw forecast + mean(actual - raw forecast)` (bias-only fix of the model): 14.753; -0.010 [-0.032, +0.011] vs raw, +0.143 [+0.091, +0.204] worse than fix (c).
  So a constant shift of the line beats the raw model and ties fix (c) and the line; a bias-only fix of the model does not help. Nothing here beats the line.

### 5. Code (step 2; nothing deployed)

Files changed: `team/market_models.py`, `team/train_baseline.py`, `team/model_registry.py`, and tests `tests/team/test_market_models.py`, `test_train_final_models.py`, `test_model_registry.py`, `test_model_provenance.py`.
- `TotalLevelCalibrator.fit(pred, opening_total, y)` / `.predict(pred, opening_total)` (+ `.weight`, `.params`): E18 fix (c), weight clipped to [0, 1], no opening total keeps the raw prediction; rows with a missing prediction, line or outcome are ignored in the fit.
- `predict_markets` applies `bundle.calibrator_total` only when it is present (`getattr` default None), after the total forecast, before anything reads it. ML, spread and `total_std` paths are untouched.
- `train_and_save_all(..., total_calibration=False)`: opt-in, last argument. When on it needs `use_odds=True` and `total_target='level'` (ValueError otherwise, before anything is written); fits the calibrator on the calibration-block total forecasts, saves `calibrator_total<suffix>.pkl`, recomputes `total_std`, bias, MAE and the per-game sigma (`fit_hetero_sigma(..., calibrator=)` feeds the out-of-fold errors of the calibrated forecast) on the calibrated residuals, and writes `total_calibrator`, `total_calibrator_params`, `total_std_raw`, `total_bias_raw`, `total_mae_raw` into `calibration_meta`.
- `model_registry`: `calibrator_total` added to `HASHED_STEMS` (after `sigma_total`), `OPTIONAL_STEMS` = sigma_total and calibrator_total, `REQUIRED_STEMS` unchanged in content, `ModelBundle.calibrator_total` (last field, default None), `load_bundle` loads it if present and refuses a set whose meta declares `total_calibrator` but lacks the file.
Proofs (scripts in this folder):
- Deployed set: `check_deployed_unchanged.py save` (before the edits) and `check` (after): `_with_odds_cand_20261019` loads; manifest hash = recomputed hash = bundle hash = `0165d3361e4228ad2551a787c66e1cd82ce11bd5b9a4a258b7e79467e7cd02e0` (hash_version 1); `predict_markets` on 1,500 rows is bit-identical to the pre-change output (`deployed_check_check.json`). Test `test_the_deployed_set_still_loads_and_its_hash_equals_its_manifest` pins this.
- Default training: `check_default_training_identical.py` trains the same synthetic history with git HEAD's `team/` and the edited `team/` (tuner grid): all 11 saved files identical, `model_set_hash` identical (`default_training_identical.json`).
- Tests: calibrator equals E18's `make_fix('c')` on synthetic data (weights and predictions to 1e-9) and, on the real saved baseline run, the fold weights and fold MAE of two folds equal E18's recorded values (skipped where the gitignored predictions are absent; it ran here); clipping, no-line, empty fit; pickle round trip; `predict_markets` applies it only when present and leaves ML, spread and `total_std` bit-identical; an old set predicts exactly as before; default training leaves no calibrator file or meta key; the opt-in run differs from the default run only in `calibrator_total`, `sigma_total`, `calibration_meta` and the manifest (ML, spread, total model and feature config byte-identical); the hash changes when the file is added, rewritten or removed and returns to the old value when removed; an edit after the manifest is refused.
- `pytest tests/team tests/core -q -m "not db"`: **417 passed, 8 deselected, 0 failed** (350 s). This includes other workstreams' in-flight edits to the shared worktree at that moment.

### 6. What would change at the next retrain with the opt-in on

Scratch candidate (`scratch_retrain_with_optin.py`, tuner grid, 2026-11-01 training set, written under this folder's `scratch/`, never `models/saved/`):
- Fitted on 1,326 calibration-block games (2025-10-31 to 2026-10-30): w0 0.497, w1 +0.0252 per point, mean level 229.5, i.e. weight on the model 0.13 at a 215 total, 0.38 at 225, 0.64 at 235, 0.89 at 245.
- Calibration-block total MAE 14.705 -> 14.646, bias -0.476 -> +0.047, std 18.557 -> 18.489 (in-sample for two parameters; the walk-forward evidence is E27). Mean per-game sigma 18.59.
- New file `calibrator_total<suffix>.pkl`; `sigma_total`, `calibration_meta` (`total_std`, `total_bias`, `total_mae` now calibrated) and the manifest/`model_set_hash` all change. ML and spread files are byte-identical to a default run.
- **Slope sign.** Walk-forward folds had w1 between -0.036 and +0.015 (negative in 6 of 7 with 1,500-3,000 calibration rows; also on the last 1,300 rows: -0.005, -0.015, -0.036, +0.015, -0.014, -0.008, -0.011) and a weight at a 245 total of about 0 to 0.3 (for example 0.13 in the 2025-26 fold). The deployed one-year block gives +0.025, which shrinks high totals much LESS (0.89 at 245) than any walk-forward fold, so the calibrator trained that way would not fix the high-total bias the E18/E27 gain comes from. The slope is the weakly determined parameter on about 1,300 games (I did not compute its standard error). Walk-forward on only the last 1,300 calibration rows of each fold still gives -0.150 [-0.209, -0.098], 7/7 (E27 RESULT, sensitivity section), so the principle holds; this block's slope may be an outlier.
- Backtest consequence: about 12 total bets per season instead of about 275 at the 5% rule (rule priced at the open).

## Decisions for the owner

1. Whether to use fix (c) at all. Evidence says it makes the total level with the line and removes the high-total bias; it creates no edge, and "level with the line" can also be had with a constant shift of the line (L1) or by dropping the model's total. Fix (c) keeps the model's information (edge slope 0.70 [0.30, 1.06]) at the price of a level slope that needs a large calibration block.
2. If yes: how to treat the poorly determined level slope (w1). Options I did not build: pin w1 to a pooled walk-forward value, use a larger calibration block for the calibrator only, or use a constant weight. Check the fitted weights of the candidate before any deployment; a positive w1 as in the scratch fit should be a stop sign.
3. Keep or drop the per-game total sigma on a calibrated set (no evidence it helps).
4. Wiring that is outside my files and not done: `team/retrain.py` and `scripts/generate_season_predictions.py` call `train_and_save_all` without the argument (a retrain stays uncalibrated until someone passes it); `team/explain.py` computes the total forecast from the raw model (`bundle.total.predict`), so on a calibrated set the explanation panel would show the raw total, not the logged one; `predictions_log_v2.total_pred` would store the CALIBRATED forecast (the raw model's own number would no longer be logged), which matters for the E26 forward test of the total (PLAN step 6, E26b).
5. Live basis (brief P1, P2): E27 prices at the open. Under the live rule (judged at the 11 am line) shrinking toward the OPENING line changes which side the rule picks; the E32 workstream measures the live-analogue rule, not this.

## Not verified / limits

- The saved walk-forward runs used the pre-audit split (uncapped last-20% calibration block, no refit on all data). The deployed trainer uses a one-year block and refits on all data. No saved run has that scheme together with E1/E3/E4. The 1,300-row sensitivity approximates the block size only (the base model that made those predictions was trained on a different history).
- `E25_with_odds` is a replication under the same pre-audit scheme (16,218 calibration rows, no fold column, no split.json), not the "current harness" as my registration said (erratum recorded in the E27 RESULT).
- R2 differs from E18's saved number (rebuilt training set), E4's sigma refit does not reproduce the saved sigma (corr 0.61); both disclosed above and neither changes a verdict.
- All E27 numbers are priced at or measured against the OPENING line; E27 says nothing about live performance.
- I did not compute a standard error for w1, did not run the full retrain with FLAML, and did not train or score anything on 2026-27 games.
- Two other places read the raw total (`team/explain.py`, any script calling `bundle.total.predict` directly); I did not change or test them (not mine).
- The saved predictions are gitignored and absent from the worktree; scripts read them read-only from `C:/Users/jorda/nba-model-hybrid/reports/experiments` (override with `E27_SRC`).

## Files

- `reports/experiments/E27_candidate_form/` : `REGISTRATION.md`, `RESULT.md`, `results.json`, `decision.json`; script `experiments/e27_candidate_form.py`.
- This folder: `RESULT.md`, `a6_descriptive.py`, `descriptive.md`, `descriptive.json`, `write_registration.py`, `check_deployed_unchanged.py`, `deployed_check_save.json`, `deployed_check_check.json`, `check_default_training_identical.py`, `default_training_identical.json`, `scratch_retrain_with_optin.py`, `scratch_retrain_with_optin.json`, `scratch/` (logs and the scratch candidate; safe to delete).
