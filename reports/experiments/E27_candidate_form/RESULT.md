# E27 - totals level calibration (E18 fix c) on the deployed form: result

Registration: `REGISTRATION.md` (sha256 `a597b60a447e1dcebe50a85b69a21048640a2c514a48f1b1368c9c405e708231`), written before any outcome. Run: `python -m experiments.e27_candidate_form`.
**Verdict: KEEP** (seed 42, the brief's seed) / KEEP (E18's seed 20261017). ML and spread unchanged: yes (columns carried through bit for bit; fix (c) only touches the total).

## Data

- Primary run: `E13_combined_winners/predictions.csv`, 8899 games, 7 seasons (spread = E1, ML = E3 stack, per-game total sigma = E4, total forecast = baseline total; equals the baseline run's `total_pred` on every game: True).
- Calibration blocks: `baseline/calibration_set_predictions.csv` (pre-audit uncapped last-20% blocks, 16,218 rows), labelled per fold.
- Reproduction of E18 on the baseline run (8,916 games): c MAE 14.6193 vs E18 14.6193; c_R1 MAE 14.6266 vs E18 14.6266; c_R2 MAE 14.5509 vs E18 14.6201; baseline MAE 14.7721 vs E18 14.7721. Fix (c), R1 and the baseline reproduce E18 to the last digit; R2 does not (it uses the trailing-365-day mean line of ALL games, taken here from the rebuilt 2026-11-01 training set rather than the lost original), but its sign agrees (E18 saved R2 14.620 vs 14.772 baseline).

- Registration erratum: `REGISTRATION.md` says `E25_with_odds` was made under the current harness. File evidence says otherwise: its calibration file has 16,218 rows (the uncapped last-20% layout), no fold column and no split.json, so it is a second walk-forward run of the same scheme as the baseline (different training-set rebuild and feature selection; `total_pred` differs from the baseline run by up to 24 points on single games). It is a replication, not a test of the deployed one-year-block scheme.
- Limit: the deployed trainer fits on a one-year (about 1,300-game) calibration block and refits the base models on all data; no saved walk-forward run used that scheme together with the E1/E3/E4 pieces, so fix (c) was not scored under it.

## Keep rule (E18 rule applied to fix (c), against the raw total forecast)

- seed 42: seasons better 7/7 (rule: at least 5); pooled paired MAE diff -0.153 [-0.212, -0.102] (rule: CI below 0: True); R1 -0.146, R2 -0.147 (rule: both negative: True); seasons whose own CI excludes 0: 4/7 -> KEEP
- seed 20261017: seasons better 7/7 (rule: at least 5); pooled paired MAE diff -0.153 [-0.207, -0.100] (rule: CI below 0: True); R1 -0.146, R2 -0.147 (rule: both negative: True); seasons whose own CI excludes 0: 4/7 -> KEEP
- Replication on `E25_with_odds` (independent re-run of the baseline total on the 2026-10-19 training set, global sigma only): 7/7, -0.153 [-0.212, -0.105], R1 -0.145, R2 -0.146 -> KEEP

## Total MAE / RMSE / bias (actual - forecast), 8,899 games

| | MAE pooled [95% CI] | RMSE | bias [95% CI] | 2019-20 | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|---|---|---|---|---|---|
| raw model | 14.763 [14.527, 15.000] | 18.697 | +1.268 [+0.88, +1.67] | 15.28 / +2.44 | 14.95 / +1.00 | 14.68 / +0.11 | 14.51 / +1.80 | 14.54 / +1.51 | 14.59 / +0.80 | 14.88 / +1.34 |
| fix (c) | 14.610 [14.380, 14.847] | 18.471 | +0.507 [+0.11, +0.87] | 15.13 / +1.16 | 14.66 / -0.33 | 14.59 / -0.12 | 14.37 / +0.78 | 14.29 / +0.56 | 14.54 / +0.61 | 14.77 / +0.88 |
| opening line | 14.618 [14.385, 14.859] | 18.480 | +0.440 [+0.06, +0.80] | 15.13 / +1.12 | 14.66 / -0.53 | 14.55 / +0.00 | 14.39 / +0.60 | 14.31 / +0.21 | 14.58 / +0.73 | 14.78 / +0.93 |

Per-season cells are MAE / bias. Paired differences (negative: first is better):

| comparison | MAE diff [95% CI] | seasons first better | MSE diff [95% CI] | 2019-20 | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 (MAE diff) |
|---|---|---|---|---|---|---|---|---|---|---|
| fix (c) - raw | -0.153 [-0.212, -0.102] | 7/7 | -8.42 [-10.87, -6.21] | -0.16 | -0.29 | -0.09 | -0.14 | -0.25 | -0.05 | -0.12 |
| fix (c) - line | -0.008 [-0.029, +0.015] | 5/7 | -0.37 [-1.15, +0.48] | -0.00 | +0.00 | +0.04 | -0.02 | -0.02 | -0.04 | -0.01 |
| raw - line | +0.146 [+0.080, +0.223] | 0/7 | +8.05 [+5.21, +11.26] | +0.16 | +0.29 | +0.13 | +0.12 | +0.23 | +0.01 | +0.10 |

## Cover log loss (p(over) = Phi((forecast - line)/sigma), pushes dropped; lower is better)

| forecast + sigma | log loss |
|---|---|
| raw_E4_sigma | 0.69893 |
| raw_global_sigma | 0.69968 |
| c_global_calibrated_sigma | 0.69288 |
| c_E4_sigma_as_saved | 0.69290 |
| c_E4_sigma_refit_calibrated | 0.69285 |

| paired difference (first minus second) | diff [95% CI] | seasons first better |
|---|---|---|
| c_global_minus_raw_E4 | -0.00605 [-0.00843, -0.00368] | 7/7 |
| c_global_minus_raw_global | -0.00680 [-0.00928, -0.00436] | 7/7 |
| raw_E4_minus_raw_global | -0.00075 [-0.00122, -0.00030] | 5/7 |
| c_E4refit_minus_c_global | -0.00003 [-0.00013, +0.00006] | 3/7 |
| c_E4saved_minus_c_global | +0.00002 [-0.00010, +0.00013] | 4/7 |
| c_E4refit_minus_raw_E4 | -0.00608 [-0.00850, -0.00368] | 7/7 |

Sigma caveat: `c_E4_sigma_refit_calibrated` refits E4's per-game sigma model on forward-chained out-of-fold errors of the calibrated forecast, using the rebuilt 2026-11-01 training set; the same refit on RAW errors correlates only 0.61 with the saved E4 sigma (mean 19.93 vs 18.97), so it is a different sigma model from the one that was saved, not a reproduction.

## 5%-rule total bets priced at the open, -110 (descriptive)

| forecast | bets | over / under | hit rate [95% CI] | ROI per 100 [95% CI] |
|---|---|---|---|---|
| raw, E4 per-game sigma | 1931 | 509 / 1422 | 53.4% [51.3, 55.6] | +2.03 [-2.1, +6.2] |
| fix (c), calibrated global sigma | 81 | 47 / 34 | 54.3% [43.1, 64.6] | +3.70 [-17.7, +23.2] |

## Edge slope: (actual - open) on (forecast - open), intercept included

- raw: +0.223 [+0.096, +0.339]
- fix (c): +0.697 [+0.303, +1.056]

## Bias (actual - forecast) by opening-total bucket

| bucket | games | raw bias [95% CI] | fix (c) bias [95% CI] | line bias [95% CI] |
|---|---|---|---|---|
| <215 | 995 | -0.39 [-1.4, +0.7] | +0.22 [-0.8, +1.3] | +0.98 [-0.0, +2.0] |
| 215-225 | 3093 | -0.20 [-0.9, +0.4] | +0.14 [-0.5, +0.8] | +0.36 [-0.3, +1.0] |
| 225-235 | 3526 | +1.41 [+0.8, +2.0] | +0.78 [+0.2, +1.4] | +0.60 [-0.0, +1.2] |
| 235-242 | 1049 | +4.69 [+3.6, +5.8] | +0.70 [-0.4, +1.8] | -0.21 [-1.3, +0.9] |
| >242 | 236 | +10.21 [+7.6, +12.9] | +1.50 [-1.2, +4.2] | -0.29 [-2.9, +2.4] |

## Fitted fix (c) parameters per fold (weight w = w0 + w1 (line - mean_level), clipped to [0, 1])

| fold | w0 | w1 | mean_level |
|---|---|---|---|
| 2019-20 | 0.418 | -0.0170 | 220.0 |
| 2020-21 | 0.481 | -0.0206 | 222.3 |
| 2021-22 | 0.572 | -0.0201 | 223.7 |
| 2022-23 | 0.248 | -0.0054 | 222.3 |
| 2023-24 | 0.173 | +0.0039 | 224.4 |
| 2024-25 | 0.336 | -0.0109 | 227.3 |
| 2025-26 | 0.345 | -0.0121 | 227.0 |

## Sensitivity (not registered; no decision): fix (c) refitted on only the last 1,300 rows of each fold's calibration block

The deployed trainer fits on a one-year block (about 1,300 games); the saved runs used 1,500-3,000. This approximates the deployed block size.

- pooled MAE 14.613, bias +0.532; fix (c) - raw -0.150 [-0.209, -0.098], 7/7 seasons better; fix (c) - line -0.005 [-0.028, +0.020]
- fitted weights (w0, w1 per total point, mean level): 2019-20: 0.25, -0.0052, 221; 2020-21: 0.45, -0.0152, 222; 2021-22: 0.67, -0.0362, 224; 2022-23: 0.18, +0.0146, 221; 2023-24: 0.32, -0.0144, 228; 2024-25: 0.24, -0.0076, 227; 2025-26: 0.34, -0.0106, 226
- bias by bucket (actual - forecast): <215 +0.22, 215-225 +0.14, 225-235 +0.82, 235-242 +0.94, >242 +0.99
