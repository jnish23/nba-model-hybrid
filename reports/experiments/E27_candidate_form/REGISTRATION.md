# E27 - totals level calibration (E18 fix c) on the deployed form: registration

Written 2026-11-11 by workstream A6 of the pregame audit, BEFORE any E27 outcome was computed. Only file layouts, column names and row counts of the saved runs
were inspected, and E18's published headline (MAE 14.77 -> 14.62 for fix c, in `reports/upgrade_2026-27/PLAN.md`) had been read. No E27 residual had been computed.

## 1. The E27 section of the audit brief (verbatim)

### E27: totals level calibration (E18 fix c) on the deployed form (re-scoring saved predictions; no retraining)

As described in `reports/upgrade_2026-27/PLAN.md` step 1. Keep rule: E18's registered rule in `reports/experiments/PREREGISTRATION.md`
(better total MAE in at least 5 of 7 seasons, pooled CI below 0, and its R1/R2 robustness checks negative), with ML and spread unchanged.
If E27 fails, say so and stop before writing any calibrator code.

## 2. The exact E18 rule text applied (verbatim from `reports/experiments/PREREGISTRATION.md`, section "E18")

**Step 2 (candidate fixes, each fitted only on that fold's calibration block, applied to that fold's test games).**
...(fixes (a) and (b) omitted here; they are not part of E27)...
- (c) Level-dependent shrinkage toward the line: OLS without intercept of `actual - line` on `(pred - line)` and `(pred - line) x (opening total - calibration mean)`; weight `w = b0 + b1 (L - mean)` clipped to [0, 1]; corrected = line + w (pred - line).
- Robustness R1 ("recent only"): refit (a), (b), (c) on the later 50% of the fold's calibration block by date. R2 ("era-adjusted level"): (b) and (c) with `L` = opening total minus the mean opening total of all games in the previous 365 days (past lines only).

**Step 3 (metrics and the keep rule).**
- Total MAE (primary), per season and pooled. Cover log loss as in E4: `p(over) = Phi((pred - line)/sigma)`, outcome over versus under with pushes dropped, `sigma` = standard deviation of the calibration-block residuals after the same fix (baseline: its saved `sigma_total`).
  5%-edge total bets from `metrics.make_bets(thr=0.05)`: count, hit rate and return at -110 as secondary.
- **KEEP a fix iff** (i) its total MAE is lower than the baseline's in at least 5 of 7 seasons, **and** (ii) the pooled paired MAE difference has a day-bootstrap interval entirely below 0 (the E1-E13 main criterion), **and** (iii) in both R1 and R2 the pooled paired MAE difference
  is also negative (point estimate); otherwise "not kept", and a fix that meets (i) and (ii) but fails (iii) is labelled "era-confounded".
  The user's stricter wording ("error bars excluding zero in at least 5 of 7 seasons") is reported alongside as the count of seasons whose own paired interval excludes 0; it is not the keep rule.

sha256 of the rule text above (the block above from the Step 2 header to the end of the keep rule, exactly as printed including the omission marker, UTF-8, LF): `9f158f05cf76b8a57ea58757497aea20332e804f13faf1ab3bee8a7c2a8ca451`
sha256 of `reports/experiments/PREREGISTRATION.md` at the time of writing: `95c4383066437798b240deb3d6c9594a7cca4449d8d751acd2d8f30947881832`

Fix (c) is implemented exactly as `experiments/e18_calibration.py::make_fix('c', ...)`: OLS without intercept of `target_total - line` on `[gap, gap * (line - mu)]`
(gap = pred - line, mu = calibration-block mean opening total), weight `w = clip(b0 + b1 (line - mu), 0, 1)`, corrected = `line + w * gap`; rows without an opening total keep the raw prediction.
Fitted per fold on that fold's calibration block only (rows with finite line, prediction and outcome). R1 refits on the later 50% of the block by date; R2 uses the level
`opening total - mean opening total of all games in the previous 365 days (past lines only)`.

## 3. Choices made by A6 for E27 (fixed here, before any outcome)

- Candidate form: E1 residual spread + E3 margin-stack ML + E4 per-game total sigma (the deployed form), plus fix (c) on the total.
- Run used for the primary test: `reports/experiments/E13_combined_winners/predictions.csv` (8,899 games with both opening lines, 7 walk-forward test seasons). It is the only saved run
  that carries all three pieces: its `spread_pred` is E1's, `ml_prob` is E3's logit-margin stack on E1's margin, and `sigma_total_h` is E4's per-game total sigma. Its `total_pred` is the baseline total
  forecast (the form E18 corrected), and no saved run changes the total point forecast. It has no calibration-set file, so the fold-by-fold calibration blocks are
  `reports/experiments/baseline/calibration_set_predictions.csv` (labelled with `experiments.lib.label_calib_folds`); the total predictions there are the same model's, so they are valid for E13's
  total. Both runs were made under the pre-audit split (uncapped last-20% calibration block, no refit on all data). The deployed trainer uses a one-year calibration block and refits on all data
  (not the same scheme); `reports/experiments/E25_with_odds/` was made under the current harness and carries the baseline total forecast and global sigma only, so it is scored as a replication (same code, global sigma).
  This difference is reported as a limit.
- Reproduction check: the same code on `reports/experiments/baseline/predictions.csv` (8,916 games) must reproduce E18's `results.json` MAE for baseline and fix (c) before the E27 numbers are trusted.
- Keep rule applied to the primary run (E13): (i) fix (c) total MAE below raw in at least 5 of 7 seasons; (ii) pooled paired MAE difference (c - raw), day-bootstrap 95% interval entirely below 0;
  (iii) R1 and R2 pooled paired differences negative (point estimates); (iv) ML and spread columns unchanged (identical to E13's, bit for bit). Day bootstrap: 1,000 draws over game dates. Seed 42 is the brief's rule and
  decides; the same rule is also evaluated with E18's seed 20261017 and both are reported (if they disagree, that is stated).
- Metrics: total MAE, RMSE, bias (actual - forecast) for raw, fix (c) and the opening line; pooled and per season; paired day-bootstrap intervals for c - raw, c - line and raw - line (MAE and MSE).
  Cover log loss as in E4/E18 (`p(over) = Phi((forecast - line)/sigma)` with `betting.cover_probabilities`, pushes dropped), with sigma options: raw: E4 per-game sigma (saved) and the fold's global sigma;
  fix (c): the fold's calibrated global sigma (std of calibration-block residuals after the fix), E4's per-game sigma as saved (fitted on raw residuals), and E4's per-game sigma refitted on forward-chained out-of-fold calibrated errors
  (E4's procedure, `experiments/e04_sigma.py`, using fix (c) with the fold's parameters). "Better sigma" is read from the paired cover log loss difference with a day-bootstrap interval.
- 5% rule: `metrics.make_bets(thr=0.05, markets=('total',))` with the per-game sigma (raw: saved E4 sigma; fix (c): the sigma that won the sigma question, otherwise the calibrated global one), priced at -110 at the opening line:
  count, over/under split, hit rate, ROI per 100 with day-bootstrap intervals. Descriptive; nothing is decided from it.
- Edge slope: OLS (intercept) of `(actual total - opening total)` on `(forecast - opening total)`, raw and fix (c), day-bootstrap interval. Descriptive.
- Bias by opening-total bucket: below 215, 215-225, 225-235, 235-242, above 242 (upper edge exclusive of the next bucket); bias = actual - forecast; raw, fix (c) and the line.
- Descriptive extras (no decision): `line + mean calibration-block residual` variants against raw and fix (c); error by line level for the raw model versus the line.
