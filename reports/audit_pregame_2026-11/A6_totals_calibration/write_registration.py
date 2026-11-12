"""Writes reports/experiments/E27_candidate_form/REGISTRATION.md (before any E27 outcome is computed)."""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
brief = (ROOT / 'reports/audit_pregame_2026-11/BRIEF.md').read_text(encoding='utf-8').splitlines()
pre = (ROOT / 'reports/experiments/PREREGISTRATION.md').read_text(encoding='utf-8')
pl = pre.splitlines()
i0 = next(i for i, l in enumerate(brief) if l.startswith('### E27'))
e27 = '\n'.join(brief[i0:]).rstrip()
a = next(i for i, l in enumerate(pl) if l.startswith('- (c) Level-dependent shrinkage'))
r = next(i for i, l in enumerate(pl) if l.startswith('- Robustness R1'))
k0 = next(i for i, l in enumerate(pl) if l.startswith('**Step 3 (metrics and the keep rule).**'))
k1 = next(i for i, l in enumerate(pl) if l.startswith('- Descriptive only: MAE, bias'))
h2 = next(i for i, l in enumerate(pl) if l.startswith('**Step 2 (candidate fixes'))
rule = '\n'.join([pl[h2], '...(fixes (a) and (b) omitted here; they are not part of E27)...', pl[a], pl[r], '', *pl[k0:k1]]).rstrip()
rule_sha = hashlib.sha256(rule.encode('utf-8')).hexdigest()
hdr = '\n'.join(pl[a - 4:a - 1]).strip()   # the step-2 header and the fold-by-fold statement
step2 = pl[a - 1]
text = f"""# E27 - totals level calibration (E18 fix c) on the deployed form: registration

Written 2026-11-11 by workstream A6 of the pregame audit, BEFORE any E27 outcome was computed. Only file layouts, column names and row counts of the saved runs
were inspected, and E18's published headline (MAE 14.77 -> 14.62 for fix c, in `reports/upgrade_2026-27/PLAN.md`) had been read. No E27 residual had been computed.

## 1. The E27 section of the audit brief (verbatim)

{e27}

## 2. The exact E18 rule text applied (verbatim from `reports/experiments/PREREGISTRATION.md`, section "E18")

{rule}

sha256 of the rule text above (the block above from the Step 2 header to the end of the keep rule, exactly as printed including the omission marker, UTF-8, LF): `{rule_sha}`
sha256 of `reports/experiments/PREREGISTRATION.md` at the time of writing: `{hashlib.sha256(pre.encode('utf-8')).hexdigest()}`

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
"""
out = ROOT / 'reports/experiments/E27_candidate_form/REGISTRATION.md'
out.write_text(text, encoding='utf-8')
print(rule_sha, hashlib.sha256(out.read_bytes()).hexdigest())
