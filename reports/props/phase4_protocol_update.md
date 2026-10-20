# Phase 4 protocol update after the props-onto-main audit (2026-10-20)

Status: **binding from now on; supersedes the season roles in `phase4_plan.md` section 2.1.** The user asked for a verified assessment of the audit; the assessment is in `phase4_audit_response.md`. This note records only the protocol consequences.

## 1. 2024-25 is no longer a clean validation season

The audit is right. The plan said remedies would be designed on the dev folds and applied once to 2024-25. In practice I formed the seasonality hypothesis (R1) from 2024-25 and 2023-24 monthly bias, and I discussed the isotonic layer using what it did on 2024-25, before the remedy rules were evaluated. The rules themselves were scored on dev folds only, but a pass on 2024-25 would not be an out-of-sample result any more. The user also asked me to look at 2024-25 first, so this is a consequence of that request, not a mistake in following it.

New roles:

| Season | Role | May be used for |
|---|---|---|
| 2021-22, 2022-23, 2023-24 | development folds | every model, feature, bias-remedy and calibration decision |
| 2024-25 | **seen**: confirmation only | reporting how a frozen configuration behaves; **no decision may use its outcomes from now on**. Labelled "seen" wherever it appears |
| 2025-26 | single clean holdout | scored once, after the configuration is frozen, for models AND (see below) betting rules |

Consequences for Phase 5 (decision for the user at the Phase 5 plan): betting and market rules were to be developed on 2023-24 and validated on 2024-25. 2024-25 prices have never been used to choose a rule, so it is still clean for market rules, but the model configuration that generates the probabilities was influenced by its outcomes. Recommended: develop market rules on 2023-24 with walk-forward inside it, treat 2024-25 as a second development season for rules (not for the model), and score 2025-26 once for everything. This costs power (one final season); the Phase 5 plan will carry a power calculation.

## 2. Classes of change, and how each is adopted

- **Corrections of validity (adopted without a performance test, because they are about leakage or optimism, not accuracy):** injury publish-lag buffer (15 min), strict snapshot timing for the market environment, `env_source` never a model input, forward-chaining out-of-fold residuals with out-of-sample sd for the minutes bank, push-aware probabilities, dataset hash in manifests.
- **Design corrections judged for harm only:** refit the models on fit + calibration before predicting the test season. Adopted unless a core stat's paired CRPS difference is significantly worse on a dev fold.
- **Pre-registered candidates (tried rules, counted for Phase 5 multiple testing):**
  - R1 (drop the two season-position features): evaluated, failed (see interim report).
  - R2 (rolling 21-day ratio, prior 2,000 player-games): evaluated on the v2 pipeline, missed its bar by 3 points of bias shrink. Will be re-evaluated, with the same constants, on the corrected pipeline because its baseline changed; that second evaluation counts as one more tried rule.
  - **R3 (recency weights, half-life 730 days, new):** exponential recency weights on training rows, age relative to the end of the training window, applied to every component. Fixed now, before its result: adopt if, on 2023-24, the mean absolute bias over the core stats (points, rebounds, assists, threes, pra) shrinks by at least one third relative to the same pipeline without recency, AND no core stat's paired CRPS difference interval lies entirely above zero on 2022-23 or 2023-24.
  - R2b (season-boundary carry-over, prior 1,000): only if the user allows it.

## 3. What this changes in the existing numbers

Everything reported on 2024-25 in `phase4_interim_report.md` and `phase4_eval_2024-25.md` stays as a "seen" result. The go/no-go criteria are re-scored on the final frozen configuration on 2025-26 only, once.
