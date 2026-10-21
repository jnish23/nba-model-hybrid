# Phase 4 interim report: evaluation against outcomes (2024-25 validation)

> **Update 2026-10-20:** a later audit changed the protocol and several pipeline details; read [phase4_audit_response.md](phase4_audit_response.md) and [phase4_protocol_update.md](phase4_protocol_update.md) first. All 2024-25 figures below are now labelled *seen* (not out of sample for decisions), the seasonality hypothesis (R1) was refuted, and the go/no-go criteria are re-scored once on 2025-26 after the configuration is frozen.


Date 2026-10-19/20. Full tables: [phase4_eval_2024-25.md](phase4_eval_2024-25.md). Plan and pre-registered rules: [phase4_plan.md](phase4_plan.md). **2025-26 is untouched.** The feature-group ablations are still running (section 5); everything else in the plan is done except the decision in section 6.

## 1. Headline

The model **clearly beats every baseline** on 2024-25 (criterion 1 passes in all 8 stats, intervals well clear of zero), but it **over-predicts the mean by 3-6%** in every stat, so the bias and calibration criteria (2 and 3) fail. Neither pre-registered remedy was adopted. The plan's own fallback applies: Phase 5 can proceed on the information test and CLV rather than on raw model edge, with the bias reported.

## 2. Pre-registered go/no-go criteria on 2024-25 (final configuration = `v2_base` method)

| # | Criterion | Result |
|---|---|---|
| 1 | CRPS beats the best baseline, paired 95% game-clustered interval below 0, all 8 stats | **PASS** 8/8 |
| 2 | Mean bias within +/-2% for points, rebounds, assists (or an adopted rolling recalibration meets it) | **FAIL**: points -4.2%, rebounds -2.9%, assists -5.0% (intervals exclude 0); no remedy adopted |
| 3 | ECE of P(over) at the ladder lines at most 0.02 in each stat | **FAIL**: 0.021 to 0.037 raw (worst pra) |
| 4 | CRPS at T-5 not significantly worse than at T-240 | **PASS** (no stat worse; see 3.4) |

Correction: my first run of criterion 4 encoded "not worse" as "significantly better" and reported FAIL; the plan says "not worse", the code is fixed and the committed script and report are the corrected ones.

## 3. Findings

**3.1 Accuracy (CRPS, 2024-25, model vs best of three baselines).** points 3.168 vs 3.322 (-4.7%), rebounds 1.301 vs 1.370 (-5.1%), assists 0.892 vs 0.931 (-4.1%), threes 0.585 vs 0.601 (-2.6%), pra 4.177 vs 4.428 (-5.7%), pr -5.1%, pa -5.4%, ra -5.6%. All paired intervals (game-clustered; date-clustered as a check) lie entirely below zero. The new baseline, the player's own last-20-games distribution rescaled to his expected minutes, is worse than the better of the Poisson and Normal baselines in all 8 stats, so it never set the bar. The Normal baseline's sd is fitted on the season itself, which favours it.

**3.2 Bias.** Observed / predicted - 1: points -4.2% [-4.8, -3.6], rebounds -2.9%, assists -5.0%, threes -5.9%, pra -4.0%. It is present from the first month (October -2%, November -3%) and grows to about -5% by January, then holds. By projected minutes it is worst for bench players (under 15 min: points -10%, threes -10%) but present for starters too (28+ min: points -4.0%, assists -5.2%, threes -6.4%). By position see the full report.

**3.3 My seasonality hypothesis was wrong.** I proposed that the model had learned a late-season minutes uplift through two season-position features (`ctx_team_game_number`, `rest_games_played_season`). R1 tested exactly that on the 2023-24 dev fold: removing them did not reduce the bias (mean absolute bias over the core stats 3.17% to 3.40%) and made CRPS significantly worse for points, rebounds and pra. R1 failed its pre-registered rule clearly, and the cause of the over-prediction is still unexplained.

**3.4 Late information adds almost nothing in this model.** CRPS at T-5 vs T-240 differs by at most 0.4% and the paired difference is not significant in any stat (points -0.0028 [-0.0066, 0.0008]). With injury status, lineup-relevant availability and team context in the features, the model's accuracy barely depends on information arriving between 4 hours and 5 minutes before the tip. This matters for H1 (news lag): the model itself does not extract much from late news, so any H1 edge would have to come from the market being slow, tested in Phase 5 against prices, not from this accuracy comparison.

**3.5 Calibration.** Raw P(over) at the ladder lines: log loss 0.46 (threes) to 0.67 (pra), ECE 0.021-0.037, i.e. systematically too many overs. The bias explains it. The isotonic layer I removed after the dev folds would have cut ECE to 0.003-0.012 on this season (shown in the Phase 3 smoke run) because its calibration window already contained the bias; I am not switching it on because of what the validation season shows.

## 4. Pre-registered remedies, decided on the development folds

Rules (fixed before any result): see `scripts/props_phase4_remedies.py`. Intervals are 95% game-clustered.

| Rule | 2023-24 mean abs bias, core stats | CRPS harm? | Verdict |
|---|---|---|---|
| R1: drop the two season-position features, retrain | 3.17% to 3.40% (needs -33%) | points, rebounds, pra significantly worse | **FAIL** |
| R2: rolling 21-day ratio, prior 2,000 player-games | 3.17% to 1.68% = **-47%** (needs -50%) | none on 2022-23 or 2023-24; CRPS significantly **better** on 2023-24 (points -0.0116 [-0.0142, -0.0090]) | **FAIL by 3 points of shrink** |
| R1+R2 | not run: R1 is harmful, so combining adds nothing to decide | | not adopted |

R2 narrowly misses the bar I set. Part of the shortfall is built in: with about 3,600 player-games in a 21-day window and a prior worth 2,000, a steady bias can be removed by at most about 64%, and in the early weeks of a season the window is nearly empty. I have **not** changed R2's constants to make it pass. Tried rules so far: 2 evaluated (R1, R2), 1 skipped (R1+R2).

## 5. Still running

Feature-group ablations (injury, teammates, opponent, team_context, rest, schedule, role, conversion; one group dropped at a time; frozen `v2_base` families): 2023-24 first, then 2024-25, about 12 minutes per fit, `data/raw/ablation_reports/`. They are descriptive (H4 reads from the opponent and team_context rows) and nothing is selected from them. Component ablations (no widening, point minutes instead of the mixture, single rate family) are planned and not started.

## 6. Decision for you

Criteria 2 and 3 fail, so per the plan Phase 5 would proceed with the bias reported and conclusions drawn from CLV and the information test, not from raw model edge. One option is open and it is yours to choose:

- **A. Stand by the rules.** R2 is not adopted. Phase 5 uses the fallback. Cleanest for the multiple-testing budget.
- **B. Allow one more pre-registered variant, R2b**, designed and judged on the dev folds only, then applied once to 2024-25. The only changes from R2: the window carries across the season boundary (the previous season's last completed games count, which fixes the empty-window start) and the prior is worth 1,000 player-games instead of 2,000. Same adoption rule as R2 (at least half the bias removed on 2023-24, no CRPS harm on 2022-23 or 2023-24). Tried-rule count would be 3 evaluated (R1, R2, R2b), which Phase 5 counts.

I recommend B: the bias is large, persistent and visible in the first month, R2 helped CRPS significantly in the one fold that had the bias, and the fix is cheap and strictly point-in-time. If you choose A I stop here and write the Phase 4 gate report once the ablations finish.
