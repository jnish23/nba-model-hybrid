# Phase 4 plan: evaluation against outcomes (2024-25 validation)

Date 2026-10-19. Nothing below has been run except the single 2024-25 fold described in section 1. 2025-26 is untouched.

## 1. First, the bias measurement you asked for (2024-25, frozen v2_base method)

Run: `scripts/train_props_models.py --version v2_val_2024-25 --folds 2024-25 --families-from v2_base` (rate families frozen from the dev folds, same hyperparameters, same dispersion method; the script now refuses 2024-25/2025-26 without `--families-from`). Diagnostics: `scripts/props_phase3_diagnostics.py`. Registry: `data/props_models/v2_val_2024-25/`.

**The over-prediction is real and persistent.** Observed / predicted - 1, played rows, all three decision times:

| | 2022-23 | 2023-24 | 2024-25 |
|---|---|---|---|
| minutes (obs - pred, min) | +0.10 | -0.39 | -0.45 |
| points | -0.3% | -5.0% | -4.2% |
| rebounds | -0.7% | -1.6% | -2.9% |
| assists | +1.1% | -1.2% | -5.0% |
| threes | -1.7% | -4.4% | -5.9% |
| pra | -0.2% | -3.7% | -4.0% |

(Negative = the model predicts more than happened.) By segment in 2024-25: players projected under 15 minutes -9% on points, 15-28 min -2.5%, 28+ min -4.4%; the three decision times are alike (T-240 -3.9%, T-60 -4.2%, T-5 -4.6%). The dispersion fix works out of sample: variance ratios are 0.95-1.02 for every stat, combos included.

**What the month breakdown says** (points bias, 2023-24 then 2024-25): October -2.5% / -2.0%, November -2.8% / -2.7%, then -5% to -7% in January-April / -4% to -5%. So the model is already high from the first month and drifts further as the season goes on. Minutes are 0.3-0.6 too high in every month. The league itself did not change in a way that explains it: the share of played rows under 5 minutes (8-9%) and the mean minutes per played row (22.6-23.0) are flat across 2019-20 to 2025-26.

**Working hypothesis (H-S, seasonality learned from the past).** In March-April of 2023-24 the model predicts about 1.2-1.5 minutes more than the players' own recent average (23.2-23.6 against 22.0), and about 1.0-1.3 more in 2024-25. In 2019-20 to 2022-23 minutes rose late in the season, and the model has season-position inputs (`ctx_team_game_number`, `rest_games_played_season`, `ctx_is_playoff`) that let it learn that. If that pattern stopped (load management rules from 2023-24 are the obvious suspect), a model trained on the older years would carry the old late-season uplift forward. I have **not** tested this; it is a hypothesis that fits the monthly shape, not a finding.

**One more observation, not a decision.** On 2024-25 the isotonic layer I dropped *helps*: over-probability ECE 0.021-0.037 raw vs 0.003-0.012 calibrated, log loss better in all 8 stats. It works because the calibration window (late 2023-24) already contains the bias. On the dev folds it hurt because those windows had none. So whether to apply it depends on whether a recent window can be trusted to represent the next season, which is exactly the rolling-recalibration question below. I will not switch it on because of what 2024-25 shows (that would be choosing on the validation season).

Accuracy against the baselines holds on 2024-25: CRPS model vs best baseline, points 3.168 vs 3.322, rebounds 1.301 vs 1.370, assists 0.892 vs 0.931, threes 0.585 vs 0.601, pra 4.177 vs 4.428, pr 3.836 vs 4.042, pa 3.492 vs 3.691, ra 1.795 vs 1.901 (4-6% better in every stat, no intervals yet).

## 2. What Phase 4 will do

**2.1 Evaluation set and rules.** 2024-25 is the validation season; the run above is its first use and used the frozen method unchanged. Anything below that changes the model is designed on the dev folds (2021-22 to 2023-24) first and applied to 2024-25 once. 2025-26 stays untouched until the configuration is frozen at the end of Phase 4/start of Phase 5.

**2.2 Metrics.** For all 8 stats: CRPS, bias of the mean, variance ratio and PIT tails; log loss, Brier and ECE of P(over) at the ladder lines (floor(mean) -2..+2 plus 0.5); reliability bins. Real market lines are Phase 5. Segments: decision time, projected-minutes bucket, position group, month, and starter-like vs bench. Every segment cell is reported (no cherry-picking); the number of cells is stated and intervals for them are Holm-adjusted.

**2.3 Baselines.** The two existing ones (rolling-average Poisson and Normal around rate x expected minutes), plus one new: the player's own empirical distribution over his last 20 games, with the same minutes expectation. The model must beat the best of them in every stat.

**2.4 Uncertainty.** Paired differences (model minus baseline) with a cluster bootstrap over games (all players and decision times of a game resample together), 2,000 draws, 95% intervals; a sensitivity run clustering by calendar date. Game-level clustering ignores correlation across games on the same night, hence the check.

**2.5 Ablations (descriptive, run on the dev fold 2023-24 and on 2024-25).** Drop one feature group at a time: injury, teammates, opponent, team_context, rest, schedule, role, conversion (minutes and rates are the core and are not dropped). Component ablations: no dispersion widening, no minutes mixture (point minutes = `m_min_h15`), LightGBM-only and hierarchical-only rate families. That is 8 + 3 = 11 extra fold fits, about 15 minutes each, no cost. H4 (opponent effects add value) is read from the `opponent` and `team_context` ablations. **Nothing is removed from the model because of an ablation on 2024-25**; simplifications are decided on 2023-24 and then checked once on 2024-25.

**2.6 Pre-registered remedies for the over-prediction (designed on dev folds, applied to 2024-25 once).**

- **R1 (seasonality):** retrain without `ctx_team_game_number` and `rest_games_played_season` (needs a per-feature exclusion option, small code change). Adopt if, on 2023-24, the mean bias shrinks by at least a third and CRPS is not worse beyond its interval. Then run once on 2024-25.
- **R2 (rolling recalibration):** per stat, scale the predicted mean by the ratio of observed to predicted over the trailing 21 days of completed games (only games that finished before T), shrunk toward 1 with a prior worth 2,000 player-games, mean-preserving rescale of the pmf. Design and its two constants are fixed now; I evaluate it on 2022-23 (no bias; it must not hurt) and 2023-24 (bias; it must help) and then apply it once to 2024-25. Adopt if it removes at least half the bias on 2023-24 without raising CRPS on 2022-23 beyond its interval.
- R1 and R2 can be combined; only the pre-registered combinations are tried (R1, R2, R1+R2), so the count of tried rules for Phase 5's multiple-testing correction is 3.

**2.7 Go / no-go for Phase 5.** All must hold on 2024-25 in the final configuration:

1. CRPS beats the best baseline in each of the 8 stats with the paired 95% interval excluding zero (stops the project otherwise, as you specified).
2. Mean bias within +/-2% for points, rebounds and assists over the season, **or** a rolling recalibration (R2) that meets that bar is part of the live design. If neither is achievable, Phase 5 proceeds only with the bias reported and bets judged by CLV rather than model edge.
3. ECE of P(over) at ladder lines at most 0.02 for each stat.
4. Information sanity: CRPS at T-5 is not worse than at T-240 for any stat (later information must not hurt).

## 3. What I need from you

1. Approve this plan, including the three pre-registered rules (R1, R2, R1+R2) and the go/no-go criteria. Change any threshold you disagree with now; they get frozen.
2. The clustering default (game) with a date-cluster sensitivity run: fine?
3. Is the new baseline (player's last-20 empirical distribution) worth the extra code, or stay with the two existing ones? (I recommend including it: it is the hardest simple benchmark.)

Estimated work after approval: R1/R2 design runs on dev folds (about 1 hour compute), ablations (about 3 hours compute, mostly unattended), bootstrap evaluation (minutes), report. No external cost.
