# Phase 5 plan: evaluation against the market, H1-H4, simulated betting

Date 2026-10-20. Nothing here has been run. Configuration: **frozen** ([frozen_config.json](frozen_config.json)): v3 pipeline + R2b. Protocol: [phase4_protocol_update.md](phase4_protocol_update.md) (2024-25 is *seen*; 2025-26 is scored once for the model and for the betting rules). Void rules: [void_rules.md](void_rules.md). Audit lessons: [phase4_audit_response.md](phase4_audit_response.md).

## 0. What the evidence says before we start (so the plan is honest about the odds)

On 2023-24 (dev, v2_base, consensus book line): the market beats the model in log loss in every stat (+0.003 threes to +0.023 pra). The model's disagreement with the market carries some information: slope 0.20 [0.15, 0.25] of realized-minus-market on model-minus-market (threes 0.37, rebounds 0.29, assists 0.28, points 0.12). Overs were overpriced in the market by about 1.8 points that season. A bettor needs the part of the gap that is real to beat the vig (about 4-5% round trip at -110). **The prior that a profitable rule exists is modest; the plan is built so that a null result is a clean, reportable answer.**

## 1. Data and definitions (point in time, always)

- **Decision times** T-240, T-60, T-5 before the NBA-schedule reference tip (the clock of Phase 1). Everything used for a bet at T is stamped at or before T: the model's features (injury reports usable only if stamped at least 15 minutes earlier), and the prices.
- **Market information price:** consensus main line (at least 3 agreeing books; stale, one-sided, wide quotes excluded) with the de-vigged P(over) (multiplicative; power and Shin as sensitivity).
- **Bettable price:** the best price per side among the allowed books at the snapshot stamped at or before T (quote not stale: updated within 60 minutes), at that book's own line. The opening line is never an entry price. Lines that differ across books are handled per book (a bet is on a book's line, not on the consensus line).
- **Model probability:** P(over | no push) from the 20,000-draw simulation at the bet's line, after R2b. Rows compared with a market use 20,000 draws (measured noise 0.35 points per run, against 0.77 at 4,000); saved 4,000-draw pmfs are used only for CRPS-type metrics.
- **Closing reference:** the T-5 consensus. It is not the true closing line (the snapshot grid ends at T-5), so CLV is "to T-5" and says so everywhere. For bets at T-5 there is no later reference; they are judged on realized results only.
- **Outcome and settlement:** over wins if the stat exceeds the line, under wins if below, a push refunds. A player who does not play voids the bet at every book. Every simulated result is reported under two bounding treatments (from `void_rules.md`): (a) the bet stands whenever the player took the court; (b) at books with an early-exit rule, an over is voided if minutes fall below 50% of expected minutes (proxy for an early injury exit) while the under stands.

Decision needed from you (section 8, item 1): which books can you actually bet at.

## 2. Protocol and stages

| Stage | Season | Use |
|---|---|---|
| Develop | 2023-24 (first market season) | rule family, shrinkage factors, thresholds, H1-H4 definitions refined only here |
| Validate | 2024-25 (seen for the model, never used to choose a betting rule) | the frozen rules run once. If anything is changed after looking, 2024-25 becomes development and only 2025-26 validates |
| Final | 2025-26 | scored once, after both the model and the rules are frozen: the Phase 4 go/no-go criteria and the betting results together |

2023-24 and 2024-25 predictions come from the frozen pipeline's own folds (`v3_dev`, `v3_seen_2024-25`); 2025-26 from a new fold of the same pipeline.

## 3. The tests, in order

**3.1 Information test (primary, before any betting rule).** Realized hit minus market probability regressed on (model minus market), through the origin, game-clustered 95% intervals, by stat and pooled; plus a logistic version with market logit and model-minus-market logit as regressors. Pass: slope interval above 0 in the final season. This decides whether any gap-based rule can pay.

**3.2 CLV (the primary edge measure, per your rule 6).** For bets at T-240 and T-60: change in the de-vigged consensus probability toward the bet side from the bet price to the T-5 consensus (and in log odds), game-clustered intervals, by rule and stat. Positive CLV with non-positive ROI is reported as such (variance), not hidden.

**3.3 H1, news lag on teammate status changes.** Event: a teammate with `m_min_h15` of at least 25 changes status between consecutive decision times to Out or Doubtful (or the reverse). Affected players are those on the same team whose forecast for minutes or usage moves by at least a pre-set amount (0.5 sd of the model's own change distribution, fixed on 2023-24). Test: after the event, does the market move toward the model's updated forecast by the next decision time (regression of the market's logit change on the model's logit change, event rows vs non-event rows), and is CLV positive for bets in the direction of the model's update at the pre-move price? Needs prices at consecutive decision times for the same player.

**3.4 H2, prop lines under-react to team total and spread moves.** Snapshot era only (2023-24 on). Regress the change in the prop consensus line (and in P(over)) from T-240 to T-5 on the change in the game total and in the team spread (consensus), by stat; compare the market's elasticity with the structural elasticity of the outcomes (outcome ~ environment, estimated on the dev season and checked on the next). Where the market's elasticity is lower, bet in the direction of the environment move at the earlier price, judged by CLV. The `with_env` model (env features at T, `env_source` excluded, trained on the snapshot era only) is used only for this test.

**3.5 H3, discrete distribution gives better calibration at specific lines.** Compare calibration of the market consensus and of the model by line value for the low-count stats (threes 0.5/1.5/2.5, assists and rebounds at their common lines, integer versus half lines), using the no-push probability, pooled over 2023-24; does the model correct the market's bias at those lines (the model's slope on the gap, by line)? Line groups are fixed in advance and counted.

**3.6 H4, team-level opponent effects add value.** Ablations of the `opponent` and `team_context` groups (already trained on the v2 pipeline for 2023-24 and 2024-25; retrained on the frozen pipeline for 2023-24 only): paired change in CRPS and in the information slope versus the market with bootstrap intervals. No player-versus-team splits, as specified.

## 4. Simulated betting (pre-registered family, no peeking)

For each (player, game, stat, T) and each side, with p_mkt the consensus de-vigged probability and p_mod the model's no-push probability:

- Shrunk probability p* = p_mkt + k x (p_mod - p_mkt), with k fixed per stat family on 2023-24 as the information slope (floor 0, cap 0.5) and never refit on later data.
- Bet the side with the higher EV when EV >= e at the best allowed price, with pushes as refunds; one bet per player-game-stat (the best decision time among those allowed by the rule), flat 1 unit.
- Rule grid (all counted as tried): e in {0.02, 0.03, 0.05}; decision time in {T-240, T-60, T-5, earliest with positive EV}; min books {3, 5}; the primary rule is e = 0.03, earliest time with positive EV, 3 books, pooled over stats, designated before any result. Secondary: per stat (8) at the primary settings. At most 3 x 4 x 2 + 8 = 32 tried rules; Holm adjustment across the family, and the primary alone is the headline.
- Correlated bets: results reported with intervals clustered by game (and by player-game); a per-game exposure cap sensitivity (at most 2 bets per game); correlated combos from one player use a joint re-simulation to size risk, not independence.
- Winner's-curse guard: thresholds and k come only from the development season; the validation and final seasons never feed back.

## 5. Power (what the final season can detect)

One unit-stake bet has a standard deviation of about 1.0, so the standard error of ROI over N independent bets is about 1/sqrt(N). Detectable ROI at 80% power and a 5% two-sided level is about 2.8/sqrt(N): N = 5,000 independent bets -> 4.0%; 10,000 -> 2.8%; 20,000 -> 2.0%. A player-game appears at three decision times and in up to eight related stats, so the effective N is roughly a third of the raw count. In 2023-24 there were 334k comparable quotes (about 110k independent player-game-stats); a rule that fires on 10% of them gives about 11k independent bets, enough to detect an ROI near 2.7%. **Smaller true edges cannot be confirmed in one season**; the report will say "not distinguishable from zero" rather than "no edge" in that case, and CLV over three seasons will carry the weight.

## 6. Gates (wording approved by the user 2026-10-20, after the audits)

Go to Phase 6 (paper trading, live parity) only if, on 2025-26 and under all three settlement treatments, the HEADLINE configuration (v3 pipeline, no recalibration; R2b reported beside it as a declared sensitivity) shows ALL of:

1. **Information:** slope of realized-minus-market on model-minus-market with a 95% interval above zero.
2. **CLV net of the placebo:** real CLV to T-5 minus the CLV of the no-information placebo (same k, rule and prices; at least four runs) is positive, with a positive conservative lower bound (the real model's lower bound minus the placebo's largest upper bound). Raw CLV alone is not a criterion, because selection against consensus noise produces positive CLV by itself.
3. **ROI:** the primary rule's ROI at the best bettable price has a lower bound above zero, AND the ROI at the **median** bettable price has a non-negative point estimate, with the best-price ROI always quoted next to it and never alone.
4. The 2024-25 validation did not contradict the development result, as defined below.

Criteria 2 and 3 of the Phase 4 list (bias, ECE) are met only through the plan's fallback (bias reported; bets judged by net CLV and these criteria) unless the configuration passes them outright on 2025-26.

**What counts as contradicting, fixed before 2024-25 was loaded:** the validation (frozen k and rule from the no-R2b development run) contradicts the development result if the information-slope interval includes zero or is below it, OR the net-of-placebo conservative CLV lower bound is not positive, OR the best-price ROI interval lies entirely below zero. Anything else is reported as consistent, with the magnitude differences shown (development numbers are in-sample for k and the rule, so a smaller validation ROI is expected and is not a contradiction).

## 7. Pitfalls handled

Selection from pulled lines (a quote that disappears when a player is ruled out is still a bet that voids; all quotes stay in the store and are settled); void and DNP rules; pushes; time zones (all UTC with the schedule reference clock); name mismatches (703 rows still unresolved, mostly one ambiguous name, excluded not guessed); correlated bets; multiple testing (32 rules, Holm); judging by win rate (never: ROI and CLV with intervals); stale quotes; limits (props limits are small, so results are per-unit, not dollars).

## 8. What I need from you before building

1. (answered: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Fanatics, Unibet; offshore books only in the consensus) **Which sportsbooks can you actually bet at** (state and accounts)? The backfill holds DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Bovada, BetOnline, Fanatics, Unibet and others. Default if you do not answer: the US-regulated ones only, offshore books excluded from the bettable set (still used in the consensus).
2. Approve the three-stage protocol in section 2 and the 20,000-draw simulations (about 30 minutes of compute per season, no cost).
3. Approve the primary rule (e = 0.03, earliest decision time with positive EV, at least 3 books, pooled) as the headline, designated now.

Build order after approval: 20,000-draw prediction script from the saved bundles -> bettable-price loader -> settlement with both treatments and tests -> information test and CLV on 2023-24 -> rule family on 2023-24 -> report and stop at the gate.
