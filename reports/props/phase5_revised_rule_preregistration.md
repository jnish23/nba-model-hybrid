# Pre-registration: the revised Phase 5 rule and the single 2025-26 scoring

Written 2026-10-20 **before** the pooled k was computed and before any 2025-26 market data was loaded. Authorised by the user (option B after the 2024-25 validation, `phase5_validation_summary.md`). Committed before the run so its timestamp precedes every 2025-26 number.

## What changes, and nothing else

1. **Stats:** the rule bets only **rebounds, assists, threes and ra** (rebounds + assists). Points, pr, pa and pra are dropped: across the two development seasons they show little or no information (validation slopes 0.05, 0.00, 0.00, 0.01).
2. **k per stat:** the information slope (realized hit minus market probability regressed through the origin on model minus market probability), estimated on the **two seasons pooled** (2023-24 + 2024-25, no recalibration, 20,000-draw simulations, game-clustered), floored at 0 and capped at 0.5 as before. Computed once by `scripts/props_phase5_revise.py`; the result is written to `phase5_revised_rule.frozen_rule.json` together with the code commit and input hashes. Nothing is re-estimated afterwards.

**Unchanged:** configuration (v3 pipeline, no rolling recalibration, rate families from v2_base, refit on fit + calibration, dispersion floor 1); primary rule parameters (e = 0.03, earliest decision time with positive EV, at least 3 agreeing books, one bet per player-game-stat, best price among DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Fanatics, Unibet at the consensus line); prices stamped at or before T and not stale; pushes refund; the three settlement treatments; the placebo (four runs, same k, rule and prices); exposure caps; the approved gate (section 6 of `phase5_plan.md`).

## Tried rules for the multiple-testing count

35 before; the stat restriction and the pooled k are two more: **37**. Family-wise intervals on the final use 37. Nothing else will be tried before the final scoring.

## The single 2025-26 scoring

Run once, after the rule file is committed, in this order, with no code changes in between: train the 2025-26 fold of the frozen pipeline; load 2025-26 prices; 20,000-draw predictions for the quoted rows; the revised rule (headline); then the declared sensitivities, which are reported beside the headline and never instead of it:

- **S1, rolling recalibration R2b** (corrected, per game type, carry-over from the 2024-25 fold of the same pipeline), because the user wants to decide on R2b after seeing 2025-26;
- **S2, the original unrestricted rule** with the 2023-24 k file, to show what the restriction changed.

Also on 2025-26, once: the four Phase 4 criteria for the headline configuration (and for R2b as a sensitivity). The gate is evaluated on the headline revised rule only: information slope above zero; net-of-placebo CLV with a positive conservative lower bound; best-price ROI lower bound above zero together with a non-negative median-price ROI point estimate (best price always quoted beside it); validation did not contradict (already met on 2024-25 for the unrestricted rule).

## Not part of this scoring

H1 (news lag), H2 (environment moves) and any betting rule they might suggest are **not** evaluated on 2025-26 here, because a rule defined after seeing development data would use up the clean season. They can be explored on 2023-24 and 2024-25, and any rule that comes out of them is to be tested forward on the live 2026-27 season (Phase 6), which is a genuine holdout. H3 and H4 are descriptive and need no holdout.

## Pre-declared caveats for the report

Execution risk (limits, quotes that move, account restrictions) is not priced; the best-price assumption needs accounts at several books; Unibet no longer quotes in 2025-26.
