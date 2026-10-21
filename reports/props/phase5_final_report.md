# Phase 5 final report: the single 2025-26 scoring

Date 2026-10-21 (run 2026-10-20 evening). Pre-registration: [phase5_revised_rule_preregistration.md](phase5_revised_rule_preregistration.md) (commit 761705d); frozen rule: [phase5_revised_rule.frozen_rule.json](phase5_revised_rule.frozen_rule.json) (commit da902d2); no code change between the freeze and the run. 2025-26 was untouched until this run and each step ran once. Tables: [phase5_final_2025-26.md](phase5_final_2025-26.md) (headline), `phase5_final_2025-26_S1_R2b.md`, `phase5_final_2025-26_S2_original.md`, `phase4_final_2025-26.md`, `phase4_final_2025-26_R2b.md`; bets: `phase5_final_2025-26.primary_bets.csv`.

## 1. Verdict

**The approved Phase 5 gate is not passed.** Three of its four conditions pass; the fourth, the median-price ROI, comes out at **-0.5%** against a required non-negative point estimate. By the wording you approved that is a **no-go for live betting at ordinary prices**. It is a narrow miss with a wide interval that spans zero, and the other evidence is positive, so the honest summary is: *the model has real, replicated information, but at the prices an ordinary bettor gets it is worth about nothing; the profit in the simulation depends on taking the best of several books' quotes.*

## 2. The gate, condition by condition (headline: revised rule, no recalibration, 15,593 bets on 10,145 player-games)

| Condition (approved wording) | 2025-26 result | |
|---|---|---|
| 1. Information slope, 95% interval above zero | four traded stats pooled **0.204 [0.149, 0.260]** (all eight stats 0.121 [0.072, 0.171]); threes 0.34, rebounds 0.24, assists 0.17, ra 0.09 [-0.02, 0.20] | pass |
| 2. CLV to T-5 net of the placebo, conservative lower bound above zero | real +0.509 [0.454, ...] vs placebo mean +0.329 (largest upper bound 0.363): excess **+0.181, conservative lower bound +0.091** | pass |
| 3a. Best-price ROI, lower bound above zero | **+4.63% [+2.96%, +6.25%]**; +4.79% / +5.43% if overs are voided on early exits at flagged books / all books | pass |
| 3b. Median-price ROI, point estimate non-negative | **-0.48% [-2.07%, +1.06%]** | **FAIL** |
| 4. The 2024-25 validation did not contradict development | not contradicted (slope 0.097 [0.047, 0.150], net CLV lower bound +0.115, best-price ROI +3.3% [+1.7%, +5.0%]) | pass |

## 3. Replication across the three seasons (same pipeline, k and rule frozen before each later season)

| | 2023-24 (dev) | 2024-25 (validation) | 2025-26 (final) |
|---|---|---|---|
| Best-price ROI of the primary rule | +4.9% [+3.1%, +6.5%] | +3.3% [+1.7%, +5.0%] | **+4.6% [+3.0%, +6.2%]** |
| **Median-price** ROI, same bets | +0.8% [-0.9%, +2.4%] | -1.7% [-3.2%, -0.1%] | **-0.5% [-2.1%, +1.1%]** |
| Placebo (no-information model) ROI | about -4% | -3.3% to -3.8% | -1.4% to -2.5% |
| CLV excess over placebo (points) | +0.20 | +0.19 | +0.18 |
| Blind unders, best price / median price | -0.9% / -3.1% | -2.0% / -4.6% | -0.6% / -3.5% |
| Market-only rule (k = 0, no model), best price | +4.3% (1,536 bets) | +2.3% (2,266) | +7.0% [+2.1%, +11.7%] (1,907) |

(2023-24 and 2024-25 use the unrestricted rule as it was run then; the 2025-26 headline is the revised four-stat rule.)

## 4. What the three seasons add up to

- **The signal is real.** The placebo loses money every season at the same prices where the model makes it; the CLV excess over the placebo is stable at about +0.2 points of consensus probability; the information slope is positive in all three seasons (0.20 in the four traded stats this season). The market is still better than the model overall (log loss 0.6891 vs 0.7067), as it should be.
- **At average prices there is no edge.** The same bets priced at the median bettable quote are worth +0.8%, -1.7%, -0.5%: statistically zero.
- **The edge is price dispersion across books, not forecasting.** Taking the best of five or six books' quotes is worth about 5 points of ROI. The market-only rule, which ignores the model entirely and only buys a quote that beats the consensus fair probability by 3 points, earned +4.3%, +2.3% and +7.0% in the three seasons. The model adds selection on top (the blind-under benchmarks lose), but most of what this simulation earns is available without it.
- **Executability is the whole question and history cannot answer it.** Whether a quote that looks best in a snapshot is still there when a bet is placed, what the limits are, and how fast accounts get restricted when they win are all outside this simulation. The simulation also bets 1.5 times per player-game on average; capping at one bet per player-game gives +6.5% [+4.4%, +8.4%] on 10,145 bets (family-wise [+3.8%, +9.4%]).
- **Sides:** unders +6.1% [+3.8%, +8.3%] on 10,031 bets, overs +2.1% [-0.9%, +4.8%] on 5,562. The market's lean (overs overpriced by about 1.4 points in the comparable sample) persisted.

## 5. Declared sensitivities (reported beside the headline, none selected)

| | Bets | Best-price ROI | Median-price ROI | Net CLV, conservative lower |
|---|---|---|---|---|
| **Headline**: revised rule, no recalibration | 15,593 | +4.63% [+2.96%, +6.25%] | -0.48% [-2.07%, +1.06%] | +0.091 |
| S1: with R2b (corrected, carry-over from 2024-25) | 15,525 | +4.87% [+3.14%, +6.54%] | -0.25% [-1.90%, +1.34%] | +0.099 |
| S2: the original unrestricted rule (eight stats, 2023-24 k) | 21,855 | +3.69% [+2.05%, +5.37%] | -0.97% [-2.52%, +0.63%] | +0.074 |

The restriction to the four stats helped as intended (S2 vs headline: +0.9 points of best-price ROI, +0.5 of median-price ROI), within the noise. R2b changes nothing that matters for betting and does not rescue the median-price condition.

## 6. Phase 4 criteria on 2025-26 (the same season, scored once)

| Criterion | Headline (no recalibration) | With R2b |
|---|---|---|
| 1. CRPS beats the best baseline, 8 of 8 stats | pass | pass |
| 2. Bias within +/-2% (points / rebounds / assists) | **pass** (-0.1% / -0.7% / +1.5%) | pass (-0.1% / -0.2% / +0.4%) |
| 3. ECE of P(over) at the ladder lines at most 0.02 | **pass** (worst 0.014, threes) | pass (0.008) |
| 4. T-5 not significantly worse than T-240 | pass | pass |

All four pass without recalibration, so the Phase 4 gate is met outright on the clean season. The over-prediction that appeared in 2023-24 and 2024-25 did not recur in 2025-26 (points -0.1%). It stays unexplained; **R2b was not needed this season**, which is the evidence you wanted for the R2b decision: it would have shaved the assists bias from +1.5% to +0.4% and nothing else of consequence.

## 7. Limits

Only the best-price assumption separates a positive from a zero result, and it is untestable here; the settlement of early exits is bounded, not known; Unibet no longer quotes; the list of tried rules is 37; 2025-26 can never be used again as a clean season; H1 (news lag) and H2 (environment moves) were deliberately not tested on it.

## 8. Decisions for you

1. **Gate result:** accept the no-go as worded, or decide that the median-price clause was too strict (an override would be a documented deviation, and I would not recommend it on this evidence).
2. **What to do with a model that has information but no ordinary-price edge.** Options: (a) stop here and treat Phases 0-5 as the answer; (b) a **forward paper-trading test on the live 2026-27 season** (the collector is already running) that records, with your actual books, the price that could really be placed, its size limits and how fast it moves, which is the one thing the history cannot show; this is Phase 6 without money; (c) H1/H2 as new rule candidates, tested forward only. I recommend (b) first: the open question is purely price realism.
3. **R2b:** not needed in 2025-26; I suggest closing the R2b question as not adopted, with the 2025-26 evidence on record.
