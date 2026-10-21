# Phase 5 development stage: summary and gate (2023-24 only) - corrected after the second audit

Date 2026-10-20. Supersedes the first version of this file (its R2b window contained only playoff games and its CLV had no placebo). Tables: [phase5_dev_2023-24.md](phase5_dev_2023-24.md) (corrected R2b) and [phase5_dev_2023-24_noR2b.md](phase5_dev_2023-24_noR2b.md) (no recalibration, sensitivity); frozen parameters: [phase5_dev_2023-24.frozen_rule.json](phase5_dev_2023-24.frozen_rule.json); bets: `phase5_dev_2023-24.primary_bets.csv`. 20,000-draw simulations for the 50,322 quoted player-game-times. **2024-25 and 2025-26 market data have not been loaded.** Everything below is in-sample for k and for the choice among the tried rules, so it is optimistic by construction.

## 1. Results (R2b with per-game-type windows)

**Information test.** The market still beats the model at the book's own line (log loss 0.6876 vs 0.7015 pooled; interval above 0 in every stat except threes). The model's disagreement carries real information: slope of realized-minus-market on model-minus-market **0.23 [0.18, 0.27]** pooled (threes 0.42, assists 0.32, rebounds 0.30, ra 0.26, pr 0.21, points 0.16, pa 0.16, pra 0.15). The market also over-priced overs by 1.8 points that season.

**Primary rule** (e = 0.03, earliest time with positive EV, at least 3 books, pooled; k per stat = slope, cap 0.5): **27,934 bets on 12,198 distinct player-games** (2.3 bets per player-game, up to 8), win rate 52.5%, **ROI +4.8% [+3.0%, +6.5%]** with the bet standing; +5.2% / +5.7% if overs are voided on early exits at DraftKings / at every book. 73% of the bets are unders (unders +5.8% [+4.0%, +7.8%], overs +2.0% [-0.8%, +4.9%]).

**Placebo check (new, from the audit).** A model with no information (each market's probability swapped with a random other market's of the same stat; same k, rule and prices) **loses 3.3% to 3.6%** in four runs. Its CLV is not zero: selection against consensus noise that reverts by T-5 gives it **+0.19 points** of consensus-probability movement toward its side. The real model's is +0.41 [0.37, 0.45], so the excess is **+0.22 points, conservative lower bound +0.16** (real lower bound minus the largest placebo upper bound). When the line moved by T-5 (15% of bets) it moved +0.13 points in the real model's favour against -0.50 for the placebo, an excess of 0.63 points. So roughly half of the headline CLV is mechanical and the other half is signal.

**Exposure (new).** Capping at one bet per player-game (the highest-EV one): **12,198 bets, ROI +6.5% [+4.6%, +8.5%]**; at two per player-game 18,877 bets, +5.4%; at two per game 2,634 bets, +7.6% [+3.2%, +11.5%] (family-wise interval touches zero). The honest unit for statistical power is the 12k player-games, not the 28k bets.

## 2. Where the ROI comes from (unchanged in substance)

| Strategy (bets stand) | Bets | ROI | 95% interval |
|---|---|---|---|
| **Primary rule, best price among the six bettable books** (designated headline) | 27,934 | **+4.8%** | [+3.0%, +6.5%] |
| The same bets repriced at the **median** bettable price | 27,934 | **+0.9%** | [-0.8%, +2.6%] |
| Market-only rule (k = 0), best price vs consensus fair probability, no model | 1,536 | +4.3% | [-1.5%, +10.0%] |
| All unders at T-60, median price / best price (no model) | 108,721 | -3.1% / -0.9% | |
| All overs at T-60, median price / best price (no model) | 108,721 | -10.0% / -7.8% | |
| Placebo model (four runs) | about 55,500 | -3.3% to -3.6% | |

About **3.9 of the 4.8 points come from taking the best of six books' prices**; at average prices the edge is about +0.9% and not distinguishable from zero. Against blind sides at the same pricing the model's selection is worth roughly 4 points, consistent with the information slope. Staleness is not the driver: every selected quote was at most 15 minutes old.

## 3. How much does R2b matter for the betting results?

Almost not at all. Without any recalibration the primary rule gives **+4.9% [+3.1%, +6.5%]** on 22,723 bets (median price +0.8%, unders +6.5%, overs +1.9%), slope 0.21, CLV excess over placebo +0.20 points, one bet per player-game +6.7% on 11,526 bets. R2b changes how many bets clear the threshold and how well the forecasts are centred, not the betting verdict: k and the EV threshold already absorb a level bias. R2b matters for forecast accuracy and calibration (section 4), which is what the Phase 4 criteria and any live use need.

## 4. R2b after the second audit: needs your decision again

The first R2b was adopted by a documented deviation on evidence from a window that contained only playoff games. With per-game-type windows (a validity correction):

| | 2023-24 (the biased fold) | 2022-23 (no bias) |
|---|---|---|
| Mean absolute bias, core stats | 3.57% to 1.62% = **-55%** (bar 50%): pass | 0.68% to 0.60% |
| CRPS vs unrecalibrated | better in every stat, points -0.0099, pra -0.0111 (intervals exclude 0), none worse | **assists +0.0006 [0.0001, 0.0012] significantly worse**; ra +0.0010 [0.0003, 0.0017] (not a core stat); the others neutral |
| Registered non-harm rule | pass | **fail (assists)** |

The previous waiver covered pra on 2022-23 at 0.026% of CRPS in a different implementation; this failure is a different stat at about 0.07%. I have not carried the waiver over.

## 5. Fixes from the second audit and what they did

- **R2b window by game type:** done (regular season and postseason separated; the previous season's regular-season tail carries over, a postseason never carries). Result in section 4.
- **Dispersion scaler may narrow:** implemented, tried, **not adopted**: the calibration windows still ask for widening (c above 1 on all dev folds), so the change was a near no-op and slightly worse for rebounds. The over-width the audit measured (variance ratios 0.91-0.99) is real; its cause is that the scaler and NB dispersion are fitted on the fit-only models, whose errors are larger than those of the refit models that predict the test season. A rolling dispersion calibration on completed games would address it but is a new tried rule; not done.
- **Placebo CLV and caps:** done (section 1). The CLV gate becomes "real minus placebo".
- **Merge guards in the dataset build:** done; a rebuild is identical to the v3 dataset table for table.
- **Tuned on one season; same-player exposure:** reported, not fixable. The validation stage reports ROI by side and the market's lean.

## 6. Decisions for you

1. **R2b:** (a) adopt the corrected R2b as a second documented deviation, or (b) stand by the rule: R2b is not adopted, the frozen configuration is the v3 pipeline without recalibration, the forecast bias (about -5% for points in 2023-24) is reported, and the Phase 4 bias criterion is met only through the plan's fallback. I recommend (b) for the betting validation: the verdict is the same either way (section 3), it avoids a second deviation, and live forecast centring can be revisited in Phase 6.
2. **Gate wording for Phase 5:** replace "positive CLV" with "real minus placebo CLV above zero with a positive conservative lower bound"; and require the ROI at the median bettable price to have a non-negative point estimate, quoting the best-price ROI beside it. Tried rules are now 35 (adjusted intervals use 35).
3. **Run the 2024-25 validation stage once** with the frozen rule file unchanged, after (1).
