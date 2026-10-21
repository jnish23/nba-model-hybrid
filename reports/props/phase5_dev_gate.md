# Phase 5 development stage: summary and gate (2023-24 only)

Date 2026-10-20. Full tables: [phase5_dev_2023-24.md](phase5_dev_2023-24.md); frozen parameters: [phase5_dev_2023-24.frozen_rule.json](phase5_dev_2023-24.frozen_rule.json); bets: `phase5_dev_2023-24.primary_bets.csv`. Configuration: the frozen v3 pipeline + R2b, 20,000-draw simulations for the 50,322 quoted player-game-times. **2024-25 and 2025-26 market data have not been loaded.** Everything below is in-sample for k and for the choice among the 32 tried rules, so it is optimistic by construction.

## 1. Results

**Information test.** The market still beats the model at the book's own line (log loss 0.6876 vs 0.7016 pooled, interval above 0 in every stat except threes, whose lower bound is 0.0000). But the model's disagreement carries real information: slope of realized-minus-market on model-minus-market **0.23 [0.18, 0.28]** pooled (threes 0.43, assists 0.33, rebounds 0.30, ra 0.26, pr 0.21, points 0.17, pa 0.16, pra 0.15). The market also over-prices overs by 1.8 points that season (mean hit minus market probability -0.018 [-0.023, -0.013]).

**Primary rule** (e = 0.03, earliest time with positive EV, at least 3 books, pooled; k per stat = the slope above, cap 0.5): 28,582 bets, win rate 52.8%, **ROI +5.1% [+3.4%, +6.8%]** with the bet standing; +5.6% [+3.8%, +7.3%] if overs are voided on early exits at DraftKings; +6.0% [+4.3%, +7.8%] if at every book. 74% of the bets are unders. CLV to T-5, line unchanged: the consensus fair probability moved **+0.41 points** toward the bet side (n = 17,011, interval [+0.37, +0.45]); when the line moved (16% of bets) it moved in the bettor's favour by 0.12 points [0.03, 0.21]. Correction made on the way: my first CLV line-move figure had the wrong sign; it is fixed and tested.

**The 32 tried rules:** every one is positive, from +3.5% (e = 0.02, T-5) to +7.5% (e = 0.05, T-240, 5 books). At the family-wise (Bonferroni) level the primary rule's interval is [+2.5%, +8.1%]. By stat the primary rule is positive everywhere (+3.4% pr to +6.5% rebounds); per-stat intervals touch zero at the family-wise level for points, pr, pra, pa and threes.

## 2. Where the ROI comes from (the part that changes the reading)

| Strategy (bets stand) | Bets | ROI | 95% interval |
|---|---|---|---|
| **Primary rule, best price among the six bettable books** (designated headline) | 28,582 | **+5.1%** | [+3.4%, +6.8%] |
| The same bets repriced at the **median** bettable price | 28,582 | **+1.3%** | [-0.4%, +2.9%] |
| Market-only rule (k = 0): best price vs consensus fair probability, no model | 1,536 | +4.3% | [-1.5%, +10.0%] |
| All unders at T-60, median price (no model) | 108,721 | -3.1% | [-4.3%, -2.1%] |
| All unders at T-60, best price (no model) | 108,721 | -0.9% | [-2.1%, +0.2%] |
| All overs at T-60, median / best price | 108,721 | -10.0% / -7.8% | |
| Primary rule, unders only / overs only | 21,226 / 7,356 | +6.0% / +2.6% | [+4.2, +8.0] / [-0.4, +5.8] |

Reading: about **3.9 of the 5.1 points come from taking the best price across six books**, which is line shopping, not forecasting; at average prices the edge is +1.3% and cannot be told from zero. The model's contribution shows in the comparison with blind sides at the same pricing: median-priced blind unders lose 3.1%, the model's selection at median prices makes +1.3%, a gap of about 4.4 points, consistent with the information slope. Staleness is not the driver: every selected quote was at most 15 minutes old (the 60-minute stale cut-off never binds), and restricting to quotes at most 15 minutes old changes nothing.

## 3. What this does and does not establish

- It **does** show a model signal that survives de-vigging and the vig at a market that is otherwise efficient (positive slope, positive CLV with a tight interval, positive ROI under all three settlement treatments).
- It does **not** show a bettable edge at ordinary prices. The headline ROI depends on being able to place the best of six books' quotes at the stamped price. Real execution has latency, per-book limits (prop limits are small), account restrictions and quotes that move. The honest range on this season is **+1.3% (median price) to +5.1% (best price)**, both in-sample for k and the rule family.
- Power: 28k bets here, but only about a third of that many independent bets; the validation season will show whether +1% to +5% survives.

## 4. Recommended addition to the Phase 5 gate (needs your approval)

The approved gate asks for a positive lower bound on ROI, positive CLV and a positive slope on 2025-26. Given section 2 I recommend adding a pricing-realism requirement and reporting both prices as the headline: **ROI at the median bettable price must have a non-negative point estimate on 2025-26, and the best-price ROI is quoted next to it, never alone.** Without it, a "go" could rest on line shopping that may not be executable.

## 5. Gate: what I am waiting for

Stage 2 loads 2024-25 prices for the first time and runs the frozen rule once (no changes afterwards, or 2024-25 becomes development). Nothing has been touched for it except building the code.

1. Approve running the validation stage on 2024-25 with `phase5_dev_2023-24.frozen_rule.json` unchanged (k per stat, primary rule, no tuning).
2. Approve or change the gate addition in section 4.
