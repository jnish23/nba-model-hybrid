# Phase 5 validation stage: 2024-25, run once (frozen no-recalibration rule)

Date 2026-10-20. Full tables: [phase5_validation_2024-25.md](phase5_validation_2024-25.md); bets: `phase5_validation_2024-25.primary_bets.csv`. k per stat and the primary rule were read from [phase5_dev_2023-24_noR2b.frozen_rule.json](phase5_dev_2023-24_noR2b.frozen_rule.json) and **nothing was re-estimated**; the file is unchanged. 20,000-draw simulations for 50,825 quoted player-game-times of the 2024-25 fold of the same pipeline. 2024-25 market data is now *seen*; 2025-26 is untouched. Bettable books present: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, and Fanatics for the second half of the season (30k quotes, 55 primary bets); Unibet no longer quotes.

## 1. Verdict against the rule fixed before the run

| Test (pre-declared) | Result | |
|---|---|---|
| Information slope interval above zero | 0.097 [0.047, 0.150] | **not contradicted** |
| Net-of-placebo CLV, conservative lower bound above zero | real +0.441 [0.398, 0.484] vs placebo mean +0.248 (max upper 0.283): excess +0.193, lower bound **+0.115** | **not contradicted** |
| Best-price ROI interval not entirely below zero | +3.3% [+1.7%, +5.0%] | **not contradicted** |

So the validation does not contradict the development result. That is the whole of the good news; the magnitudes are materially smaller, and one approved gate criterion looks like it will fail.

## 2. What held and what did not (development 2023-24 vs validation 2024-25)

| | Development (in-sample for k and rule) | Validation |
|---|---|---|
| Information slope, pooled | 0.21 [0.16, 0.26] | **0.10** [0.05, 0.15] |
| Primary rule: bets / distinct player-games | 22,723 / 11,526 | 22,059 / 11,909 |
| ROI at the **best** price among the books (headline) | +4.9% [+3.1%, +6.5%] | **+3.3% [+1.7%, +5.0%]** |
| ROI at the **median** bettable price | +0.8% [-0.9%, +2.4%] | **-1.7% [-3.2%, -0.1%]** |
| Early-exit treatments (flagged books / all books) | +5.4% / +6.0% | +3.6% / +4.3% |
| CLV excess over placebo | +0.20 points | +0.19 points (conservative lower +0.12) |
| Line move in the bettor's favour, real vs placebo | +0.08 vs -0.58 | +0.18 vs -0.59 |
| Placebo ROI (four runs) | about -4% | -3.3% to -3.8% |
| Blind unders at T-60, median / best price | -3.1% / -0.9% | -4.6% / -2.0% |
| Overs overpriced by the market (hit minus market) | -1.8 points | -1.0 points |
| One bet per player-game | +6.7% (11,526 bets) | +4.6% [+2.7%, +6.5%] (11,909 bets) |

**CLV and the placebo comparison replicated** (that is the strongest evidence that the signal is not noise). **The information shrank by half and the median-price ROI turned significantly negative.**

By stat (information slope validation, development in brackets): rebounds 0.23 (0.29), assists 0.25 (0.32), threes 0.20 (0.41), ra 0.18 (0.26) hold up, at about half to two thirds of their development size; points 0.05 [-0.02, 0.12] (0.12), and the combos pr 0.00, pa 0.00, pra 0.01 (0.17, 0.13, 0.11) show **no information at all**. Primary-rule ROI by stat: ra +5.0% [+1.3%, +8.7%], assists +4.2% [+1.5%, +7.0%], rebounds +4.1% [+1.6%, +6.6%], threes +1.9% [-1.0%, +4.7%], pr +1.5%, points +1.4%, pa -4.3%, pra +7.5% on only 431 bets; the points and combo intervals all include zero. By side: unders +4.8% [+2.5%, +7.2%] on 11,829 bets, overs +1.5% [-0.7%, +3.8%] on 10,230.

## 3. What it means

- The model carries **real but modest information in the low-count markets** (rebounds, assists, threes, ra) and essentially none that survives out of sample in points and the combos, where the development season's k (0.11-0.17) over-trusted it.
- The headline ROI depends on **taking the best of five or six books' quotes**. At the median bettable price the same bets lose 1.7% with an interval entirely below zero. The approved gate (median-price ROI point estimate non-negative) would therefore **fail on this season**; it is a criterion for 2025-26, but the two seasons put the median-price ROI within about 2 points of zero on either side (+0.8%, -1.7%).
- Line shopping itself is a small positive but not distinguishable from zero here: the market-only rule (k = 0) made +2.3% [-2.1%, +7.0%] on 2,266 bets (development +4.3% on 1,536).
- Practical caveats that this simulation does not price: best quotes may disappear before placement, prop limits are small, and accounts that win are restricted. The honest range of attainable ROI on these two seasons is roughly **-2% to +4%**, with the top of the range requiring multi-book execution.

## 4. Where this leaves the project, and the decision for you

2024-25 market data has now been looked at once. The pre-registered final test (2025-26) is still clean. Options:

**A. Freeze exactly as is and spend 2025-26 now.** Cleanest statistically. Cost: the evidence says criterion 3 (median-price ROI non-negative) is likely to fail, so this would probably end in a documented "no-go" for ordinary-price betting.

**B. Revise the rule using both 2023-24 and 2024-25 as development, then spend 2025-26 once on the revised rule (recommended).** Pre-registered revision, fixed before 2025-26: (1) bet only in the markets where both seasons show information (rebounds, assists, threes, ra); (2) re-estimate k per stat on the two seasons pooled; (3) keep everything else and the approved gate unchanged. 2024-25 then becomes development (as the protocol says it must if the rule changes) and 2025-26 is the only clean test. Costs: two more tried rules (37 in total), a smaller bet universe (about 85% of the validation's primary bets are already in these markets), and the risk of selecting on noise, which the pooled-k estimate and the clean final season guard against.

**C. Stop Phase 5 here** and treat the result as the answer: modest information in low-count markets, executable edge only through multi-book price dispersion, no ordinary-price edge.

I recommend B: it spends the one clean season on the best-supported version of the idea instead of on a rule whose weak parts (points and combos) are already known to carry no information.
