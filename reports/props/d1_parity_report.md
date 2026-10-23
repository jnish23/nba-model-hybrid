# D1.1 parity report: can the live pipeline reproduce the training inputs?

Date 2026-10-22. Pre-registration: [d1_parity_preregistration.md](d1_parity_preregistration.md), [addendum 1](d1_parity_preregistration_addendum1.md), [addendum 2](d1_parity_preregistration_addendum2.md). Code: `props/live/`, `scripts/props_d1_parity.py`. Outputs: `data/props_live_parity/attempt1|2/` (not committed). Model for A5/A6: `v3_final_2025-26`, 20,000 draws, common random numbers.

## 1. Verdict

**The acceptance criteria are not met, and the one that fails hardest (A4, teammate availability) cannot be met by any rule that uses only information available before the game.** Rule v2 gets the injury inputs identical (A3), every historical, role, rest, rate and context column identical (A2) and coverage to ~100% from November on, but the teammate-availability columns agree with the training values on only 71-90% of rows, and that moves the model: the mean absolute change in P(over) at the consensus line is 2.1 points (criterion 0.5), and the rule's qualified bets overlap with a Jaccard of 0.72 (criterion 0.85). Under the pre-registered policy the live signal job does not start until A1-A6 pass on a fresh date set. They cannot, so the next step is the owner's decision (section 5), not a third roster rule.

## 2. Results

| Criterion | Attempt 1: registered rule, 40 dates | Attempt 2: rule v2, 40 new dates | Needed |
|---|---|---|---|
| A1 coverage of traded rows | 99.8% pass | **95.8% fail** (Oct 60%; Nov-Feb 100%; Mar+ 99.9%) | 99% |
| A2 T-independent features identical | fail: 86 columns, 3-6 of 20,605 rows each | **pass, 191/191 columns** | 100% |
| A3 injury status and play probability | pass (100%) | **pass (100%)** | 99% |
| A4 teammate/opponent availability | fail, worst column 6% | **fail, worst 71%** (shares 71-77%, `n_out` 89-91%) | 95% every column |
| A5 predicted mean within 1% | fail (4-10% of rows) | **fail** (41-58% of rows) | 95% |
| A5 mean abs change in P(over) | not given | **2.06 pts**; p95 9.4; max 36.2 (2,812 of 64,114 quotes above 10) | 0.5 / 2 / 10 |
| A6 Jaccard / side / EV correlation | 0.43 / 93% / 0.44 | **0.72 / 99.0% / 0.90** | 0.85 / 98% / 0.98 |

Development check of v2 on 30 dates of 2024-25 with the 2024-25 fold (no 2025-26 date used): the same picture (A1 99.9%, A3 100%, A4 worst 68%, Jaccard 0.75, EV correlation 0.86).

Attempt 1 failed for the reason predicted: the registered live roster (last 15 games + report + quotes) is a superset of the training roster, mostly injury-report "Out" players who never appear in the box score. Its A2 failures (3-6 rows per column) were one builder defect, a player who changed teams being placed on two rosters on the same date; fixed with a test, and A2 passes in attempt 2. The other defects found on the way (a player's last-season team used as his current team; empty-roster dtype; failure listing) are in addendum 2.

## 3. Why A4 cannot pass

The training roster of a game is the set of players **listed in its box score**, which is only known after the game. Measured on 2024-25: a player the injury report lists as Out is listed 36% of the time if he was in the team's previous game and 3% if not; a player on the report as anything else is listed 72-84%; a player listed in the previous game and not on the report is listed 98%. So for every Out or Questionable teammate, the training features count him in or out depending on a game-day fact that no pre-game rule can reproduce; v2 takes the majority answer for each class and is still wrong for about a quarter of rows on the share columns (the median difference is ~0.005, the 95th percentile ~0.2: most rows match, a minority differ a lot).

## 4. What the mismatch costs (information, not a criterion)

Realised ROI per unit staked, settlement "stands", the selected quote's price, frozen revised rule, 40 new dates (no k or rule was fitted on them):

| Bets | n | ROI | Units |
|---|---|---|---|
| Made with training features | 3,490 | +5.60% | +195.6 |
| Made with live (v2) features | 3,509 | +3.91% | +137.2 |
| Made with both | 2,935 | +5.82% | +170.9 |
| Live only | 574 | -5.86% | -33.7 |
| Training only | 555 | +4.74% | +26.3 |

Same pattern on the development dates (4.17% vs 5.02%). Reading, with the usual caution (roughly 3,500 bets, standard error ~1.5 points per row; the live-only vs training-only difference is about 1.8 standard errors): (a) the edge is mostly still there with pre-game rosters, about 1.7 points lower; (b) the bets that appear only with training features win more, and the ones that appear only with live features lose, which is what a small look-ahead in the training roster would produce (a teammate's box-score listing carries game-day information). If so, the Phase 5 backtest ROIs (best price +4.6%, median price -0.5% on 2025-26) are somewhat optimistic for a live version of this model, and the median-price figure, already below zero, would be lower. The 2025-26 backtest was scored once and is not re-opened; this is a measurement of the live-vs-training gap, and it argues for treating the forward paper record as the real test.

## 5. Options (owner's decision)

1. **Start the live pipeline with the frozen model and roster rule v2 and waive A1, A4, A5, A6 as preconditions** (disclosed in every report; October coverage ~60% flagged as low-information). The forward paper record is out-of-sample whatever the backtest says, so starting does not contaminate it; expect roughly 1.7 points less than the backtest and a median-price ROI at or below zero. Recommended: it costs nothing and produces the evidence that matters.
2. **Rebuild the training dataset with point-in-time rosters (rule v2 at each decision time), retrain, and re-run the Phase 4 audits and a Phase 5 evaluation** (a v4 model). This removes the mismatch and the possible look-ahead, but 2025-26 has been scored, so the only clean test of v4 is forward data; cost: hours of compute (dataset rebuild ~1 h, folds ~30 min each, 20,000-draw predictions), a new gate and a new pre-registration. Can run alongside option 1 as a shadow model compared on the same live slates.
3. **Stop here** and keep the dashboard on backtest data.

## 6. Housekeeping

Attempts: 2 on the pre-registered 40-date sets (plus a 30-date 2024-25 development set and one discarded date draw, addendum 2). Nothing about the model, the frozen rule file or any criterion changed. `v3_live_2026-27` is trained (D1.2) and waits on this decision.
