# D1.1 pre-registration: the live feature builder, the roster rule and the parity test

Written 2026-10-22, before any live-style feature was compared with a training feature. Approved by the user (D1.1 with acceptance criteria written in advance). Context: [live_pipeline_plan.md](live_pipeline_plan.md).

## 1. What is being built

`props/live/`: for the games of a date and a decision time T, produce the model-input rows (the exact `FeatureSpec('base')` columns) for every player in the traded set, using only information stamped at or before T. The T-independent block (history, role, rest, rates, conversion, team and opponent context) is computed by the SAME functions as training (`history.add_history_features`, `context.add_context`) on the history through the previous day plus one placeholder roster row per player per upcoming game; team form, travel and zone profiles for games that are not in the box-score tables come from the team builder's existing "latest features" entry points and are checked component by component. The T-dependent block (injury status at T with the 15-minute publish lag, play probabilities, teammate and opponent availability) uses the training functions on the live roster.

## 2. The roster rule (fixed now)

For team X in game G on date D, decision time T, the live roster is the union of:

1. players who appeared in X's box score (played or listed) in any of X's last **15** games before D;
2. players on an injury report for G stamped at least 15 minutes before T;
3. players with a consensus quote for G at T (the traded set; always included).

Training rosters were the box-score lists of game G itself. The differences this causes (players in the training list but not in the live roster, and the reverse) are exactly what the test measures.

## 3. The test

- **Dates:** 40 historical dates drawn once with seed 20261022 from 2025-26 game dates, stratified: 10 from October-November, 15 from December-February, 10 from March-April, 5 from the playoffs. The list is written to `reports/props/d1_parity_dates.json` before the first run.
- **As-if rule:** for each date D and T in {240, 60, 5}, only history with game_date < D, injury reports stamped at least 15 minutes before T, and quotes from snapshots at or before T are visible. The harness filters explicitly (it does not rely on the code ignoring later data).
- **Compared with:** the stored training rows of `data/props_features_v3.db` for the same (game, player, T), on the intersection of rows present in both.

## 4. Acceptance criteria (all must hold; fixed now)

| # | Criterion |
|---|---|
| A1 | **Coverage:** at least 99% of traded-set players (consensus quote at T) get a live row with a complete base feature set (rookies and new arrivals without history are counted as failures here and listed). |
| A2 | **T-independent features are identical:** for every column of the history, role, rest, rate, conversion and position blocks, 100% of compared rows agree within 1e-9. For the team and opponent context block (form, travel, zone, schedule): 100% agree within 1e-6, except where the training value is itself missing. Any difference is a defect, not a tolerance question. |
| A3 | **Injury status at T:** the status code is identical for at least 99% of compared rows; the play probability for at least 99% within 1e-9. |
| A4 | **Teammate and opponent availability** (`av_*`, `opp_av_*`): for share-type columns, 95% of rows within 0.03 of the training value; for counts (`*_n_out`, `*_n_questionable`), 95% within 1; for `*_miss_starter_minutes`, 95% within 3 minutes. |
| A5 | **Model outputs:** with the frozen 2025-26 fold models (`v3_final_2025-26`), the predicted mean of points, rebounds, assists and threes changes by less than 1% in at least 95% of rows; the mean absolute change in P(over) at the consensus line is under 0.5 points, its 95th percentile under 2 points and its maximum under 10 points (anything above 10 is listed and explained). |
| A6 | **Decisions:** applying the frozen rule with live features instead of training features on the same dates, the sets of qualified bets overlap with a Jaccard index of at least 0.85, the side agrees on at least 98% of shared bets, and the correlation of EV across shared bets is at least 0.98. |

## 5. If it fails

The fix is the roster rule or a feature-construction defect, never the model, the rule file or the criteria. Each change is logged; a re-run uses a NEW set of 40 dates drawn with a new seed (also written down first); the number of attempts is reported. The live signal job does not start until A1-A6 pass on a fresh date set.

## 6. Also fixed now

The harness and the live builder are written and unit-tested before the 40-date run. No criterion above changes after the first run. Results go to `reports/props/d1_parity_report.md`, including every failed criterion and its cause.
