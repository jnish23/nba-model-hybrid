# D1.1 pre-registration, addendum 2: roster rule v2 and attempt 2

Written 2026-10-22 after attempt 1 and before attempt 2. No criterion changes. [Addendum 1](d1_parity_preregistration_addendum1.md) holds the definitions and the attempt log; the criteria are in [d1_parity_preregistration.md](d1_parity_preregistration.md).

## 1. Attempt 1 (registered roster rule, 40 fixed 2025-26 dates, code commit be25ef0)

A1 pass (99.8% of traded rows have a complete live row). A2 fail (86 of 191 columns; each differs on 3-6 of 20,605 rows: one player placed on his last season's team, and a few role/rest edge cases). A3 pass (status code and play probability identical on 100%). A4 fail (teammate-availability shares within 0.03 on as few as 6% of rows). A5 fail (the predicted mean moves by under 1% in only 5-10% of rows). A6 fail (Jaccard 0.43, EV correlation 0.44). Cause: the registered live roster is a superset of the training roster.

## 2. Why the roster rule has to change

The training roster of a game is the set of players listed in its box score. Measured on 2024-25 (50,097 candidate team-game-players = last 15 games + report): a player the injury report lists as Out is listed in the game 36% of the time if he was listed in the team's previous game and 3% if not; a player listed in the previous game and not on the report is listed 98% of the time; a player on the report with any status other than Out is listed 72-84% of the time; a player in neither is listed 17% of the time (opening game of the season: 61% from the previous season's last game).

## 3. Rule v2 (the roster rule for attempt 2)

For team X in game G on date D at decision time T the live roster is:
1. players listed in X's most recent game of the same season before D (whose latest team is X), **minus** players the usable report lists as Out for G; a team with no game yet this season has none;
2. **plus** players on the usable report (stamped at least 15 minutes before T) with a status other than Out;
3. **plus** every player with a consensus quote for G at T (always included, even if the report lists him as Out).

Selected by looking at listing rates on 2024-25 (not at any 2025-26 outcome) and checked on 30 development dates of 2024-25 (`d1_roster_dev_dates.json`, seed 20261023): one alternative rule was compared (the registered one); no parameters were tuned. Development results (2024-25, no 2025-26 date used): v2 raises the worst teammate-availability agreement from 8% to 68%, keeps A1 at 99.9%, A3 at 100%; A2 differs on 4 columns in 3 of 19,322 rows; A5 and A6 still fail (mean |change in P(over)| 2.2 points, Jaccard 0.75, EV correlation 0.86); realised ROI of live-feature bets 4.17% vs 5.02% for training-feature bets (3,085 and 3,034 bets).

## 4. Attempt 2

Dates: `d1_parity_dates_attempt2.json` (40 dates, seed 20261024, disjoint from attempt 1), same stratification. Rule v2. Model `v3_final_2025-26`. All criteria and definitions unchanged. The harness additionally reports, **as information and not as a criterion**, the realised ROI per unit staked (settlement "stands", the selected quote's price) of the bets made with live features, with training features, and of the bets only one side makes. Expectation written down now: A4 will still fail because the training roster depends on who is listed in the box score, which is not known before the game; A5 and A6 are expected to fail narrowly or clearly. The result decides what to ask the owner, not whether to relax a criterion.

## 5. Attempt log

| Attempt | Date set | Roster rule | Result |
|---|---|---|---|
| 1 | `d1_parity_dates.json` | registered | A1 pass, A2-A6 fail |
| 2 | `d1_parity_dates_attempt2.json` | v2 | see `d1_parity_report.md` |
