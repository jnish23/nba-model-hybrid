# D1.1 pre-registration, addendum 1: operational definitions and the attempt log

Written 2026-10-22, before the first 40-date run, after the harness was specified and before it produced any result. It does not change a criterion in [d1_parity_preregistration.md](d1_parity_preregistration.md); it fixes the words that document left open so the harness cannot choose them after seeing numbers. (Informal look-ups on three dates while the builder was being developed showed that the live roster is a superset of the training roster and that the availability columns differ; attempt 1 is run with the roster rule exactly as registered anyway, and its failure, if it fails, is reported as the first attempt.)

## 1. Populations

- **Traded set:** the distinct (game, player, decision time T) with a consensus quote (at least 3 books at one line) in `data/props_betting/prices_2025-26.pkl`, for the games of the 40 dates.
- **Intersection rows:** (game, player, T) present in both the live frame and the stored training frame (`data/props_features_v3.db`; the training frame holds every player listed in the game's box score). A2-A4 use all intersection rows.
- **Model rows (A5, A6):** intersection rows, restricted to players who played (`lbl_played = 1`, minutes > 0, as every backtest prediction was) and that carry at least one consensus quote. A6 additionally counts a bet made on training features whose live row is missing (it is a non-overlap).

## 2. Definitions

- **A1 complete row:** the live row exists at (game, player, T) and has no missing value in any base-spec column that is missing in fewer than 1% of the training rows of the 40 dates. Players with a quote but no live row, or an incomplete one, are listed with their reason (no history, not placed on a team, other).
- **A2 columns and tolerance:** every base-spec column whose feature group is not `injury` or `teammates` (those belong to A3 and A4). Groups `team_context`, `opponent` and `schedule` use an absolute tolerance of 1e-6; every other group 1e-9. A row where the training value is missing is not compared; a live value missing where the training value is present counts as a disagreement. Categorical columns must be equal. The criterion needs 100% agreement on every column.
- **A3:** pooled over the three decision times; status code equal; play probability within 1e-9 absolute.
- **A4:** per column, the share of intersection rows within the tolerance (0.03 for share columns, 1 for `n_out` and `n_questionable`, 3 minutes for `miss_starter_minutes`); both missing counts as agreement, one missing as a disagreement. The criterion needs at least 95% on every one of the `av_*` and `opp_av_*` columns.
- **A5:** 20,000 draws, seed 21 for both feature sets, identical rows in identical order so the simulation noise is common to both. Relative change of the predicted mean = |live - training| / training mean (no floor), for points, rebounds, assists and threes separately; each stat must have at least 95% of rows under 1%. P(over) is the model's P(over | no push) at the consensus line of each quote, over all eight stats; change measured in percentage points: mean under 0.5, 95th percentile under 2, maximum under 10; every quote above 10 is listed.
- **A6:** the frozen revised rule (`phase5_revised_rule.frozen_rule.json`, sha256 recorded in the report) applied to the quotes of the 40 dates with the A5 predictions (and, for rows in only one frame, predictions from that frame alone). Bet identity = (game, player, stat). Jaccard over identities; side agreement and the correlation of `best_ev` over shared bets.

## 3. Fixed inputs

Models: `v3_final_2025-26` (trained on seasons before 2025-26, so every 2025-26 date is out of sample for it). Play-rate tables: built from the stored dataset by the training function, season 2025-26 using only earlier seasons. Reports: only those stamped at least 15 minutes before T are passed to the builder (the harness filters, the function filters again). Quotes: the stored consensus rows at each T (snapshot at or before T, the filter inside `prices.load_season`).

## 4. Attempt log

| Attempt | Date set | Roster rule | Result |
|---|---|---|---|
| 1 | `d1_parity_dates.json` (seed 20261022) | as registered (last 15 games + report + quotes) | see `d1_parity_report.md` |

Mechanics dry run (attempt 0, first two dates of the set only, 2,000 draws, not a result): found and fixed two builder defects before attempt 1, neither a change to the roster rule or a criterion. (1) A player named on an injury report for the other team than the one his last box score gives was placed on both teams; he is now placed once, on the report's team. (2) A quoted player whose only listing is last season's, with last season's team being one of the two teams in the game, was placed on that team; players with no listing this season and no report entry are now left unresolved and listed. The dry run also showed what the registered rule is expected to fail on: the 15-game look-back reaches into last season in October and November, so a player who changed teams in the offseason has no row (or the old team's), and the live roster is a superset of the box-score roster, which moves the availability columns.

