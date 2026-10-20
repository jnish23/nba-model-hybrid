# Phase 2 gate report: player-game dataset and features

Status: ready for your review (2026-10-19). Numbers below come from the built dataset (`data/props_features.db`, 684 MB) and are reproduced in
[`phase2_dataset_report.md`](phase2_dataset_report.md); the feature dictionary is [`phase2_feature_dictionary.md`](phase2_feature_dictionary.md) / `.csv`.
Plan and decisions: [`phase2_plan.md`](phase2_plan.md). Build command: `python scripts/build_props_features.py`.

## 1. Gate criteria

| Criterion | Status |
|---|---|
| Dataset, one row per (player, game, decision time T) | **Done.** 232,782 player-games (2019-20 to 2025-26; every listed player, played or not; 80% to 84% played per season) x 3 decision times (T-240, T-60, T-5) = 698,346 rows |
| Feature dictionary | **Done.** 223 features, each with group, source, meaning and availability by season; a test fails if any column lacks an entry |
| Leakage tests, including "shuffling future rows changes no past feature" | **Done, all pass** (section 3) |
| Parity tests | **Partly.** There is no live props pipeline yet, so a literal training-vs-live comparison cannot exist. What is tested instead is that every feature is a function of pre-game information only (section 3); the same functions will be called by the Phase 6 live path, where the literal parity test gets added |
| Full test suite | **186 passed**, including the tests that need the real database |

## 2. What was built (all in `props/features/`, tests in `tests/props/`)

| Module | Purpose |
|---|---|
| `rows.py` | The row universe (v3 box-score roster of record, including DNP/inactive players and their reasons) and the per-game stat history from 2013-14. Outcomes are named `y_*` / `lbl_*` so a test can find them |
| `decay.py` | Point-in-time exponentially decayed sums: each row sees only earlier played games, never itself |
| `history.py` | Minutes (decayed mean, spread, blowout vs close split), role (share of games played, share started), rest and returns, 23 per-minute rates and 3 shooting percentages at half-lives 5, 15, 40 games, shrunk toward league priors built from earlier seasons by position group and minutes bucket |
| `context.py` | Own team and opponent: ratings, pace proxy, scoring, rebounding, assists, what the defense allows, shot-zone profile allowed, travel, rest, altitude. Reuses the team model's pre-game builder (the one allowed `team` import, enforced by a test) |
| `injury_at_t.py` | Latest official injury report at or before T, status, per-T probability of playing, report age; teammate and opponent availability |
| `market_env.py` | Game spread, total and win probability at T from our snapshots; implied team totals; blowout probability |
| `build.py`, `dictionary.py` | Assembly into three tables (`player_game_labels`, `player_game_base`, `player_game_at_T`) and the dictionary |

Also done for this phase: **8,442 hourly injury reports downloaded** (2021-22 to 2025-26, 99.93% of the slots requested) into their own folder, separate from the team pipeline.

## 3. Leakage and consistency tests (what each one proves)

1. **Decay arithmetic equals a naive loop**; changing a row's own value or anything after it leaves its state unchanged.
2. **Mask test on real data:** blank every outcome dated on or after a date D (stats, minutes, started and played flags) and rebuild: the features of every game on D are identical.
3. **Shuffle test on real data:** permute all outcomes dated on or after D among themselves and rebuild: the features of every game before D are identical.
4. **No outcome in a feature table:** no feature column is named like a label, and label stats cannot appear in the feature tables.
5. **Injury reports:** the report used for T-X is never newer than T-X (report age is non-negative for every row); a report published after tip is invisible at any T.
6. **Play probabilities learn only from earlier seasons:** shuffling the labels of 2024-25 changes no 2024-25 probability but does change 2025-26's (so the shuffle bites, and the table is not seeing its own labels).
7. **Market snapshots:** no snapshot taken later than T is used (found 8 of 11,871 that were, from revised tip times; they are now excluded and set to missing).
8. **Zone features:** the opponent shot-zone value equals an independent recomputation from raw data using only earlier games.
9. **Dataset consistency:** unique keys, exactly 3 rows per player-game, probabilities in [0, 1], market snapshots exist only from 2023-24 and the SBR proxy only before it.

## 4. Findings in the data

- **Status means something different at each T, and differs from the spec's numbers.** Questionable players actually play 72% to 73% of the time (the spec said 57%); Doubtful 7% to 12%; Out 0.6% (T-5) to 5% (T-240, before later updates). The probability is therefore estimated per T, per status and per minutes bucket from earlier seasons only. On 2023-24 onward at T-60 it is well calibrated (predicted 0.399 vs realized 0.409 in the lowest decile, 0.994 vs 0.994 in the highest).
- **Injury information is staler in older seasons.** Median age of the latest report: 2019-20 125 min at T-5; 2020-21 95; 2021-22 to 2024-25 35 to 40; 2025-26 10 to 20. Report age is a feature on every row; models trained mostly on older seasons will see fresher reports in production.
- **Market environment before 2023-24 is a proxy, and a weaker one for T-240.** The SBR closing line matches our T-5 snapshot to 0.20 points on the spread (0.32 on the total) over 3,918 games, so it is a good T-5 stand-in. The SBR opening line differs from our T-240 snapshot by 1.22 points, so it is a loose stand-in. There is nothing for T-60 before 2023-24. Flag: `env_source`. In 2023-24 onward 0.62% of rows (26 games) have no snapshot and stay missing.
- **The features carry signal** (a sanity check, not a model evaluation): trailing minutes rank-correlate 0.76 with actual minutes, and rate x minutes correlates 0.69 with points, 0.63 with rebounds, 0.66 with assists, 0.60 with threes.

## 5. Known limits

- **Teammate on/off is the weighted-missing-share version only** (decision D3). Pairwise with/without shifts are deferred until the Phase 4 ablation shows whether the group matters.
- **Teammates are the game's listed players** (the box-score roster of record, which includes inactive players). That set is knowable before the game apart from rare call-ups, who carry no history weight.
- **No BPM weights:** the teammate weights use trailing minutes, usage, rebound chances, assist creation and scoring (the BPM cache covers only 2023-24 onward). **"Potential assists" does not exist in the tracking data**, so assist creation uses assists plus secondary assists, with passes and touches as rates. Hustle data is sparse early and unused.
- **Position is the static listing** (`player_info`), used only to choose league priors; a player who changed position gets his current one.
- **15-minute injury reports are not used.** The hourly archive covers 2021-22 on; 2019-20 and 2020-21 stay sparse.
- **Hyperparameters set by judgement, not tuned:** half-lives 5/15/40, shrinkage pseudo-counts (200 minutes, 25 attempts), blowout threshold 15 points, margin sd 12. They are fixed in `HYPER` and the tuning belongs on 2019-22 only (Phase 3/4).
- **The package rule changed:** `props` may now import exactly one team module (the pre-game team builder). It is enforced by a test; nothing else from `team/` is reachable.

## 6. Recommendation and what comes next

**Pass the gate.** Phase 3 (models) is the next step, and I will bring its plan and the decisions first, as in this phase. The ones I expect to need you for: how to model the minutes distribution (including the early-exit probability), whether the rate models are LightGBM or a hierarchical shrinkage model (the spec asks for a comparison), and how the final distribution is formed (Monte Carlo from minutes and rates versus a direct count model), plus the walk-forward folds. Still outstanding from Phase 1: confirming the live collector on Oct 20/21.
