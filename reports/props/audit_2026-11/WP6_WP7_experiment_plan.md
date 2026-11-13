# WP6 and WP7 experiment plan (pre-game roster, k re-estimation, de-vig, environment, exposure, minutes, dispersion)

**IN FORCE for steps S1 to S8. Accepted by the owner on 2026-11-12, before any step was run.** The owner accepted all ten recommendations of section G (G1: S1-S8 approved now, the rest after the S5 reproduction check; G2: forward record only; G3: A1 and A2, A0 conditional; G4: v2nq; G5: 2023-24 + 2024-25; G6: `--registry-root`; G7: one bundled candidate; G8: game date; G9: Kelly evaluated only; G10: compute window). After acceptance, the keep rules, thresholds, season roles and comparison count do not change; a change is a written, dated amendment made before the affected step runs.

**Amendments of 2026-11-12, made before any step ran** (owner: the margins in this market are small, so the dev stage must not be more conservative than its job requires; the dev stage only proposes a candidate and the forward test is the gate):

1. C.5 items 2 and 3 (and so item 4): the adoption margin on median-price ROI is **+0.25 points**, not +0.5, for C2 - C1, C2 - P0 and C3/C4 against the current pick. Reason: on about 15,000 paired bets a season the paired difference has a standard error near 0.3 to 0.4 points, so a +0.5 bar would reject most true improvements of the size this market allows. The 7c tolerances (a cap or sizing rule may cost up to 0.5 points) are unchanged: they are allowances, not hurdles.
2. 7d item 5: the bound is the **unadjusted 95% lower bound**, not the Bonferroni one. Reason: passing 7d changes no rule; it only keeps the hypothesis as a forward sub-hypothesis, which the signed forward pre-registration already contains (H2), so the strict bar bought nothing.
3. CLV stays a reported secondary (A.4 item 4). The owner declined to add it to the keep rules.
4. File renamed from `WP6_WP7_experiment_plan_DRAFT.md`; references to `forward_test_preregistration_DRAFT.md` mean `reports/props/forward_test_preregistration.md` (signed 2026-11-12).
5. 7a item 7: the keep rule is judged on the **four bet stats** (rebounds, assists, threes, ra); the same rule on all eight markets is printed beside it and decides nothing. Reason: the plan did not say, and the market probability enters a bet only on those four. Tie-break among passing variants, which the plan called fixed without stating it: lowest pooled log loss, then the order current, multiplicative, power, Shin, each with all-books mean, bettable-only mean, all-books median. Coverage is measured on all consensus rows of a variant, before outcomes are joined. Pushes and players who did not play are dropped, as in `market_bench.compare`.
6. A.5: with `--registry-root` the configurations of a research run are appended to `<registry root>/config_log.jsonl`, not to `data/props_models/config_log.jsonl`, so research entries stay out of the production log. The count of 60 tried rules does not come from that log.
7. C.2 and C.5, points the plan left open, fixed as implemented in `props/betting/a11_rules.py`: when the clip of a slope binds, each intercept is re-fitted with the slope held at the clipped value; a gap of exactly 0 takes the non-negative-side intercept; E3's c is profiled on the four bet stats, ties to the smaller c; C.5 item 2 is evaluated only if C1 is the pick after item 1, and items 3 and 4 compare against the running pick; the stop condition uses the pick's own slope with intercept on the four bet stats (E1's if C0 is the pick) and 'includes 0' means lo <= 0 <= hi; the S5 gate requires both the clipped k and the pooled slope within 0.05 of the frozen values.
8. 7c item 6: 'Sharpe at least 1.10 times flat's' is read as flat + 0.10 x |flat|, so the bar does not get easier when flat's Sharpe is negative. A per-game cap trims the marginal bet to the room left. '2023-24 not reversing' means the variant's Sharpe is at least the reference's on 2023-24; those 2023-24 numbers are in-sample for k and labelled so.
9. 7e item 6: 'pooled dev' is all played rows of the three dev folds, every slot, one pooled game-clustered bootstrap; the CRPS clause is the upper bound of the paired difference divided by the pooled `widen` mean CRPS, at most 0.0005 for each bet stat; the P(0) clause uses point estimates and a tie passes. Known before running: the negative binomial discards the minutes-mixture shape that `widen` keeps, and the simulated variance of threes and ra carries about 2% noise. The experiment runs as registered; if it fails, widening inside the mixture is a new experiment.
10. Not built for S1 to S8, reported as gaps: ROI by blowout margin in 6b (the inputs carry no final margin); placebo CLV for every candidate (computed for the C.5 pick only). If 7a adopts a variant whose consensus line differs from the current one, S1 must first write a price file for that variant; until then S6 keeps only the bets on the variant's line and logs how many it dropped.

House style follows `phase5_revised_rule_preregistration.md`: plain, numbered, fixed thresholds, written before the data that will judge them is computed. Each threshold has its reason in one line. Where a fact could not be checked in code or data, the text says so in place.

Scope: research only. No change to `config/props_frozen_rule.json`, `config/props_live.json`, `data/props_features_v3.db`, `data/props_models/v3_live_2026-27r1/`, the live roster rule, or any scheduled job. A change that passes here becomes part of **one candidate** (model + rule file under new names and hashes). That candidate goes to the owner, and the 2026-27 forward record is the confirmatory test (section A.2).

## 0. What the code and data say before we start

Read for this plan: `props/models/{pipeline,folds,data,registry,market_bench,evaluate,dispersion,minutes}.py`, `props/features/{build,rows,injury_at_t,history,decay,market_env}.py`, `props/live/{roster,at_t,base,parity,monitor}.py`, `props/betting/{prices,rules,evaluate}.py`, `props/analysis.py`, `core/devig.py`, `scripts/{train_props_models,build_props_features,props_phase5_dev,props_phase5_predict,props_phase5_revise,props_d1_parity,train_props_live}.py`, `scripts/run_phase5_final_chain.sh`, and the reports `phase5_plan.md`, `phase5_revised_rule_preregistration.md`, `phase4_protocol_update.md`, `phase4_ablations.md`, `d1_parity_report.md`, `d1_parity_preregistration_addendum2.md`, `forward_test_preregistration_DRAFT.md`. The data checks below were small read-only queries (`mode=ro`): grouped counts on `props_features_v3.db` and the `snapshot_log` table of `props_odds.db`.

These findings change the design. Each is used below.

1. **Rule v2 removes Out players from the member set** (`props/live/roster.py` `rule_members`, lines 92-98: previous-game roster minus report-Out, plus non-Out report players, plus quoted players). `injury_at_t.teammate_availability` (lines 130-167) aggregates only over the rows it is given, so live `av_n_out` is about 0 **because of how the rule is built**, not because data is missing. If rule v2 is applied to training as written, `av_n_out` and `opp_av_n_out` become constant at about 0 in training too. The `av_miss_*` shares then reflect only Questionable/Doubtful/Probable teammates, so an Out star teammate is no longer visible to the model. That removes the skew by removing the information. Section B therefore runs two arms: a pre-game rule that matches the live rule, and a pre-game rule that keeps Out players with p_play of about 0.
2. **Share of training rows with `av_n_out > 0`**, measured on `player_game_at_T`: 15.3% at T-240, 16.0% at T-60, 19.1% at T-5, pooled over 2019-20 to 2025-26. At T-60 by season: 15.7, 18.8, 10.1, 9.0, 14.3, 24.7, 19.0%. The audit's 22% matches only some slices (2024-25 and 2025-26). The direction of the audit fact holds.
3. **Injury-report coverage holds from 2019-20.** The main archive starts at `Injury-Report_2019-10-21`, with 216 PDFs in 2019, 373 in 2020 and 1,100-1,638 a year from 2021. The hourly archive starts at 2021-10-19. `inj_report_missing` is 0 at T-60 in every season (0.09% in 2023-24). The weaker part is staleness: the mean `inj_report_age_min` at T-60 is 121 and 114 minutes in 2019-20 and 2020-21, against 50-59 minutes from 2021-22 on. Previous-game rosters come from the box score, from 2013-14 on (`rows.FIRST_HISTORY_SEASON`). **Quotes**, rule v2's third component, exist only from 2023-24 (`snapshot_log` hist modes start 2023-10).
4. **The game environment before 2023-24 has no T-60 value.** `market_env.sbr_proxy` gives T-240 the SBR opening line, T-5 the SBR closing line (which is later than T-5), and T-60 nothing. Measured: `env_total` is non-null on 0% of T-60 rows in 2019-20 to 2022-23 and on 99.9% from 2023-24. The live path has no environment code at all (`props/live/at_t.live_at_t`: "minus the market environment"). This constrains 7b.
5. **Through-origin bias, sign.** For r = a + b g + e, the through-origin slope estimates b + a·E[g]/E[g²]. The audit gives a < 0 (market over-bias: hit runs 1.4-2.5 points below p_market) and E[g] < 0 (mean gap about -0.03). With both negative, the bias term is **positive**, so the frozen k is expected to be **overstated** and should fall under an intercept estimator. The simulation in `tests/props/test_wp5_monitor.py` uses a mean gap of +0.02 and so shows the downward case. The point stands (biased when both are non-zero); the expected direction here is the opposite of that test's. Not checked: whether the held-out 2025-26 slopes quoted by the audit (0.239, 0.336, 0.173, 0.089) are through-origin. `market_bench.information_slope` is through-origin, so they probably are.
6. **Bettable books per season** (from `snapshot_log.books`, not checked on `prop_snapshots` itself): 2023-24 had DraftKings, FanDuel, BetMGM, Caesars, BetRivers and Unibet. 2024-25 had DraftKings, FanDuel, BetMGM, Caesars, BetRivers and Fanatics (from mid-season); Unibet is absent. 2025-26 had DraftKings, FanDuel, BetMGM, Caesars, BetRivers and Fanatics. "Best of 7 books" is in practice best of 5-6. Offshore books in the consensus: Bovada, BetOnline and MyBookie in 2023-24; Bovada and BetOnline later. PointsBet (US, 2023-24 only) is held but is not in `prices.BETTABLE`.
7. **The dev and frozen artefacts are gone.** `data/props_models/` holds only `v2_base/manifest.json` (the rate families) and `v3_live_2026-27r1`; `data/props_betting/` does not exist. The current `props_features_v3.db` (sha256 `7350a26f...`, rebuilt 2026-11-03, +90 KB) is not the file behind the frozen rule (`2d221a8d...`). Regenerated dev predictions will not reproduce the frozen k exactly, which is why step S5 runs a reproduction check.
8. **Uncommitted WP2 edits in this worktree change features.** `props/features/history.py` and `decay.py` now drop played games with a missing stat from that stat's rate. A features DB rebuilt with this code is not byte-identical to v3 in the base table. Section B builds a control DB to measure that difference.
9. **This worktree's `data/` is a real directory, not a link.** It holds only `raw/nba_schedule`. `CLAUDE.md` says every worktree reaches `data/` through a link; this one does not. No step can run from here as it stands (section E.0), and this folder must never be deleted recursively.
10. **The exposure caps have already been tried and seen.** `props_phase5_dev.py` counted the three exposure caps among its 35 tried rules. On 2025-26, cap 1 per player-game gave +6.45% against +4.63% uncapped (best price, `phase5_final_2025-26.md`). 7c is a re-test of an old idea at the median price, not a fresh hypothesis.

## A. Evaluation protocol shared by all experiments

### A.1 Season roles

| Season | Model experiments (6a, 7b, 7e) | Rule experiments (6b, 7a, 7c, 7d) | Why |
|---|---|---|---|
| 2019-20, 2020-21 | training windows only | not used (no quotes) | first two dataset seasons (`folds.FIRST_SEASON`) |
| 2021-22, 2022-23 | **dev test folds** (CRPS, log score at fixed lines) | not used (no quotes) | development folds since Phase 3 |
| 2023-24 | **dev test fold**; information slope at book lines | **training season for walk-forward k**; in-sample rule results reported | dev fold for the model; first market season |
| 2024-25 | **seen**: reported, never decides a model choice (`phase4_protocol_update.md` section 1) | **the out-of-sample season for rules**: k from 2023-24 only, scored here | used before to choose the 4 stats and the pooled k, so it is a development season for rules, not a confirmation |
| 2025-26 | **excluded**: no fold trained or scored, no prices loaded | **excluded** | scored as the final, again for S1/S2, the Phase 4 criteria and parity attempt 2; its numbers motivated this audit |
| 2026-27 forward | | **the confirmatory test** | the only data no choice here has seen |

Enforced in code: every new script refuses `2025-26` in `--folds`, `--season`, `--seasons` and `--score`, and refuses a date range ending after 2025-06-30. Exception: the final live candidate (step S17) is trained through 2026-06, like `v3_live_2026-27r1`. That is a fit, not a choice.

Walk-forward for k: k for season s comes only from seasons before s. With quotes starting in 2023-24, that gives exactly **one** out-of-sample season for any k estimator (2024-25, k from 2023-24). This is a power limit (A.5); it is stated, not worked around.

### A.2 Is there a confirmatory look left on 2025-26?

1. **Against using it.** 2025-26 has been scored under the headline rule, S1 (R2b), S2 (the original rule) and the Phase 4 criteria, and re-used for parity attempt 2 (40 dates). Its per-stat slopes, its best-vs-median gap, its no-model +7.0% and its cap results are among the facts that motivated WP6 and WP7. A pipeline re-specified knowing those numbers is not independent of them: a 2025-26 score of the candidate would reward the very changes that 2025-26 suggested (fitting to the holdout by another route). One look would also be a single, unpaired-with-forward number that would invite a second look if it disappointed.
2. **For using it.** It is the only full season of quotes left, about 15,600 bets under the revised rule. The forward record grows by about 70 stakes a night, so a 2-point ROI difference takes most of a season to resolve.
3. **Recommendation: only the 2026-27 forward record confirms.** 2025-26 stays untouched until the forward test of the candidate has reached its pre-registered horizon. After that it may be scored once, labelled "seen", as a description. Owner decision G2.

### A.3 Primary metric

1. **ROI per unit staked at the median bettable price**, settlement treatment `stands`, for the rule's own bets. Bets are selected exactly as in the frozen rule (EV at the best bettable price, `rules.select`) and **priced** at the median bettable price of the same market, side and slot (`props_phase5_dev.market_level` / `decompose`, "repriced at the MEDIAN"). Owner decision of 2026-11-12: median primary, best secondary. Selection stays on best-price EV because the owner intends to rest limit orders at the best price.
2. **Interval: bootstrap over game dates** (2,000 replicates, percentile, seed 20261112). Paired differences between two rules are computed on the **same date resamples**, which gives a paired interval. Why dates, not games: bets on one night share injury news, market-wide line moves and model inputs; 59% of live bets share a player-game, and games are nested in dates. Clustering by date is the more conservative choice, and it is the forward pre-registration's cluster, so dev and forward numbers are comparable. A season has about 165-200 game dates, enough clusters. The repo's `props/models/evaluate.cluster_bootstrap` clusters by whatever array it is given: it is called with `game_date` for ROI. The game-clustered interval (what the Phase 5 reports used) is printed next to it as a secondary. `props/live/monitor.cluster_bootstrap_roi` already does date clustering and is reused.
3. **Model-quality metrics** (CRPS, log score) keep the game cluster (`evaluate.compare_versions`), as in Phase 4. They are averaged over rows, not stakes, and game clusters keep them comparable with every earlier report.

### A.4 Secondary metrics (reported every time, decisive only where a keep rule names them)

1. ROI at the best bettable price (same bets, same intervals); ROI at flat -110; the other two settlement treatments.
2. **Information slope with an intercept**, per stat and pooled (stat-specific intercepts), date-clustered. Reuse `props/live/monitor.information_slope`, which returns `slope_origin`, `slope_icpt`, `intercept`, `over_bias` and their intervals. Rows are the `market_bench.compare` rows: consensus line, every slot, pushes dropped.
3. **Log loss and Brier of p_model and p_market at the book's line** (`market_bench.score_table`), and **CRPS** on all played test rows (`evaluate.compare_versions`, paired, game-clustered).
4. **CLV to the T-5 consensus** (`props/betting/evaluate.clv_table`), real minus the four-seed placebo (`placebo_quotes`, seeds 11-14), as in Phase 5.
5. **Calibration by side**: expected wins under p* against actual wins, overs and unders separately, with clustered SE; Brier of p* on the bet side.
6. ROI by claimed-EV bucket (3-4, 4-5, 5-7, 7%+), by stat, by slot, and by blowout (|final margin| ≥ 20, descriptive only, since the margin is not known before the game).

### A.5 Multiple comparisons

Planned comparisons (each line is one candidate judged against a reference):

| # | Experiment | Comparison |
|---|---|---|
| 1 | 6a | A1 (rule v2, pre-game) vs R (deployed model scored on pre-game features) |
| 2 | 6a | A2 (pre-game, Out players kept) vs R |
| 3 | 6a | A2 vs A1 |
| 4 | 6a | A0 (no teammate group) vs R; conditional, counted anyway |
| 5 | 6b | C1 (k with intercept) vs C0 (through-origin k) |
| 6 | 6b | C2 (intercept also in p*) vs C1 |
| 7 | 6b | C3 (per-side intercept) vs current pick |
| 8 | 6b | C4 (saturating gap) vs current pick |
| 9 | 6b | per-slot k (secondary, reported) |
| 10-17 | 7a | 8 de-vig x consensus variants vs current (multiplicative, all-books mean) |
| 18 | 7b | with_env vs reference model |
| 19, 20 | 7c | cap 1, cap 2 per player-game vs uncapped |
| 21 | 7c | fractional Kelly with per-game cap vs flat |
| 22 | 7d | unders, expected minutes < 20 vs ≥ 20 |
| 23 | 7e | negative-binomial refit vs `widen()` |

**23 comparisons.** Correction: any statement in a report that a candidate is "significantly better" uses a **Bonferroni level of 0.05/23 = 0.0022** (99.78% intervals), and those adjusted intervals are printed for every comparison. The keep rules themselves use pre-declared margins plus unadjusted 95% non-harm bounds. The dev stage only proposes candidates, and the forward test confirms them as **one** comparison (bundled candidate vs frozen rule). Cumulative tried-rule count carried into the next forward pre-registration: 37 (Phase 5) + 23 = **60**. Every model configuration trained here is also appended to `data/props_models/config_log.jsonl` by `common.log_config`, as before.

### A.6 One candidate, one forward segment

Every change adopted under sections B to D goes into a single candidate: model version, features DB, live-builder options and rule file, each with its own hash. Changes are not deployed one by one. The candidate is shadowed on the same slates as the frozen rule as a new numbered segment (`forward_test_preregistration_DRAFT.md` section 11). Starting it is an owner decision.

### A.7 Reported whatever the result

Every comparison in A.5 is reported with the same table layout, including failures and comparisons whose keep rule was not reached. A step that is skipped because an earlier step failed is listed as skipped, with the reason.

## B. WP6a: pre-game roster in training

### B.1 The defect (code)

1. Training rows: `rows.load_history` (line 95) takes the roster from `player_advanced_stats_v3`, i.e. everyone **listed in the box score** after the game, played or not.
2. `build.build_at_t` (lines 72-94) calls `injury_at_t.at_t_block` (lines 173-179) on those rows. So `teammate_availability` sums (1 - p_play) x trailing weight over teammates who were listed, which is a post-game fact. An Out teammate is listed 36% of the time if he was in the previous game and 3% if not (`d1_parity_preregistration_addendum2.md` section 2).
3. Live, `props/live/at_t.live_at_t` runs the same `at_t_block` on the member set of `roster.rule_members('v2', ...)`.

### B.2 Arms and references (all trained with the frozen configuration: `base` spec, families from `v2_base`, `--refit-all`, no recency, no recalibration)

| Name | Training at-T features | Scored on (test rows) | Role |
|---|---|---|---|
| **L** | box-score rosters (v3 DB) | v3 DB | the leaky backtest view; descriptive (measures the look-ahead on dev data) |
| **R** | box-score rosters (v3 DB) = model `a11_r0` | **pre-game rule v2nq features** (pg1 DB) | the deployed situation: leaky training, pre-game serving. **Reference.** |
| **A1** | pre-game rule **v2nq** (pg1 DB) | pg1 DB | train/serve identical, live rule unchanged except dropping the quote component (B.4.3) |
| **A2** | pre-game rule **pregame_all** (pg2 DB) | pg2 DB | train/serve identical, Out players kept with p_play about 0; needs the matching live option |
| **A0** | v3 DB with `--exclude-groups teammates` | v3 DB (no av_* used) | fallback: no teammate inputs at all. Trained only if A1 and A2 both fail non-harm |

Target rows (who is predicted and scored) stay the box-score rows in every arm, so test rows are identical across arms and every comparison is paired. Only the teammate-set used inside `av_*` and `opp_av_*` changes.

### B.3 Membership rules, at each decision time T, for team X in game G on date D

1. `v2nq` (A1): players listed in X's previous game of the same season before D whose latest team before D is X (`roster.previous_game_rosters`), **minus** players the usable report lists Out, **plus** report players with any other status. This is `rule_members('v2')` without component 3 (quotes).
2. `pregame_all` (A2): the same previous-game roster, **plus every** report player, Out included. An Out player enters with his report status, so p_play is about 0 and his trailing weight counts as missing.
3. Usable report: stamped at least `PUBLISH_LAG_MIN` = 15 minutes before T (`injury_at_t.latest_report_rows`), the latest such report per game. Report players are placed on the team the report names; a player the report does not place goes to his latest team this season if that team is in the game, and otherwise is dropped. This is the logic of `roster.resolve_extra_players`.
4. The first game of a season has no previous game: members are the report players only, as live.
5. For every target row the aggregates are taken over the members. The player's own contribution is subtracted only when he is a member: a target not in the member set (an Out player who was listed, or a late call-up) gets the team aggregate as it stands. Opponent aggregates are taken over the opponent's members.

### B.4 Historical inputs

1. **Injury reports:** present for every game from 2019-20 (section 0 item 3). Use the existing linked cache `data/processed/props_injury_linked.pkl` (2026-11-03 10:52, the one the v3 rebuild used). Record its sha256 in the new DB's manifest. **Never pass `--refresh-reports`**: that rewrites the shared parse caches in `data/processed/` that the live jobs read.
2. **Previous-game rosters:** from `load_history` (box score) for all seasons. They are pre-game information for game G because the previous game was finished before D.
3. **Quotes:** only from 2023-24. Using them would make the feature definition change with the era, the same problem as `env_source`. Proposal: **drop the quote component from the teammate-set definition in all seasons** (v2nq); quoted players still create live target rows. Before training, step S11 measures how much the quote component moves live `av_*` on the 30 dev dates (`--roster-rule v2` against `v2nq`). Owner decision G4.
4. **Trailing weights for members who are not listed in game G** (they have no box-score row there): `WEIGHTS` and `m_bucket` use only `m_min_h15` and `r_*_h15`. These are decays over PLAYED rows (`decay.decayed_before`, played mask), and the league priors come from earlier seasons' played sums. So a placeholder row (not played, stats NaN) appended to the history cannot change any real row's weight columns. Plan: append one placeholder per (G, member not listed in G) to the full history, run `history.add_history_features` **once**, and keep only the placeholders' weight columns and `m_bucket`. Real rows keep the base table of the normal build. Test L4 proves the invariance on the whole dataset. Skip a placeholder if the player is listed in any game on date D (same-day trade edge).
5. **Missing pieces:** none for the rule itself. The staleness of 2019-20 and 2020-21 reports (about 2 hours at T-60) is real but is already in v3 and is a model input (`inj_report_age_min`). It is not changed here (section F, item 5).

### B.5 New code (on a branch from a clean commit; the deployed paths keep their defaults)

1. `props/features/pregame_roster.py` (new):
   - `previous_game_members(hist)`: a vectorised `roster.previous_game_rosters` plus `current_team_only` for every team-game.
   - `members_at_t(prev, report_rows_T, rule)` for `v2nq` and `pregame_all`.
   - `placeholder_weights(hist, members, pos)` (item B.4.4).
   - `availability_over_members(targets, members_status, weights)`, a generalisation of `teammate_availability` with the self-subtraction of B.3.5. For the box-score rule it must give, float for float, what `teammate_availability` gives today (test).
2. `props/features/build.py`: `build_at_t(..., membership='box')`. `'box'` is today's behaviour and the default, and a test asserts the default output is unchanged. `build_all` and `scripts/build_props_features.py` pass `--membership`.
3. `props/live/roster.py`: add rule options `'v2nq'` and `'pregame_all'` to `rule_members` and `RULES`. `config/props_live.json` keeps `roster_rule: v2`, so live behaviour does not change.
4. `scripts/props_d1_parity.py`: new flags `--features-db` (today hard-coded to `data/props_features_v3.db`) and `--out` (today `data/props_live_parity/attempt{N}`), and the new rule options. The pre-registered attempt log is not extended: these runs are development checks, written to the experiment folder.

### B.6 Building the new databases (never touching `data/props_features_v3.db`)

1. `scripts/build_props_features.py --out-db <new path> --dictionary-dir <new folder> --membership <rule>`. Without `--dictionary-dir` the script overwrites `reports/props/phase2_feature_dictionary.*`, so the flag is mandatory. A guard added to the script refuses `--out-db` equal to `data/props_features_v3.db` or `data/props_features.db` and refuses an existing file.
2. Three builds at one code commit: `ctl` (membership `box`, the control: today's rule, today's code), `pg1` (`v2nq`), `pg2` (`pregame_all`). Each new DB's sha256 is written to `<db>.manifest.json` with the code commit and the linked-cache hash.
3. **Control check:** compare `ctl` with v3 cell by cell (base, labels, at-T; tolerance 1e-9). If more than 0.1% of base cells differ (expected only from the WP2 missing-stat fix), `ctl` replaces v3 as the reference DB, and R and L are re-trained on it (step S3 repeated, +2 h). Reason: the arms must differ from the reference only in `av_*`.

### B.7 Leakage tests (must pass before any arm is trained; `tests/props/test_a11_pregame_roster.py`)

1. **L1 roster-listing invariance:** on 200 sampled games and every T, perturb the box-score listing of non-target players **for the same game**: remove a listed did-not-play player, and add a fabricated listed player. Every other row's `av_*` and `opp_av_*` must not change (exact). Under membership `box` this test must **fail** (it documents the leak); under `v2nq` and `pregame_all` it must pass.
2. **L2 future-report invariance:** add a report stamped later than T - 15 minutes (with an Out status for a starter). No feature at T changes.
3. **L3 outcome invariance:** change `y_*`, `minutes` and `lbl_*` of game G. No at-T feature of G changes.
4. **L4 placeholder invariance:** every base column of every real row is identical with and without the placeholders (whole dataset, exact).
5. **L5 equivalence:** `previous_game_members` equals `roster.previous_game_rosters` on 50 sampled dates of each dev season; `availability_over_members` with the box-score member set equals `teammate_availability`.
6. **L6 parity A4** (run, not a unit test; step S11): `scripts/props_d1_parity.py --no-model` on the 30 dev dates of 2024-25 (`reports/props/d1_roster_dev_dates.json`, no 2025-26 date), live builder with the matching option against each new DB. **Pass: A3 = 100% and every `av_*`/`opp_av_*` column within the A4 tolerances on at least 99% of rows; A2 identical for all base columns.** Reason: both sides run the same rule, so anything under 99% is an implementation difference, not information. A failure stops section B (section F).

### B.8 Runtime and disk (sources in brackets)

1. Build with the cached reports: the base step took about 10 s and the at-T and write steps about 1 minute in the 2026-11-03 restore log (`data/raw/build_props_features_restore.log`; the 38 minutes in that log were PDF parsing, which is not repeated). The membership and placeholder steps are new. Estimate 20-40 minutes per DB, peak 6-8 GB (one history pass over about 300k listed rows plus placeholders). `d1_parity_report.md` gives "dataset rebuild ~1 h", used as the upper bound.
2. Parity on 30 dates: "the build takes ~30 minutes" for 40 dates (`props_d1_parity.py` comment). Estimate 25 minutes per run, 3 runs.
3. Training one version on the four folds 2021-22 to 2024-25: "folds ~30 min each" (`d1_parity_report.md`). The live fold took 1,666 s on 685k rows with refit (`train_props_live_restore.log`). Estimate **2 h per version**. Peak memory: not stated; estimate 8-12 GB (frame about 700k x about 260 columns, about 1.5 GB as float64, plus LightGBM copies). The machine has 32 GB and 16 threads.
4. 20,000-draw predictions on quoted rows: "about 30 minutes of compute per season" (`phase5_plan.md` section 8). 4,000-draw all-played predictions for the cross-scoring of R: not stated; estimate 5 minutes per fold.
5. Disk: each features DB about 690 MB (v3 is 684 MB); each model version with predictions about 0.6-1 GB (the live bundle alone is 29 MB; the pmfs dominate). Section B total about 5 GB. Free space on C: 87 GB.

### B.9 Keep rule (written before running; dev folds 2021-22, 2022-23, 2023-24 decide; 2024-25 reported as seen)

Preconditions: L1-L6 pass. Without them nothing below is computed.

1. **Non-harm against R** (for A1, A2, and A0 if trained). For each bet stat (rebounds, assists, threes, ra), take the paired CRPS difference (arm - R) over all played test rows of the three dev folds pooled, game-clustered. Its **95% upper bound must be at most +0.10% of R's mean CRPS for that stat.** Reason: one tenth of the value of the whole teammates group (0.78-0.99% of CRPS, `phase4_ablations.md`), below any size that moves bets. In addition, the **information slope with intercept** pooled over the four stats on 2023-24 (date-clustered) must be **at least R's minus 0.02.** Reason: about 0.8 standard errors of a one-season pooled slope (the frozen two-season pooled interval is ±0.037).
2. **A2 over A1:** A2 passes item 1, and the pooled-four-stat paired CRPS difference (A2 - A1) has a **95% upper bound below 0**, and no bet stat has a point difference above +0.05%. Reason: A2 needs a live-builder change, so it must show a real gain, not a tie.
3. **Choice, in this order:** A2 if items 1 and 2 hold; otherwise A1 if item 1 holds; otherwise train A0 and take it if item 1 holds; otherwise no change (the skew stays, disclosed as now).
4. This is a validity correction (`phase4_protocol_update.md` section 2): it is adopted for non-harm, not for a gain. It is expected to **lower** backtest numbers against L, and that is not a reason to reject it.

### B.10 Reported whatever the result

L vs R (the size of the look-ahead on dev data: CRPS, slope, and median and best ROI with the frozen k on 2023-24 and, as seen, 2024-25); A1, A2 and A0 against R on every A.4 metric; the share of rows whose member set differs between `box` and each rule, by season and T; the `av_n_out > 0` share under each rule; the parity tables; and the quote-component effect from S11.

## C. WP6b: re-estimating k

### C.1 Rows

`market_bench.compare` rows: one row per (game, player, slot T, stat) with a consensus quote and a model pmf, at the consensus line, pushes dropped. These are the rows the frozen k used: all slots and all eight stats, with only the four bet stats entering the rule. g = p_model - p_market, r = hit - p_market. The p_market is the 7a choice (current method unless 7a adopts another; 7a runs first).

### C.2 Estimators (per stat s, least squares, date-clustered intervals; seasons as in C.3)

| Code | Model | Parameters per stat |
|---|---|---|
| E0 | r = b g (through the origin; `market_bench.information_slope`, the frozen method) | b |
| E1 | r = a + b g | a, b |
| E2 | r = a⁺·1[g ≥ 0] + a⁻·1[g < 0] + b g (separate intercept by the side the model leans to) | a⁺, a⁻, b |
| E3 | r = a + b·clip(g, -c, c), **the only saturating form**; c is common to all stats, chosen by profile least squares on the training season over the grid {0.02, 0.03, ..., 0.12}, with a and b re-fitted per stat for each c | a, b, plus one shared c |
| E4 | E1 separately by slot (240, 60, 5) | a, b per slot (secondary, reported only) |

Every slope is floored at 0 and capped at 0.5 (`rules.clip_k`), as in the frozen rule. E1-E3 reuse `props/live/monitor.information_slope` for point and intervals; E2 and E3 need new code (`scripts/a11_k_estimation.py`).

### C.3 Walk-forward

1. Out-of-sample: each estimator is fitted on 2023-24 and scored on 2024-25. This is the only out-of-sample comparison available (A.1).
2. In-sample 2023-24 numbers are reported and labelled in-sample. They decide nothing.
3. Final candidate (step S16): the chosen estimator is re-fitted on 2023-24 + 2024-25 pooled, as the frozen k was, for the **final candidate model's** predictions. k depends on the model, so it is re-estimated after 6a and 7b settle the model. Whether 2025-26 joins that final fit is owner decision G5 (recommendation: no).

### C.4 Bet rule formulas (EV as in `rules.ev_over_under`, pushes refund, same selection `rules.select`, e = 0.03, earliest slot, 3 books)

| Candidate | p* |
|---|---|
| C0 (frozen method) | p_mkt + b₀ (p_mod - p_mkt) |
| C1 | p_mkt + b₁ (p_mod - p_mkt), with b₁ from E1 and the intercept **not** applied |
| C2 | clip(p_mkt + a + b₁ (p_mod - p_mkt), 0.01, 0.99) |
| C3 | clip(p_mkt + a^{sign(g)} + b₂ (p_mod - p_mkt), 0.01, 0.99) |
| C4 | clip(p_mkt + a + b₃·clip(p_mod - p_mkt, -c, c), 0.01, 0.99) |
| P0 (placebo) | clip(p_mkt + a, 0.01, 0.99): **k = 0 with the intercept** |
| P00 (reference) | p_mkt (k = 0, no intercept: the "market-only" rule of `props_phase5_dev.decompose`) |

**Is the intercept a disguised "bet unders" rule?** Yes, in part. With a ≈ -0.02 for every stat, C2 lowers p* on every market and so tilts every decision toward the under, with or without the model. P0 is exactly that tilt with no model. So every C2-C4 result is printed next to P0 on the same quotes and the same date resamples, and the model's contribution is reported as **(candidate - P0)**. P00 shows what price dispersion alone earns (+7.0% at the best price on 2025-26, per the audit). If C2 beats C1 only by about what P0 earns, the gain is the market over-bias, not the model. Fixed now: that gain is then attributed to the over-bias in the report, and C2 is not adopted (C.5 item 2).

### C.5 Keep rule (2024-25 out-of-sample; median-price ROI; paired date-clustered differences)

1. **C1 replaces C0** unless C1 is **clearly worse**: point ΔROI_med (C1 - C0) below -0.5 points **and** its 95% upper bound below 0. Reason: E0 is a biased estimator (section 0 item 5); correcting it is a validity fix, adopted unless it demonstrably costs money.
2. **C2 replaces C1** only if ΔROI_med (C2 - C1) is at least +0.25 points (point), (C2 - P0) is at least +0.25 points (point), and C2's calibration by side (expected wins under p* against actual) is within 2 clustered SE for both overs and unders. Reason: the P0 condition separates model from tilt; the margin is +0.25 by the amendment of 2026-11-12 (was +0.5).
3. **C3 replaces the current pick** only if ΔROI_med is at least +0.25 points (point; amended 2026-11-12, was +0.5) with a 95% lower bound of at least -1.0 point. Reason: one extra parameter per stat needs a margin, and a lower bound of -1.0 rules out a likely loss.
4. **C4 replaces the current pick** only if item 3's conditions hold **and** the realised median-price ROI of its 7%+ claimed-EV bucket is at least the current pick's (point). Reason: C4 exists to fix the over-trusted tail, so it must improve the tail.
5. **E4 (per slot)** is reported only. Not adopted in this round.
6. **Stop condition:** if the chosen estimator's pooled slope-with-intercept on 2024-25 has a 95% interval that includes 0, the model shows no information beyond the market's bias. Rule work stops there and is reported (section F, item 9).

### C.6 Files

`config/props_frozen_rule.json` and `reports/props/phase5_revised_rule.frozen_rule.json` are **not edited**. The candidate is written as `reports/props/audit_2026-11/candidate_rule_a11.json`, with the same keys as the frozen file plus `estimator`, `intercept_by_stat` (if C2-C4), `c` (if C4), `developed_on`, `tried_rules: 60`, `features_db_sha256`, `model_version` and `code_commit`. Its sha256 goes into a sidecar `candidate_rule_a11.json.sha256`. The live job reads only `config/props_live.json`, which is not touched.

## D. WP7 pre-registrations (dependency and cost order: 7a, then 7c/7d/7e on the regenerated base predictions, then 7b after 6a)

### 7a. De-vig method and consensus set

1. **Hypothesis:** the market's over-bias of 1.4-2.5 points is partly a measurement artefact of multiplicative de-vig and of averaging in offshore books that cannot be bet. Another method or set gives a better-calibrated p_market.
2. **Exact change:** 3 de-vig methods (`core.devig` multiplicative, power, Shin, via `analysis.add_devig(method=...)`) x 3 consensus sets: (i) mean over all books at the consensus line (current, `analysis.consensus`), (ii) mean over bettable books only (`prices.BETTABLE` as present that season; consensus line recomputed on those books; at least 3 books), (iii) median over all books. 9 variants, 8 comparisons.
3. **Files:** new `scripts/a11_cache_quotes.py` and `scripts/a11_devig_calibration.py`; `analysis.consensus` gains `agg='mean'|'median'` and `books=None` (defaults unchanged).
4. **Data:** 2023-24 and 2024-25 quotes at hist_T-240/-60/-5, main lines, the usual quality flags; outcomes from `player_game_labels`. No model.
5. **Query cost and chunking:** `prop_snapshots` has no index on `commence_time_utc`. `prices.load_season` filters on `substr(commence_time_utc,1,10)`, which makes SQLite scan all 4.2 GB for every call. Instead, take event ids per calendar month from the `events` table (`et_game_date`, 4,118 events) and query with `p.event_id_vendor IN (...)`, which uses `ix_props_event_ts` (checked with `EXPLAIN QUERY PLAN`). Hist-mode rows from `snapshot_log.n_prop_rows`: about 3.5M in 2023-24 and about 2.1M in 2024-25, all lines included; `is_alt_line = 0` reduces this. Estimate 1-3 minutes of read per season month by month, and peak memory under 3 GB per monthly chunk (a whole season at once would be about 2 GB of rows plus 2-3 copies inside `add_devig` and `flag_main_lines`). The consensus is per snapshot, so month chunks give exact results. Outputs: one consensus table per variant and season, plus `prices_<season>.pkl` in the format `prices.load_season` returns, for all later steps. Run in the off-window anyway: the snapshot collector writes to this file during the day.
6. **Metric:** on rows where every variant has a consensus at the **same** line, pooled over slots: log loss and Brier of p_market against the outcome; mean bias (hit - p); a logistic recalibration hit ~ logit(p) (intercept and slope; ideal 0 and 1); bias by p_market bucket {<0.35, 0.35-0.45, 0.45-0.50, 0.50-0.55, 0.55-0.65, ≥0.65}; all by season and by stat. Paired variant-minus-current differences, date-clustered. Coverage = rows with a consensus, as a share of the current method's.
7. **Keep rule:** variant V replaces the current method if, in **each** of 2023-24 and 2024-25 separately, its paired log-loss difference has a 95% upper bound below 0, its |mean bias| is no larger than the current method's (point), and its coverage is at least 95% of current. If several pass, take the lowest pooled log loss (tie-break fixed now). Reason: p_market enters every later step, so it must be better in both seasons, not on average.
8. **Reported regardless:** the full 9 x metrics table by season, stat and bucket; how much of the over-bias each variant removes; the bettable-only coverage loss by season.

### 7b. Game environment as model input (`with_env`)

1. **Hypothesis:** spread and total at T move minutes (blowouts) and rates. The `base` spec cannot see them (`data.BASE_GROUPS` excludes `market_env`), and under profits concentrated in games decided by 20+ points (+53 units against -38 elsewhere).
2. **Exact change:** `--spec with_env` (existing; adds the `market_env` group to every component: availability, minutes, rates). `env_source` stays out (`data.NOT_FEATURES`). Nothing else changes.
3. **Files:** none for training. If adopted, a live environment builder is needed (`props/live/at_t.py` or a new `props/live/env.py`) using `market_env.team_view` on the live `team_lines` snapshot at T, plus a parity test. It is built only after adoption and before any forward use.
4. **Data:** the 6a winner's DB (or v3 if 6a adopts nothing), dev folds 2021-22, 2022-23, 2023-24; 2024-25 reported as seen. **Era caveat:** test rows of 2021-22 and 2022-23 carry SBR-proxy environment (T-240 opening line, T-5 closing line, T-60 empty); 2023-24 carries snapshot environment, which is what live will see.
5. **Metric:** paired CRPS (four bet stats, plus minutes MAE as a diagnostic) against the reference model on the same DB; information slope with intercept on 2023-24; calibration of p_model at the book line in games with |env_team_spread| ≥ 10 at T (2023-24; a pre-game proxy for blowouts).
6. **Keep rule:** adopt if the pooled-four-stat paired CRPS difference (with_env - reference) over the three dev folds has a 95% upper bound below 0, **and** its point estimate on the 2023-24 fold alone is below 0, **and** no bet stat is worse with a 95% upper bound above +0.10%, **and** the 2023-24 slope with intercept is at least the reference's minus 0.02. Reason: a new live builder must earn its keep; 2023-24 is the only fold with live-like environment data.
7. **Reported regardless:** per fold and per T (T-60 separately, since it has no environment before 2023-24); results in |spread| ≥ 10 games; under-bet ROI split by final margin (descriptive).

### 7c. Per-player-game cap and sizing

1. **Hypothesis:** correlated bets on one player-game (ra together with rebounds or assists) add variance with little ROI; a cap or Kelly sizing improves ROI per unit of variance.
2. **Exact change:** on the rule's selected bets: cap 1 and cap 2 per player-game, keeping the highest EV (`rules.cap_per_player_game`); and sizing stake = min(0.25·f*, 0.02)·B with f* = EV/(d - 1) at the best price, a fixed bankroll B = 100 units (no compounding), and the total stake per game capped at 0.05·B (lower-EV bets cut first). The parameters 0.25, 0.02 and 0.05 are fixed now and not tuned.
3. **Files:** new `scripts/a11_rule_eval.py` (uses `rules`, `props/betting/evaluate`, `props/live/monitor`).
4. **Data:** base-model predictions (S4); k from the 6b choice, walk-forward (2023-24 fit, 2024-25 scored); 2023-24 in-sample reported.
5. **Metric:** median-price ROI; date-level Sharpe = mean / SD of daily profit per unit staked; maximum drawdown divided by total stake; the number of stakes.
6. **Keep rule (2024-25):**
   - A **cap** is adopted if its daily Sharpe is at least the uncapped rule's and its median ROI is no more than 0.5 points below (point), and 2023-24 shows its Sharpe no lower than uncapped. Reason: the point of a cap is less correlated variance; 0.5 points is the price we accept for it.
   - **Kelly** is adopted if its Sharpe is at least 1.10 times flat's, its max drawdown / total stake is at most flat's, and its median ROI is no more than 0.5 points below flat's, with 2023-24 not reversing the Sharpe ordering. Reason: sizing on p* repeats any over-trust of the tails (fact 2), so it needs a clear variance gain.
   - If 6b adopts C4, Kelly is evaluated on C4's p*.
7. **Reported regardless:** all three variants and flat, on both seasons, by stat; the share of bets that share a player-game. Already seen elsewhere (section 0 item 10): the caps were in Phase 5's tried rules, and their 2025-26 numbers are known.

### 7d. Unders on players with expected minutes under 20

1. **Hypothesis (post hoc, from the live record of October-November 2026):** among the rule's under bets, those on players with expected minutes under 20 return more than the others.
2. **Exact test, the only one:** among the rule's selected **under** bets (6b's k choice, walk-forward), compare median-price ROI for `expected_minutes` < 20 against ≥ 20. `expected_minutes` is the model's mean minutes given the player plays (`idx['minutes_mean']`, joined in `props/betting/evaluate.quote_table`). That it is the same quantity as the live `prop_bets.expected_minutes` column is inferred from the column name and the shared quote path; it is to be confirmed in code before the run. Threshold 20, fixed.
3. **Files:** `scripts/a11_rule_eval.py`.
4. **Data:** 2023-24 and 2024-25 backtest quotes, pooled, plus each season separately.
5. **Keep rule ("the effect exists in backtest quotes"):** the pooled difference (under 20 minus ≥ 20) has an **unadjusted 95% lower bound above 0** (amended 2026-11-12, was the Bonferroni bound), and its point estimate is positive in each season. The Bonferroni interval is still printed. Passing does not change the rule. It makes "min-under-20 unders" a pre-registered sub-hypothesis of the forward test. Failing drops it.
6. **Reported regardless:** the split for overs too; the no-model version (every under at T-60, median price, expected minutes under 20), descriptive.

### 7e. Negative-binomial dispersion in place of `widen()`

1. **Hypothesis:** stretching the piecewise-linear CDF (`dispersion.widen`) distorts low counts. On Poisson(2.5) with c = 1.09 it puts P(0) at 0.101, against 0.091 for a variance-matched negative binomial. That distortion matters at lines 0.5 and 1.5.
2. **Exact change:** for each row and stat with dispersion factor c > 1 (`DispersionScaler.c`, unchanged), replace `widen(pmf, sqrt(c))` with a negative binomial having the raw pmf's mean μ and variance c·σ², computed from the same raw pmf. Where c·σ² ≤ μ (no negative binomial exists), keep the `widen` output. Nothing is re-fitted. The c values stay as fitted on the calibration window.
3. **Files:** `props/models/dispersion.py` gains `nb_match(pmf, c)` and `DispersionScaler(method='widen'|'nb')` (default `widen`); `scripts/props_phase5_predict.py` gains `--save-raw` (stores pre-dispersion pmfs as `raw_<stat>.npy`) and `--all-played` (no price file; every played test row); new `scripts/a11_nb_dispersion.py`.
4. **Data:** raw pmfs of the base model's dev folds 2021-22, 2022-23, 2023-24 (4,000 draws, all played rows); 2024-25 reported as seen; book lines 0.5 and 1.5 from the 2023-24 quotes.
5. **Metric:** log score of P(Y > 0.5) and P(Y > 1.5) for threes, assists and rebounds on all played dev rows; the same at actual book lines 0.5 and 1.5 (2023-24); calibration of P(Y = 0); CRPS of the four bet stats.
6. **Keep rule:** adopt if, for threes, the pooled-dev paired log-score difference (nb - widen) at **both** 0.5 and 1.5 has a 95% upper bound below 0; no bet stat's CRPS is worse with a 95% upper bound above +0.05%; and |mean predicted P(0) - observed P(0)| for threes is no larger than under `widen`. Reason: the defect is at low counts, so the gain must show there without costing elsewhere. Threes is the stat whose book lines sit at 0.5 and 1.5.
7. **Reported regardless:** all three stats and both lines; the share of rows where the negative binomial does not exist; the effect on the rule's bets at those lines (descriptive).

### D.6 Order, sharing, dependencies

1. **7a first.** It needs no model, it is cheap, and its p_market feeds 6b, 7c and 7d.
2. **One regeneration of dev predictions** (model `a11_r0`, frozen configuration on the v3 DB, or `ctl` if B.6.3 says so) serves 6b, 7c, 7d and 7e, and is reference R/L for 6a. It needs: training of the four folds; 20,000-draw quoted predictions for 2023-24 and 2024-25 with `--save-raw`; and 4,000-draw all-played raw predictions for 2021-22 to 2024-25 (for 7e).
3. **6b runs before 7c and 7d**, because their bets depend on k.
4. **6a waits for its code and for L1-L6.** **7b waits for 6a**, because the environment model is trained on the DB that 6a picks, so no two models are compared on different rosters.
5. **The final k (S16) waits for 6a and 7b**, because k belongs to the final model. 7c and 7d are re-reported on the final model, descriptive only, with no second decision.

## E. Compute plan

### E.0 Preconditions (no compute)

1. Owner answers section G and signs this file.
2. **Run worktree:** a new worktree (for example `C:\Users\jorda\nba-props-a11`) on a branch from a clean commit after the WP1-WP5 changes are committed, with a `data` directory junction to `C:\Users\jorda\nba-model-hybrid\data` (`mklink /J data C:\Users\jorda\nba-model-hybrid\data`). The current worktree (`markouts-b5f108`) has a real `data` folder and cannot run these steps. Before that worktree or the run worktree is ever removed, **unlink the junction first** (the 2026-10-31 data loss).
3. **Tests in a worktree whose `data/` is NOT linked.** Tests that train models write `data/props_models/config_log.jsonl` relative to the working directory (`common.CONFIG_LOG`); with a junction they would append test entries to the real try count (compare `config_log.TEST_RUN_2026-11-03_not_a_real_count.jsonl`).
4. **Registry location:** `registry.ROOT` is the relative `data/props_models`. Proposal (G6): `--registry-root` flags on `train_props_models.py`, `props_phase5_predict.py` and `props_phase5_dev.py` (writes and reads of the new versions), while `--families-from v2_base` keeps reading from `data/props_models`. All models go to `OUT/models/`. Without the flag they would land as new, immutable version directories in `data/props_models/` (no overwrite is possible, since `write_manifest` refuses an existing version).
5. **Window:** heavy steps only when `run_live_bot.py` is not running (check before each step), between about 23:00 and 15:30 PT, and preferably finished by about 11:00 PT so that they do not overlap the props signal slots.

Notation: `PY=C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe`, `OUT=data/props_experiments/audit_2026-11`, `REG=$OUT/models`, `RPT=reports/props/audit_2026-11`. The working directory is the run worktree. "NEW" marks a script or flag that does not exist yet.

### E.1 Run list

| Step | What | Command | Wall time (source) | Peak memory | Outputs |
|---|---|---|---|---|---|
| S1 | Quote caches and price files, 2023-24 and 2024-25 | `$PY scripts/a11_cache_quotes.py --props-db data/props_odds.db --seasons 2023-24 2024-25 --out $OUT/quotes --prices-out $OUT/prices` (NEW) | not stated; estimate 30-45 min | estimate < 3 GB (monthly chunks) | `$OUT/quotes/<season>/cons_<devig>_<set>.pkl`, `$OUT/prices/prices_<season>.pkl` |
| S2 | 7a | `$PY scripts/a11_devig_calibration.py --quotes $OUT/quotes --features-db data/props_features_v3.db --out $OUT/results/7a --report $RPT/7a_devig/report.md` (NEW) | estimate 15 min | < 3 GB | tables + report |
| S3 | Base regeneration `a11_r0` (R, L) | `$PY scripts/train_props_models.py --version a11_r0 --features-db data/props_features_v3.db --folds 2021-22 2022-23 2023-24 2024-25 --families-from v2_base --refit-all --registry-root $REG --report $RPT/r0/training_report.md` (`--registry-root` NEW) | about 2 h (4 folds, "~30 min each", d1 report) | not stated; estimate 8-12 GB | `$REG/a11_r0/<season>/` bundles + 4,000-draw pmfs |
| S4a | 20,000-draw quoted predictions | `$PY scripts/props_phase5_predict.py --version a11_r0 --season <2023-24, 2024-25> --prices $OUT/prices/prices_<season>.pkl --out-version a11_r0_p5 --registry-root $REG --save-raw` (flags NEW) | 2 x 30 min (`phase5_plan.md`) | estimate 6 GB | `$REG/a11_r0_p5/<season>/` |
| S4b | Raw all-played pmfs for 7e | `$PY scripts/props_phase5_predict.py --version a11_r0 --season <each of 2021-22..2024-25> --all-played --n-sims 4000 --out-version a11_r0_raw --registry-root $REG --save-raw` (NEW) | estimate 4 x 5 min | estimate 6 GB | `$REG/a11_r0_raw/` |
| S5 | Reproduction check and baseline | `$PY scripts/a11_k_estimation.py --mode reproduce --p5-version a11_r0_p5 --seasons 2023-24 2024-25 --registry-root $REG --prices-dir $OUT/prices --features-db data/props_features_v3.db --out $OUT/results/r0_reproduce.json` (NEW), then `$PY scripts/props_phase5_dev.py --season <s> --p5-version a11_r0_p5 --full-version a11_r0 --no-r2b --prices $OUT/prices/prices_<s>.pkl --frozen-rule config/props_frozen_rule.json --registry-root $REG --out $OUT/results/r0_baseline_<s>.md` | estimate 20 min | < 6 GB | `--full-version` is required by `props_phase5_dev.py` even with `--no-r2b` (it loads it). **Gate:** the pooled through-origin slopes of the four stats must lie within ±0.05 of the frozen `pooled_slopes`; otherwise stop (F.4) |
| S6 | 6b | `$PY scripts/a11_k_estimation.py --mode walkforward --p5-version a11_r0_p5 --registry-root $REG --prices-dir $OUT/prices --pmarket <7a choice> --train 2023-24 --score 2024-25 --estimators E0 E1 E2 E3 E4 --out $OUT/results/6b --report $RPT/6b_k/report.md` | estimate 20-30 min (placebos, 2,000 bootstraps) | < 6 GB | k tables, chosen estimator |
| S7 | 7c, 7d | `$PY scripts/a11_rule_eval.py --p5-version a11_r0_p5 --registry-root $REG --prices-dir $OUT/prices --k-file $OUT/results/6b/k_walkforward.json --experiments 7c 7d --out $OUT/results/7cd --report $RPT/7cd/report.md` (NEW) | estimate 15 min | < 6 GB | |
| S8 | 7e | `$PY scripts/a11_nb_dispersion.py --raw-version a11_r0_raw --base-version a11_r0 --registry-root $REG --folds 2021-22 2022-23 2023-24 --seen 2024-25 --prices-dir $OUT/prices --out $OUT/results/7e --report $RPT/7e_nb/report.md` (NEW) | estimate 20 min | estimate 8 GB | |
| S9 | 6a code and unit tests L1-L5 | `$PY -m pytest tests/props/test_a11_pregame_roster.py -q` (NEW), **in the unlinked worktree** | minutes | small | must pass |
| S10 | Build `ctl`, `pg1`, `pg2` | `$PY scripts/build_props_features.py --membership <box, v2nq, pregame_all> --out-db $OUT/features/props_features_a11_<ctl, pg1, pg2>.db --dictionary-dir $OUT/features/dict_<name>` (`--membership` NEW; no `--refresh-reports`), then `$PY scripts/a11_compare_dbs.py --a data/props_features_v3.db --b $OUT/features/props_features_a11_ctl.db --out $OUT/results/ctl_vs_v3.json` (NEW) | 3 x 20-40 min (upper bound 1 h each, d1 report) | estimate 6-8 GB | 3 DBs about 690 MB each + manifests. B.6.3 decides whether S3 is repeated on `ctl` |
| S11 | Parity L6 | `$PY scripts/props_d1_parity.py --attempt 0 --season 2024-25 --dates reports/props/d1_roster_dev_dates.json --roster-rule <v2nq, pregame_all, v2> --no-model --features-db $OUT/features/props_features_a11_<pg1, pg2, pg1>.db --prices $OUT/prices/prices_2024-25.pkl --out $OUT/parity/<pg1, pg2, pg1_quotes>` (flags NEW) | 3 x about 25 min ("~30 minutes" per 40 dates) | estimate 6 GB | A1-A4 tables; the third run measures the quote component (G4) |
| S12 | Train A1, A2 | as S3 with `--version a11_pg1` / `a11_pg2` and `--features-db $OUT/features/props_features_a11_pg1.db` / `pg2.db` | 2 x 2 h | 8-12 GB | |
| S13 | Score R on pre-game features; 20,000 draws for the arms | `props_phase5_predict.py --version a11_r0 --features-db $OUT/features/props_features_a11_pg1.db --all-played --n-sims 4000 --season <each dev fold and 2024-25> --out-version a11_r0_on_pg1 --registry-root $REG`; the same with `--prices ... --out-version a11_r0_on_pg1_p5` for 2023-24 and 2024-25; and `--version a11_pg1` / `a11_pg2` with their own DB, `--out-version a11_pg1_p5` / `a11_pg2_p5` | 4 x 5 min + 3 x 1 h | estimate 6 GB | |
| S14 | 6a evaluation | `$PY scripts/a11_eval_models.py --experiment 6a --reference a11_r0_on_pg1 --candidates a11_pg1 a11_pg2 --leaky a11_r0 --folds 2021-22 2022-23 2023-24 --seen 2024-25 --registry-root $REG --prices-dir $OUT/prices --out $OUT/results/6a --report $RPT/6a_roster/report.md` (NEW) | estimate 30 min | < 8 GB | decision per B.9; **only if A1 and A2 both fail:** S12/S13 for `a11_noav` (`--features-db data/props_features_v3.db --exclude-groups teammates`), +3 h, then S14 again |
| S15 | 7b | `train_props_models.py --version a11_env --spec with_env --features-db <6a winner DB> ...` (as S3), then 20,000-draw quoted predictions as S4a (`--out-version a11_env_p5`), then `a11_eval_models.py --experiment 7b --reference <6a winner version>` | 2 h + 1 h + 20 min | 8-12 GB | |
| S16 | Final k and candidate rule file | `$PY scripts/a11_k_estimation.py --mode final --p5-version <final model>_p5 --train 2023-24 2024-25 --estimator <6b choice> --pmarket <7a choice> --write-rule $RPT/candidate_rule_a11.json`, then `a11_rule_eval.py` with the final model (7c and 7d re-reported, descriptive) | 30 min | < 6 GB | candidate rule + sha256 |
| S17 | **Owner approval required:** live candidate model through 2026-06 | `$PY scripts/train_props_live.py --version v4_live_2026-27c1 --next-season 2026-27 --features-db <winner DB> --families-from v2_base`, as a new immutable version | about 35 min (`live_pipeline_plan.md`), 1,906 s in the restore log | 8-12 GB | not part of this plan's approval |

**Total compute**, without A0, without repeating S3 on `ctl`, and without S17: S1-S8 about 5.5 h; S10-S11 about 2.5-3.5 h; S12-S14 about 7.5 h; S15 about 3.3 h; S16 0.5 h. That is **about 19-20 h, over three night windows**:

1. Night 1: S1-S8, then S10.
2. Night 2: S11-S14.
3. Night 3: S15-S16.

Worst case: +3 h for A0 and +2 h for S3 on `ctl`. Any step that runs longer than twice its estimate, or goes above 24 GB of memory, is stopped and reported.

Disk: about 8-10 GB under `$OUT` (four features DBs at about 0.7 GB, 6-7 model versions at 0.6-1 GB each, quote caches about 1 GB). Nothing under `data/` outside `$OUT` is written, except `config_log.jsonl` appends (intended) and the `data/raw/nba_schedule` cache that `props.schedule` may refresh. Not checked whether `load_schedule` writes when the cache is complete.

## F. Risks and stop rules

1. **A leakage test (L1-L5) fails:** stop section B. Fix, re-run the tests, record the fix. Nothing is trained on a DB that has failed them.
2. **Parity L6 below 99% on any av column:** stop section B and diagnose (rule mismatch, report placement, trade edge cases). A third roster rule is not invented to pass: the arms are fixed in B.3.
3. **`ctl` differs from v3 on more than 0.1% of base cells:** re-train R/L on `ctl` before any 6a comparison (+2 h). If `ctl` differs in labels or in non-WP2 columns, stop: the build is not reproducible.
4. **S5 reproduction fails** (a pooled through-origin slope outside ±0.05 of the frozen value for any of the four stats): stop and report to the owner before 6b. The comparison inside 6b would still be paired, but a drift that size means the regenerated pipeline is not the one behind the frozen rule.
5. **Injury reports for the early seasons:** coverage is complete from 2019-20 (section 0 item 3); only staleness differs (about 2 h at T-60 in 2019-21). Pre-declared: keep 2019-20 and 2020-21 in training as v3 does. Fallbacks, if L6 or the per-season share of differing member sets shows that those seasons behave differently (more than twice the 2021-22 to 2024-25 rate of rows whose member set differs from the box score): (a) set `av_*` to NaN for 2019-21 rows (LightGBM handles missing values), counted as one more comparison; (b) start training at 2021-22, which costs the 2021-22 dev fold. Recommendation if needed: (a).
6. **Linked-report cache missing or changed:** stop. Do not regenerate it with `--refresh-reports` (it rewrites caches shared with the live jobs); restore it from `D:\nba_backups` or ask the owner.
7. **Odds-store contention:** read-only connections with `timeout=120`, run in the off-window. If the collector reports a lock error during S1, stop S1.
8. **Data safety:** outputs only under `$OUT` and `$RPT`. Never delete a worktree's `data` recursively; unlink the junction first.
9. **No information beyond the market's bias** (C.5 item 6): if the slope-with-intercept interval on 2024-25 includes 0, stop the rule work and report. No candidate rule is written. The no-model price-dispersion result is a separate question for the owner, not something to rescue with this model.
10. **7a changes p_market:** every later step uses the chosen p_market. If 7a is decided after S6 by mistake, S6 is re-run. The order is fixed.
11. **Power:** one out-of-sample season for k (2024-25; about 15,700 bets under the revised rule, ROI interval ±1.6 points game-clustered). A candidate that passes here can still fail forward, and that is expected; the forward test is the confirmation.
12. **7b era confound:** environment features change source at 2023-24. If with_env passes overall but its 2023-24 point estimate does not, it is not adopted (the keep rule already requires this).
13. **Live code needed by an adopted change** (A2 membership option, 7b live environment): it is built and parity-tested before the forward segment starts. Until then the candidate is not complete and is not started.

## G. Owner decisions needed before step S1

1. **Approve dev-fold training for experiments** (`CLAUDE.md`: no retrain without the owner).
   - Options: (a) approve up to 7 model versions in `$OUT/models` (`a11_r0`, `a11_pg1`, `a11_pg2`, `a11_env`, conditional `a11_noav`, conditional re-train on `ctl`), plus prediction versions, with nothing deployed; (b) approve only 7a and the base regeneration (S1-S8) first.
   - Pros of (a): one pass, three nights. Cons: commits about 20 h of compute up front.
   - **Recommendation: (b), then (a)** after the S5 reproduction check passes. It is a cheap stop point.
2. **Confirmatory look.**
   - Options: (a) forward 2026-27 only; (b) score the frozen candidate once on 2025-26 as a confirmation.
   - (b) gives a large sample quickly, but it is contaminated by the audit (A.2).
   - **Recommendation: (a).** 2025-26 may be scored, labelled "seen", only after the forward horizon.
3. **6a arms.**
   - Options: (a) A1 only (matches today's live rule); (b) A1 + A2, with A0 conditional; (c) A0 only (drop teammate features).
   - (a) changes nothing live, but it trains the Out information out of the model (section 0 item 1). (b) costs about 3.5 h more, and A2 needs a live-builder option if it wins. (c) is cheapest, but the teammates group is the strongest single group in the ablations.
   - **Recommendation: (b).**
4. **Quote component of the teammate set.**
   - Options: (a) `v2nq` in all seasons, with the live builder switched to `v2nq` for the teammate set (quoted players still predicted); (b) v2 with quotes from 2023-24 and without before.
   - (a) gives one definition for every era, at the cost of a small live change. (b) keeps live unchanged, but the feature definition then shifts by era.
   - **Recommendation: (a)**, unless the S11 quotes run shows the component changes fewer than 1% of av values. In that case the difference is moot and live can stay on v2.
5. **Seasons for the final k of the candidate.**
   - Options: (a) 2023-24 + 2024-25, as the frozen k was; (b) also 2025-26.
   - (b) has about 50% more rows and does not contaminate the forward test, but it touches 2025-26, against A.1.
   - **Recommendation: (a).**
6. **Where models are written.**
   - Options: (a) new `--registry-root` flags, with everything under `$OUT/models`; (b) new version names in `data/props_models`.
   - (a) keeps the production registry clean but adds three small flags; (b) needs no code but mixes research versions into the production registry.
   - **Recommendation: (a).**
7. **Bundling and forward start.**
   - Options: (a) one bundled candidate, shadowed against the frozen rule as a new segment; (b) a forward segment per change.
   - (a) gives one comparison and one clock; (b) shows which change matters, but it multiplies the comparisons and splits a small forward sample.
   - **Recommendation: (a).** Per-change attribution comes from the dev results.
8. **Primary cluster.**
   - Options: (a) game date; (b) game.
   - (a) is conservative and matches the forward pre-registration; (b) matches the Phase 5 reports.
   - **Recommendation: (a)**, with (b) printed next to it.
9. **Kelly in 7c.**
   - Options: (a) include, as fixed in 7c; (b) defer until the exchange fill study says what size can be filled at the best price.
   - Props limits are small and the owner plans to rest limit orders, so stake size may be set by fills rather than by Kelly.
   - **Recommendation: (a)** as an evaluation only, with no adoption before the fill study exists.
10. **Compute window.** Confirm that heavy steps may run about 23:00-11:00 PT on three nights, with a check before each step that the live bot is not running. **Recommendation: yes.**

## Appendix: new code at a glance (all on a branch; deployed defaults unchanged; each with tests)

| File | Change |
|---|---|
| `props/features/pregame_roster.py` | new (B.5.1) |
| `props/features/build.py`, `scripts/build_props_features.py` | `--membership` (default `box`), an out-path guard |
| `props/live/roster.py` | rule options `v2nq`, `pregame_all` |
| `scripts/props_d1_parity.py` | `--features-db`, `--out` |
| `props/analysis.py` | `consensus(agg=..., books=...)` |
| `props/models/dispersion.py` | `nb_match`, `DispersionScaler(method=...)` |
| `scripts/train_props_models.py`, `scripts/props_phase5_predict.py`, `scripts/props_phase5_dev.py` | `--registry-root`; predict also `--save-raw`, `--all-played` |
| `scripts/a11_cache_quotes.py`, `a11_devig_calibration.py`, `a11_k_estimation.py`, `a11_rule_eval.py`, `a11_nb_dispersion.py`, `a11_eval_models.py`, `a11_compare_dbs.py` | new; every one refuses 2025-26 (A.1) |
| `tests/props/test_a11_*.py` | L1-L5, the default-unchanged tests, the estimator tests (E1-E3 recover a simulated truth with a ≠ 0 and E[g] < 0) |

After any scripted edit of a Python file: `ast.parse`. Staging: explicit file paths only.
