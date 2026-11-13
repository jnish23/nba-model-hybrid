# A2 (`pregame_all`) live candidate: build plan

Written 2026-11-13, before any of the work below was run. Owner instruction: "Let's get A2 live" (2026-11-13). Source result: `6a_roster/report.md` (keep rule B.9 picks A2).

## What "live" needs

| Step | What | Depends on | Who |
|---|---|---|---|
| P1 | Production live frame path for the pre-game rules (`v2nq`, `pregame_all`), with a row for every quoted player | nothing | subagent A, audited |
| P2 | Candidate tooling: live training script and config pin accept a spec, a features DB and a roster rule; candidate config file separate from `config/props_live.json` | nothing | subagent B, audited |
| P3 | Parity run of the production path against `props_features_a11_pg2.db` (30 dev dates of 2024-25) | P1, free run worktree | main session |
| P4 | 7b result decides the spec (`base` or `with_env`) | 7b run (started 08:40 PT) | main session |
| P5 | S16: final k on the final model, candidate rule file | P4 | main session |
| P6 | S17: live candidate model `v4_live_2026-27c1`, trained through 2025-26 on a candidate features DB with the `pregame_all` membership | P2, P4, bot off | main session |
| P7 | Dry run of the signal job with the candidate config on a past date, in a worktree, no publish | P1, P2, P6 | main session |
| P8 | Switch: merge to main, pin the candidate config, signed amendment to the forward-test pre-registration (new segment) | owner | owner |

Nothing in P1 to P7 changes `config/props_live.json`, `config/props_frozen_rule.json`, the deployed model, or the main checkout. The current rule `v2` must behave exactly as before.

## P1 design (quoted players who are not members)

Measured on the 30 parity dates: 59 quoted player-games (1.6% of quoted) have no live row under `pregame_all`. 54 are regulars who missed the team's previous game and are not on today's report (median 29.6 minutes played that night); 2 changed teams; 3 have no box-score identity at all.

Design: **a quoted player always gets a target row; he never becomes a member because he is quoted.**
- Members (the set the teammate aggregates run over) stay exactly the rule's set: previous-game roster plus report players. Quotes never change it.
- A quoted non-member gets a base row (live history path) and an at-T row whose `av_*` / `opp_av_*` are aggregated over the members of his team with no self-subtraction. This is what the training builder does for a box-score target who is not a member, so no retraining is needed.
- His team: the team the usable report names, else his latest listed team this season when that team plays in the game, else he is skipped and logged with a reason.

Rejected: (a) adding quoted players to the member set (rule `v2`): training cannot reproduce it before 2023-24, which is the train/serve skew 6a removed; (b) skipping them: loses a non-random 1.6% of markets (returning regulars).

## Keep rules (fixed now)

- **P1 unit tests:** (i) under `v2` the frames of a fixture date are identical before and after the change; (ii) adding or removing a quote for a non-member changes no other row's features (exact); (iii) a report stamped after T - 15 min changes nothing; (iv) a quoted non-member's row equals the training builder's row for the same player (teammate columns exact, on a synthetic fixture).
- **P3 parity:** A3 = 100%; every `av_*` / `opp_av_*` column within the A4 tolerances on at least 99% of rows, **computed including the newly covered quoted non-members**; A2 identical for base columns; A1 coverage at least 99.5% (the 3 identity-less player-games are the allowed shortfall). A failure stops the work; no new rule is invented to pass.
- **P7 dry run:** every slot of the date completes; `players_without_features` holds only skipped-with-reason players; no write outside the dry-run store.

## Not decided here (owner)

1. Whether the candidate replaces the current model (new forward segment from the switch date, signed amendment) or runs beside it. **Decided by the owner 2026-11-13: replace.** A parallel record needs the bet key to include the model version and doubles the work in the T-5 slot.
2. Whether 2025-26 joins the final k fit (plan section C.3).
3. Merge order: PR 49 first, then this branch.
