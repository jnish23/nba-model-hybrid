# Pre-registration: forward (paper) test of the props candidate, segment 2

**IN FORCE: signed by the owner on 2026-11-13, effective date E2 = 2026-11-14 (section 8).** The text below is as signed; the last edits before the signature (2026-11-13) set the 137-date schedule count and the implementation notes of section 7.

Segment 1 (`reports/props/forward_test_preregistration.md`, signed 2026-11-12, effective 2026-11-13) is closed by this change under its section 11 (items 1, 2 and 4: new model, new rule file, new roster rule and price definition). Its bets never count towards segment 2. Every section of the segment 1 text applies here unchanged unless this document replaces it. Replaced: sections 1, 4, 6, 7, 11 and 14.

## 1. What is being tested (replaces section 1)

One bundled candidate, decided by the experiments of `reports/props/audit_2026-11/` (plan `WP6_WP7_experiment_plan.md`, keep rules written before the runs):

1. **Model** `v4_live_2026-27c1` (bundle sha256 `4cc85ee601c261ee0c82c71f649638b8d93243816c8f403653041563e750fdf0`), base feature spec, trained through 2026-06-13 on `data/props_features_v4_pregame_all.db` (sha256 `ee64d102c6ba1abcdc29251666826eec6784df0bc4e828721d020a4576079bd6`).
2. **Roster rule** `pregame_all` (experiment 6a): teammate-availability features over the team's previous-game roster plus every player on the usable pre-game injury report; the same code in training and live. A quoted player who is not a member still gets a row; quotes never change the member set.
3. **Rule file** `reports/props/audit_2026-11/candidate_rule_a11.json` (sha256 `fd026afd076e728cf38ea8a1aaf2321cce21ad1837256dd55b6a670878dd941d`), which has no cap:
   - stats rebounds, assists, threes, ra;
   - market probability p_market: Shin de-vig, mean over all books (experiment 7a); a bettable quote not on that consensus line is dropped;
   - p* = clip(p_market + a + k (p_model - p_market), 0.01, 0.99), with a = a_pos when p_model >= p_market and a_neg otherwise (form C3, experiment 6b); k: rebounds 0.1304, assists 0.2434, threes 0.2911, ra 0.1688; intercepts as in the file;
   - bet when EV at the best bettable price is at least 3%, earliest slot, at least 3 books at the consensus line, one bet per player-game-stat, flat 1-unit stakes.
4. **Recorded bets are uncapped; the capped rule is a marked subset.** The cap of one bet per player-game (experiment 7c) is not applied by the live job. The monitor derives the capped subset from the recorded bets with `props.betting.rules.cap_per_player_game_causal(bets, 1)`: slots in time order, a player-game with a bet takes no later bet, within one slot the highest best-price EV wins (ties: stat name). The subset is therefore fixed by the record and cannot be chosen afterwards.
5. Slots, closing reference and the question are those of segment 1.

Not part of the candidate: game-environment features (experiment 7b, not adopted), the expected-minutes filter (7d, dropped), the dispersion change (7e, not adopted), Kelly sizing (evaluation only).

## 4. Evidence so far (replaces section 4; descriptive)

- **No out-of-sample test of the whole bundle exists.** The final k and intercepts are fitted on 2023-24 and 2024-25, so both seasons are in sample for the rule's parameters; 2025-26 was excluded from every experiment and has not been scored.
- Model and roster rule alone, frozen k, 2024-25 (k out of sample): median-price ROI -0.3% (95% interval -2.0% to +1.4%), best price +5.2%. Against the deployed model's live behaviour: +0.3 points (-0.7 to +1.5).
- Whole bundle, in sample, median-price ROI: uncapped +2.3% on 2023-24 and -0.3% on 2024-25 (21,564 and 19,715 bets); capped subset +3.1% and +0.7% (12,042 and 11,881 bets). Best price: uncapped +6.2% and +4.7%; capped +7.2% and +6.0%.
- Forecast error (CRPS) of the model against the deployed model's live behaviour, dev folds: -0.2% to -1.1% by stat, intervals excluding zero.
- Segment 1 record at the switch: reported by the monitor as it stands.

## 6. Precision and horizon (replaces section 6)

Precision depends on the number of game dates, not on a stake count: the standard error is date-clustered, and the schedule has 137 game dates from 2026-11-13 through 2027-04-12. Measured on the candidate's own backtest bets (date-clustered, scaled to 137 dates; chances from a simulation of the three-look rule of section 7):

| Record | Stakes per date (backtest) | 95% half-width, median-price ROI | Chance of PASS if the true ROI is +3% |
|---|---|---|---|
| Uncapped (primary) | 93 to 104 | about +/-1.9 points | about 88% |
| Capped subset (secondary) | 56 to 58 | about +/-2.2 points | about 74% |

Live volume has been lower than the backtest (72 uncapped stakes per date in the segment 1 record). If that persists the uncapped half-width is about +/-2.2 points and the chance of PASS at +3% about 76% (95% at +4%); the capped subset about +/-2.7 points and 58%.

**Horizon (fixed now, no extension): every bet on a game date from E2 through 2027-04-12, the end of the regular season.** There is no stake target. **OWNER:** accept the date horizon.

## 7. Decision rule (replaces section 7)

Primary metric: median-price ROI of the **uncapped** recorded bets, date-clustered bootstrap as segment 1 section 3 (4,000 resamples, seed 20261112). Z = ROI / its clustered standard error.

1. **Three looks, with early success allowed** (group-sequential, O'Brien-Fleming boundaries for three equally spaced looks, one-sided false-PASS rate 2.5% over the whole test):

   | Look | Data used | PASS if Z is above | Observed ROI this needs at live volume (about) |
   |---|---|---|---|
   | 1 | the first 50 game dates from E2 (about early January) | 3.471 | +6.4% |
   | 2 | the first 100 game dates (about late February) | 2.454 | +3.2% |
   | Final | every game date through 2027-04-12 | 2.004 | +2.2% |

   A PASS at look 1 or 2 ends the confirmatory test; the record keeps running and is reported, but the outcome is not revisited. Chance of a PASS by look 2 at live volume: about 44% if the true ROI is +3%, about 73% if it is +4%.
2. **Futility, at looks 1 and 2 only:** STOP-FUTILITY if the conditional power of a PASS at the final look, with a true ROI of +3.0% on the remaining dates, is below 10% (same formula as segment 1 section 7, with the final boundary 2.004).
3. At least 30 game dates for any PASS. Otherwise CONTINUE; at the final look the result is PASS or FAIL. FAIL means "not shown", as in segment 1 section 8.
4. The looks are counted in game dates with at least one settled stake (voids and pushes do not make a date count). The boundaries assume equal information per look; dates differ in size, which the report states (stakes per look) and does not adjust for.
5. **As implemented** (`props.live.monitor.group_sequential_dates`, `scripts/props_forward_monitor.py --segment 2`):
   - Z uses the analytic date-clustered standard error of the ROI (the same one segment 1 uses for conditional power); the bootstrap gives the intervals of the secondary tables only, so the decision is deterministic.
   - A look waits while any recorded bet on or before its cut-off date is unsettled; a bet unsettled for more than 3 days (a postponed game) stops holding the look and is listed. This can delay a look, never create a PASS.
   - Nothing is stored: every run recomputes the looks in order from the data through each cut-off, and the first PASS or STOP-FUTILITY is the outcome on every later run.
   - The information fraction for futility is game dates so far divided by the schedule's count of game dates from E2 through 2027-04-12 (137 from 2026-11-13).
   - False-PASS rate by simulation: 2.4% to 2.5% with a known standard error; about 2.7% to 3.0% when the standard error is estimated from 50 to 137 dates, as it will be. The owner accepts that as the test's level.

**Secondary, pre-declared, never decisive:** the capped subset (section 1 item 4) on the same looks, its paired difference from the uncapped record, the best-price ROI of both, and closing-line movement. These are the inputs to any separately pre-registered limit-order pilot; this document sets no gate for one.

**OWNER:** uncapped as primary; looks at 50 and 100 game dates; the boundaries above; futility floor 10% at +3.0%.

## 11. Configuration that defines the segment (replaces section 11)

(model_version, model sha256, rule sha256, roster rule) = (`v4_live_2026-27c1`, `4cc85ee6...`, `fd026afd...`, `pregame_all`). The void list of segment 1 section 11 applies, with the price definition now including the p_market method, and with one change:

- **A refit by the same recipe does not end the segment.** Recipe: `scripts/train_props_live.py` with the base spec, families from `v2_base`, a features database built by the same code with the `pregame_all` membership, and more recent games as the only difference; the rule file unchanged. The new version and its sha256 are listed in the monitor report, and results are shown per model version. No refit is scheduled this season. Anything else that changes the model, the features' definitions, the rule file, the roster rule, the slots or the price definitions ends the segment.
- **The test does not continue into 2027-28.** Building a second season into this design would raise this season's final boundary from 2.004 to about 2.9 (an observed ROI of about +3.1% instead of +2.2%). A 2027-28 test is segment 3, with its own document.

## Switch procedure (before the effective date)

1. PR 49 merged to main, then the branch `props-audit-a11-experiments` (live frame path, rule support, tooling, this document).
2. `config/props_live.json` replaced by the content of `config/props_live_candidate.json` with the effective date below; the dashboard methodology line updated in the same commit. The monitor implements this document's rule as `--segment 2` (built and tested 2026-11-13; segment 1 stays the default and its output is unchanged); from E2 it is run with `--segment 2 --since E2`.
3. The switch happens between game days (after the last settlement of one date, before the first T-240 slot of the next), so no bet is placed under one rule and closed against the other's signals.
4. First run checked by the owner or the assistant: slot log complete, `players_without_features` only with reasons, bets carry the new four identifiers.

## 8. Signature (replaces section 14)

Effective date E2: **2026-11-14** (the switch commit sets `effective_from` in `config/props_live.json`; it applies if the main checkout runs that commit before the first slot of 2026-11-14, otherwise E2 is the next game date and this line and the segments file are corrected). As proposed: the first game date on which all three slots run under the configuration of section 11 (proposed 2026-11-14 if the switch is done before that day's first slot; otherwise the next game date).

Signed: Jordan Nishimura   Date: 2026-11-13
