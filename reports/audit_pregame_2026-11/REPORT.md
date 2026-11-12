# Pregame team-model audit, 2026-11-11: report

Scope: the moneyline, spread and total LightGBM models run by `scripts/run_inference_v2.py`, their back-tests, this season's bets and the dashboard.
Work was done by nine Sonnet workstreams (A1-A7, B1-B4) in the worktree `C:\Users\jorda\nba-pregame-audit` (branch `pregame-audit-20261111`); the lead
checked each result, re-ran the key numbers and decided what was adopted. Each workstream's own write-up is in its folder beside this file.

## Bottom line

1. **The back-tests are valid.** No leakage, no fold or settle error was found, and the published accuracy numbers reproduce.
2. **The model is as accurate as the opening line it reads, and no better.** It does not beat the closing line, and nothing tested makes it do so.
3. **The live bet rule was never what the back-tests measured.** The model reads the opening line; live bets are judged against the line at run time; the back-tests judged against the opening line. On 8,400 back-test games, pricing the rule at a later line gives a 50.0% hit rate and loses exactly the vig.
4. **The model's stated edges are several times too large.** It claims a 60-63% win probability on spread and total bets that win 50-51% of the time.
5. **This season (66 bets, -9.9 units) is too small to read.** Closing-line value is the only measure that can answer within a season.
6. **Several production defects were found and fixed on the audit branch** (below). Two changes were made in production with your approval: the task time and the bet-log backfill.
7. **Where an edge could realistically come from** is price, not forecasting: a zero-fee venue, off-market prices against a sportsbook consensus, or making markets. The test plan for Polymarket US and Novig is in `VENUE_TEST_PLAN.md`.

## 1. Are the back-tests valid? (A2)

| Check | Result |
|---|---|
| Feature leakage | Pass. 125 team-form features recomputed from raw rows for 784 team-games match to 1e-13; roster BPM matches on all 16,876 games; scrambling all later box scores changes nothing earlier. |
| Same-day ordering, playoffs, play-in | Pass. |
| Injury-report timing | Pass. The report used is never after noon ET or after tip minus 30 minutes; miss rates by status use earlier seasons only. |
| Odds identity and signs | Pass. Opening lines appear a median 25 hours before tip. |
| Fold integrity, calibration, sigma | Pass. |
| Metric reproduction and settle logic | Pass. Bet tables equal the project's own to the bet. |
| Placebo | Pass. Shuffled predictions and random sides return about -4.5%. |

Caveats on which numbers to quote:
- `E25_with_odds` is the pre-adoption form (raw-margin spread model, isotonic calibrator). Use `E13_combined_winners` for the deployed form.
- The model reads one book's opener; evaluations use a cross-book median. The moneyline edge slope is 0.39 against the single book and 0.19 against the consensus.
- Open-priced ROI includes movement no one could capture: the same totals bets are +1.8% settled at the open and -2.0% settled at the close.
- The training snapshot called "noon" was a different report in different eras (8:30 pm the night before in 2019-21, 11:30 am in 2021-25, noon from 2025-26).
- **The pre-2019 archive odds file is missing since the data wipe.** The rebuilt training set has no odds before 2019-20, so a retrain today would not reproduce the deployed model's inputs.

## 2. The live rule against its back-test (E32, A5; lead's recount matches to the bet)

Later line = the close; 8,416 spread games, 8,419 total games, 8,366 moneyline games; day-bootstrap 95% intervals.

| Market | Rule | Bets | Hit rate | ROI at -110 | ROI at zero vig |
|---|---|---|---|---|---|
| Spread | live analogue | 2,366 | 50.1% | -4.4% (-8.4 to -0.6) | +0.1% |
| Spread | own view | 644 | 50.2% | -4.1% (-12.0 to +3.4) | +0.5% |
| Total | live analogue | 3,546 | 50.0% | -4.6% (-7.6 to -1.3) | -0.1% |
| Total | own view | 1,942 | 51.3% | -2.0% (-6.2 to +2.5) | +2.6% |
| Moneyline | live analogue | 4,006 | wins at fair price | -5.2% at real closing prices | -1.2% |

- 94% of the spread bets and 86% of the total bets go against the line's move since the open.
- "Own view" rules (bet only when the model itself disagrees with the opener) cut volume and stop the fading, but none beat the live rule with an interval excluding zero, and none has a hit-rate interval above 50%.
- The model's own disagreement with the opener carries about a tenth of its face value (slope +0.09 spread, +0.13 total, about 0 moneyline).
- This season, against the real run-time line, the rule fires on 11% of games for spreads and 20% for totals, so the fading is milder than the close-based analogue, but it is the same mechanism: 95% of this season's spread and moneyline bets fade the move.

## 3. This season's bets (A1)

| Market | Bets | W-L | Units | ROI (95% interval) | Mean closing-line value |
|---|---|---|---|---|---|
| Moneyline | 18 | 7-11 | -3.87 | -21.5% (-69 to +19) | +1.4 probability points before vig |
| Spread | 19 | 11-8 | +2.02 | +10.6% (-28 to +52) | 0.00 points |
| Total | 29 | 11-18 | -8.04 | -27.7% (-59 to +1) | +0.60 points (+0.22 to +1.25) |

- **When is the model more trustworthy?** Nothing can be said yet. Seven splits were fixed before looking at outcomes; none survives a multiple-comparison correction, and hit rate does not rise with edge size.
- Totals are the one flag: stated edges implied 17.7 wins and 11 happened (p = 0.009), the same overconfidence the back-tests show.
- A 2-point ROI edge needs about 18,600 bets to detect; a 0.3-point closing-line edge needs about 110-136.
- 42 of the 66 bets (Oct 20-31) are reconstructed, because the daily log before Nov 1 was lost and the prediction log did not store sides, prices or sigmas.

## 4. Methodology questions

- **Targets.** Spread as "margin minus the line" is right. Total as a level with the line as a tree feature is wrong: trees cannot follow the line at high totals (lines of 235+ are 14% of games and 66% of the excess error). Outcomes are also a very noisy target; "later line minus entry line" is the bet-relevant one and needs forward data.
- **Include the line?** Yes, as the starting point the model adjusts, and it should be the line at decision time and a consensus, not one book's opener. A model without the line has margin MAE 10.99 against the line's 10.64.
- **EWM windows 5 and 20.** Never tested before. Without the market, span 40 is better for margin by 0.057 MAE in 7 of 7 seasons and 5+20 is about optimal for totals. Once the opening line is known no window adds anything (10.64 in every case), so it does not matter for the deployed model.
- **BPM and injuries.** Calculated correctly, not used well: roster BPM describes last game's lineup, the deployed spread model uses no injury magnitude at all, and the in-house BPM predicts margin worse than a plain point-margin average. The injury value lost is worth about half a point per standard deviation against the opener and nothing reliable against the close.
- **Capping outliers.** No effect on spread (misses are bell-shaped). A small totals gain (0.05) that works the same way as the totals shrink, which gains 0.15. Dropping games is worse.
- **Totals shrink (E27, A6).** Passes its keep rule: total MAE 14.76 to 14.61 in 7 of 7 seasons, level with the line (14.62). Built as an opt-in, off by default. Not ready to switch on: on the trainer's one-year block the fitted weight barely shrinks high totals.
- **Is the model as good as it can be?** With this information set, essentially yes. Every feature experiment (E5-E8, E28-E31) and the checks above land inside noise.

## 5. When to run

For 7-9 pm ET tips: at noon ET 70% of eventual absences are official and 41% of the spread's remaining move (28% of the total's) is still to come; at 5 pm 86% are official and 7% of the move remains. The line moves ahead of the official report, so no hour gives the report an advantage over the line. Recommendation: 12:15 ET, which matches the training snapshot. Running later with the current model changes only the line it is compared with. The venue test records two later shadow times to settle this with closing-line value.

## 6. What outside evidence says (A4)

- NBA closing lines for sides and totals are close to efficient; published anomalies come from 1995-2014 samples and did not replicate. No peer-reviewed NBA closing-line study for 2015-25 was found.
- Long-term winners mostly win on price: soft openers at low limits, news speed, stale prices against a sharp book, promotions, liquidity provision. Originating a better number or winning on news speed is not realistic for a noon batch run on public data.
- No public evidence that any named data source (EPM, DARKO, tracking, referees, rest) adds value beyond the close.
- Fees: Polymarket US 0.095 x p x (1-p) at the lowest tier (about the same as -110 at a 50c price); Novig none on pregame straight trades. Novig's own pages disagree on excluded states.

## 7. What changed

**In production, with your approval**
- The "NBA Daily Inference" task now fires at 9:15 AM Pacific local time (12:15 ET all year). It had drifted to 11:15 ET on Nov 1 because its trigger carried a fixed UTC offset.
- `bet_recommendations_log` and `bet_recommendations_provenance` were created in the production database and filled for Oct 20 to Nov 11 (170 games). Insert-only; `predictions_log_v2` is unchanged (1,491 rows); integrity check passes.

**On the audit branch (not merged, not committed)**

| Area | Change |
|---|---|
| Bet log | Every run appends the opening line read, the current line, price and book, sigmas, bet side and edge, own view and line move, and the injury report used. |
| Run-time guard | The script waits for the noon ET report (up to 90 minutes) and exits 3 without predicting if started too early; it logs an error if the report used is more than 30 minutes behind. |
| Reliability | An odds or schedule outage exits 4 instead of looking like an off day; exact team-name lookup; unpredicted games recorded in `inference_skipped_games`; all date decisions in Eastern time; a missing or stale injury report withholds the bets and exits 5. |
| Dashboard | The record is the logged recommendation (first run with a line), graded at its own line and price, pushes as pushes; closing-line value beside it; the "model lean on every game" displays are removed. |
| Matchup features | The two rating pairs that were never built now use the column that exists. The deployed model uses no matchup feature, so predictions cannot change. |
| Stale CSV | Each score update reconciles every settled game; this repairs the 25 rows stuck as pending. |
| Totals calibrator | E18 fix (c) as an opt-in, off by default; the deployed set's hash and predictions are unchanged. |
| Experiments | E32 and E27 with registrations, results and decisions under `reports/experiments/`. |
| Docs | `CLAUDE.md` and `team/CLAUDE.md` updated. |

Tests on the branch, final state: 586 passed, 0 failed (all of `tests/team` and `tests/core`, database tests included; run by the lead after the last change).

## 8. Decisions that are yours

1. **Merge the audit branch.** At the next run after merge: predictions and bets are identical on a normal day; the bet log starts filling from real runs; the 25 stale CSV rows are repaired; the wrapper reports exit codes.
2. **Restore the pre-2019 archive odds file** (look in `D:\nba_data_rescue\`) before any retrain, and restore a training set into the production `data\processed` (the explanation step and the weekly retrain need it).
3. **The bet rule.** The evidence says the live rule mostly fades line movement and that no tested alternative is better. Options: keep it and treat the dashboard as a measurement; switch to the own-view rule (fewer bets, no fading, same expected result); or stop recommending spread and moneyline bets until a forward test shows closing-line value. Any change needs a registered experiment under the repo's own rule.
4. **Totals shrink.** Decide how to pin the weight (pooled value, longer block, or a constant) before a candidate is trained with it on.
5. **The Odds API fallback** feeds the model the current line as if it were the opener (26 games on Oct 23-24). Decide whether such games should be predicted but not bet.
6. **Props settle task** still has a fixed-offset trigger at 9:30 AM, 15 minutes after inference now starts; its own wrapper says 10:30.
7. **The venue test:** effective date, the Polymarket US decision-time snapshot, and whether to obtain a Novig read key.
8. **Matchup feature formula.** As defined (offence minus defence) it cancels what it should capture; changing it is a model change that needs an experiment.

## 9. Not verified

- The bet log, the guard and the reliability paths were tested with stubbed sources and a fake clock, not on a real scheduled run. Read `bet_recommendations_log` and the job log after the first run following a merge.
- `run_daily_inference.bat` was read, not executed.
- The hosted dashboard page, the real publish and the page layout were not seen; the real `app.py` ran under streamlit's test harness without errors.
- No retrain was done. The pre-adoption back-test cannot be reproduced from current data (lost training set and archive odds).
- Many of A4's secondary claims rest on search summaries; the fee figures were read from the venues' own pages on 2026-11-11.
- The lead's own checks in `lead_checks/` (timing, EWM windows, BPM and injuries, capping) are exploratory and decide nothing.

## 10. Where things are

- This report and `VENUE_TEST_PLAN.md`: `reports/audit_pregame_2026-11/`
- Shared brief and fixed keep rules: `BRIEF.md`
- Workstreams: `A1_season_bets`, `A2_backtest_validity`, `A3_code_fixes`, `A4_research`, `A5_E32_live_rule`, `A6_totals_calibration`, `A7_venue_recon`, `B1_dashboard_record`, `B2_cutoff_guard`, `B3_reliability`, `B4_drop_lean`
- Lead's checks: `lead_checks/`
- Experiments: `reports/experiments/E32_live_rule`, `reports/experiments/E27_candidate_form`
