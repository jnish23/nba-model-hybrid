# Pregame team-model audit (2026-11-11): shared brief for every workstream

The lead auditor wrote this. Read it fully before starting. Your own task comes in your prompt; this file holds the rules, the context and the
findings so far. You report to the lead, who checks your work and decides what is adopted.

## What is being audited

The pregame LightGBM models for NBA moneyline, spread and total. Daily entry point `scripts/run_inference_v2.py`; code in `team/` and `core/`;
dashboard `dashboard/app.py`. Deployed model set: `team/model_registry.DEPLOYED_SUFFIX = '_with_odds_cand_20261019'`.
Read `CLAUDE.md` and `team/CLAUDE.md` first. Earlier audits and experiments: `reports/audit_fixes/TEAM_AUDIT_2.md`,
`reports/experiments/FINAL_REPORT.md`, `reports/experiments/PREREGISTRATION.md`, `reports/upgrade_2026-27/PLAN.md`. Do not read `reports/` wholesale.

How the live run works (verified by the lead):
- It runs about 11:19 ET each game day. Model inputs are the OPENING lines from SBR (`odds_*`), team form (EWMA span 5 and 20 of box-score
  stats, running across seasons), travel, roster BPM, and the injury report as of min(now, noon ET).
- The spread model predicts `-opening_spread + f(X)`. The ML probability is a logistic stack of the raw classifier and the spread margin.
  The total model predicts the level of the total with the opening total as one feature.
- Bets are then judged against the CURRENT line at run time (`curr_odds_*`): one rule, probability edge above 5% over the price's break-even.
- `predictions_log_v2` stores the model's numbers and the CURRENT line only. It does not store the opening line the model read, the bet side,
  the edge, the price or the sigma. `todays_picks.csv` is overwritten daily. `data/raw/daily_inference.log` prints the full picks table every day.

## Hard rules

1. Work only inside `C:/Users/jorda/nba-pregame-audit` (branch `pregame-audit-20261111`). `C:/Users/jorda/nba-model-hybrid` is the production
   checkout: scheduled jobs run from it. You may READ files there. Never write, never run scripts from it, never switch its branch.
2. Python is `C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe`. Run it with the worktree as the current directory, so `data/` is the
   worktree's own snapshot. `statsmodels`, `streamlit` and `seaborn` are not installed. Install nothing.
3. `data/` in this worktree is a real folder holding a snapshot (not a link): `data/nba_stats.db` (copied 2026-11-11 14:28 PT),
   `data/processed/training_set_v1_reduced_2026-11-01.csv`, `data/processed/injury_reports_parsed.csv`, `data/raw/daily_inference.log`.
   Never delete `data/`. The deployed set was trained on a 2026-10-19 training set that was lost in a data wipe; the 2026-11-01 file is a later
   rebuild (16,876 rows, through 2026-10-30) and has a different hash.
4. No paid or outward calls. Never call The Odds API. Never run `scripts/run_inference_v2.py`, anything with `--publish`, or any scheduled `.bat`.
   Never print or read `.env`. Do not download files.
5. Never write into `models/saved/`. Never retrain into an existing suffix. Never change `DEPLOYED_SUFFIX`. Anything that trains writes to a
   scratch folder under your own report folder.
6. Do not commit, stage, stash or push. Leave your changes in the working tree. The lead reviews and commits.
7. Touch only the files your prompt assigns to you. Other workstreams edit this worktree at the same time.
8. For tests run `-m "not db"` unless you need the database, and only `tests/team tests/core` (props tests write into `data/`).
   After editing a Python file through a script, run `ast.parse` on it. Use Windows-style paths (`C:/Users/...`) in Bash.
9. Experiments: the keep rules below were fixed by the lead before any run. Do not change them. Report the result whatever it is.
10. Be exact about what you verified. For every number give the command or script that produced it. Say plainly what you did not check and
    what failed. Use day-bootstrap intervals (resample game dates, 1,000 draws, seed 42) for every ROI, hit rate or loss difference.
    A result without an interval is not a result.

## Deliverable

Write `reports/audit_pregame_2026-11/<your workstream folder>/RESULT.md` plus the scripts that produced it (in the same folder, or where your
prompt says). `RESULT.md` starts with a "Bottom line" of at most 8 lines, then findings ordered by importance, each with evidence, then
"Not verified / limits". Your final message to the lead is a summary of at most 40 lines: findings with numbers, files written, open questions.

## Lead's preliminary findings (verify them where your task overlaps; do not assume they are right)

P1. **Live bets are judged on a different basis from the backtests.** The model reads the opening line; live edges are measured against the
    11:19 ET current line; backtests measure against the opening line. On this season's 141 games with open, 11 am and closing lines,
    corr(edge versus the 11 am line, open minus 11 am line) is 0.79 for spreads and 0.58 for totals, and about 85-90% of larger-edge games are
    on the side that fades the line's move since the open. Scratch script: the lead's `season.py` (not in the repo; rebuild independently).
P2. **The closest historical analogue of the live rule loses the vig.** On `reports/experiments/E25_with_odds/predictions.csv` (7 walk-forward
    seasons, about 8,400 games with closing lines), pricing the 5% rule at the CLOSE instead of the open: spread bets 629 -> 2,338, hit rate
    0.500, ROI about -4.5%; total bets 1,924 -> 3,526, hit rate 0.500, ROI about -4.6%; 94% / 86% of those bets fade the open-to-close move.
    Regression of (result - close) on (model - open) gives +0.09 (se 0.10) for spread and +0.13 (se 0.06) for total. Simple standard errors only.
P3. `GameFeatureBuilder.MATCHUP_PAIRS` names `def_defensiveRating_5g_away` / `def_defensiveRating_20g_away`. Those columns never exist
    (`defensiveRating` is built under the `off_` prefix), so two of the three matchup features have never been built, in training or live.
    The 2026-11-01 training set has only `matchup_3pa_vs_opp_3pa_5g`.
P4. Season to date (160 completed games, lead's rough reconstruction at -110 with one global sigma): ML 18 bets 7-11, spread 17 bets 10-7,
    total 32 bets 13-19. Mean closing-line value: spread +0.29 pts, total +0.57 pts, ML +1.4 probability points. Model versus lines on 141 games:
    ML log loss model 0.5442 / open 0.5510 / 11 am 0.5464 / close 0.5451; total MAE model 14.57 / open 14.56 / 11 am 14.27 / close 14.12.
    Nothing here is statistically distinguishable from zero.
P5. `predictions/predictions_2026-27.csv` in the production checkout shows 25 rows (2026-10-26 to 10-28) still PENDING while the database
    table `predictions_log_v2` has them COMPLETED. The "close" in `odds_snapshots` is the last pre-game snapshot, about 30 minutes before tip.
P6. Since the data wipe there is no training set in the production `data/processed`, so the daily explanation step logs
    "No league-wide reference" every day, and the weekly retrain could not run.

## Pre-registered rules (fixed 2026-11-11 by the lead, before any run)

### E32: the live rule's historical analogue and the "own view" rule (measurement on saved predictions; no retraining)

Data: `reports/experiments/E25_with_odds/predictions.csv`, all 7 folds, games with both an opening and a closing line.
Definitions: `own = model - open`; `move = close - open`; the "later line" is the close. Sigma is the fold's saved sigma.
Spread and total bets are priced at -110 and settled at the later line; pushes refund. Each arm is also scored at zero vig (even money).
Moneyline uses the de-vigged closing probability as the later price: zero-vig only, plus -110-equivalent break-even 52.38% for reference.
- R0 (live analogue): bet when the model's probability edge against the LATER line exceeds 5%.
- R1 (own view, not crossed): bet when the probability edge against the OPEN exceeds 5% and the later line has not crossed the model's number
  (sign(model - close) equals sign(own)). Priced at the later line.
- R2 (own view transferred): fair value = later line + own. Bet when its probability edge against the later line exceeds 5%. Priced at the later line.
Readouts per arm and market: bets, W-L-P, hit rate, ROI at -110 and at zero vig, day-bootstrap 95% CI, per-season table, share of bets fading the move.
Decisions:
- H1: "the live analogue loses" is confirmed for a market if R0's ROI at -110 has a CI entirely below 0.
- H2: R1 or R2 is "better than R0" only if the paired ROI difference at -110 (bootstrap over days on the union of bet days) has a CI excluding 0
  and the arm's ROI is higher in at least 5 of 7 seasons.
- H3: an arm is "expected to pay on a zero-fee venue" only if its pooled zero-vig hit-rate CI has a lower bound above 50%. Otherwise any
  forward test of it is a measurement, not an expected edge.
Stated limit: the close is the extreme later line; live entry is 11 am. Only this season's snapshots have 11 am lines.

### E27: totals level calibration (E18 fix c) on the deployed form (re-scoring saved predictions; no retraining)

As described in `reports/upgrade_2026-27/PLAN.md` step 1. Keep rule: E18's registered rule in `reports/experiments/PREREGISTRATION.md`
(better total MAE in at least 5 of 7 seasons, pooled CI below 0, and its R1/R2 robustness checks negative), with ML and spread unchanged.
If E27 fails, say so and stop before writing any calibrator code.
