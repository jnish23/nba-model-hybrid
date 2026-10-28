# 2026-27 upgrade plan: totals calibration, edge shrinkage, line shopping, then re-freeze E26

Written 2026-10-24. Nothing here has been done yet. The deployed set stays `_with_odds_cand_20261019` until step 5, and step 5 needs the owner's go-ahead.

## What this buys, honestly

| change | backtest evidence | what it does | expected effect |
|---|---|---|---|
| E18 totals level calibration (fix c) | total MAE 14.77 -> 14.62, better in 7/7 seasons, pooled CI below 0, robust to recent-only and era checks | pulls the totals forecast toward the line, more so at high totals | the totals model goes from **worse** than the opening line (14.76 vs 14.62 on the same games) to **level with it** (14.61). It fixes a weakness; it does not create an edge |
| Edge shrinkage (E25/E26) | edges overstated 2.5-10x in every market; beta CIs exclude 1 | scales each edge by beta before the bet rule | far fewer bets. On 7 past seasons with the 5% rule: about 35 totals, 6 ML, 0 spread bets in total |
| Line shopping (E12) | +1.7 ROI points totals, +3.3 spread at the best opening quote (an upper bound: quotes may not have been simultaneous) | judges and places each bet at the best book you can use | better prices on the bets you do make |

Read together: after these changes the system will bet rarely, and that is the point. The 5% rule today mostly fires on overconfident edges (E10: model P(win) 56-63% against a 48-53% hit rate).

## Decisions needed from you before step 2

1. **Which E18 fix.** I recommend (c), level-dependent shrinkage toward the line: MAE 14.62 vs (b)'s 14.69, and the bias above a 242 total drops from +10.4 to +1.6 points (b: +5.4).
   (c) already shrinks totals edges, so the separate totals beta in step 3 is re-estimated after (c) and should come out near 1. (b) keeps more totals bets (1,485 vs 81 over 7 seasons), but most of them would be cut by the shrinkage in step 3 anyway.
2. **Books for line shopping.** Which of draftkings, fanduel, betmgm, bet365, caesars and bet_rivers_ny you can actually bet at. A price you can't take must not be used.
3. **Bet threshold after shrinkage.** I recommend keeping the 5% rule for now. With honest probabilities a smaller buffer would be defensible, but changing it is a separate choice and should be registered, not tuned on the backtest.
4. **Shrink ML and spread too, or totals only?** I recommend all three, because all three betas are well below 1. The consequence is almost no ML or spread bets.

## Steps

### 1. E27: register and run the candidate form on the 7 past seasons (no retraining, about half a day)
- Register E27 in `PREREGISTRATION.md` before running. The candidate is the deployed form (E1 spread, E3 ML stack, E4 sigma) plus E18 fix (c) on totals.
  E13's totals are the baseline totals, so E18's fold-by-fold fits apply directly; this re-scores saved predictions and retrains nothing.
- What it measures:
  - total MAE and cover log loss (should reproduce E18);
  - whether E4's per-game sigma still helps on the calibrated residuals, or a single sigma does;
  - **the new beta per market** for the candidate form, with intervals;
  - bet counts and hit rates under the new rule with shrinkage and line shopping (descriptive).
- Keep rule: E18's rule (better in at least 5 of 7 seasons, pooled CI below 0, R1/R2 negative), and ML/spread unchanged.
- If E27 fails, **stop**: don't promote, keep E26 running as it is, and ship only step 3 on the current model.

### 2. Code the totals calibrator (about a day; nothing deployed)
- `team/market_models.py`: a `TotalLevelCalibrator` with fit on the calibration block and predict(pred, opening total), using fix (c) exactly as E18 registered it.
  The weight is clipped to [0, 1]. A game with no opening total keeps the raw prediction (the live run already skips games without an opening line).
- `team/train_baseline.train_and_save_all`: fit the calibrator on the calibration-block total predictions, as the E3 stack is fitted, and save `calibrator_total<suffix>.pkl`.
  Recompute `total_std` (and E4's sigma, or turn it off, per E27) on the calibrated residuals.
- `team/model_registry.py`: load the optional file and include it in the set hash. `predict_markets` applies it.
- Tests: fit/predict against E18's numbers on one fold, save/load round trip, hash changes when the file changes, old sets without the file load unchanged.

### 3. Betting-rule changes (about a day; no model change, so this can ship before or without step 5)
- **Shrinkage.**
  - The betas live in a new `team/edge_shrink.json`, keyed by `model_set_hash`. `calibration_meta` is part of the set hash, so editing it would change the deployed set.
  - `team/betting.py`:
    - spread/total: the shrunk edge is beta x (pred - line), with the cover probability from `sigma_shrunk`;
    - ML: logit p = logit q + beta x (logit p_model - logit q), where q is the de-vigged price of the same book.
  - A set with no entry is not shrunk (current behaviour), and the picks file shows the raw and the shrunk edge.
  - **The logged predictions stay raw.** `predictions_log_v2` must keep the model's own numbers, because the forward test measures beta from them.
- **Line shopping.**
  - `team/live_odds.py` / `run_inference_v2.py` currently judge bets at the first book in priority order.
  - Evaluate each side at every allowed book's own line and price, after the `core/odds_clean` cross-book outlier filter so one stale quote can't win, and pick the best EV. The picks file names the book.
  - The model's inputs stay the opening line, which is unchanged.
- The dashboard and `scripts/backtest_analysis.py` use the same `team/betting.py` functions, so they stay consistent; add tests for both rules.
- Caveat: the betas were estimated against the opening line, while live edges are measured against the current line. The forward test (step 6) measures beta against the noon entry line and is the check on this.

### 4. Build the candidate and gate it (an hour of compute)
- `scripts/retrain_candidate.py` with the calibrator gives a new suffix and `reports/retrain/retrain_<tag>.json`.
- The gate is a **no-harm check only**: it fails a candidate only if it is worse by more than 0.15 MAE / 0.004 log loss on the last 120 days, which is coarser than E18's gain. E27 is the evidence; the gate guards against a broken build.
- Smoke test on today's slate with the candidate. `run_inference_v2.py --suffix` exists but **writes to the live log**, so add a `--no-log` option first.

### 5. Promote (your go-ahead; a one-line reviewed commit)
- Set `DEPLOYED_SUFFIX` to the candidate, with the `edge_shrink.json` entry for its hash in the same commit. It takes effect at the next 9:15 run. Record the date.
- **Then freeze the model for the season.** The weekly retrain keeps producing candidates, but none is promoted until the season ends, except to fix a real bug.
  Every promotion restarts the forward test's clock, which is why model changes belong at the start of the season.

### 6. Re-freeze E26 as E26b (half a day; before the first game on the new model)
- Register E26b in `PREREGISTRATION.md`: the same design as E26, with:
  - the betas from E27's candidate-form predictions (the same numbers the bet rule uses, so the forward test checks exactly the rule in use);
  - the new `model_set_hash`;
  - a start on the first full day after promotion.
- Close E26's (a) tests with the games they have (likely none) and say so.
- **E26's B-total test carries over unchanged**, because the no-odds model and the blend don't depend on the deployed model.
- `experiments/e26_score.py`: take the frozen folder, hash and start date as options, then commit it before the first look. `e26_freeze.py` refuses to overwrite, so E26b gets its own folder.
- Check that E15, E20, E22 and E24 depend only on market lines (I expect so) and record it in the E26b section.

### 7. Optional: closing-line-value tracker
For each bet the rule makes, does the line move toward it between noon and the close? This gives a usable reading in 1-2 months, compared with 1-2 seasons for results. It uses the `odds_snapshots` already being collected.

## Order and timing

1 -> (2 and 3 in parallel) -> 4 -> 5 -> 6. Steps 1-4 are about 3-4 working days. About 7 games a day are played early in the season, so each week of delay is roughly 50 games of the new model the forward test won't see.

If E27 fails or you'd rather not change the model this season, do step 3 alone on the current set with E26's frozen betas. E26 then keeps running exactly as registered.

## Not done without your yes

Promoting (step 5), retraining or overwriting any existing set, regenerating `predictions_2025-26`, changing scheduled tasks, and committing.
