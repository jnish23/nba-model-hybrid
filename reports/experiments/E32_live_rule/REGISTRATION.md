# E32 registration (verbatim copy, made before any outcome was computed)

Source: `reports/audit_pregame_2026-11/BRIEF.md`, section "E32" (lines 75-93 of that file), copied verbatim between the rules below.
sha256 of the section text (UTF-8, LF line ends, no trailing newline): `6a3fe07005b37b74dec8b67b51179d3aacc2555faeeb90c7f825a627069355a1`

---
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
---

## Descriptive readouts from the work order (sha256 `8efdec70fbd24aec65dd5b9e6aeedb028c88d2053f89d0dc8d76ec43ea1df6e4`)

Readouts a-e below come from the lead's work order for workstream A5 (registered as descriptive; they decide nothing):
a. Edge-slope tables: regress (result - close) on own and on (open - close) jointly, per market, with day-bootstrap CIs; the same against the open. For ML a logistic regression with offset logit(close).
b. Threshold sweep for R2 from 0% to 10% probability edge in 1-point steps: bets, zero-vig hit rate with CI. No winner is picked.
c. How often the close crosses the model's number, by market, and the result on those games.
d. This season's forward evidence with a real 11 am line (predictions_log_v2 2026-27 joined with odds_snapshots): R0, R1, R2 with the 11 am line as the later line, settled at the 11 am line; counts, hit rates, CLV to the close, CIs. About 160 games cannot decide anything.
e. Bets per season each of R0, R1, R2 would have recommended (how many bets a year to expect if R1 or R2 replaced the live rule).

## Operational choices (sha256 `25ff166f49f51095a56a9739b515ce05bca9a3b94d67ead24da9fc63a95fa92a`)

Operational choices fixed before any outcome was computed (they fill gaps in the text above; they change no arm, threshold or decision rule):
1. Data: reports/experiments/E25_with_odds/predictions.csv is read from the production checkout (read-only) because the worktree holds only fold_info.json for that folder. Game rows are kept per market only when the opening and the closing line of that market are present (ML: open_home_prob and close_home_prob).
2. Spread units: home-margin points. model M = spread_pred, home line L (-3.5 = home favoured), home covers when margin + L > 0. own = M + L_open (= model margin minus open-implied margin). Edge against a line L = M + L. Side +1 = home. Total: own = total_pred - open total, side +1 = over.
3. Probability edge uses team.betting.cover_probabilities (push-aware on whole lines, line = the line the edge is measured against) minus the -110 break-even 0.5238. R0 and R2 measure it at the close line, R1 at the open line. Selection ">" 5 points strictly.
4. R1 "not crossed": sign(model - close) must equal sign(own) and be non-zero. R2: point edge = |own| applied at the close line (fair = close + own).
5. Settlement at the close line: win +100/110, loss -1, push 0 (-110); win +1, loss -1, push 0 (zero vig). ROI = profit / bets placed (pushes count as bets). Hit rate = W / (W + L). Model P(win) reported at the settled line (R1: model vs close, R2: transferred edge).
6. "Fades the move": spread bet fades if the market moved against the bet side between open and close (home bet and the home margin line moved away from home; away bet and it moved toward home); total: over bet and close < open, under bet and close > open; ML: home bet and close prob < open prob. Zero-move bets are reported separately.
7. Moneyline: probabilities. own = logit(ml_prob) - logit(open_home_prob). R0: side with model prob - close fair prob > 0.05. R1: |model - open| > 0.05 on side sign(own), and the model prob is still on that side of the close (home: model > close; away: model < close). R2: fair = expit(logit(close) + own), |fair - close| > 0.05, side sign(own). Priced at the side's fair closing probability q: win (1-q)/q, loss -1. ML has no -110 ROI in the registration, so H1 and H2 for ML are read on the zero-vig ROI, and H3 on the zero-vig hit rate (literal rule: lower bound above 50%); the zero-vig ROI CI is reported beside it. Second scoring at real closing prices (core.odds_clean.load_clean_odds close_home_ml/close_away_ml, the median across books) where they exist.
8. Bootstrap: day-bootstrap, 1,000 draws, np.random.default_rng(42). For single-arm statistics the resampled units are all game dates of the market sample; for the paired H2 difference they are the union of the two arms' bet days (as registered). Per-season CIs resample dates within the season. 95% CI = 2.5 and 97.5 percentiles.
9. H2 season rule: arm ROI (-110; ML zero vig) higher than R0 in at least 5 of 7 seasons, comparing seasons where the two arms' ROI are both defined; a season where an arm has no bets counts as not higher.
10. Sigma = the fold's saved sigma_spread / sigma_total (per row). Forward readout d uses calibration_meta_with_odds_cand_20261019.json spread_std 14.287 and total_std 18.590 (one global value; the per-game total sigma is not logged).
