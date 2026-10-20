# Experiment pre-registration (written before any experiment was run)

Committed before the first experiment result existed. Keep criteria below are **not changed after seeing results**;
if a criterion turns out to be badly specified, that is reported as a limitation, not edited.

## Common protocol

- **Baseline** (`reports/experiments/baseline`): the post-audit deployed configuration. Opening odds as inputs, two-stage
  feature selection on the first 80% of model-train rows, LightGBM hyperparameters chosen **inside each fold** from a fixed
  8-config grid on a time-ordered validation block (last 20% of model-train), isotonic calibration and residual sigma from the
  last 20% of the training window. Seeds fixed (42), LightGBM deterministic.
- **Folds**: test seasons 2019-20 ... 2025-26 (7 folds), training = all earlier seasons. Identical for every experiment.
  Exception: E8 (see there).
- **One change at a time** against the baseline. Experiment code lives in separate modules and the baseline code path is not
  edited; the baseline is re-run at the end and must reproduce bit-for-bit.
- **Lookahead**: any new feature gets a test that corrupts all rows dated on/after a game's date and asserts the feature for that
  game is unchanged (`tests/test_experiment_lookahead.py`).
- **Primary metrics**: ML log loss, Brier; spread and total MAE, RMSE, bias (actual - pred); calibration (reliability by decile, ECE);
  closing line value (points for spread/total bets) where closing lines exist. **Secondary**: ROI, bet count, hit rate, max drawdown
  (units of a $100 stake) with day-bootstrap 95% CIs.
- **Bootstrap**: resample game days with replacement, 1000 draws, fixed seed. Paired comparisons use per-game losses on the
  common games of baseline and experiment.
- **Main criterion (MC) for a metric**: the metric improves in **at least 5 of 7** test seasons (per-season mean paired difference
  has the favorable sign) **and** the pooled paired day-bootstrap 95% CI of the difference excludes zero in the favorable direction.
- **No-harm rule (NH)**: no non-targeted primary metric (ML log loss, Brier, spread MAE, total MAE) is worse than baseline by more than
  0.5% (relative, pooled point estimate).
- A decision is KEEP only if stated criteria hold; otherwise REJECT. "Inconclusive" is reported as REJECT (not adopted).

## A. Model structure

**E1 market-residual targets.** Spread target = margin + opening spread, total target = total - opening total, ML via init_score =
logit(de-vigged opening home probability; where the ML price is missing, an in-fold logistic of home win on the opening spread).
Same selected features as the baseline. Keep **per market**: KEEP-SPREAD if MC holds for spread MAE; KEEP-TOTAL if MC holds for total MAE;
KEEP-ML if MC holds for ML log loss; each also requires NH. Tail analysis (|spread| > 12, opening total > 242) is descriptive:
bets under the 5% rule, bet count and ROI baseline vs E1.

**E2 calibration.** Variants: Platt on log-odds (fit on the last 20% split), cross-fitted isotonic (4 time blocks over the last 2
training seasons, out-of-fold raw predictions). Keep a variant if (i) ECE is lower in at least 5/7 seasons, (ii) the pooled paired
log-loss difference has CI upper bound < +0.0005 (not worse), and (iii) either distinct output probabilities >= 3x the baseline's or
MC holds for log loss. If both qualify, the lower pooled ECE wins.

**E3 coherence.** (a) descriptive: share of games where ML and spread disagree on the winner (ml_prob > .5 vs predicted margin > 0), and
ML vs total-implied-by-nothing is not defined (only ML/spread pair is tested). (b) blend = w * ml_prob + (1-w) * Phi(margin_pred / sigma_spread),
w from a 0..1 grid (step 0.1) by log loss on the calibration set inside each fold. (c) stack: logistic regression on
[logit(ml_prob_raw), margin_pred/sigma_spread] fitted on the calibration-set predictions (out-of-sample for the base models), then isotonic-free.
Keep (b) or (c) if MC holds for ML log loss, NH holds, and the contradiction rate does not increase.

**E4 heteroskedastic sigma.** Second LightGBM (small) on out-of-fold |residual| (4 forward-chaining blocks inside model-train), features:
FGA-EWMA pace proxies, 3PA share, injury counts/missing flags, |line|, total line. sigma_i = E|e|_i * sqrt(pi/2).
Metric: mean log-likelihood of cover outcomes (spread: home covers; total: over; pushes dropped), p = Phi((pred +/- line)/sigma_i).
Keep per market if the log-likelihood improves in >= 5/7 seasons with a pooled CI excluding zero **and** the ECE of cover probabilities
(10 bins) is not worse than the global-sigma ECE by more than 0.005.

## B. Features (extra candidate columns added to the baseline pool; selection decides)

**E5 pace-normalised, opponent-adjusted team stats.** Per-100-possession versions of 10 box-score stats, minus the opponent's entering
EWMA allowance (span 20, strictly prior games), EWMA spans 5/20, home-away diffs. Keep if MC holds for at least 2 of {ML log loss,
spread MAE, total MAE} and NH holds for the third.

**E6 offseason regression to the mean.** kappa in {1.0, 0.75, 0.5} selected per fold by the validation block (mean of the three
targets' validation ranks), applied to team EWMAs. Keep if MC holds for at least 2 of the 3 targets on all games, NH holds, and the
October-November subset point estimate is not worse on any of the three metrics. October-November results are reported separately.

**E7 bench-depth replacement level.** Replace constant REPLACEMENT_BPM = -2.0 in the player value with the team's own healthy bench BPM
(minutes-weighted mean BPM of its rotation players ranked 6-10 by expected minutes who are not on the report as Out/Doubtful), strictly
prior information. Keep if on games with >= 1 Out player (either team) MC holds for spread MAE or total MAE, and NH holds on all games.

**E8 fresh vs chronic absences.** Step 1 (hypothesis test on discovery seasons 2019-20..2021-22 only): regress the market residual
(margin + opening spread) on home-minus-away fresh and chronic lost value (fresh = played in at least one of the team's last 3 games; chronic =
missed >= 5 consecutive team games) with day-cluster bootstrap CIs. Supported iff the fresh coefficient is negative with CI excluding zero
and |chronic| < 0.5 |fresh| (or its CI includes zero). Only if supported, step 2: add the features (plus decayed expected minutes for
chronic absentees: e_minutes * 0.5^((gap-1)/5), a fixed constant, not tuned) and evaluate on folds 2022-23..2025-26 (4 folds, to keep the
discovery window out of the evaluation). Keep if improves in >= 3/4 folds with a pooled CI excluding zero for >= 1 of the three
primary losses and NH.

**E9 official PDF injuries in live inference.** Not a backtest. Done when: the ESPN scrape no longer exists on any live path, a
noon-to-latest downgrade flag exists with a lookahead test, both sources are logged side by side for new games, and the disagreement rate on
the historical overlap is reported. No performance claim.

## C. Betting strategy (on baseline predictions; no retraining)

**E10 selection rule.** Baseline rule = 5% probability edge. Alternatives: (a) flat |pred - line| > 1.0 pt for spread/total (ML keeps the 5%
edge rule), (c) per-fold edge threshold from {0, .01, .02, .03, .04, .05, .075, .10} maximising ROI on the calibration-set predictions
(fallback .05 if < 100 calibration bets). Keep an alternative if its pooled ROI exceeds the baseline rule's by >= 1.0 point with a day-bootstrap
CI of the ROI difference excluding zero, in each market separately; hit rate of selected bets must be within 3 pp of the model's average P(win).

**E11 same-game exposure.** Quarter-Kelly stakes from the model's win probability and the actual price. Strategies: independent (baseline),
one best-EV market per game, joint Kelly via a Gaussian copula (rho from in-fold outcome indicators of ML vs spread bets; total independent).
Bootstrap (1000 resamples of game days) bankroll simulation. Keep a strategy over independent Kelly if median log-growth is higher with a
paired CI excluding zero and its 95th-percentile max drawdown is not higher.

**E12 line shopping and consensus.** `odds_snapshots` has no rows in this database, so multi-book data = the per-book columns of
`nba_odds_opening` / `nba_odds` (six books, cleaned by `odds_clean.py`); coverage reported. (a) consensus de-vigged median probability as the
market reference; (b) bets settled at the best available price/line across books. Decision: adopt best-price settlement as a *reporting*
option only if it improves ROI with a CI excluding zero **and** the control "bet both sides of every game at best price" has ROI <= 0.
Price improvement is measured per bet by key number (3, 7, other), not assumed.

## Winners combined

After all experiments, only KEEP decisions are re-run together (one combined run) and compared with the sum of individual effects.
