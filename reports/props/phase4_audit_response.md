# Response to the props-onto-main audit (2026-10-20)

Every item was checked against the code and the data before accepting it. Branch `props-phase4` (stacked on `props-onto-main`, PR #17). Protocol consequences: [phase4_protocol_update.md](phase4_protocol_update.md). Development folds only were used for every decision below; 2024-25 is "seen" and 2025-26 is untouched.

## Verdicts

| # | Audit item | Verdict | Evidence | Action |
|---|---|---|---|---|
| B1 | Fit window is the first 80% of dates; models are never refit on the calibration window | **Genuine; minor contributor to the bias** | 2024-25 fold: 44% of fit rows from 2019-21, 2% from 2023-24 (2023-24 fold: 55%, 0 of 2023-24). Refitting on fit + calibration (`--refit-all`) on 2023-24: CRPS better for points -0.0047, pra -0.0051 (intervals exclude 0), mean absolute bias 3.17% to 2.89% | Option added and used in the corrected pipeline |
| B1 | ... "weight recent seasons more" | **Tested, rejected** | R3, recency half-life 730 days (fixed in advance): 2023-24 bias 3.57% to 3.65% (needed -33%); CRPS significantly worse for points, threes, pra | Not adopted. The old-seasons-taught-a-wrong-correction story is therefore not supported at this half-life |
| B2 | 2024-25 can no longer serve as clean validation | **Genuine** | I formed the R1 hypothesis from, and discussed isotonic using, 2024-25 results | Protocol update: 2024-25 "seen", 2025-26 the single clean holdout; needs user sign-off for Phase 5 |
| W1 | Minutes residual bank: in-sample sd and non-forward OOF | **Genuine** (the code said so in a comment) | `sd = spread.predict(training rows)`; `blocked_oof` used later blocks | Forward-chaining OOF residuals + forward-chaining out-of-sample sd (`common.forward_oof`), tests |
| W2 | Never compared with the betting market | **Genuine** | CRPS vs rolling averages is weak evidence. 2023-24 dev, v2_base, consensus book line: the market beats the model in log loss in every stat (diff +0.003 threes to +0.023 pra, all intervals above 0). Information slope 0.20 [0.15, 0.25] | `props/models/market_bench.py`, `scripts/props_phase4_market.py`; report `phase4_market_2023-24.md`. Market metrics become Phase 5 gate metrics |
| W3 | Calibration measured at model-chosen lines | **Genuine** | Ladder at floor(model mean) +/- 2 | Book-line calibration by model-minus-market gap added (same report). The ladder is still used for Phase 4 shape checks only |
| W4 | Injury report time = slot stamp, not posting time | **Genuine** | 28% of rows at every decision time used a report stamped 5-10 minutes before T; min age at T-5 = 5.0 | 15-minute publish-lag buffer in `latest_report_rows`; dataset rebuilt (`props_features_v3.db`). Effect is small: injury status changes for 0.07% (T-240), 1.1% (T-60), 0.8% (T-5) of rows |
| W5a | `snapshot_env` accepts snapshots up to 1 minute after T | **Genuine, negligible** | 0.05% of snapshots (about 2 per grid point) | Made strict |
| W5b | `env_source` encodes the era; SBR proxy timing differs | **Genuine for `with_env`** (not trained yet) | `env_source` was a model input | Removed from model inputs (`NOT_FEATURES`). Phase 5 will run H1/H2 on the snapshot era (2023-24 on) and treat the proxy era as sensitivity only |
| W6 | Simulation noise is as large as the edges | **Genuine** | Measured (fixed lines across seeds, rotation players, P(over) in 0.2-0.8): 0.77 percentage points per run at 4,000 sims, 0.35 at 20,000 for points, threes, pra, pr; 0 for rebounds and assists (exact mixtures) | Phase 5 requirement: every row compared with a market uses 20,000 sims (about 30 min per season). Saved 4,000-sim pmfs remain fine for CRPS-type metrics |
| W7 | Pushes counted as losses | **Genuine for Phase 5** | `prob_over` = P(X > line) | `prob_push`, `prob_under`, `prob_over_no_push`, `ev_per_unit` (push refunds) with tests; `market_bench` uses the no-push probability and drops pushes |
| W8 | Combo stats drawn independently given minutes; correlation 0.21-0.35 left out | **Not genuine as stated** | The simulation shares one minutes draw across stats and that carries the dependence. Model-implied vs observed covariance, 2023-24: points-rebounds 4.37 vs 4.51 (corr 0.29 vs 0.31), points-assists 1.94 vs 1.84 (0.18 vs 0.18), rebounds-assists 0.87 vs 0.89 (0.19 vs 0.20); same on 2024-25. PIT histograms of combos look like those of singles (the skew is the mean bias) | No model change. Limit: the stored pmfs are marginals; a multi-prop portfolio needs a joint re-simulation (the simulator has the joint) |

Not in the audit but found while checking: the one-to-one merge guard in `market_bench` caught real duplicates (name variants across books, 1,705 of 339,854 consensus rows), fixed by keeping the quote most books agree on. The props dataset build has many merges without `validate=`; the finished tables are checked for duplicate keys in tests but not at each merge. That is still open (below).

## What the corrected pipeline does on the development folds

`v3_dev` = v3 dataset (publish lag, strict env timing) + forward-chaining minutes bank + refit on fit + calibration; frozen v2_base rate families; no recency, no R2.

| Fold | CRPS vs v2_base | Mean absolute bias, core stats |
|---|---|---|
| 2021-22 | better or equal in all stats (points -0.0056 significant) | 2.31% to 2.30% |
| 2022-23 | better or equal (rebounds, assists, threes significant) | 0.81% to 0.68% |
| 2023-24 | equal, except rebounds slightly worse (+0.0012) and assists slightly better (-0.0013) | **3.17% to 3.57%** |

The corrections that remove optimism are adopted regardless (protocol update), and they leave the 2023-24 bias where it was.

## The bias, restated

- It is created by the model, not by the league: the simple shrunk-rate x recent-minutes baseline is not biased in 2023-24 (points obs/baseline +0.2%) or 2024-25 (+2.0%). The model's per-minute rates sit 4.1% (points), 3.4% (assists), 7.8% (threes) above its own features in 2023-24, and its predicted minutes 1.6% above recent-average minutes while actual minutes are flat.
- Tested and not the cause (or not fixed by): season-position features (R1), the stale fit window (refit: partial), recency weighting (R3), the minutes residual bank, injury timing.
- Still unexplained. Rolling recalibration (R2, constants fixed before any result) halves it: on the corrected pipeline it removes 49% of the 2023-24 bias (bar: 50%), improves CRPS significantly in 2023-24 (points -0.0102), and does not harm 2022-23. By the rule it is **not adopted**; the constants have not been touched. Tried rules so far: R1, R2 (twice, same constants), R3, plus design corrections that were adopted on validity grounds.

## Still open

1. Decision for the user: R2b (season-boundary carry-over, prior 1,000), or stand by the rules.
2. Decision for the user: the protocol change (2024-25 seen; 2025-26 scored once for both model and betting rules).
3. Per-merge `validate=` guards in the props dataset build.
4. Rerun the 2024-25 numbers on the final configuration, as "seen" results, once the configuration is frozen.
5. Phase 5 plan: 20,000-sim rows, book-line metrics as gate criteria, snapshot-era-only H1/H2, joint re-simulation for portfolio risk.
