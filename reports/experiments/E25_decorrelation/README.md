# E25 - decorrelation from the market: edge slope and blend

Registered in `reports/experiments/PREREGISTRATION.md` (section E25) before any run. Scripts: `experiments/e25_regen.py` (three walk-forward runs), `experiments/e25_decorrelation.py` (checks + diagnostic).
Numbers: `results.json`. Run 2026-10-24 on `training_set_v1_reduced_2026-10-19.csv`, folds 2019-20..2025-26, 8,897 games common to all three runs.

## 1. Were the existing no-odds files usable? No - regenerated

| file | problem |
|---|---|
| `models/saved/backtest_predictions_no_odds_v2.csv` (Jan 2026) | corrupt attached lines: 230 games with a moneyline beyond +-2000, 355 with both moneylines the same sign, 553 spread sign flips vs the open |
| `models/saved/backtest_predictions_no_odds.csv` (Mar 2026) | one season only; feature config has `roster_bpm_ewma_x/_y` (duplicated-merge artifact) |
| both | older code than every E-series run (pre audit-fixes / team-audit-2) |

Regenerated with the E-series harness: `E25_with_odds` (market as a feature), `E25_residual` (market as an offset, E1 hooks), `E25_no_odds` (no market information). Deployed models untouched.

**Deviations from the registration (disclosed):** E1's ML fallback could not handle the 44 games with no opening spread in the 10-19 set, so `e25_regen.init_scores` fits it on rows with a spread (home-win rate when both lines are missing); the residual run drops those 44 games (its targets need a line). The analysis uses only games present in all three runs.

### Validity checks

| check | result |
|---|---|
| 1 no selected feature contains `odds`/`close` | pass |
| 2 same games (residual = with-odds minus the 44 no-line games) | pass |
| 3 training ends before each test season | pass, all 7 folds |
| 4 metrics within tolerance of the audit's no-odds walk-forward | **fail** in 2 seasons, in the *better* direction: 2020-21 total MAE 0.62 lower, 2023-24 ML log loss 0.052 lower |
| 5 market proxies (feature correlation > 0.9 with a line) | none; highest `diff_off_plusMinus_20g` with the spread, 0.83-0.85 |

Check 4 investigated: the audit reference used untuned neutral parameters, and per-fold tuning improved the audit's own with-odds run in the same seasons (-0.21 total MAE, -0.021 log loss), though by less. The decisive leak test passes: **the no-odds model is worse than the opening line in every season and every market**
(total MAE +0.24..+0.55, spread +0.21..+0.36, log loss +0.008..+0.051). A leak would beat the line. Results are read with this disclosed.

## 2. Edge slope beta (actual - line regressed on model - line), confirmation 2022-23..2025-26, opening line

| market | with odds | residual | no odds | no-odds edge SD vs with-odds | error corr. with market (with / no odds) |
|---|---|---|---|---|---|
| total | 0.19 [0.05, 0.34] | 0.10 [-0.02, 0.21] | 0.19 [0.09, 0.28] | 5.4 vs 3.2 pts | 0.985 / 0.958 |
| spread | 0.27 [0.01, 0.55] | 0.76 [0.34, 1.16] | 0.16 [0.05, 0.28] | 3.4 vs 1.4 pts | 0.994 / 0.970 |
| ML (log-odds) | 0.39 [0.09, 0.72] | 0.53 [0.14, 0.91] | 0.18 [0.04, 0.32] | - | - |

- Almost every model's edges carry *some* information (lower bound > 0), and **all are overstated by about 2.5-10x** (beta well below 1). This is the E10 overconfidence, measured directly. The residual spread model's 0.76 is the exception, but its discovery beta is 0.02, so it is unstable.
- Against the **closing line every beta is about 0** (-0.05..+0.05, all intervals include 0): the close already contains whatever the models know.

## 3. Blend L + k(model - L), k fitted on discovery, scored on confirmation (opening line)

| market | k (no odds) | no-odds blend minus line | with-odds blend minus line | no-odds minus with-odds blend | Holm p (no odds) |
|---|---|---|---|---|---|
| total (MSE, pts^2; line 335.3) | 0.13 | **-0.95 [-1.67, -0.23]**, p 0.012 | -0.21 [-0.83, 0.43] | **-0.75 [-1.27, -0.25]**, p 0.004 | **0.036** |
| spread (MSE; line 187.8) | 0.17 | -0.30 [-0.78, 0.14], p 0.17 | -0.12 [-0.29, 0.04] | -0.18 [-0.55, 0.18] | 0.18 |
| ML (log loss; line 0.6026) | 0.11 | -0.0006 [-0.0012, 0.0001], p 0.09 | -0.0004 [-0.0008, -0.0000] | -0.0002 [-0.0007, 0.0004] | 0.18 |

Regular season only: total no-odds blend -0.88, p 0.006. Walk-forward k: the totals no-odds blend helps in 4 of 6 seasons (2020-21, 2022-23, 2023-24, 2024-25) and hurts in 2021-22 and 2025-26. Against the close no blend helps (total +0.04, p 0.65).

## 4. Decisions (as registered)

- **H_DECOR not supported in any market.** It required the no-odds beta to exceed the with-odds beta; for totals they are equal (0.19 vs 0.19).
- Its second leg holds for **totals**: the no-odds blend beats the opening line out of sample (Holm p 0.036) and beats the with-odds blend.
  Post hoc (not a registered test): what decorrelation bought is *more* informative disagreement at the *same* reliability (edge SD 5.4 vs 3.2 at equal beta), not more reliable edges. My registered statistic (the beta difference) was the wrong one to discriminate on.
- Size: the totals gain is 0.28% of MSE (RMSE 18.31 -> 18.29). The blended shift has an SD of about 0.7 points against roughly 1.1 needed to beat -110, so only the tails could be bettable. The decile table is asymmetric (top decile +2.1 points, bottom decile -0.1).
- Status: the totals no-odds blend is a **candidate** (forward test or the feature route), not a deployment. Shrinking all edges by beta is the robust practical lesson.

## 5. The caveat that matters most

The models read the game-day noon injury report; the opening line is usually posted before it (`metrics.OPEN_BASIS_CAVEAT`). An edge against the open that vanishes against the close is exactly what *injury news the model saw first* looks like, and an opening price may no longer be on offer at noon.
These historical numbers therefore overstate what is achievable. The real test is against the entry line (first line at or after the feature cutoff, `core.odds_clean.load_entry_odds`). That needs `odds_snapshots`, which only exists from 2026-10-19, so it is a forward test.
