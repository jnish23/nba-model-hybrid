# Experiment report: E1–E12 against the post-fix baseline

Branch `experiments` (cut from `audit-fixes`). Baseline = the post-audit retrained model with its walk-forward backtest:
7 folds (test seasons 2019-20 … 2025-26, each trained on all earlier seasons), seed 42, deterministic LightGBM, hyperparameters
chosen inside each fold on a time-ordered validation block, 8,916 test games. Saved in `reports/experiments/baseline/`.
**Re-running the baseline reproduces it bit-for-bit** (predictions, calibration-set predictions and selected features identical).
Keep criteria were written before any run (`PREREGISTRATION.md`) and not changed. CIs are day-bootstrap (1,000 draws);
differences are paired per game. "MC" = improves in ≥ 5 of 7 seasons and CI excludes 0; "NH" = no more than 0.5% relative worse.

## Baseline (pooled and by season)

| ML log loss | ML Brier | ML ECE | spread MAE / RMSE / bias | total MAE / RMSE / bias |
|---:|---:|---:|:--|:--|
| 0.6199 | 0.2132 | 0.0107 | 10.70 / 13.71 / −0.15 | 14.77 / 18.71 / +1.29 |

| season | ML log loss | Brier | ECE | spread MAE | spread bias | total MAE | total bias |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 2019-20 | 0.6320 | 0.2206 | 0.0566 | 10.096 | −0.19 | 15.293 | +2.47 |
| 2020-21 | 0.6361 | 0.2226 | 0.0280 | 10.931 | −0.60 | 14.952 | +1.00 |
| 2021-22 | 0.6288 | 0.2154 | 0.0316 | 11.000 | −0.57 | 14.680 | +0.11 |
| 2022-23 | 0.6509 | 0.2211 | 0.0441 | 9.944 | +0.51 | 14.507 | +1.80 |
| 2023-24 | 0.6059 | 0.2056 | 0.0323 | 10.786 | +0.14 | 14.597 | +1.58 |
| 2024-25 | 0.5959 | 0.2050 | 0.0264 | 10.870 | +0.03 | 14.582 | +0.82 |
| 2025-26 | 0.5931 | 0.2044 | 0.0237 | 11.225 | −0.45 | 14.884 | +1.34 |

Reference point: de-vigged market consensus log loss on 8,366 games is 0.6076 at the open and 0.6001 at the close, versus 0.6193
for the model on the same games. The market is still better than the model.

**Selection-noise floor** (baseline rerun with other feature-selection seeds, nothing else changed): ML log loss moves
+0.0001 / +0.0032, spread MAE +0.010 / +0.011, total MAE +0.005 / +0.005; no CI excludes zero. Feature-pool experiments
(E5–E8) can only be read beyond about these sizes.

## A. Model structure

### E1 Market-residual targets — KEEP for spread only
Changed: spread model predicts margin + opening spread (prediction = −spread + f(X)); total model predicts total − opening total;
ML uses init_score = logit of the de-vigged opening home probability. Same features and tuning protocol.
- Spread MAE −0.0239 (CI −0.048 … −0.001), better in **7/7** seasons (−0.004 … −0.048); MSE −0.77 (CI excludes 0). Bias −0.15 → −0.04.
- ML log loss −0.0022 (CI −0.009 … +0.005), 4/7 seasons: not significant → reject for ML.
- Total MAE **+0.27** (CI +0.20 … +0.34), worse in 7/7 → reject for total (the total market is already close to the best available total prediction and the offset adds noise).
- Tails: for |spread| > 12 (655 games) the 5%-rule spread bets fall from 157 to 18; for total > 242 the model is still biased high (+10 → +12 points). So the large-spread "edges" largely disappear for spread, not for total.

### E2 Calibration method — Platt wins (nested inside E3)
Changed: only the probability calibrator. Isotonic (baseline) vs Platt vs cross-fitted isotonic over the last two seasons.
- Platt: log loss −0.0071 (CI −0.0128 … −0.0020), Brier −0.0010, ECE 0.0107 → 0.0082 (better in 5/7), distinct output probabilities 256 → 8,855. Per season −0.0079, −0.0067, −0.0088, −0.0187, −0.0098, −0.0001, +0.0022.
- Cross-fitted isotonic: log loss +0.0032, ECE 0.0166 → rejected.
- Explanation: isotonic on a single 20% block is a staircase fitted to ~1,500–2,000 noisy rows; a two-parameter logistic map is smoother and generalises better. The gain is concentrated in 2019-23 and vanishes in 2024-26.

### E3 Cross-model coherence — stack KEEP, blend REJECT
(a) The baseline contradicts itself on 6.6% of games (ML > 0.5 but predicted margin < 0), 11.2% in 2019-20 down to 3.7% in 2025-26.
- (b) Blending with Φ(margin/σ): log loss +0.0060, 2/7 seasons → reject.
- (c) Logistic stack on [logit raw ML, margin/σ]: log loss −0.0077 (CI −0.0134 … −0.0025), Brier −0.0012, 7/7 seasons, ECE 0.0071, contradictions 2.7% → **KEEP**.
- Explanation: the spread model carries information the classifier lacks, and the logistic stack uses it while also replacing the isotonic step (so E2 and E3 largely measure the same improvement).

### E4 Heteroskedastic sigma — total KEEP (tiny), spread REJECT
Changed: σ_i from a second LightGBM on out-of-fold |residual| using pace/3PA/injury/line features, instead of one global σ.
- Total: cover-probability log loss −0.00075 (CI −0.0012 … −0.0003), 5/7 seasons, ECE 0.0369 → 0.0355. Correlation of σ_i with realised |residual| is only 0.03.
- Spread: loss +0.00036 (CI excludes 0 in the wrong direction), 0/7 seasons, correlation 0.01 → reject.
- Explanation: residual size is almost unpredictable from pre-game information; the total gain is real but economically negligible.

## B. Features (all REJECT)
Common pattern: re-running feature selection with a changed pool already moves results by the noise floor above, and none of these exceeded it consistently.

| exp | what changed | ML log loss | spread MAE | total MAE | seasons better (LL / spread / total) | decision |
|:--|:--|--:|--:|--:|:--|:--|
| E5 | 84 pace-normalised, opponent-adjusted features | +0.0038 | −0.009 | −0.010 | 3 / 5 / 3 | REJECT |
| E6 | offseason shrinkage κ∈{1,.75,.5} chosen per fold on validation | +0.0022 | +0.001 | +0.003 | mixed | REJECT |
| E7 | bench-depth replacement BPM instead of −2.0 | mostly identical | +0.002 | +0.014 | 1 (total) | REJECT |
| E8 step 2 | fresh and decayed-chronic absence features | +0.005 | +0.016 | +0.008 | 0 / 0 / 0 (4 folds) | REJECT |

Per-season ML log loss difference vs baseline:
E5: −0.0052, −0.0025, +0.0029, −0.0143, +0.0383, +0.0014, +0.0043 · E6: −0.0053, −0.0048, +0.0276, −0.0154, +0.0124, 0.0000, −0.0007 ·
E7: 0, 0, 0, 0, +0.0034, +0.0003, +0.0108 · E8: n/a for 2019-22 (used for the hypothesis), 0, +0.0204, +0.0030, −0.0025.
Full per-season tables for MAE/Brier are in each `RESULT.md`.

- **E5:** nothing is consistent; the −0.0143 and +0.0383 seasons are the size of the selection noise.
- **E6:** October–November is *not* helped (ML log loss +0.0137, total MAE +0.076 on 1,795 early-season games), so the pre-registered Oct/Nov no-harm check also fails. Season-start carry-over appears to be adequately handled by the existing EWMA.
- **E7:** the new replacement level is selected in only 3 folds; on the 7,144 games with ≥ 1 Out player nothing improves. Injury features carry < 1.2% of gain in every fold.
- **E8:** the hypothesis test was run first on 2019-20 … 2021-22 (2,558 games): market residual falls 0.17 points per unit of home-minus-away *fresh* lost value (CI −0.29 … −0.05) and not for chronic absences (−0.02, CI −0.16 … +0.12). So the market under-prices fresh absences. Adding the features in 2022-26 did not help (spread MAE 0/4 seasons better), which matches the weak, small effect and the very small injury-feature gain. The separate request to fix stale `exp_min` for long-term injured players was **not implemented**: the E8 variant handles chronic absences by decaying their value rather than changing `exp_min`.
- **E9 (no backtest claim):** live inference now reads the official NBA injury-report PDFs (`models/live_rotation_v2.py`); ESPN is only a logged comparator; noon-to-latest downgrade flags are written to the picks file and tested for lookahead; PDF vs ESPN disagreement on the historical overlap and the lookahead-ordering check are in `E09_pdf_injuries_live/RESULT.md`. Caveat: for 16 of 8,824 games the production tip time differs from the late report's own tip time (pre-existing behaviour).

## C. Betting strategy (all REJECT as model changes; E12 adopted as a reporting option)
Bets use opening lines/prices as in the baseline backtests. Baseline unified 5%-edge rule: ML 1,771 bets, ROI +0.74% [−5.7, +7.4]; spread 644 bets, −3.66% [−11.3, +3.7]; total 1,973 bets, +1.12% [−2.9, +5.2]. Every CI includes zero.

- **E10 selection rule.** (a) flat 1.0 pt: spread ROI −1.23% on 3,802 bets (+2.4 vs 5% rule, CI −3.7 … +9.2); total −1.27% (−2.4, CI −5.8 … +1.1). (c) walk-forward threshold: ML −1.12% on 2,755 bets (−1.9, CI −6.7 … +3.2); spread −2.76% (+0.9); total +2.27% (+1.15, CI −1.6 … +4.0). None met the pre-registered rule (ROI gain ≥ 1.0 pt, CI excludes 0, hit rate within 3 pp of mean model P(win)). The 5%-edge bets are heavily overconfident: mean model P(win) 56–63% vs hit rate 48–53%, which is why a larger edge threshold does not mean a better bet.
- **E11 same-game exposure.** 706 of 3,614 bet-games have more than one bet. Independent quarter-Kelly, one best-EV market per game and joint Kelly (Gaussian copula, latent ML/spread ρ ≈ 0.87–0.91) all lose heavily on the realised bet record (log-growth −5.1, −4.2, −3.8; drawdown ≈ 100% of bankroll), because the underlying bets have no demonstrated edge. Paired growth gains of +1.0 and +1.3 have CIs including zero → REJECT; sizing cannot fix a bet set with no edge.
- **E12 line shopping.** `odds_snapshots` is empty in this database, so I used the per-book opening quotes in `nba_odds_opening` (≥ 2 books for 99–100% of test games, 4–5 books on average) with the `odds_clean.py` outlier filter. Settling the 5%-rule bets at the best available line: spread ROI −3.66% → −0.35% (+3.30 pts, CI +1.40 … +5.32); total +1.12% → +2.77% (+1.66, CI +0.61 … +2.71); ML best price +1.44 pts (CI +0.45 … +2.35). Mean improvement 0.27 pts (spread) and 0.45 pts (total); the bets crossing key numbers 3 and 7 gain most (31 bets, +15 to +38 ROI pts, tiny n). The both-sides control at best prices loses (ML −1.34%, spread −1.62%), so the gain is not a book artifact. **Caveat:** these are opening quotes across books, which may not have been simultaneously available; this is an upper bound on achievable shopping, and shopping improves price, not the model.

## Winners together (E13)
Combination: spread from E1; total point forecast from baseline with E4's σ_i; ML = logistic stack on [logit raw ML, E1 margin / σ]. E2 is nested in the E3 stack and was not applied a second time. Evaluated on the 8,899 games common with E1.

| metric | combined − baseline | 95% CI | seasons better |
|:--|--:|:--|:--|
| ML log loss | −0.0085 | [−0.0145, −0.0033] | 7/7 |
| ML Brier | −0.0015 | [−0.0023, −0.0007] | 7/7 |
| Spread MAE | −0.0239 | [−0.048, −0.001] | 7/7 |
| Spread MSE | −0.765 | [−1.445, −0.163] | 7/7 |
| Total MAE/MSE | 0 (unchanged) | – | – |
| Total cover log loss (E4) | −0.00075 | [−0.0012, −0.0003] | 5/7 |

ECE 0.0104 → 0.0070 and contradiction rate 6.6% → 2.3%. **Additivity:** spread and total reproduce their single-experiment numbers exactly, since each market is touched only by its own winners. ML is slightly *more* than additive: E3 alone −0.0076, combined −0.0085, combined-minus-E3 −0.0009 (CI −0.0016 … −0.0001), because E1's lower-error margin improves the stack input. Betting under the 5% rule (descriptive only): bets ML 1,771 → 447, spread 644 → 141, total 1,973 → 1,915; ROI differences are +0.8 [−10.8, +13.3], −5.6 [−22.7, +9.4] and +1.0 [−0.4, +2.5], none distinguishable from zero. The better-calibrated, less extreme probabilities mostly remove bets rather than make the remaining ones win.

## Overall conclusions
1. Kept: E1 for spread, the E3 stack (which supersedes E2's Platt), E4 for total σ (negligible), and E12 as a reporting option.
2. Everything that adds information to the feature set (E5–E8) is inside the selection-noise floor and was rejected. E8 shows a real but small and hard-to-use effect on the market residual.
3. Calibration and structure gains are in log loss/MAE terms (ML −0.0085, spread MAE −0.024); they do not translate into a demonstrable betting edge. The model still trails the closing consensus on log loss and every strategy ROI CI includes zero.
4. Not done / limits: no live out-of-sample test; E9 has no backtest; E12 is based on opening quotes; the feature experiments have a noise floor of ≈ 0.003 ML log loss and ≈ 0.01 MAE from selection alone; kept changes are not yet integrated into `models/train_baseline.py` or the live pipeline.
