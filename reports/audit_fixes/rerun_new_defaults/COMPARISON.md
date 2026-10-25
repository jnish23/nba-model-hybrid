# E28 / E29 re-run under the post-audit defaults

Run 2026-10-24 by `run.py` (this folder). Nothing under `reports/experiments/` or `data/processed/experiments/` was touched; the outputs are in
`reports/experiments/` here (prediction CSVs and `data/` are not committed - `python reports/audit_fixes/rerun_new_defaults/run.py variant regen screen explore e29`
rebuilds them, ~35 minutes). Changes against the original runs: the references `E25_with_odds` / `E25_residual` are regenerated with the harness defaults (residual
feature selection, 1,300-row calibration block, refit on model_train + calibration, Platt ML), and the variant file carries the corrected `trk_prev_opp_pace`
(the previous game's pace shock; only the three `trk_prev_opp_pace` columns differ from the saved variant). The registered scripts are not edited; only their
paths are re-pointed, and E29's pre-written feature lists are passed to the harness with `features_from`. This is a sensitivity re-run, not a new registration:
the hashes in `E28_tracking/registration.sha256` no longer match `features_tracking.py`.

## E28 screen: tiers old -> new

Unchanged except the rows below. (a) = residual of the deployed-form out-of-sample prediction; (b) = residual against the opening line.

| test | old | new | why |
|---|---|---|---|
| (a) F1 total | CANDIDATE (find p 0.0044, gamma 0.96, q 0.015) | CANDIDATE (p 0.0207, gamma 0.59, q 0.060) | weaker, same tier |
| (a) F3 margin | not confirmed (find p 0.088) | fails find step (p 0.125) | corrected pace column |
| (b) F3 margin | CANDIDATE (find p 0.084, gamma 0.52) | **fails find step (p 0.116)** | corrected pace column (F3 contains it) |
| (b) F4 margin | CANDIDATE (q 0.082) | not confirmed (q 0.110) | own numbers identical (p 0.065, gamma 0.42); the BH correction changed because F3 margin dropped out |
| (b) F1 margin, F5 total | CANDIDATE | CANDIDATE, numbers identical | do not depend on the reference or the pace column |

Mechanism (c): no change. By the registered rule the "fatigue story" is still "supported" for F1 and F2 (pace and, for F1, FT%), and the exploratory pace-controlled
version still removes it (F1 pace gamma -0.09, F2 pace -0.01; F1 total gamma 0.12, 95% CI [-2.0, 2.0]; old 0.06).

## E29: forced inclusion on the total (arm - reference; negative = better)

| arm | old total MAE | new total MAE | verdict old / new |
|---|---|---|---|
| E29_F1_total (registered) | +0.0140 [-0.003, +0.032], 2/7 | +0.0032 [-0.019, +0.029], 5/7 | REJECT / REJECT |
| E29x_pace_total | +0.0107 [-0.003, +0.026], 1/7 | -0.0004 [-0.013, +0.012], 3/7 | REJECT / REJECT |
| E29x_F1pace_total | +0.0176 [+0.002, +0.035], 1/7 | +0.0021 [-0.022, +0.029], 5/7 | REJECT / REJECT |
| E29x_F1pool_total (fresh selection) | -0.0136 [-0.043, +0.013], 3/7 | **+0.0357 [+0.010, +0.061], 2/7** | REJECT / REJECT |

ML and spread reproduce the reference exactly in the three forced arms (validity checks pass). The F1 columns take 2-12% of total-model gain in the forced arm and
about 0 in the fresh-selection arm, which never picks more than two F1 columns. The fresh-selection arm differs from the reference by selection alone (list overlap
0.39-0.63): total MAE moves by +0.036 here and -0.014 before, both far above the 0.005 seed-only floor, which was measured on the old selector and a fixed pool.
Treat any comparison that changes the candidate pool as carrying about +-0.03 total-MAE selection noise.

## Reading

No decision flips for the team models: F1 on the total is still the only (a) candidate, it is still pace-confounded, and it is still rejected by the walk-forward test.
The corrected pace feature no longer predicts the total on its own (+0.011 -> -0.000). Of the four opening-line candidates frozen for the E30 forward test, F1 margin and F5 total
are unchanged; F3 margin no longer passes the find step and F4 margin is borderline only through the multiplicity correction.
