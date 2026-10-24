# E26 - forward test: edge shrinkage and the totals no-odds blend vs entry lines

Registration: `reports/experiments/PREREGISTRATION.md`, section E26 (2026-10-24). Frozen inputs: `frozen/params.json`, `frozen/no_odds_total.joblib` (made by `experiments/e26_freeze.py`, which refuses to overwrite).
Forward games: 2026-10-25 onward. Nothing in the deployed model or the live pipeline is changed.

## Scoring

The (a) tests, from the live log and entry lines:

    .venv\Scripts\python.exe -m experiments.e26_score

B-total also needs point-in-time features for the forward games. Rebuild the training set first (the scorer refuses to run if the feature code changed since the freeze):

    .venv\Scripts\python.exe scripts\generate_training_set.py
    .venv\Scripts\python.exe -m experiments.e26_score --features data\processed\<new training set>.csv

Output: `results_<date>.json`. Looks at 500 / 1,000 / 1,300 games decide nothing; decisive looks are at 1,300 games (A-total) and 2,500 games (A-spread, A-ml, B-total).
Pipeline check on pre-start games: `--start 2026-10-20 --dry-run` (writes `dry_run.json`; 16 games, all eligible, on 2026-10-24).

## Frozen values

| | all seasons (primary) | discovery only (reported) |
|---|---|---|
| beta spread | 0.474 [0.201, 0.780] | 0.044 [-0.425, 0.481] |
| beta total | 0.223 [0.099, 0.347] | 0.226 [0.053, 0.400] |
| beta ML | 0.320 [0.061, 0.581] | 0.134 [-0.280, 0.504] |
| k totals blend | 0.166 | 0.133 |
