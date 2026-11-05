# E30 - forward test of the four E28 opening-line candidates

Registration: `reports/experiments/PREREGISTRATION.md`, section E30 (2026-10-24). Frozen inputs: `frozen/params.json` (made by `experiments/e30_freeze.py`, which refuses to overwrite). Hashes: `reports/experiments/E28_tracking/registration.sha256`.
Forward games: regular season from 2026-10-25. Nothing in the deployed model, the live pipeline or the database is changed; the scorer is read-only.

## Scoring (about 2 minutes: it rebuilds the tracking features from the database)

    .venv\Scripts\python.exe -m experiments.e30_score --allow-code-change

Output: `results_<date>.json`. Decisive looks use the first 2,460 and 4,920 eligible games of each test (Holm over the four tests, one-sided alpha 0.025 each look); looks at 600 and 1,230 decide nothing.
`--allow-code-change` is required since amendment 1 (PREREGISTRATION.md, E30): `features_tracking.py` was changed after the freeze (the F3 pace column), with a measured effect of 0.001 points on the T2 signal; the results file records it. `dry_run_after_amendment.json` is the pipeline check with the current code (`dry_run.json` is the original, untouched).
The scorer stops if `features_tracking.py`, `e30_signals.py` or `core/travel_builder.py` changed since the freeze (score from a checkout of the registration commit, or pass `--allow-code-change` and report it).
Pipeline check on pre-start games: `--start 2026-10-20 --dry-run` (writes `dry_run.json`; 18 games, far too few to estimate anything).

## Tests

T1 F1 running load -> margin; T2 F3 physical previous game -> margin; T3 F4 schedule density -> margin; T4 F5 interactions -> total.
Expected slopes are 0 to 0.6, not 1 (archive-season slopes 0.55, 0.62, 0.28, 1.06 with SEs 0.2-0.4); power is low (about 50% at 2,460 games if the slope is 1), so a "not supported" result is not evidence of no effect.
