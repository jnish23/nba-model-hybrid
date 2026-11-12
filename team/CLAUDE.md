# Team model (moneyline, spread, total)

Scope: `team/`, `models/saved/`, `dashboard/`, `experiments/e01_*.py` ... `e13_*.py` and later E-series, `scripts/` entry points for training and daily inference. Shared inputs come from `core/`.
Not in scope: `props/`, `dashboard_props/`, and anything named `live_*`.

Layout
- `team/`: feature builders (`team_features_v2`, `game_features_v2`), live row assembly (`live_features`, `live_odds`), betting math (`betting`, `betting_simulation`), market-aware components (`market_models`), training and backtests (`train_baseline`, `backtest`), model and config pairing (`model_registry`).
- `models/saved/`: artifacts only. The deployed set is `model_registry.DEPLOYED_SUFFIX`; loading checks the feature list against each pickle.
- `scripts/run_inference_v2.py` is the daily inference entry point and `scripts/run_daily_inference.bat` the scheduled wrapper.
- `dashboard/bet_record.py`: the dashboard's betting record (pure logic, unit-tested); `dashboard/app.py` only displays it.
- `reports/audit_pregame_2026-11/`: the November 2026 audit (`REPORT.md`, one folder per workstream).
- Experiments: `experiments/lib.py` (walk-forward harness), one module per experiment, results under `reports/experiments/<name>/` (`RESULT.md`, `decision.json`).

Things to keep true
- Training and live inference build features with the same code. `tests/team/` and `tests/core/` include train/serve parity tests (`db`-marked ones need `data/nba_stats.db`); run them after any feature change.
- One bet rule for all three markets: probability edge above 5% over the price's break-even. Do not add per-market thresholds without a pre-registered experiment.
- CLV and ROI figures that price bets at the opening line are not valid (the line moves before the model's cutoff). Price entries with `core.odds_clean.load_entry_odds`. Comparisons with the closing line (log loss, MAE) are valid.
- The model reads the OPENING line but a live bet is judged against the CURRENT line, so most live spread and moneyline edges are the line's own move since the open. Priced at a later line the 5% rule hits 50% and loses the vig (E32). Do not quote open-priced backtest ROI as the live rule's record; report closing-line value.
- For claims about the deployed form use `reports/experiments/E13_combined_winners`; `E25_with_odds` is the pre-adoption baseline.
- The daily run must happen after the injury report the model was trained on (noon ET on game day). The task fires at 09:15 Pacific local time; the script waits for the report (up to 90 minutes) and exits 3 if started too early, 4 if games are scheduled but nothing could be predicted, 5 if the injury report is missing or stale (predictions logged, bets withheld).
- Every run appends to `bet_recommendations_log` (opening line read, current line and price, sigmas, bet, own view, line move, injury report used) and unpredicted games to `inference_skipped_games`. `predictions_log_v2` keeps one row per game with the current line only. Never drop, reorder or rename columns of the bet log; add columns at the end through `ensure_bet_log_schema`.
- The dashboard's record is the logged recommendation of the first run that had a line, graded at its own line and price, pushes as pushes. Never recompute it with the `betting.py` default sigmas or a flat -110 when a logged row exists.
- Features must use only information available before tip-off; keep `tests/` leakage guards passing.
- Walk-forward only: tune inside each fold and never on the season being scored.
