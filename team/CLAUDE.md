# Team model (moneyline, spread, total)

Scope: `team/`, `models/saved/`, `dashboard/`, `experiments/e01_*.py` ... `e13_*.py` and later E-series, `scripts/` entry points for training and daily inference. Shared inputs come from `core/`.
Not in scope: `props/`, `dashboard_props/`, and anything named `live_*`.

Layout
- `team/`: feature builders (`team_features_v2`, `game_features_v2`), live row assembly (`live_features`, `live_odds`), betting math (`betting`, `betting_simulation`), market-aware components (`market_models`), training and backtests (`train_baseline`, `backtest`), model and config pairing (`model_registry`).
- `models/saved/`: artifacts only. The deployed set is `model_registry.DEPLOYED_SUFFIX`; loading checks the feature list against each pickle.
- `scripts/run_inference_v2.py` is the daily inference entry point and `scripts/run_daily_inference.bat` the scheduled wrapper.
- Experiments: `experiments/lib.py` (walk-forward harness), one module per experiment, results under `reports/experiments/<name>/` (`RESULT.md`, `decision.json`).

Things to keep true
- Training and live inference build features with the same code. `tests/team/` and `tests/core/` include train/serve parity tests (`db`-marked ones need `data/nba_stats.db`); run them after any feature change.
- One bet rule for all three markets: probability edge above 5% over the price's break-even. Do not add per-market thresholds without a pre-registered experiment.
- CLV and ROI figures that price bets at the opening line are not valid (the line moves before the model's cutoff). Price entries with `core.odds_clean.load_entry_odds`. Comparisons with the closing line (log loss, MAE) are valid.
- Features must use only information available before tip-off; keep `tests/` leakage guards passing.
- Walk-forward only: tune inside each fold and never on the season being scored.
