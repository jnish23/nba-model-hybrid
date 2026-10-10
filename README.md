# NBA Betting Model: Hybrid Team + Player Model, Evaluated Against the Market

I built a full pipeline for predicting NBA moneylines, spreads and totals:
- data ingestion
- point-in-time features (team form, player ratings, travel and rest, official injury reports)
- gradient-boosted models
- a production schedule that runs daily during the season

I built the project around one question: **can a model built from public data beat the betting market?** I backtested using walk-forward validation using accuracy, ROI and closing line value as metrics.

**[Live dashboard →](https://nba-model-betting-dashboard.streamlit.app/)** 

## What I found

| Seasons 2019-20 to 2025-26 (7,936 games) | My model | Opening line | Closing line |
|---|---|---|---|
| Moneyline log loss | 0.644 | 0.613 | 0.606 |
| Spread MAE (pts) | 11.06 | 10.61 | 10.48 |
| Total MAE (pts) | 15.65 | 14.55 | 14.26 |

- **The market is more accurate than my model in every season and every market.** My model adds a small amount of information beyond the opening line (t ≈ 2) and none beyond the closing line.
- **Where my model disagrees with the opening line, the disagreement is informative.** Lines move toward my model by the close: +0.26 to +0.42 points of closing line value (CLV) on totals (t ≈ 7–8) and +0.12 to +0.23 on spreads. These edges are real but smaller than the ~4.5% vig.
- **Injury news is the main thing that moves NBA lines after the open, and the close prices it completely.**
  - Reported absences explain 21–25% of open-to-close line movement (t up to −49).
  - The closing line's error is unrelated to them (t ≈ 0).
  - No snapshot of the injury report I tested beats the opening line at a statistically credible level. The early-report win rates are about 53%, with 95% CIs of roughly 43–62%.
  - The usable signal is timing: about +1.8 points of CLV, if you act before the market moves. I now run a timestamped odds collector in-season to test whether that is achievable.


The reports behind these numbers are in [`reports/`](reports/), for example [`market_eval_calibrated.txt`](reports/market_eval_calibrated.txt) and [`injury_report_eval_v3.txt`](reports/injury_report_eval_v3.txt).

## How it works

```
nba_api box scores ─┐
SBR opening/closing ├─► SQLite DB ─► point-in-time features ─► LightGBM (FLAML-tuned) ─► predictions ─► dashboard
NBA injury PDFs ────┘                                                │
                                                                     └─► walk-forward backtest ─► market evaluation (accuracy, CLV)
```

**Features.** Every feature uses only information that was available before tip-off:
- **Team form:** exponentially weighted box-score and efficiency stats (5- and 20-game spans), offense vs. defense matchups, and recent record.
- **Players:** rolling Box Plus/Minus for each player, aggregated to the roster and centered on a trailing league mean.
- **Schedule:** rest, back-to-backs, games in the last 7 days, travel distance, time zones and altitude.
- **Injuries:** I archived and parsed the official NBA injury report PDFs and use the report as published at a fixed time on game day. Each absence is weighted by the player's value and by the empirical probability that a player with that categorization status sits out.
  - Out plays 0% of the time, Doubtful about 2%, Questionable about 57%, Probable about 93%.
  - Training and live inference build these features with the same code. My parity tests match to floating-point precision.

**Models.** I train one LightGBM model each for win probability, margin and total.
- Feature selection in two steps: I cluster correlated features, then keep only features that beat a shuffled-target importance baseline.
- FLAML hyperparameter tuning.
- Isotonic calibration of win probabilities on a time-ordered holdout.

**Evaluation.**
- Walk-forward by season: train on past seasons, test on the next.
- I compare every result with the de-vigged market consensus, not raw win/loss records.
- I use closing line value as the main measure of edge, because it is far less noisy than results.

## Repository layout

| Path | Description |
|---|---|
| `dashboard/app.py` | Streamlit dashboard: today's picks, recent performance, strategy simulator, team and league breakdowns |
| `models/` | Feature builders (`team_features_v2`, `player_features`, `bpm_builder`, `travel_builder`, `game_features_v2`), injury reports (`injury_report_parser`, `injury_report_features`), odds cleaning (`odds_clean`), training (`train_baseline`) |
| `scripts/` | DB sync (`update_database`), training set (`generate_training_set`), daily inference (`run_inference_v2`), evaluation (`evaluate_vs_market`, `evaluate_injury_reports`), collectors (`scrape_injury_reports`, `snapshot_odds`) |
| `models/saved/` | Deployed models (`*_with_odds_v2_injury`) with calibrator, feature lists and metadata |
| `reports/` | Backtest and market-evaluation outputs |

## Running it

**Dashboard (no database needed):**
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```
Without the local database, the app loads predictions from the `dashboard-data` branch, which my daily run updates. Offline, it uses a snapshot bundled in `dashboard/snapshot/`.

**Full pipeline:** this needs `data/nba_stats.db`, which is gitignored (~1 GB, built from nba_api and SBR).
```bash
pip install -r requirements.txt
python scripts/scrape_injury_reports.py            # official injury report archive (2019-20 onward)
python scripts/generate_training_set.py            # point-in-time training set
python models/train_baseline.py                    # walk-forward backtest, then train final models
python scripts/evaluate_vs_market.py --preds models/saved/backtest_predictions_calibrated.csv
python scripts/run_inference_v2.py                 # sync today's data and predict
```

**In season:** I run the pipeline with two Windows Task Scheduler jobs.
- `scripts/run_snapshot_odds.bat` runs every 30 minutes on game days. It records timestamped lines from each sportsbook and the latest injury report.
- `scripts/run_daily_inference.bat` runs daily at 12:15 PM ET. It syncs the data, predicts the day's games and publishes the dashboard data.

GitHub Actions runs smoke checks only, because the database is not in the repo.

## Limitations and next steps

- **Beating the closing line from public box scores is not realistic.** The avenues I see left are timing (information before the market reacts) and softer markets, such as player props.
- **A direct "line adjustment" model** that predicts the opening line's error from injury news should extract more than adding injury features to my general model. That ablation was a statistical null.
- **The early-season cold start** is handled by exponentially weighted averages that carry across seasons. My backtests show the model does not fall further behind the market early in the season. Priors based on rosters and the market are a possible improvement.

## Tech stack

Python, pandas, LightGBM, FLAML, scikit-learn, SQLite, pdfplumber, nba_api, Streamlit, Plotly.
