# NBA Betting Model: Hybrid Team + Player Model, Evaluated Against the Market

I built a full pipeline for predicting NBA moneylines, spreads and totals:
- data ingestion
- point-in-time features (team form, player ratings, travel and rest, official injury reports)
- gradient-boosted models
- a production schedule that runs daily during the season

I built the project around one question: **can a model built from public data beat the betting market?** I backtested using walk-forward validation using log loss, MAE, ROI and closing line value as metrics, and I treat every change to the model as an experiment with a keep rule written down before it runs.

**[Live dashboard →](https://nba-model-betting-dashboard.streamlit.app/)** 

## What I found

All numbers are walk-forward: each season is predicted by models trained only on earlier seasons, with hyperparameters tuned inside each fold. The comparison uses the 8,357 games from 2019-20 to 2025-26 that have an opening and a closing line.

| 8,357 games | Model before the experiments | Current model | Opening line | Closing line |
|---|---|---|---|---|
| Moneyline log loss | 0.6194 | 0.6108 | 0.6077 | 0.6002 |
| Spread MAE (pts) | 10.73 | 10.70 | 10.68 | 10.52 |
| Total MAE (pts) | 14.71 | 14.71 | 14.57 | 14.26 |

"Current model" is the baseline plus the three changes that passed their keep rules (see [Experiments](#experiments)). The market lines are the de-vigged consensus probability and the quoted spread and total.

- **The market is still more accurate than my model, but the gap on moneyline and spread is nearly gone.** The opening line beat the earlier model on moneyline log loss in 7 of 7 seasons; it beats the current one in 6 of 7. The moneyline gap to the open fell from 0.0117 to 0.0031 and the spread gap from 0.044 to 0.016 points. Totals did not improve (the model is 0.14 points behind the open). Nothing I tried beats the closing line.
- **Where my model disagrees with the opening line, the disagreement is informative.** For bets that clear a 5% probability edge, the line moves toward my model by the close: +0.29 points of closing line value (CLV) on spreads (95% CI +0.10 to +0.49, 609 bets) and +0.73 on totals (CI +0.58 to +0.89, 1,896 bets). These are real but not enough to overcome the ~4.5% vig.
- **That edge does not show up as profit.** Under one bet rule for all three markets (bet when the model's win probability beats the price's break-even by more than 5%), ROI is +0.7% on moneylines (1,771 bets, 95% CI −5.7% to +7.4%), −3.7% on spreads (644 bets, CI −11.3% to +3.7%) and +1.1% on totals (1,973 bets, CI −2.9% to +5.2%). Every interval includes zero. The bets the rule selects are overconfident: the model says 56–63% and they win 48–53%, so a larger edge threshold does not mean a better bet.
- **Injury news is the main thing that moves NBA lines after the open, and the close prices it completely.** These results are from the earlier evaluation in [`reports/injury_report_eval_v3.txt`](reports/injury_report_eval_v3.txt) and were not re-run after the audit and experiments.
  - Reported absences explain 21–25% of open-to-close line movement.
  - The closing line's error is unrelated to them.
  - The usable signal is timing: about +1.8 points of CLV, if you act before the market moves. I run a timestamped odds collector in-season to test whether that is achievable.
  - In the experiments, absences that are new that day are under-priced by the opening line (0.17 points of residual per point of lost player value, 95% CI 0.05 to 0.29), while long-term absences are not. Turning that into model features did not improve the model (see E8).

The reports behind these numbers are in [`reports/`](reports/): [`experiments/FINAL_REPORT.md`](reports/experiments/FINAL_REPORT.md) for the experiments and [`audit_fixes/REPORT.md`](reports/audit_fixes/REPORT.md) for the code audit that preceded them.

## How it works

```
nba_api box scores ─┐
SBR opening/closing ├─► SQLite DB ─► point-in-time features ─► LightGBM models ─► predictions ─► dashboard
NBA injury PDFs ────┘                                                │
                                                                     └─► walk-forward backtest ─► market evaluation (log loss, CLV, ROI)
```

**Features.** Every feature uses only information that was available before tip-off:
- **Team form:** exponentially weighted box-score and efficiency stats (5- and 20-game spans), offense vs. defense matchups, and recent record.
- **Players:** rolling Box Plus/Minus for each player, aggregated to the roster and centered on a trailing league mean.
- **Schedule:** rest, back-to-backs, games in the last 7 days, travel distance, time zones and altitude.
- **Opening lines:** the opening moneyline, spread and total are model inputs. The live path feeds the same opening line the training set used (current lines are used only for edges and bet sizing).
- **Injuries:** I archived and parsed the official NBA injury report PDFs and use the report as published at a fixed time on game day. Each absence is weighted by the player's value and by the empirical probability that a player with that categorization status sits out.
  - Out plays 0% of the time, Doubtful about 2%, Questionable about 57%, Probable about 93%.
  - Live inference reads the same official PDFs. ESPN's injury page is only logged next to them for comparison.
  - Training and live inference build these features with the same code (team form, travel, roster BPM and injury features, then one shared difference/matchup step). `tests/test_parity.py` rebuilds the training set and checks that, for historical dates, every live feature column equals the training-set value.

**Models.** I train one LightGBM model each for win probability, margin and total.
- Feature selection in two steps: I cluster correlated features, then keep only features that beat a shuffled-target importance baseline. Selection stops before the block used for hyperparameter validation.
- FLAML hyperparameter tuning for the deployed models.
- The deployed set uses three market-aware components (`models/market_models.py`), each adopted only after it passed its pre-registered keep rule:
  - **Spread learns the residual against the opening line.** The model is trained on `margin + opening spread` and predicts `−opening spread + f(X)`, so it only has to explain what the line misses. Without an opening spread there is no spread forecast.
  - **Moneyline probability is a logistic stack, not isotonic.** It combines the classifier's raw score with the spread model's margin divided by its residual σ, fitted on a time-ordered holdout. This replaced isotonic calibration: moneyline and spread disagree on 2.7% of games instead of 6.6%, and the output is a smooth probability instead of ~250 distinct values. Games with no margin fall back to Platt scaling on the raw score.
  - **Total uses a per-game σ.** A second model predicts the absolute error of the total from pace, three-point rate, injury load and the lines, using forward-chaining out-of-fold errors. The cover probability uses that σ instead of one global standard deviation.

**Betting rule.** One rule for all markets (`models/betting.py`): bet when the model's win probability exceeds the price's own break-even probability by more than 5%. For spreads and totals the win probability is Φ(point edge / σ) at −110. Each moneyline side is judged against its own price.

**Evaluation.**
- Walk-forward by season (7 folds, 2019-20 to 2025-26): train on past seasons, test on the next. Hyperparameters are chosen inside each fold on a time-ordered validation block, never on a test season.
- I compare every result with the de-vigged market consensus, not raw win/loss records.
- Intervals are day-bootstrap (resampling game days, 1,000 draws) and model comparisons are paired per game.
- Closing line value is the main measure of edge, because it is far less noisy than results.

## Experiments

After a 29-item code audit (train/serve skew, betting math, leakage, data integrity; [`reports/audit_fixes/REPORT.md`](reports/audit_fixes/REPORT.md)), I fixed the post-audit retrained model and its walk-forward backtest as the baseline and ran twelve experiments against it, one at a time. All used the same folds, seeds and data.

- Each experiment's keep rule was written before it ran and never changed ([`reports/experiments/PREREGISTRATION.md`](reports/experiments/PREREGISTRATION.md)). The main rule is "improves in at least 5 of 7 seasons and the paired bootstrap CI excludes zero", with a no-harm check on the companion metric.
- Every new feature has a lookahead test (corrupt all data from a date on and check that features on or before that date are unchanged).
- Re-running the baseline reproduces it bit for bit.
- Re-running feature selection with a different seed moves ML log loss by up to 0.003 and spread MAE by about 0.01. Feature-pool experiments only count beyond that noise.

| | Experiment | Result |
|---|---|---|
| E1 | Market-residual targets | **Kept for spread** (MAE −0.024, 7/7 seasons, CI excludes 0). Total got worse (MAE +0.27), ML inconclusive |
| E2 | Platt vs isotonic vs cross-fitted isotonic | Platt beats isotonic (log loss −0.007); its job is done by E3's stack |
| E3 | Moneyline/spread coherence | **Kept: logistic stack** (log loss −0.0077, 7/7 seasons). Blending with Φ(margin/σ) rejected |
| E4 | Per-game σ | **Kept for total** (cover log loss −0.0008, 5/7 seasons; tiny gain). Rejected for spread |
| E5 | Pace-normalized, opponent-adjusted stats | Rejected |
| E6 | Offseason regression to the mean | Rejected (October–November not helped) |
| E7 | Bench-depth-aware replacement level | Rejected |
| E8 | Fresh vs chronic absences | Hypothesis supported on 2019–22; model features rejected on 2022–26 |
| E9 | Official PDF injuries in live inference | Implemented, no backtest claim |
| E10 | Bet-selection rule (flat points, walk-forward threshold) | Rejected; the 5% rule stays |
| E11 | Same-game exposure (one market per game, joint Kelly) | Rejected: no bet set with this little edge is rescued by sizing |
| E12 | Line shopping across 4–5 books | **Adopted as a reporting option**: best-price settlement improves ROI by +3.3 pts on spreads and +1.7 on totals (CIs exclude 0), using opening quotes that may not all be available at once |

**Winners together.** Spread from E1, total σ from E4, moneyline from the E3 stack on E1's margin (E2 is nested inside E3, so it is not applied twice):

| Combined vs baseline | Difference | 95% CI | Seasons better |
|---|---|---|---|
| ML log loss | −0.0085 | −0.0145 to −0.0033 | 7/7 |
| ML Brier | −0.0015 | −0.0023 to −0.0007 | 7/7 |
| Spread MAE | −0.024 | −0.048 to −0.001 | 7/7 |
| Total cover log loss | −0.0008 | −0.0012 to −0.0003 | 5/7 |

The gains are additive: each market is touched only by its own winners, and the moneyline is slightly more than additive (the better margin improves the stack by a further 0.0009). Calibration error fell from 0.0104 to 0.0070.

On the one out-of-sample check through the production training path (2025-26, trained on earlier seasons, FLAML 120 s per model), moneyline log loss was 0.5896 against 0.6111 for the previous pipeline. That run is one season with different tuning, so it is a sanity check, not a second estimate of the gain.

## Repository layout

| Path | Description |
|---|---|
| `dashboard/app.py` | Streamlit dashboard: today's picks, recent performance, strategy simulator, team and league breakdowns |
| `models/` | Feature builders (`team_features_v2`, `player_features`, `bpm_builder`, `travel_builder`, `game_features_v2`), live row assembly (`live_features`, `live_odds`), injury reports (`injury_report_parser`, `injury_report_features`, `injury_downgrades`, `injury_source_compare`), odds cleaning (`odds_clean`), betting math (`betting`), market-aware components (`market_models`), training (`train_baseline`), model/config pairing (`model_registry`) |
| `scripts/` | DB sync (`update_database`), training set (`generate_training_set`), daily inference (`run_inference_v2`), evaluation (`evaluate_vs_market`, `evaluate_injury_reports`), collectors (`scrape_injury_reports`, `snapshot_odds`) |
| `experiments/` | Walk-forward harness (`lib.py`), metrics and bootstrap (`metrics.py`, `report.py`, `decide.py`) and one module per experiment (`e01_residual.py` … `e13_combined.py`) |
| `models/saved/` | Deployed models (`model_registry.DEPLOYED_SUFFIX`, currently `*_with_odds_v6`) with calibrator, per-game total σ model, feature lists and metadata; loading checks that the feature list matches each pickle. Earlier sets (`_v5`) stay available through `--suffix` |
| `tests/` | pytest suite (feature builders, betting math, market-aware components, train/serve parity, leakage guards) |
| `reports/` | Backtest and market-evaluation outputs; `reports/experiments/` holds each experiment's result, decision and the final report |

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
python models/train_baseline.py                    # walk-forward backtest (fixed neutral hyperparameters), then a no-odds research model
python models/train_baseline.py --skip-backtest --no-research-model --deploy   # train the deployed (market-aware) model set
python -m pytest tests                              # unit tests; the db-marked parity tests need data/nba_stats.db
python scripts/evaluate_vs_market.py --preds models/saved/backtest_predictions_calibrated.csv
python scripts/run_inference_v2.py                 # sync today's data and predict
```

**Experiments:**
```bash
python -m experiments.run_baseline                 # the walk-forward baseline every experiment is compared with
python -m experiments.e01_residual                 # one module per experiment (e01 … e12)
python -m experiments.e13_combined                 # the winners together
```
Each run writes its predictions, `RESULT.md` and `decision.json` under `reports/experiments/<name>/`. The walk-forward harness tunes a fixed grid of eight LightGBM configurations inside each fold; the deployed set tunes with FLAML, so deployed accuracy will not match the experiment numbers exactly.

**In season:** I run the pipeline with two Windows Task Scheduler jobs.
- `scripts/run_snapshot_odds.bat` runs every 30 minutes on game days. It records timestamped lines from each sportsbook and the latest injury report.
- `scripts/run_daily_inference.bat` runs daily at 12:15 PM ET. It syncs the data, predicts the day's games and publishes the dashboard data.

GitHub Actions runs the smoke checks and the pytest suite; the parity tests that need the database skip themselves there.

## Limitations and next steps

- **Beating the closing line from public box scores is not realistic.** The model is now within 0.003 log loss of the opening line on the moneyline but not ahead of it, and 0.011 behind the close. The avenues I see left are timing (information before the market reacts) and softer markets, such as player props.
- **The CLV is not profit.** Lines move toward the model, but the bets it selects are overconfident and no ROI interval excludes zero. Edge thresholds and bet sizing do not fix that (E10, E11).
- **Feature work has hit a noise floor.** None of four feature experiments beat the selection noise. The fresh-absence signal in the opening line is real but small, and the model could not use it.
- **Line shopping is an upper bound.** The E12 gain uses opening quotes across several books, which may not be simultaneously available; the timestamped odds collector will show how much is achievable.
- **Not yet implemented:** a fix for stale expected minutes of long-term injured players, and a live out-of-sample season for the market-aware set.

## Tech stack

Python, pandas, LightGBM, FLAML, scikit-learn, SQLite, pdfplumber, nba_api, Streamlit, Plotly.
