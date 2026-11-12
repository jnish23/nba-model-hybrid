# B1: the dashboard's season record is now the record of the recommended bets

Run everything from the worktree root. Reproduce with `SCRATCH=<empty folder> sh reports/audit_pregame_2026-11/B1_dashboard_record/run_all.sh` (copies the snapshot DB, about 1 GB, into SCRATCH; never touches the production DB or the network).

## Bottom line

1. With `bet_recommendations_log` backfilled (applied to a COPY of the snapshot DB) the dashboard at its default settings shows exactly A1's reconstruction: ML 18 bets 7-11-0 -3.87u, spread 19 bets 11-8-0 +2.02u, total 29 bets 11-18-0 -8.04u, same bet sets, same result on every bet, same profit on every bet (0 differences), in the local-DB path and in the hosted path (published files). Mean CLV beside it: ML +1.36 prob. pts, spread 0.00 pts, total +0.60 pts (A1: +1.36 / 0.00 / +0.60).
2. With no log table at all (the fallback; also what the hosted page shows between the merge and the backfill) it is not exact: ML 18 bets 7-11 -3.87u (exact), spread 17 bets 10-7 +2.09u, total 32 bets 13-19 -7.18u. Cause, bet by bet: spread and total prices are not stored (the page uses -110, A1 the matched book's price), and one total flips on the sigma stand-in. 13 bets differ, 8 spread and 5 total (list below); none of the old causes (wrong sigma, push as loss, slider) is left.
3. Every place that showed a pick on every game is now either restricted to recommended bets (Today's Games bet columns, live-score picks and "Today's record", Recent Performance headline, breakdowns and bets table, Team Breakdown, League Summary) or relabelled "NOT bets" (the old all-games hit rates sit in a labelled box on Recent Performance; Today's Games has a separate lean column).
4. Deviation from the brief: the pure record logic is a NEW file `dashboard/bet_record.py` (not in your allowed list): `team/betting.py` was the only other place for pure helpers and the logic is about 400 lines. If you object, it can be pasted into `app.py` and the tests re-pointed; nothing else depends on the file name.
5. Not run: the backfill against production, any publish, the hosted page itself. `streamlit` 1.66.1 and `plotly` ARE installed in the venv (CLAUDE.md and the brief say not), so I ran the real `app.py` with `streamlit.testing.AppTest` (no exception, table contents read back).

## What changed (file:line)

`dashboard/bet_record.py` (new). Pure pandas, imports only `team.betting` (no scipy). `first_line_runs` :129 (the first-run rule), `attach_recommendations` :180 (adds `rec_<market>_*` columns to the predictions frame), `decide` :237 (recorded vs what-if, per-game points conversion), `grade` :71 / `bet_profit` :89 (push-aware, own price), `clv` :111, `summarize` :272, `override_pick_sides` :337, `read_log_from_db` :361, `deployed_meta_path` :374.
Where a recommendation comes from, per game and market: (1) `run`: the `bet_recommendations_log` row of the FIRST run at which the market had a current line (ML: both prices; spread: home line; total: total line); a later run, including one that bets where the first said no bet, never replaces it; (2) `printed` / `recomputed`: rows from the backfill, with provenance from the new table; (3) `page`: no row, the live rule recomputed from the stored prediction and current line, ML at its own two prices, spread/total at -110 (marked `*`), the deployed model set's sigmas.

`dashboard/app.py`
- :110-190 `_read_recommendations`, `_read_sigmas`, `load_record_context`: the log from the local DB, else `DATA_BASE_URL + bet_recommendations.csv`, else `dashboard/snapshot/bet_recommendations.csv`; sigmas from the repo's `calibration_meta<DEPLOYED_SUFFIX>.json` (local), else `record_meta.json` (published, bundled), else the repo file (tracked in git, so it exists on Streamlit Cloud), else betting.py's defaults flagged "approximate". Every step tolerates absence.
- :389-418 `load_data` attaches the recorded bets to the frame.
- :420-532 `compute_model_accuracy_and_roi` (the lean grader): a push is no longer a loss (None, counted in `pushes`); docstring says it is the lean, not bets.
- :536-595 `simulate_strategies(df, threshold, bet_type, recorded=True, points_edge=None)`: recorded bets graded at the logged line and own price, push as Push (`bet_result` 0); what-if keeps the old rule with the sigma/price/line of the record; `record_of` summary.
- :597-675 `_team_model_record`, `build_league_summary(…, recorded)`; :676-860 `build_team_breakdown(…, recorded, points_edge)`: model bets are the recorded ones, W-L-P.
- :870-915 `render_record_panel` (season record: W-L-P, units, ROI, CLV, "N logged / M recomputed"), `render_record_metrics`, `recommended_bets_table`.
- Sidebar (:955-990): at the default sliders `recorded=True` (the slider default is compared to `round(points_threshold(model-set sigma, 5%), 1)`, so "default" is exactly the live rule); any other value switches every tab to a captioned what-if. Sigma is the model set's, the points conversion is per game and line-aware (`points_to_prob_threshold`).
- Today's Games (:1030-1080): columns `ML Bet / Spread Bet / Total Bet` show only recommended bets with line, price and edge (`*` = -110 assumed) and a separate `Model lean on every game (not bets)` column; live-score picks and "Today's record (recommended bets)" use `override_pick_sides` (:207), so they count bets only.
- Recent Performance (:1091-1185): headline, breakdowns, bets table = recommended bets of the last 14 days; the old all-games grading is in an expander "Model lean on every game … (NOT bets …)".
- Simulation tab (:1189): record panel on top, caption "RECOMMENDED bets: the record above" or "WHAT-IF … NOT the record"; metrics show W-L-P, win rate and ROI over non-push bets; Detailed Log has side, line, price, source.

`scripts/export_dashboard_data.py`: `closing_lines` :54 (last pre-game `odds_snapshots` fetch, cross-book median of spread, total, de-vigged home probability), `load_bet_recommendations` :74 (all log rows, provenance joined, closing line added, empty if no table), `load_record_meta` :100, `write_files` :139 and `publish` :196 take two optional arguments (existing columns, names and the predictions export are unchanged; a missing source removes the stale file); `main` wraps the new part in try/except so a failure never stops the predictions publish.

`scripts/maintenance/backfill_bet_recommendations.py` (new, dry run by default). One row per game 2026-10-20 to 2026-11-11 (170) from A1's `bets.csv` + `predictions_log_v2` + `prediction_explanations` (opening line; the 14 games of Oct 20-21 use the `odds_snapshots` opening proxy for spread/total, nothing for ML). Provenance goes in the new table `bet_recommendations_provenance(game_id, run_ts_utc, prov_ml, prov_spread, prov_total, open_source, note, …)` because the A3 table has no column for it: `printed` for games of Nov 1 on (side and edge read from the daily log's printed picks), `recomputed` before; the note names the price source and the total-sigma stand-in 18.94 (A1's implied median; the per-game sigma is not stored). Idempotent (a game with a provenance row is skipped), refuses to touch a game that has a log row with no provenance row (a real run), only INSERTs, one transaction. Edges of non-bet markets are recomputed at -110.

`tests/team/test_dashboard_record.py` (new, 30 tests; list below).

## Before / after (default settings, 2026-10-20 to 2026-11-11, settled bets)

| | old dashboard (A1 F7) | after, backfilled (local DB and hosted) | after, no log table (fallback) | A1 reconstruction |
|---|---|---|---|---|
| ML | 18, 7-11, -3.87u | 18, 7-11-0, -3.87u, CLV +1.36 pp (17 with a close) | 18, 7-11-0, -3.87u | 18, 7-11, -3.87u |
| Spread | 20, 12-8, +2.91u | 19, 11-8-0, +2.02u, CLV 0.00 pts (19) | 17, 10-7-0, +2.09u | 19, 11-8, +2.02u |
| Total | 33, 13-20, -8.18u | 29, 11-18-0, -8.04u, CLV +0.60 pts (24) | 32, 13-19-0, -7.18u | 29, 11-18, -8.04u |

Commands: `verify_record.py local|hosted|fallback` (outputs `verify_*.json`, `verify_*_common_bets.csv`, `log_verify_*.txt`). Backfilled path source split: ML 6 printed + 12 recomputed, spread 8 + 11, total 10 + 19 (the page says "N printed in the daily log, M recomputed (backfill)"). Local-DB mode has no CLV (see limits).

Fallback, bet by bet (`fallback_differences.md`, `explain_fallback_diffs.py`): spread page-only 0022600015 (home, W +0.91), 0022600089 (away, L), 0022600188 (away, L); A1-only 0022600004 (L), 0031 (W +0.95), 0162 (W +0.93), 0179 (L), 0195 (L). Total page-only 0085 (L), 0118 (W), 0119 (L), 0129 (W); A1-only 0132 (L). All differ by price (A1: -103 to -118) except total 0118 (edge 0.0492 at 18.94, 0.0505 at 18.59). The 60 bets both sides hold differ in profit on 14 (6 spread, 8 total; 0.05u and 0.035u net): A1's price versus -110.

## What the owner sees on the hosted page

After merge and the first publish (the daily `--publish` now also pushes `bet_recommendations.csv` and `record_meta.json` to `dashboard-data`), before any backfill:
- Season record (Simulation tab top) is the fallback column above, labelled "recomputed by this page", spread 17 bets 10-7 and total 32 bets 13-19 (was 20 / 12-8 and 33 / 13-20). The page says how many bets are logged versus recomputed, and that spread/total prices are -110.
- From the first run on the merged code each new game is a `run` bet graded at its logged price, and the count of logged bets grows daily.
- Recent Performance no longer shows a spread/total record on every game as its headline: it shows recommended bets (last 14 days), with the old numbers (ML 4-8, spread 55-49, totals 52-52) in a labelled box "NOT bets". Today's Games shows bets and, separately, the lean.
- The sidebar says the defaults are the recorded bets; moving a slider is a labelled what-if.
After the backfill is applied to production and a publish: the "backfilled" column (A1's numbers, CLV shown).

## Needs the owner's decision

1. New file `dashboard/bet_record.py` (above).
2. Run the backfill on production? It writes 170 rows plus 170 provenance rows (INSERT only, dry run default, refuses real-run rows). The Oct 20-31 rows are A1's reconstruction (30 spread/total bets not exactly recoverable; the total sigma 18.94 and the recovered prices are inferences, flagged in each row's note). Needs a database backup first (the table must exist: it is created by the first run of the merged `run_inference_v2.py`, or by the script itself).
3. Keep the "model lean on every game" boxes (Recent Performance expander, Today's Games column) or drop them. They are clearly labelled now.
4. Rule choice: the bet of a game is its first run with a line. A game whose first run said "no bet" stays "no bet" even if a re-run bets (the daily log would show that later bet). Alternative: last run before tip. I chose first as instructed; the number of games where a re-run changes the decision is not knowable until the log has data.
5. Seasons before 2026-27 are all "recomputed by this page" with the CURRENT deployed model set's sigmas (they were produced by older sets).

## Tests

`pytest tests/team/test_dashboard_record.py`: 30 passed (push grading incl. W-L-P and ROI base; own-price grading and logged line; missing price marked -110; ML own price; first-run rule incl. out-of-order rows, per-market first line, no-bet stays no-bet; page recompute marking; provenance to labels; fallback sigma (a 2.55-pt spread edge bets at betting.py's 13.4, not at 14.287); sigma source chain and flagged defaults; slider conversion on a whole-number line (a number the old half-point formula accepted is rejected, a half-point line unchanged); what-if never changes the record; CLV signs; export with closing lines, in-game rows ignored, missing table, stale file removal; backfill dry run / apply / idempotent / real-run row untouched / opening line / provenance; missing recommendations file and sigma file; simulation push and own price in `app.simulate_strategies`; lean grader push; live-score picks are bets; the real page under AppTest with and without the log, "NOT bets" box present, what-if wording after moving the slider).
`pytest tests/team tests/core -q -m "not db"`: 475 passed, 1 failed, 8 deselected (186 s; `pytest_not_db.out`). The one failure, `tests/team/test_run_time_guard.py::test_constants_are_what_the_notes_say` (`rm.DEFAULT_MAX_WAIT_MINUTES` is 90, the test expects 120), is another workstream's new untracked test, unrelated to the dashboard. Dashboard-related files alone (`test_dashboard_record`, `test_dashboard_refresh`, `test_dashboard_live_scores`, `test_explain`): 107 passed. An earlier full run showed 5 failures that were MY test leaking state into the process (a patched `live_feed.scoreboard` and cached loaders); fixed (monkeypatch + cache clear), reruns clean. `ast.parse` ok on every edited or new Python file (app.py, bet_record.py, export_dashboard_data.py, backfill_bet_recommendations.py, test_dashboard_record.py, the two verification scripts).

## Not verified / limits

- The hosted Streamlit Cloud page itself; the `dashboard-data` publish (never run). I ran `app.py` under `AppTest` against the snapshot DB (with `core.live_feed.scoreboard` stubbed: no network) and the temp DBs of the tests; layout and widget look were not inspected.
- Local-DB mode has no CLV: closing lines come only from the export (the page runs no `odds_snapshots` query). The hosted page gets CLV once the production table is backfilled and published.
- CLV is against the last pre-game snapshot (23-60 min before tip, cross-book median), not the final line; 18 of 170 games have no close.
- "Pre-tip": the log has no tip time, so the first-run rule cannot exclude a manual run after tip. Exporting every run (not only the latest) was needed for the first-run rule; it is a deviation from "latest pre-tip row".
- The backfilled numbers are A1's; they inherit A1's limits (30 bets recomputed with inferred prices/sigma; Oct 23 late-API and Oct 24 API-open games included).
- Backfilled `ml_own_view` / `ml_line_move` assume a constant overround between open and run (away opening price not stored); the opening line of Oct 20-21 games is the snapshot proxy.
- The page's `*` marks -110 assumed; ML never assumes a price.
