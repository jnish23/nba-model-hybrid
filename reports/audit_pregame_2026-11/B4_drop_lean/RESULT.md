# B4: drop the "model lean on every game" displays (owner decision)

## Bottom line
The team dashboard now shows recommended bets only. Every lean display is gone from `dashboard/app.py`; recommended-bet columns, "Why?", the first-run rule,
the what-if sliders and the closing-line-value figures are untouched. Dashboard record numbers are unchanged (see "Verified").

## Removed (dashboard/app.py, line numbers before the edit)
1. Sidebar selectbox "Model lean panel (not bets): ML leans with edge above" and `ml_filter`, `ml_pick_threshold`, `model_metrics`, `model_df` (991-996).
2. Today's Games: `get_ml_lean`, `get_spread_lean`, `get_total_lean`, the `Lean` column and the table column "Model lean on every game (not bets)",
   and the caption sentence describing it (1026-1056, 1076, 1082-1083). The caption now ends with the "Why?" sentence.
3. Recent Performance: the expander "Model lean on every game, last 14 days (NOT bets ...)" with its three metrics (1177-1187), and the sentence
   "The model's lean on every game is further down, labelled as not bets." (1094).
4. `compute_model_accuracy_and_roi` (420-534). Callers checked (grep of *.py/*.md): only the two lean calls above, the deleted test, and the A1 audit script
   `reports/audit_pregame_2026-11/A1_season_bets/06_dashboard.py` (an audit script that extracts the function from app.py by name; it will no longer find it
   in the current tree; its recorded output is not affected). Nothing else on the page used any part of it, so nothing was kept.
5. Imports made unused by the removal: `get_ml_bet`, `calculate_payout` (import line and the comment under "Helper functions moved to team.betting").
   `calculate_bet_ev` was already unused before this change; left alone.
6. Two comments that said "not a lean on every game" (render_live_scores docstring and the `override_pick_sides` line) reworded.
`dashboard/bet_record.py`: nothing there exists solely for the lean displays; unchanged.

## Tests (tests/team/test_dashboard_record.py)
- Removed `test_lean_panel_does_not_grade_a_push_as_a_loss` (it tested the deleted function; push grading of recommended bets is covered by the other tests).
- Renamed the AppTest page test to `..._shows_only_recommended_bets_and_labels_what_ifs`; replaced its `any('NOT bets' in e.label ...)` assertion (that was the
  lean expander) by a walk of every element of the rendered page (main, sidebar, every tab/expander; proto text, labels, help, options, table columns and cells)
  asserting none contains "lean" in any case, with and without the log, at the default view and after the what-if slider moves.
- Added to the same test a second render with `core.season_utils.nba_today` patched to 2026-11-02 (the test DB has no games for the real today, so Today's Games
  was never rendered by that test): asserts the Today's Games table columns are exactly Why?, teams, win %, moneyline, ML/Spread/Total Bet, and no "lean" anywhere.
  The detector was checked on a toy page that does contain "lean" in a selectbox help, an expander table header and a cell: it flags them.
- `test_live_score_picks_are_the_recommended_bets_not_a_lean_on_every_game` unchanged (still asserts the picks are the recommended bets).

## Verified
- `pytest tests/team/test_dashboard_record.py tests/team/test_dashboard_live_scores.py tests/team/test_dashboard_refresh.py -q -p no:cacheprovider`: 74 passed.
  `dashboard/test_dashboard_logic.py` does not exist in this tree (the only such file is `tests/props/test_dashboard_logic.py`, a props test), so the command
  with that path errors with "file or directory not found"; it was left out.
- `pytest tests/team tests/core -q -m "not db" -p no:cacheprovider`: 577 passed, 9 deselected, 0 failed (250 s).
- `verify_record.py fallback data/nba_stats.db`: output identical (modulo CRLF) to `log_verify_fallback.txt`; `verify_fallback.json` and
  `verify_fallback_common_bets.csv` byte-identical. ML 18 bets 7-11, spread 17 bets 10-7, total 32 bets 13-19.
- `ast.parse` OK on dashboard/app.py and tests/team/test_dashboard_record.py.

## Left dead elsewhere (not edited, outside the allowed files)
- `dashboard/live_scores.py`: `pick_sides` still computes a lean on every game (ML only past 5% edge, spread and total always the model's side) and its module
  docstring (line 4) says so. It is still called, and `bet_record.override_pick_sides` replaces every side with the recommended bet, so it is a seed, not dead;
  but the docstring/lean seeding could be simplified later. Not a page element.
- `reports/audit_pregame_2026-11/A1_season_bets/06_dashboard.py` and `RESULT.md` / `B1_dashboard_record/RESULT.md` describe the lean panel (historical).
## Not verified
- No browser/visual check of the page (AppTest only).
