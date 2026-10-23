# Props dashboard: outline and plan

Date 2026-10-21. Status: plan for approval, nothing built. Context: the team dashboard (`dashboard/app.py`) is a single-file Streamlit app deployed to Streamlit Community Cloud; it reads `predictions_log.csv` from the `dashboard-data` branch (published daily), falls back to a local database or a bundled snapshot, and depends only on streamlit, plotly, pandas, numpy.

## 1. Separate from the team dashboard? Yes: a separate app, same repo, cross-linked

| | Team | Props |
|---|---|---|
| Rows per day | about 10 games | about 55 to 95 qualifying bets (10,145 player-games and 15,593 bets over about 180 game days last season), from about 11,000 bettable quotes |
| Refresh | once a day | three times per game (T-240, T-60, T-5), so intraday |
| Question the user asks | "what does the model pick today?" | "what is worth a look, did last night's bets work, is the signal still alive?" |
| Failure impact | live, in use | new, will change a lot while Phase 6 is built |

Reasons: different cadence and size, and the team app is in use and should not be destabilised by a fast-changing sibling. Streamlit Cloud can serve a second app from the same repo (`dashboard_props/app.py`, its own requirements and its own data branch `props-dashboard-data`). A one-line link in each sidebar joins them. A multipage merge would give one URL but couples the deployments; I would not do it now.

## 2. Handling clutter: three levels, defaults that show almost nothing

The raw universe (8 markets x about 14 rotation players x about 10 games, x 3 times, x 6 books) is unreadable; it never appears by default.

1. **Board (default page): only bets that pass the frozen rule**, one row per player-game-stat, ranked by expected value, upcoming games only, grouped by game in collapsible sections, **top 15 shown, the rest behind "show all"**. Near-miss "watchlist" (EV 0.02-0.03) is hidden unless toggled.
2. **Drill-down on click** (player or game): the full distribution of that stat against the line, all eight markets for that player, the quote ladder across books, line history T-240 to T-5, injury status and minutes outlook.
3. **Market explorer** (search box, opt-in): any player, any market, no rule filter; exploratory only and clearly labelled as such.

Other clutter controls: filters for stat, side, minimum EV, time window to tip; one pricing mode on screen at a time (section 4); settled bets live on a separate page; every table downloadable as CSV.

## 3. Pages

| Page | Content |
|---|---|
| **Board** | countdown banner (next tip), data-freshness badge ("collector last ran 4 min ago, 8 books, credits OK"), the ranked table. Columns: tip time, game, player, stat, side, line, price (generic and best), model P, market P, EV, time first flagged, state (new / line moved / pulled). |
| **Player / Game** | the drill-down above. |
| **Tracker** | open and settled paper bets; cumulative units curve with 95% bands; ROI, win rate, CLV **net of placebo**; by stat, side, book, decision time; reliability plot (model P vs hit rate at the book line). Both pricing modes side by side, always. |
| **Model health** | rolling 21-game-date forecast bias per stat (observed/predicted), CRPS vs the rolling-average baseline, calibration, coverage (share of rotation players with a consensus line), pipeline errors. |
| **Rules and about** | the frozen rule and its hashes (model version, dataset, rule file), the pricing assumptions, the pre-registered gate and where the project stands. |

Wireframe of the Board:

```
[ Props | Team ]            Updated 18:42 ET  ·  8 books  ·  next tip 19:30 (BOS @ NYK, 48 min)
Filters: stat [all v] side [all v] min EV [3%] window [next 4 h v]       Pricing: (o) generic  ( ) best
 GAME        PLAYER        BET             LINE   PRICE     MODEL  MKT    EV     FLAGGED   STATE
 BOS @ NYK   J. Brunson    UNDER assists   6.5    -112      0.55   0.50   +4.1%  T-240     line moved +0.5
 BOS @ NYK   K. Towns      OVER rebounds   11.5   +105      ...
 [ 12 more · show all ] [ watchlist ] [ download CSV ]
```

## 4. Pricing: your "generic bookie / line", and the thing the history says about it

The last three seasons say the signal is real but the profit depends on price shopping: the same bets earn about +4% at the best of six books and about 0% at the median bettable price (+0.8%, -1.7%, -0.5%). A generic line therefore measures the "ordinary" case, which is the conservative and honest default. The dashboard will show, for every bet:

- **Generic** (default): the consensus line with the median bettable price (this is roughly -115 on the typical side, slightly worse than a flat -110). This is the paper-trade number.
- **Best available**: the best quote among the bettable books and which book it was, so you can see what shopping would have added.
- Optional **flat -110** mode if you want the textbook generic bookie.

The Tracker leads with **CLV net of the placebo** because it does not depend on which price you could get, and shows ROI with intervals under both modes. A warning banner shows the number of settled bets and the standard error of ROI (about 1/sqrt(N): after 100 bets it is about 10 points), so a hot or cold week is not over-read.

## 5. Data flow (what has to exist)

The dashboard is the easy part; the live pipeline behind it is Phase 6 and does not exist yet.

1. **Live pipeline** (new): after each snapshot at T-240, T-60, T-5 for each game, build the features at T point in time (the same code as training, with a parity test), run the frozen v3 pipeline for the quoted players (20,000 draws, a few seconds per game), apply the frozen rule file, and write signals. Provenance on every row: code commit, model version, dataset hash, rule file hash. A paper bet is recorded the first time a market qualifies, at that price, and is **immutable** (no overwriting, the lesson of the team audit).
2. **Settlement job** (daily, after the box scores sync): actuals, minutes, status under the three settlement treatments, profit under each pricing mode, CLV to T-5.
3. **Tables** (SQLite `data/props_live.db`, published as small CSVs to `props-dashboard-data`; raw quotes are never published): `prop_signals`, `prop_bets` (immutable), `prop_results`, `model_health_daily`. A compact distribution summary (mean, sd, a few quantiles, or the pmf truncated to about 30 numbers) is stored only for flagged markets, which keeps drill-downs fast and the files small (a few MB per season).
4. **App**: Streamlit, pandas, numpy, plotly (no scipy, no modelling stack), data from the branch with a local-database override, like the team app.

## 6. Build order (each step usable on its own)

| Step | Delivers | Needs |
|---|---|---|
| D0 | The app on **backtest data** (2023-26 bets and results already computed), clearly labelled "backtest": Board layout, Tracker, drill-down, pricing toggle. Proves the layout and clutter controls with real volume. | nothing new |
| D1 | Live signals: the live pipeline for one game day, written to `prop_signals` and the Board | live feature parity work (the largest piece) |
| D2 | Paper-bet log and nightly settlement, Tracker on live data | D1 |
| D3 | Model-health page and alerts for pipeline failures | D2 |
| D4 (optional) | Notifications when a new bet qualifies; manual "I took this" flag stored locally | D1 |

D0 and D1 can run in parallel. Because the 2026-27 season is under way and the paper-trading record only starts when D1 exists, **D1 is on the critical path and D0 is not**.

## 7. Decisions for you

1. Separate app in the same repo with its own data branch (recommended) or a page inside the team app?
2. Generic pricing = **median bettable price** (recommended), flat -110, or both selectable?
3. Hosting: public Streamlit Cloud with the data branch public (like the team app), or local only? Publishing live picks is harmless to the model but exposes the strategy.
4. Start D0 (backtest-data dashboard) now, and D1 (live pipeline) in parallel as the Phase 6 build, or D1 first?
