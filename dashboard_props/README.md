# Props dashboard (paper-trading view)

Separate from the team dashboard (`dashboard/`). D0 shows BACKTEST data: the frozen rule replayed on 2023-24 to 2025-26.

```bash
# 1. build the data once (reads the odds store, the saved predictions and the frozen rule; a few minutes)
python scripts/export_props_dashboard_data.py

# 2. run it, bound to this machine only (use the anaconda Python that has streamlit)
streamlit run dashboard_props/app.py --server.address localhost
```

Data lives in `data/props_dashboard/` (gitignored): `bets.csv`, `pmf.json`, `books.json`, `season_summary.csv`, `reliability.csv`, `bias_monthly.csv`, `meta.json`.
Set `PROPS_DASH_DATA` to point the app at another folder. Plan: `reports/props/dashboard_plan.md`; live pipeline: `reports/props/live_pipeline_plan.md`.

## Private hosted copy (named viewers only)

The live record can be viewed away from this PC through Streamlit Community Cloud, from a **private** GitHub repository (`jnish23/nba-props-dashboard`) that holds a copy of the app and the exported data. Nothing is published to the public model repository.

- `scripts/publish_props_dashboard.py` refreshes the export (settled bets, **odds now** for pending bets) and pushes it to the private repo; it refuses to push unless GitHub confirms the repo is private.
- Automatic: the "NBA Props Settle" task (10:30 local) settles finished bets, then publishes. By hand, any time: double-click `scripts\run_props_publish.bat` (or run `python scripts/publish_props_dashboard.py`). The deployed app picks the new data up within a few minutes.
- The odds collector snapshots about every 30 minutes, so "odds now" is never fresher than the last snapshot; a refresh at 6 pm shows 6 pm odds, not the morning's.
- One-time deployment (needs your Streamlit login, so it is done in the browser): share.streamlit.io -> sign in with GitHub and allow access to private repositories -> New app -> repository `jnish23/nba-props-dashboard`, branch `main`, main file `dashboard_props/app.py` -> Deploy. Then the app's Share menu -> "Only specific people can view this app" -> add the viewers' emails. Streamlit's free tier limits how many private apps you can have; the Share dialog shows what applies to you.
