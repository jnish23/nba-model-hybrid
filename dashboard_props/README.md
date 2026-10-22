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
