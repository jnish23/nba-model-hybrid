# Props model (player props)

Scope: `props/`, `dashboard_props/`, `tests/props/`, `reports/props/`, and the `scripts/*props*` entry points. Shared inputs come from `core/`.
Not in scope: `team/`, `dashboard/`, and anything named `live_*`.

Layout
- `props/features/`: point-in-time feature builders (`build`, `context`, `history`, `injury_at_t`, `market_env`) and leakage `guards`. Decision times are T-240, T-60 and T-5 minutes before tip.
- `props/betting/`: prices, the frozen bet rule (`rules`), evaluation, settlement, export.
- `props/live/`: live path (current quotes, freshness, health, parity, predict, refresh, publish).
- Odds collection: `props/collect.py`, `props/odds_store.py`, `scripts/snapshot_props.py`, backfill via `scripts/backfill_props.py`.
- Plans, gate reports and pre-registrations are in `reports/props/` (for example `dashboard_plan.md`, `live_pipeline_plan.md`, `frozen_config.json`).

Rules
- The work runs in phases with a gate report at each stop. Do not start the next phase or change a frozen config until the owner has approved the gate.
- Held-out seasons stay held out: development used the earlier folds, and later seasons were scored once with frozen settings. Do not tune on them.
- Odds API credits cost money. The backfill is done. State the exact upper-bound cost and ask before any call that spends credits; `scripts/backfill_props.py` needs `--confirm-credits` with the exact number.
- The props stores under `data/` (`props_odds.db`, `props_features*.db`, `props_models/`) are large. Query them; do not copy or load them whole.
- The props dashboard publishes from the main checkout by a scheduled job, and the publisher refuses non-private remotes. Do not run `--publish` paths or change `dashboard_props/app.py`, `dashboard_props/logic.py` or `dashboard/requirements.txt` outside a worktree.
- The vendor's historical tip time can be wrong by about an hour. Use the NBA schedule as the reference clock for T.
