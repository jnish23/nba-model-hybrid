# NBA betting models: working notes

This repo holds two pre-game model projects plus shared code. Work on one area at a time and ignore the rest.
Each area has its own `CLAUDE.md` (loaded when you work in that folder).

| Area | Folders | Reports | Notes |
|---|---|---|---|
| Team model (moneyline, spread, total) | `team/`, `models/saved/`, `dashboard/`, `experiments/e*.py` | `reports/experiments/`, `reports/audit_fixes/` | `team/CLAUDE.md` |
| Props model (player props) | `props/`, `dashboard_props/` | `reports/props/` | `props/CLAUDE.md` |
| Shared | `core/` (ingestion, injuries, odds cleaning), `scripts/`, `tests/` | | used by both |

`reports/` is large. Do not read it wholesale: start from the README links (`reports/experiments/FINAL_REPORT.md`, `reports/audit_fixes/REPORT.md`) or the specific report you are asked about.
If you meet `live_*` code or reports that are not described here, they belong to a different project; leave them alone unless asked.

## Environment
- Python: `C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe` (3.12, pandas, lightgbm, flaml, pytest). Plain `python` on PATH is 3.9 without the dependencies.
- From Bash use Windows-style paths (`C:/Users/...`), not `/c/Users/...`.
- Tests: `.venv/Scripts/python.exe -m pytest tests -q`; add `-m "not db"` to skip tests that need `data/nba_stats.db`.
- `streamlit` and `seaborn` are not in the venv; check dashboard files with `ast.parse`.
- `data/` is gitignored (databases, odds snapshots, props stores up to several GB). Every worktree reaches it through a `data` link or copy. Never delete a worktree's `data` folder recursively: unlink it first.

## Working rules
- The main checkout (`C:\Users\jorda\nba-model-hybrid`) must stay on `main`. Windows Task Scheduler jobs (daily inference, odds and props snapshots, props signals/settle/publish, weekly retrain) run scripts from it, and one of them publishes the props dashboard from its working tree. Do branch work in a separate git worktree; never run a `--publish` mode from a worktree.
- Stage explicit file paths with `git add <file>`, never `git add <dir>` or `-A`: the working tree has many untracked model artifacts and reports that belong to the owner.
- Write commit messages with `git commit -F - <<'EOF' ... EOF`; backticks inside `-m "..."` are executed by the shell.
- Never print API keys. They live in the gitignored `.env` (`core/env.py` loads it). The Odds API spends paid credits: state the upper-bound cost and ask before any call that is not free.
- The deployed model set is `model_registry.DEPLOYED_SUFFIX`. Do not retrain, regenerate predictions, or change the suffix without the owner's approval.
- Experiments follow the same process: write the keep rule down before running, report the result whatever it is, and keep one folder per experiment under `reports/experiments/<name>/`.
- After editing a Python file through a script, run `ast.parse` on it. The Write tool strips trailing whitespace, so use the Edit tool for changes that depend on exact whitespace.
