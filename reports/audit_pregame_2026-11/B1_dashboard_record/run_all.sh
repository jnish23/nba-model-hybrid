#!/bin/sh
# B1 verification, from the worktree root C:/Users/jorda/nba-pregame-audit. Nothing here touches the production database or the network:
# the backfill is applied to a COPY of the worktree's snapshot database (about 1 GB, put in $SCRATCH, outside the repo), the export goes to $SCRATCH/export.
# usage: SCRATCH=<an empty scratch folder> sh reports/audit_pregame_2026-11/B1_dashboard_record/run_all.sh
PY=C:/Users/jorda/nba-model-hybrid/.venv/Scripts/python.exe
D=reports/audit_pregame_2026-11/B1_dashboard_record
mkdir -p "$SCRATCH/export"
cp data/nba_stats.db "$SCRATCH/copy_nba_stats.db"
# 1. backfill: dry run, then apply to the copy, then again (idempotent: inserts nothing)
$PY scripts/maintenance/backfill_bet_recommendations.py --db "$SCRATCH/copy_nba_stats.db"          > $D/log_backfill_dry_run.txt 2>&1
$PY scripts/maintenance/backfill_bet_recommendations.py --db "$SCRATCH/copy_nba_stats.db" --apply  > $D/log_backfill_apply.txt 2>&1
$PY scripts/maintenance/backfill_bet_recommendations.py --db "$SCRATCH/copy_nba_stats.db" --apply  > $D/log_backfill_apply_again.txt 2>&1
# 2. the export files the hosted page reads (write_files into $SCRATCH/export; same functions the daily --publish uses)
$PY - <<PYEOF > $D/log_export.txt 2>&1
import importlib.util, sys
sys.path.insert(0, 'scripts')
spec = importlib.util.spec_from_file_location('exp', 'scripts/export_dashboard_data.py'); exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
db = r"$SCRATCH/copy_nba_stats.db"
rec = exp.load_bet_recommendations(db)
exp.write_files(r"$SCRATCH/export", exp.load_predictions(db), None, recommendations=rec, record_meta=exp.load_record_meta())
print('exported', len(rec), 'recommendation rows;', 'closing line present for', int(rec['close_total'].notna().sum()))
PYEOF
# 3. the dashboard's season record at default settings in the three data situations, against A1
$PY $D/verify_record.py local    "$SCRATCH/copy_nba_stats.db" > $D/log_verify_local.txt 2>&1     # local database with the table
$PY $D/verify_record.py hosted   "$SCRATCH/export"            > $D/log_verify_hosted.txt 2>&1    # no database: published files
$PY $D/verify_record.py fallback data/nba_stats.db            > $D/log_verify_fallback.txt 2>&1   # no table at all (the worktree snapshot DB, read only)
$PY $D/explain_fallback_diffs.py                              > $D/log_explain_fallback.txt 2>&1
# 4. tests
$PY -m pytest tests/team/test_dashboard_record.py -q -p no:cacheprovider > $D/pytest_dashboard_record.out 2>&1
$PY -m pytest tests/team tests/core -q -m "not db" -p no:cacheprovider    > $D/pytest_not_db.out 2>&1
