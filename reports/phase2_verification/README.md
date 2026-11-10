# Public model Phase 2 verification

**Model refactor checks pass; production deployment remains frozen.**

Pure board logic now belongs to `core/props_board_logic.py`; the dashboard module is
a compatibility shim. The allowlisted generated bundle includes the pure module and
runs without source-only packages. Canonical props defaults/rules live under `config/`
with legacy fallback; the frozen rule remains byte-identical and its Git blob is
protected from newline conversion. Hardcoded checkout defaults were made relative
to each source root. Pickle module lookup aliases and dependency-direction gates pass.

Validation: **568 non-DB + 17 DB cases pass**, zero failures/skips; every one of the
581 baseline cases is preserved. Targeted publication fixtures use only local Git.
All **54 exact output checks** pass: feature/prediction CSVs, simulation arrays and
exports. Only two explicit build timestamp fields differ. The standalone app renders
19 tables and 13 metrics, with logic/feed imports proven to come from the bundle.
All 43 existing model artifacts load and deployed model identity remains unchanged.

See [comparisons](comparison_results.json), [regressions](regression_summary.json),
[module origins](standalone_module_origins.json), [rule Git bytes](frozen_rule_git_roundtrip.json),
and the [public comparison contract](../milestone1/COMPARISON_SPEC.md).

The injury-cache input includes deterministic augmentation from the earlier restore
rehearsal, not the seed's original bytes. [Its derivation](injury_cache_derivation.json)
was reproduced exactly from the captured seed and verified frozen PDFs.

The initial DB copy omitted injury caches and schedules; those harness failures are
retained, then resolved with verified frozen files. No application logic or guard was
weakened, and no external request was sent. Inputs are frozen restore copies, not
production writable state. Separate interpreter prefixes reuse installed dependencies
read-only; clean-machine installation is outside this validation.

Production source/config/model files and task definitions remain unchanged. The
generated daily `todays_picks.csv` legitimately advanced from November 6 to November 7;
it is operational output rather than frozen source. No deployment or deletion occurs.
The original complete repository/storage migration is still outside this mechanical
handoff. Historical training-input and conditional credit-ledger recovery gaps persist.

Reproduce with this repository's `scripts/verification/run_checks.py` and a fresh
report/restore/environment/temp root; actions are tests, db, targeted and smoke.
Use `compare_models.py` against the external fixture location recorded in
[the expected fixture index](../milestone1/model_fixture_manifest.json). Include frozen
injury caches/archives and NBA schedules for all DB parity cases. No private application
code or source checkout is needed by the model subprocesses.
