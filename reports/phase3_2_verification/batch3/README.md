# Phase 3.2 Batch 3: public models verification

Decision: **PASS** for the public source/test/ownership scope. Production remains
read-only; no package release, dashboard publication, collection or production model retraining
was performed. This supplement contains no private trading evidence.

- **589 passed**: 572 regular and 17 database parity tests; no old case retired/skipped.
- **54 exact comparisons**, five recovery smokes, 43 model artifacts loaded, standalone
  dashboard checks, deployed model hashes and active props bundle preserved.
- Public wheel rebuild/install passes. All **116 member payloads** equal the existing
  pinned 0.1.0 wheel; ZIP-container metadata can differ. No new wheel was promoted.
- Synthetic model-fit test logs now use test temp storage, preserving restored inputs.
  No public model/library calculation changed.
- Public source/results stay with their owner. Existing public artifacts were verified;
  no binary/notebook or older historical source was newly ported into public Git.
- Added a public experiment ownership catalog and run-record guide. Unknown historical
  run provenance stays unknown. The previous report index survives as ../BATCH2_INDEX.md.

[Completion gates](completion_gates.json), [per-case ledger](case_ledger.json),
[exact comparisons](final/comparison_results.json), [smokes](final/models_smoke.json),
[package contract](package_contract.json), [release payload equality](public_payload_parity.json),
[final inventory](final_inventory/inventory.json), [research guide](../../../docs/EXPERIMENTS.md).

Private raw logs, source recovery bundles and full delivery hashes are retained in the
private milestone evidence archive. A trusted auditor should receive the private
Batch 3 report to review the complete cross-repository work. Next: auditor review,
bounded report/diagnostic archiving, then measured performance baselines. Production
reconciliation and deployment require their own comparison/recovery gates.
