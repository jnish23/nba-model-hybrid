# Phase 2.3 public source ownership: auditor report

Decision: **public source gate passes**. Ready for audit and continued isolated
organization work. No production deployment, publishing, collection or retraining.

Started at `58a3b11`. Ownership commit `f55c1fe` ports
`scripts/analyze_feature_importance.py` byte-for-byte to its public owner and adds a
boundary test forbidding private packages/source directories. [Port](ownership_port.json).
No public library/data payload changed. Source libraries/dashboard modules import no
private bot, venue or scanner package; private-only source directories are absent.

Full offline suite: **589 passed** (572 non-DB, 17 DB), no skips. The new case is
`tests.test_package_boundaries::test_public_checkout_contains_no_private_packages_or_imports`.
All five model/load/prediction/export/dashboard smokes pass, including 43 model artifacts,
two team predictions, six real props feature rows and independently generated dashboard
bundles. **54 exact comparison checks pass**, with only previously declared build
metadata differences. [Smokes](models_smoke.json), [comparisons](comparison_results.json),
[completion gates](completion_gates.json), [environment](environment.json).

The public wheel builds and installs offline. Its **116 member payloads** (112 package/
data files plus metadata) match the immutable reviewed 0.1.0 wheel. ZIP metadata changes
the archive hash, so this candidate is verification-only. The original consumer pin
`817fc8165d49913fb99c1aba0b99c944c03e315f0dfc86ff6a4c0dfd119972f6` stays unchanged.
[Release equivalence](release_equivalence.json), [build/install](build_candidate/package_contract.json).

The initial verification prefix leaked optional Conda build dependencies and an installed
project-wheel path. An old xarray package conflicted with NumPy 2.4 during Plotly imports,
causing seven dashboard failures and one smoke failure. The clean source-test prefix
has no installed project wheel and uses only existing reviewed dependencies. The full
suite and smokes pass after rerun without model/dashboard code changes. Initial logs,
XML and output fixtures remain in the private evidence archive. Environment reporting
now probes the actual selected child interpreter instead of the parent orchestrator.

Production remains read-only at `0d11f83`; protected code/config/model hashes and task
XML are verified in the private preservation report. Generated daily data may advance.
Earlier evidence is preserved. No process/task/model/data cutover was performed.

For the auditor: give this public report plus the private Phase 2.3 report through an
authorized private channel. The private report contains the complete ownership/removal
ledger, installed-consumer/replay evidence, coverage mapping and production checks.
Raw logs/fixtures, wheel artifacts, final source commits and verified local Git recovery
bundle are sealed in the separate private `phase2_3-20261107` evidence archive.

Recommended next steps: audit this ownership gate; continue bounded report/experiment
archiving with retention manifests; measure performance before optimization. Production
migration needs a separate launcher/artifact/dependency and rollback rehearsal.
