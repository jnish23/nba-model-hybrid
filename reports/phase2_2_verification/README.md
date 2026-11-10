# Phase 2.2 public package verification

**Source contract verified for review. Production writes are denied; nothing is deployed.**

Defects were fixed first in `cadbfef`; see [defect findings](../phase2_1_defect_fixes/README.md).
The export now hashes its resolved rule. A None-safe configuration writer also creates
its resolved parent and is used by `pin()`. The audit's supplied rule checksum differs
from the reviewed frozen files/config/Git blob; the verified pin remains unchanged.

`nba-pregame-models==0.1.0` packages only `core`, `team`, pregame `props` and the
explicit feed override CSV. The wheel contains 116 files and excludes datasets,
model artifacts, launchers, dashboards and private code. Offline build/install uses
a fresh prefix and existing dependencies; imports resolve from that prefix under `-I`,
without a source checkout on the path. See [package evidence](package_contract.json)
and [package contract](../../PACKAGING.md).

588 unique model tests pass, including 17 DB parity cases; no failures/skips. Every
previously passed case is preserved. The four targeted cases pass. All five model
smokes pass: 43 historical artifacts load, deployed model identity is unchanged,
team features/predictions and seeded props probability arrays match, exports match,
and the standalone dashboard bundle renders. No collector or publisher is invoked.

All 54 exact comparisons pass. Only the two explicit `built_at_utc`
metadata fields differ; arrays use exact dtype/shape/bytes, with no comparison tolerance.
See [comparison results](comparison_results.json), [regressions](regression_summary.json),
[smokes](models_smoke.json) and [fixture setup](fixture_setup.json).

All 71 earlier report files retain their recorded hashes.
Initial package-build metadata failed because the runtime environment did not contain
setuptools; the final build reuses the existing local setuptools/wheel installation.
An initial verification invocation passed an extra CLI owner argument; corrected runs
are recorded separately. Initial logs are retained and archived, not treated as model failures.

Production static code/config/models/tasks were checked privately and remain unchanged;
daily generated output continues operating. Production deployment and storage cutover
remain frozen. Model/config/data paths remain deployment inputs; this wheel alone does
not replace the production artifact store. Python 3.12 is exercised using read-only
installed dependencies, not a clean-machine installation or complete version matrix.

Reproduce in a fresh workspace with `scripts/verification/verify_package.py` (explicit
local runtime and build dependency directories), `run_checks.py` (tests, db, targeted,
smoke), then `compare_models.py` against the frozen baseline. Raw XML/logs/fixtures are
ignored by Git and preserved in the private milestone archive, with public source/results
owned here. Earlier reports are not overwritten.
