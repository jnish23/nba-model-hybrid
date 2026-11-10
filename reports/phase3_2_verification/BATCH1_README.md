# Phase 3.2 batch 1: public inventory report

**Public inventory/planning batch complete; full ownership-cleanup milestone incomplete.**
[Revised scope](../../docs/PHASE3_2_MILESTONE.md).

Inspected source f1cb22d: **707 files**, including 459 tracked and
248 ignored files, in experiments/, models/saved/, notebooks/,
calibration_metadata.json and reports/. Raw hashes, sizes, capture stability and candidate
roles are recorded. New Phase 3.2 evidence, external links and secret contents are excluded.
[Inventory](inventory.json), [experiment index](experiment_catalog.json).

No existing source, artifact, experiment result or historical report was changed or
removed. Only planning/evidence and ignore rules are added. No training, collection,
network probe, publishing or deployment ran. scripts/probe_live_feeds.py imports public
dashboard_props.live_probe and stays public dashboard tooling; it was inspected statically.

The private auditor handoff contains cross-owner comparisons, dependency closure and
research baseline verification. No unreviewed private material was ported into this
public checkout. The index does not invent historical experiment baseline/input identity,
active/archive status or successful execution for command candidates.

The full public suite/DB parity/five smokes/wheel/comparison gates were not rerun for
this read-only inventory batch. They remain mandatory after later source/ownership changes;
previous Phase 2.3 results remain historical evidence. Production was observed read-only;
public head remains 0d11f83, with detailed task/hash checks held privately.

Recommended next step: use the ownership/dependency evidence to guide the adapter and
preservation batches, accepting only verified public-owned material after content review.
Do not overwrite newer public files with historical equivalents or introduce a sibling-
worktree runtime dependency. Production cutover is outside this milestone.
