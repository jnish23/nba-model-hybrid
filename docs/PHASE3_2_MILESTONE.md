# Phase 3.2 public ownership and repository organization

Status: batch 1 public inventory/planning complete; later batches pending. No public runtime changes or removals.
Work only in the isolated models worktree and verification roots. Production remains
read-only; no network, collectors, retraining, publishing, history rewrite or deployment.

Batch 1 inventories experiments/, models/saved/, notebooks/, calibration_metadata.json
and reports/, including tracked/ignored/untracked files, raw hashes and capture stability.
Inventory candidates are not deletion approvals. Record content/dependency ownership and
preservation status; do not port unreviewed private material into this public checkout.

Later batches may accept verified public-owned unique/newer source or historical artifacts
with exact preservation evidence. Large binaries need not become new Git commits; artifact
contracts/external storage remain explicit. Never create a runtime dependency on a sibling
worktree. Public API additions need a new versioned package and reviewed consumer pin.
Earlier milestone reports and historical result records stay immutable.

scripts/probe_live_feeds.py imports dashboard_props.live_probe for dashboard score/box-score
checks and remains public dashboard tooling. It is not executed against external feeds here.

Maintain an experiment catalog (ID, owner/status, baseline, inputs, command, results, decision),
leaving missing historical provenance explicit. After changes, run the full public suite,
DB parity, all five smokes, wheel checks and 54 exact comparisons, plus per-case ownership
accounting and read-only production checks. Batch 1 does not claim those full runs anew.

A separate private handoff records cross-owner dependency and preservation details. Public
reports contain only public inventory/scope evidence. Full ownership cleanup and deployment
readiness remain separate gates.
