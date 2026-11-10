# Public pipeline baseline validation

Baseline: 0d11f8344e65274b9490b28b674166aba9c30569.

The existing non-DB suite ran in an isolated worktree with temporary stores,
disabled external connections, dummy credentials and production-data guards.
The initial run reported 562 passed and two local Git fixture failures. Both
failures passed under short NTFS scratch paths; the targeted run passed four
publication fixture tests. No application code or global Git setting changed.

All 17 database-backed regression cases passed in a separate restored-data run.
Legacy dataset tests received an explicitly documented, byte-identical copy of
the restored active feature database under their expected filename.

Effective existing coverage: 564 non-DB cases plus 17 data-backed cases passed.
Detailed source hashes, inputs, outputs and restoration evidence are retained in
the owner's local recovery audit. No deployment or external publication occurred.
