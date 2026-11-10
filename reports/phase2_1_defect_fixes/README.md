# Phase 2.1 defect fixes

Verified against models `0080ce0` and trading `43a2c52`, before Phase 2.2.

- Export: reproduced the default `rule_path=None` TypeError. Resolve the rule once and use that same file for JSON and SHA-256. Minimal synthetic export now passes.
- Configuration: this revision has no `write()` function; `pin()` writes directly. Added a resolved-target writer, including parent creation, and routed `pin()` through it. Tests cover None, an explicit nested target and legacy fallback.
- Submarkets: reproduced the package import failure. Added the CLI's own root bootstrap and qualified sibling import; package import and direct runpy import work from an unrelated directory without running collection.

Audit checksum correction: the supplied `b217f7808764c85b832a85f5d155f328ca1fcbe5f1f58195181e469f86c71790` does not match the reviewed artifacts. Both frozen copies, the config pin and the `0080ce0` Git blob match `b597763ab3b13fd9cf295b1de802255ac87fe6e9bbf065103171db49eec2b6fb`. Tests assert the verified value. No rule bytes were changed.

Before: models 2 failed/3 passed; trading 1 failed/3 passed. After: models 5 passed; trading 4 passed. Initial traces, the checksum mismatch and final JUnit/logs are retained locally. All runs use the offline guard and isolated temporary files. Production writes are denied.
