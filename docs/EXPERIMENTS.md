# Research ownership and run records

See EXPERIMENT_CATALOG.json for observed sources, results, decision references and
explicitly unknown historical provenance. Public pregame research belongs with the
models project; venue/bot/scanner research belongs with the private trading project.
Shared core/team/props APIs are consumed through the pinned models wheel.

For each new experiment, record a unique ID, hypothesis, baseline commit, exact input
hashes/cutoffs, environment, command, output directory and decision. Use an isolated
run directory and explicit output paths. Do not overwrite a frozen reference or reuse
an earlier run directory. Promote a result only after comparison with its baseline.
Large models, databases and private trading evidence stay in their artifact stores;
Git contains code, small contracts and reviewable summaries. Retain failed runs when
they explain a decision. Archived status requires a recorded decision, not a filename.

The catalog does not reconstruct missing training data or establish live deployment
readiness. Historical reports and earlier milestone evidence remain immutable.
