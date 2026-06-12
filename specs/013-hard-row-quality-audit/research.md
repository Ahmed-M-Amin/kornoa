# Research: Hard Row Quality Audit

## Decision: Implement the audit as a dedicated offline analysis module

**Rationale**: The audit is a new roadmap phase with its own row-level contract, summary outputs, and safety rules. A dedicated module keeps the work isolated from training, submission, and same-split evaluation code while still allowing helper reuse.

**Alternatives considered**: Extending `src.analysis.v2_2_same_split_eval` was rejected because Phase 2 has a different output contract centered on human-audit categorization rather than on model-vs-model comparison. Embedding the audit into training code was rejected because this phase must remain analysis-only.

## Decision: Reuse existing hard-example and analysis context as governed inputs

**Rationale**: The repo already has hard-example loaders and prior V2.1/V2.2 analysis flows that can provide stable `image_id`, prediction context, and optional review evidence. Reusing those sources reduces duplication and preserves consistency with Spec 012.

**Alternatives considered**: Building a fresh input format with no relationship to prior analysis was rejected because it would require manual reconciliation and weaken reproducibility.

## Decision: Use exactly one primary audit category per row

**Rationale**: The roadmap Phase 2 requirement is to split the hard pool into actionable groups. Exactly one primary category per row makes the summary totals clear and prevents double-counting in the ranked failure-mode output.

**Alternatives considered**: Allowing multiple primary categories per row was rejected because it would make prioritization and summary totals ambiguous. Free-form labels only were rejected because they are harder to test and compare.

## Decision: Preserve optional detector and image-quality evidence as enrichments, not prerequisites

**Rationale**: Some rows may already have detector or quality signals from prior analysis, but the audit must remain valid even when those inputs are absent. Recording the absence explicitly is more robust than blocking the workflow.

**Alternatives considered**: Requiring detector or crop-quality evidence for every row was rejected because current artifact coverage is likely incomplete and would delay the audit. Ignoring optional evidence entirely was rejected because it would remove a useful diagnosis path.

## Decision: Write both a row-level audit table and an aggregated failure-mode summary

**Rationale**: The row-level table supports traceable review, while the summary is needed to decide the next branch in the roadmap. Both are necessary for Phase 2 to be decision-ready.

**Alternatives considered**: Summary-only reporting was rejected because it loses per-row explainability. Row-only reporting was rejected because it does not directly support prioritization across 442 rows.

## Decision: Keep relabel findings advisory only

**Rationale**: The roadmap explicitly allows the audit to identify likely mislabeled rows while leaving final policy decisions about relabel use to later stages. Advisory findings preserve value without violating competition constraints or prematurely changing the dataset.

**Alternatives considered**: Auto-relabeling rows during the audit was rejected because it would mix analysis with data mutation. Avoiding any mislabeled-row category was rejected because it would hide one of the most important possible blockers to `0.98+`.
