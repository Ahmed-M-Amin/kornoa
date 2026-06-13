# Research: Controlled Data Quality Audit Pipeline

## Decision: Keep Spec 016 audit-only and forbid model training inside the workflow

**Rationale**: The feature exists to improve future training data quality, not to run another experiment branch. Mixing training into the audit would repeat the Phase 3 failure pattern by letting unsafe rows move directly into model fitting before review evidence is inspected.

**Alternatives considered**: Training immediately after the audit was rejected because it collapses data governance and model evaluation into one step. Generating a submission from the audit was rejected because the feature is not an inference or competition-output workflow.

## Decision: Consume precomputed OOF predictions when available, but do not generate them in Spec 016

**Rationale**: Full-dataset label-quality scoring is materially stronger with out-of-sample predictions, but generating fresh OOF predictions is a separate training/evaluation concern. The audit should accept that evidence when it already exists and clearly downgrade reliability when it does not.

**Alternatives considered**: Requiring OOF predictions and failing without them was rejected because it would block useful audit work. Generating OOF predictions inside Spec 016 was rejected because it expands scope into training orchestration.

## Decision: Use deterministic fallback ranking when optional label-quality tooling is unavailable

**Rationale**: The repository must remain runnable without optional third-party tooling. Confidence, correctness, boundary distance, blocked-row evidence, and duplicate/crop signals provide a stable fallback ranking that can be tested locally.

**Alternatives considered**: Making Cleanlab mandatory was rejected because it adds an unnecessary environment dependency for core workflow correctness. Skipping ranking entirely when optional tooling is missing was rejected because it would remove the primary value of the audit.

## Decision: Treat `manual_review_required` as a hard exclusion from the cleaned training manifest

**Rationale**: Manual-review rows represent uncertainty that has not been resolved yet. The cleaned manifest must be conservative, so any unresolved row stays out until a later explicit review-lock artifact exists.

**Alternatives considered**: Allowing manual-review rows to enter the manifest below a score threshold was rejected because the score itself is one of the things under review. Allowing same-run reviewer notes to override bucket decisions was rejected because review locking belongs in a separate feature.

## Decision: Use Spec 014 blocked-row evidence as stricter governance than any later supportive signal

**Rationale**: Spec 014 already invested human review effort to distinguish allowed and blocked hard rows. If a row is blocked there, the audit must not weaken that decision because of a favorable probability score or image heuristic.

**Alternatives considered**: Reclassifying blocked rows as hard-valid based on new model evidence was rejected because it would undermine traceability and governance. Ignoring Spec 014 entirely was rejected because it would discard the most reliable existing row-level review evidence.

## Decision: Make ROI/crop evidence and duplicate-conflict evidence first-class row signals

**Rationale**: The roadmap and constitution both emphasize ROI integrity and defect preservation. Rows with crop damage or duplicate label conflicts are common failure modes that cannot be safely compressed into a generic anomaly score.

**Alternatives considered**: Folding ROI/crop and duplicate evidence into only one combined suspicion score was rejected because it hides the reason a row is unsafe. Deferring those checks to later training was rejected because the point of the audit is to prevent harmful rows from reaching training.

## Decision: Keep visual review outputs optional but provide deterministic review artifacts either way

**Rationale**: Contact sheets and external browsing tools are useful for humans, but the audit’s core value is the master table, bucket outputs, and ranked reports. The workflow must therefore produce CSV/JSON review artifacts even when optional visual tooling is absent.

**Alternatives considered**: Requiring FiftyOne or a browser-based review workflow was rejected because it would make the feature environment-dependent. Dropping visual outputs entirely was rejected because grouped image review materially helps label and crop audits.

## Decision: Record input inventory and reliability status as first-class outputs

**Rationale**: A cleaned manifest is only trustworthy when readers know which evidence was actually available. Explicit inventory and reliability reporting prevents false confidence when OOF predictions, embeddings, or historical row-review files are missing.

**Alternatives considered**: Logging missing inputs only to stdout was rejected because later readers may never see terminal output. Failing hard on all missing optional inputs was rejected because the audit must still provide value with partial evidence.
