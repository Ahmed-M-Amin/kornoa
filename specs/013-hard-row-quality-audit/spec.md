# Feature Specification: Hard Row Quality Audit

**Feature Branch**: `[013-hard-row-quality-audit]`

**Created**: 2026-06-12

**Status**: Draft

**Input**: User description: "Proceed to the next roadmap step after Spec 012: Phase 2 Label And Data Quality Audit For The Hardest 442 Rows."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Audit The Hardest Rows (Priority: P1)

As the model developer, I need a structured audit over the hardest 442 validation rows so I can determine whether the main barrier to higher locked-split performance is label quality, preprocessing quality, detector evidence quality, or genuinely hard visual content.

**Why this priority**: The roadmap states that the hard-example subset is the main blocker to `0.98+`. Without a disciplined audit of those rows, later model training would continue to optimize around noise instead of around the real failure modes.

**Independent Test**: Can be fully tested by loading a fixed hard-row input set, generating one row-level audit table for all audited rows, and confirming that every row receives exactly one primary audit category plus recorded evidence fields.

**Acceptance Scenarios**:

1. **Given** the hard-row source set is available, **When** the audit is run, **Then** every hard row appears exactly once in the audit table with stable `image_id` identity.
2. **Given** one hard row is reviewed, **When** the row is classified, **Then** it receives one primary audit outcome from the approved audit categories.
3. **Given** the audit completes, **When** row coverage is checked, **Then** no audited hard row is missing from the output table.

---

### User Story 2 - Capture Evidence For Each Audit Decision (Priority: P2)

As the model developer, I need each audited row to record the evidence behind its classification so I can later justify whether a row should remain unchanged, be relabeled, be excluded from special handling, or be investigated further.

**Why this priority**: An audit without explicit evidence would be too subjective to support later training, reporting, or insight claims. The evidence layer is what makes the audit reproducible and useful downstream.

**Independent Test**: Can be tested independently by verifying that sample audited rows preserve the required evidence fields, including original prediction context, hard-example type, and audit rationale, without requiring model retraining.

**Acceptance Scenarios**:

1. **Given** a row is marked as likely ambiguous, **When** the audit output is reviewed, **Then** the reason for ambiguity is explicitly captured.
2. **Given** a row is marked as likely preprocessing issue or detector-evidence issue, **When** the row record is inspected, **Then** the relevant supporting evidence fields are present or explicitly marked absent.
3. **Given** optional evidence such as detector or crop-quality indicators is unavailable, **When** the row is audited, **Then** the audit remains valid and records the missing evidence transparently.

---

### User Story 3 - Produce A Decision-Ready Audit Summary (Priority: P3)

As the model developer, I need an aggregated audit summary across the 442 rows so I can decide whether the next investment should be relabeling, preprocessing repair, detector repair, or stronger classifier experiments.

**Why this priority**: The roadmap requires Phase 2 to guide future candidate work, not just generate row notes. A summary is needed to rank the fixable failure modes and to justify the next training branch.

**Independent Test**: Can be tested independently by generating a summary report from the audited rows and confirming that category counts, ranked failure-mode groups, and action-oriented outputs match the row-level audit table.

**Acceptance Scenarios**:

1. **Given** the row-level audit table exists, **When** the summary is generated, **Then** it reports counts for each primary audit category.
2. **Given** the summary is reviewed, **When** the most important blockers are listed, **Then** the output identifies a ranked set of fixable failure-mode groups.
3. **Given** the audit identifies likely mislabeled or ambiguous rows, **When** the summary is consumed downstream, **Then** those rows are clearly separated from rows that are likely correct but intrinsically hard.

### Edge Cases

- If the hard-row source list is missing, the audit must fail clearly before creating outputs.
- If a row appears more than once in the hard-row source list, the audit must normalize and reject duplicate identity rather than double-counting it.
- If a hard row is missing optional detector or image-quality evidence, the audit must still proceed and record the absence.
- If a row could plausibly fit multiple issue types, the audit must still assign exactly one primary category while preserving secondary notes.
- If competition rules do not allow relabeling or data cleanup for final training, the audit must still identify likely label issues without applying them automatically.
- If any test labels or public leaderboard feedback are used to justify audit outcomes, the feature is invalid.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST load the governing hard-row input set for the Phase 2 audit and preserve stable row identity by `image_id`.
- **FR-002**: The system MUST reject duplicate hard-row identities in the audit input set.
- **FR-003**: The system MUST produce one row-level audit record for each governed hard row.
- **FR-004**: Each audited row MUST be assigned exactly one primary audit category from: `likely_correct_but_hard`, `likely_ambiguous`, `likely_mislabeled`, `likely_preprocessing_issue`, or `likely_detector_evidence_issue`.
- **FR-005**: The system MUST preserve the original comparison context needed for review, including target, baseline prediction context, candidate prediction context if available, and hard-example type.
- **FR-006**: The system MUST record a concise audit rationale for each row.
- **FR-007**: The system MUST support optional evidence fields for detector evidence, crop quality, blur, brightness, or similar image-quality signals without making them mandatory for audit completion.
- **FR-008**: The system MUST produce an aggregated audit summary that counts rows by primary audit category.
- **FR-009**: The system MUST produce a ranked summary of fixable failure-mode groups derived from the row-level audit results.
- **FR-010**: The system MUST identify rows that are likely label issues separately from rows that are likely correct but difficult.
- **FR-011**: The system MUST remain analysis-only and MUST NOT train models, create submissions, or apply relabeling changes automatically.
- **FR-012**: The system MUST prohibit test-label use, public leaderboard tuning, and any audit logic that depends on hidden-test outcomes.
- **FR-013**: The audit outputs MUST be sufficient to guide the next classifier, preprocessing, detector, or data-quality decision without re-reviewing every hard row manually.

### Constitution Alignment *(mandatory)*

- **Binary Output**: The audit preserves the `0 = Reusable` and `1 = Not Reusable` decision framing by reviewing difficult rows in terms of why the binary decision remains unreliable.
- **Accuracy/Speed**: The feature improves model-selection quality by identifying data and evidence issues without adding runtime to final inference.
- **ROI/Hybrid Flow**: The audit explicitly considers preprocessing and detector-evidence quality as candidate failure sources without changing production ROI or hybrid logic during the audit itself.
- **Annotation/Version Strategy**: The audit is a controlled follow-up to the V2.x same-split analysis and prepares later V2/V5 classifier decisions without creating a new final model directly.
- **Reproducibility/Leakage**: The hard-row source, audit table, category definitions, and summary outputs must be saved and must not use test labels or public-score feedback.
- **Confidentiality**: All row-level audit artifacts remain local or competition-safe and must not expose private dataset material publicly.
- **Explainability**: The audit requires row-level rationale and category evidence so later decisions can be technically justified.
- **Bias/Imbalance**: The audit focuses on the highest-risk subset where ambiguity, mislabeled examples, and difficult defect distributions may distort learning.

### Key Entities *(include if feature involves data)*

- **Hard Audit Row**: One governed hard-example row from the audit pool, identified by stable `image_id` and review context.
- **Audit Category**: The primary outcome assigned to a hard row, indicating whether the main issue is ambiguity, label quality, preprocessing quality, detector evidence quality, or true difficulty.
- **Audit Evidence Record**: The supporting row-level evidence used to justify a category assignment, including optional detector or image-quality observations.
- **Audit Summary Record**: One aggregated count or ranked failure-mode entry derived from the completed row-level audit table.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of governed hard rows appear exactly once in the row-level audit output.
- **SC-002**: 100% of audited rows receive exactly one primary audit category.
- **SC-003**: 100% of audited rows include recorded rationale text for the assigned category.
- **SC-004**: The aggregated summary reports category counts for all rows included in the audit, and the category totals equal the total number of audited rows.
- **SC-005**: The feature records zero training actions, zero submission creation, and zero test-label usage.
- **SC-006**: A reviewer can distinguish likely label issues, preprocessing issues, detector-evidence issues, and genuinely hard rows from the generated outputs without revisiting all 442 rows manually.
- **SC-007**: The audit produces at least one ranked failure-mode summary suitable for deciding the next improvement branch in the roadmap.

## Assumptions

- The hard-row audit pool is derived from prior validation-only same-split analysis and is safe to use for offline review.
- The governed hard-row pool is expected to contain 442 rows, but the feature should still work if the governed count changes slightly after normalization or deduplication.
- Optional detector or image-quality evidence may be incomplete, and missing evidence should not block audit completion.
- The audit may identify likely mislabeled rows even if final competition policy later limits whether those rows can be changed for training.
- This phase is a data-quality and evidence audit only; retraining, relabel application, and submission generation belong to later phases.
