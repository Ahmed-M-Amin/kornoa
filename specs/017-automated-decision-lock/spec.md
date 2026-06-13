# Feature Specification: Spec 017 Automated Decision Lock

**Feature Branch**: `017-automated-decision-lock`

**Created**: 2026-06-13

**Status**: Draft

**Input**: User description: "Spec 017 automated decision lock and active review reduction workflow based on Spec 016 outputs. Consume outputs/analysis/data_quality_audit/data_quality_master.csv, cleaned_training_manifest.csv, manual_review_required_rows.csv, review_decision_template.csv, contact sheets, and reports. Add evidence-upgrade support for embeddings, near-duplicate groups, cluster/outlier scores, optional Cleanlab Datalab scores, and ROI heuristic recalibration. Automatically lock high-confidence keep/exclude decisions, reduce manual review to a small adjudication queue, and produce approved_cleaned_training_manifest.csv for later training. No training, no submission, no test labels, no leaderboard tuning, no original label modification."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Lock Safe Training Rows (Priority: P1)

As the model owner, I need the system to convert Spec 016 audit outputs into a locked set of approved training rows so later training uses only rows that pass explicit safety and data-quality rules.

**Why this priority**: This is the required gate before any cleaned-data training. Training directly from the Spec 016 cleaned manifest would ignore unresolved review, ROI, duplicate, and label-risk evidence.

**Independent Test**: Run the decision-lock workflow against valid Spec 016 artifacts and confirm that an approved cleaned training manifest is produced, every approved row has a keep decision, and all blocked or unresolved rows are excluded.

**Acceptance Scenarios**:

1. **Given** valid Spec 016 audit outputs and Spec 014 blocked-row files, **When** the user runs the decision-lock workflow, **Then** the system produces an approved cleaned training manifest containing only rows with an explicit approved decision.
2. **Given** any image_id from a Spec 014 blocked file, **When** the decision-lock workflow evaluates the row, **Then** that image_id is excluded from the approved manifest and receives an exclusion decision with a traceable reason.
3. **Given** a row marked as manual-review-required by Spec 016, **When** no high-confidence automatic keep decision or explicit adjudication decision exists, **Then** the row is not included in the approved manifest.

---

### User Story 2 - Reduce Manual Review to High-Impact Adjudication (Priority: P2)

As the reviewer, I need the system to reduce thousands of manual-review candidates into a smaller, evidence-ranked adjudication queue so review effort focuses on rows that can materially affect model quality.

**Why this priority**: Spec 016 identified 3,361 rows requiring review, including 3,230 ROI/crop review rows. Reviewing all rows manually is slow and inconsistent.

**Independent Test**: Run the decision-lock workflow and confirm that review candidates are grouped, ranked, assigned suggested decisions, and separated from automatically kept or excluded rows.

**Acceptance Scenarios**:

1. **Given** manual-review-required rows from Spec 016, **When** the workflow evaluates all available evidence, **Then** it produces a ranked adjudication queue with risk reasons, suggested decisions, and evidence sources.
2. **Given** multiple rows that share the same risk pattern, **When** the queue is generated, **Then** the system prioritizes representative rows and records the group each row belongs to.
3. **Given** an unresolved or ambiguous row, **When** evidence is insufficient for automatic inclusion, **Then** the row is deferred or sent to adjudication instead of entering the approved manifest.

---

### User Story 3 - Upgrade Evidence Before Decisions (Priority: P3)

As the data-quality analyst, I need the decision process to use stronger optional evidence when available, including embeddings, near-duplicate groups, cluster/outlier scores, prediction evidence, and Cleanlab-style label-quality scores.

**Why this priority**: Stronger evidence improves decision quality and avoids over-trusting broad heuristics such as ROI flags alone.

**Independent Test**: Run the workflow with and without optional evidence and confirm that both cases complete safely, missing evidence is reported, and available evidence changes decision confidence and audit reporting.

**Acceptance Scenarios**:

1. **Given** optional embedding evidence is available, **When** the workflow runs, **Then** near-duplicate groups, visual clusters, and outlier scores are included in the decision evidence.
2. **Given** optional evidence is missing, **When** the workflow runs, **Then** the system records the missing evidence and makes conservative decisions rather than silently assuming the evidence exists.
3. **Given** a suspected severe ROI/crop issue, **When** the workflow evaluates the row, **Then** it uses corroborating evidence before making a high-confidence exclusion decision.

---

### Edge Cases

- Spec 016 outputs are missing, empty, unreadable, or have unexpected columns.
- Spec 014 blocked-row files are missing or contain image_ids that also appear in candidate keep rows.
- The approved hard-row list overlaps blocked rows.
- A row appears in multiple decision buckets.
- Duplicate or near-duplicate rows contain conflicting labels.
- Optional evidence files are missing, partially populated, or contain image_ids not present in training labels.
- Image files referenced by training labels are missing or unreadable.
- ROI/crop rules identify a large number of rows, but corroborating evidence indicates the rule is over-broad.
- A row has high model disagreement but no duplicate, ROI, or visual outlier evidence.
- A row has conflicting evidence, such as clean ROI evidence but high label-risk evidence.
- Contact sheets or review templates exist from Spec 016 but do not cover all high-risk groups.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST consume the Spec 016 master audit table, cleaned training manifest, manual-review-required rows, review decision template, contact-sheet inventory, and summary reports as the baseline evidence set.
- **FR-002**: System MUST consume Spec 014 approved hard-row and blocked-row outputs so that hard-valid rows can be considered and blocked rows can be excluded.
- **FR-003**: System MUST produce one decision record for every train row covered by the Spec 016 master audit table.
- **FR-004**: System MUST assign every decision record exactly one final decision: `auto_keep`, `auto_exclude`, `needs_adjudication`, or `defer`.
- **FR-005**: System MUST produce an approved cleaned training manifest that contains only rows with an approved keep decision.
- **FR-006**: System MUST prevent all Spec 014 blocked rows from entering the approved cleaned training manifest.
- **FR-007**: System MUST prevent unresolved manual-review-required rows from entering the approved cleaned training manifest.
- **FR-008**: System MUST prevent duplicate-conflict rows from entering the approved cleaned training manifest unless the conflict is explicitly resolved by an approved decision.
- **FR-009**: System MUST detect and report overlaps between allowed rows, blocked rows, manual-review rows, and approved rows.
- **FR-010**: System MUST verify that every approved row exists in the original training labels and has an available training image.
- **FR-011**: System MUST preserve original training labels and must not write modified labels back to the original training label file.
- **FR-012**: System MUST support optional evidence from embeddings, near-duplicate groups, cluster identifiers, cluster outlier scores, prediction evidence, and Cleanlab-style label-quality scores.
- **FR-013**: System MUST report optional evidence as missing, partial, or available, and must lower decision confidence when important evidence is missing.
- **FR-014**: System MUST group ROI/crop review rows by evidence patterns and recalibrate broad ROI/crop flags before assigning final decisions.
- **FR-015**: System MUST avoid excluding ROI/crop rows using a single broad heuristic unless corroborating evidence supports the exclusion.
- **FR-016**: System MUST produce an evidence-ranked adjudication queue for rows that are high-impact but not safe for automatic keep or automatic exclude.
- **FR-017**: System MUST include risk rank, risk bucket, primary issue, suggested decision, evidence sources, and reviewer decision fields in the adjudication queue.
- **FR-018**: System MUST produce separate row lists for automatically kept rows, automatically excluded rows, rows needing adjudication, and deferred uncertain rows.
- **FR-019**: System MUST produce a summary report with counts for all decision classes, evidence availability, blocked-row enforcement, overlap checks, and approved manifest size.
- **FR-020**: System MUST produce a rule-audit report that explains which rule or evidence group created each decision.
- **FR-021**: System MUST produce ROI recalibration reporting that shows ROI groups, group counts, representative examples, risk levels, and proposed group actions.
- **FR-022**: System MUST produce visual review aids for top-risk adjudication rows, auto-exclude representatives, and ROI group representatives.
- **FR-023**: System MUST fail safely if required Spec 016 or Spec 014 inputs are missing or inconsistent.
- **FR-024**: System MUST not start training, create submissions, use test labels, tune to leaderboard feedback, or use rejected Phase 3 outputs as a source of truth.

### Constitution Alignment *(mandatory)*

- **Binary Output**: The feature preserves the existing binary target meaning of `0 = Reusable` and `1 = Not Reusable`; it decides row eligibility for training but does not redefine class semantics.
- **Accuracy/Speed**: The feature is expected to improve future accuracy by removing or deferring risky rows before training; it has no direct inference-speed impact because it does not train or serve a model.
- **ROI/Hybrid Flow**: The feature audits ROI/crop quality and recalibrates ROI/crop review rows before deciding whether rows are safe for future classifier training.
- **Annotation/Version Strategy**: The feature consumes V2B and Spec 014 evidence where available and produces a locked manifest for the next V2B-style replay; it does not create detector or segmentation annotations.
- **Reproducibility/Leakage**: The feature writes locked decision artifacts, reports evidence availability, prevents test-label usage, and blocks leaderboard-driven decisions.
- **Confidentiality**: The feature uses private training artifacts locally and must not publish dataset images, labels, or derived contact sheets outside the project workspace.
- **Explainability**: Every decision must include a decision reason, confidence level, and evidence source so reviewers can trace why a row was kept, excluded, deferred, or sent to adjudication.
- **Bias/Imbalance**: The feature reports decision counts across labels and risk buckets so later training can evaluate whether exclusions or deferrals distort class balance.

### Key Entities *(include if feature involves data)*

- **Decision Record**: One row-level decision containing image_id, original target, evidence status, risk flags, decision, decision reason, confidence, and evidence sources.
- **Approved Cleaned Training Manifest**: The locked list of rows approved for later training after blocked rows, unresolved rows, duplicate conflicts, and unsafe evidence patterns are removed.
- **Adjudication Queue**: A reduced and ranked review list for high-impact uncertain rows that cannot be safely decided automatically.
- **Evidence Inventory**: A report describing which evidence sources were available, missing, partial, or ignored.
- **ROI Recalibration Group**: A group of ROI/crop review rows sharing similar quality, geometry, prediction, or visual-cluster evidence.
- **Decision Rule Audit**: A traceability table linking each final decision class to the rule or evidence pattern that caused it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of train rows in the Spec 016 master audit table receive exactly one final decision.
- **SC-002**: 0 Spec 014 blocked image_ids appear in the approved cleaned training manifest.
- **SC-003**: 0 unresolved manual-review-required rows appear in the approved cleaned training manifest.
- **SC-004**: 0 duplicate-conflict rows appear in the approved cleaned training manifest unless explicitly resolved by an approved decision.
- **SC-005**: 100% of approved rows are present in the original training labels and have an available training image.
- **SC-006**: The adjudication queue contains fewer rows than the full Spec 016 manual-review-required set while preserving all rows that cannot be automatically decided safely.
- **SC-007**: The summary report includes counts for all four decision classes and evidence availability status for every optional evidence category.
- **SC-008**: The rule-audit report can explain the final decision for 100% of decision records.
- **SC-009**: The workflow completes without training a model, creating a submission, reading test labels, modifying original labels, or using leaderboard feedback.

## Assumptions

- Spec 016 has already completed and produced the expected audit outputs in the project workspace.
- Spec 014 hard-row technical triage outputs are available and remain the source of truth for blocked hard rows.
- The original training label file and training images are available locally.
- Optional evidence sources may be unavailable during the first implementation, so the workflow must remain conservative when they are missing.
- The approved cleaned training manifest is an input to a later training spec, not a training trigger by itself.
- Reviewers may later fill adjudication decisions, but this feature must be useful even before manual adjudication is completed.
