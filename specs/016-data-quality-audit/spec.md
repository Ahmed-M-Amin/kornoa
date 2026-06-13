# Feature Specification: Controlled Data Quality Audit Pipeline

**Feature Branch**: `016-data-quality-audit`

**Created**: 2026-06-13

**Status**: Draft

**Input**: User description: "Spec 016 controlled data quality audit pipeline based on docs/superpowers/plans/2026-06-13-spec-016-data-quality-audit.md. Build the spec for data_quality_master.csv, OOF/V2B prediction evidence, Spec 014 hard-row integration, ROI/crop audit, duplicate conflict detection, label issue ranking, controlled buckets, reports, contact sheets, and cleaned_training_manifest.csv. No training, no submission, no test labels, no leaderboard tuning."

## Clarifications

### Session 2026-06-13

- Q: Should Spec 016 generate fresh out-of-fold predictions itself, require them strictly, or only consume them if already available? -> A: Spec 016 consumes precomputed out-of-fold predictions when available, but still runs without them and marks label-quality reliability as limited.
- Q: Can `manual_review_required` rows enter `cleaned_training_manifest.csv` during Spec 016? -> A: No. `manual_review_required` rows stay excluded until a later explicit review-lock artifact approves them.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build Full Data Quality Evidence (Priority: P1)

As the model owner, I need one complete audit table for every training row so I can understand which rows are clean, hard-but-useful, suspicious, duplicated, crop-damaged, or unsafe before starting another training run.

**Why this priority**: Phase 3 hard-example training reduced validation quality, so the next improvement step must identify harmful data before any retraining.

**Independent Test**: Can be fully tested by running the audit on a labeled training dataset and confirming every training row appears exactly once with prediction evidence, hard-row evidence, image-quality evidence, duplicate evidence, and a final data-quality bucket.

**Acceptance Scenarios**:

1. **Given** a training label file and training image folder, **When** the data quality audit is run, **Then** it produces a master audit table containing exactly one row per training image.
2. **Given** V2B validation prediction evidence and optional out-of-fold prediction evidence, **When** the audit is run, **Then** rows with available predictions include probability, predicted label, correctness, confidence, risk, and decision-boundary evidence.
3. **Given** optional prediction evidence is missing, **When** the audit is run, **Then** the audit still completes and marks the affected scoring reliability as limited rather than failing silently.

---

### User Story 2 - Separate Safe Training Rows From Risky Rows (Priority: P1)

As the model owner, I need controlled row buckets so future training can use only trusted clean rows and verified useful hard rows while excluding or holding risky rows for review.

**Why this priority**: The next model must not repeat the Phase 3 failure by blindly training on ambiguous, mislabeled, duplicate-conflict, or crop-problem rows.

**Independent Test**: Can be fully tested by providing allowed hard rows, blocked hard rows, duplicate conflicts, crop-problem rows, and confident-wrong rows, then confirming only clean and hard-valid rows enter the cleaned training manifest.

**Acceptance Scenarios**:

1. **Given** rows blocked by Spec 014 as suspected mislabeled, **When** buckets are assigned, **Then** those rows are excluded from training.
2. **Given** rows blocked by Spec 014 for ROI or other high-risk reasons, **When** buckets are assigned, **Then** those rows require manual review and are excluded from the cleaned training manifest.
3. **Given** an allowed Spec 014 hard row with no label, duplicate, or crop risk, **When** buckets are assigned, **Then** it is marked as hard-valid training data.
4. **Given** normal rows with no major risk evidence, **When** buckets are assigned, **Then** they are marked as clean training data.

---

### User Story 3 - Prepare Manual Review Evidence (Priority: P2)

As the human reviewer, I need prioritized review files and visual evidence so I can efficiently inspect the most suspicious rows before approving any cleaned dataset.

**Why this priority**: Automated scores can rank risk but must not relabel or approve ambiguous visual data without human confirmation.

**Independent Test**: Can be fully tested by running the audit and confirming it produces prioritized review outputs for confident-wrong rows, label issues, blocked rows, duplicate conflicts, crop issues, and unclear boundary cases.

**Acceptance Scenarios**:

1. **Given** suspicious training rows, **When** the audit is run, **Then** the highest-priority manual review list starts with confident wrong predictions and likely label issues.
2. **Given** rows needing visual inspection, **When** optional visual review outputs are requested, **Then** grouped visual evidence is produced for reviewer inspection without changing labels.
3. **Given** a reviewer needs to approve later training data, **When** the audit completes, **Then** a review decision template is available for a future review-lock feature.

---

### User Story 4 - Produce Safety and Traceability Reports (Priority: P2)

As the project owner, I need a complete summary of what the audit used, what it excluded, and what safety rules were enforced so I can trust the cleaned manifest before using it in later training.

**Why this priority**: Any score improvement attempt must be reproducible, leakage-safe, and explainable.

**Independent Test**: Can be fully tested by inspecting summary reports after the audit and confirming row counts, bucket counts, exclusion reasons, input availability, warnings, and safety confirmations are present.

**Acceptance Scenarios**:

1. **Given** an audit run, **When** reports are inspected, **Then** total rows, clean rows, hard-valid rows, excluded rows, manual-review rows, and top exclusion reasons are visible.
2. **Given** optional inputs are unavailable, **When** reports are inspected, **Then** missing optional inputs are listed as warnings and the audit reliability status is clear.
3. **Given** the audit is complete, **When** safety confirmations are inspected, **Then** the reports confirm no training, no submission creation, no test-label usage, no leaderboard tuning, and no original label modification.

### Edge Cases

- Missing optional V2B, out-of-fold, embedding, annotation, bottle-type, or historical hard-row evidence must not stop the audit; the audit must report the missing input and downgrade only the affected evidence type.
- Missing or unreadable training images must be excluded from future training and reported as image availability issues.
- Duplicate or near-duplicate images with conflicting labels must require manual review and must not enter the cleaned training manifest.
- Rows near the decision boundary must be ranked for review when other evidence suggests visual ambiguity, but they must not be automatically relabeled.
- ROI or crop problem rows must require review by default unless they are clearly missing or unreadable, in which case they are excluded.
- Any overlap between training-approved rows and blocked-row evidence must resolve in favor of the stricter blocked or manual-review decision.
- Optional visual review outputs must never be required for the core audit to succeed.
- The cleaned training manifest must be empty rather than unsafe if all rows are excluded or pending review.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST produce a master data quality audit table containing every training row exactly once.
- **FR-002**: The master audit table MUST include the original binary target value without modifying the source label file.
- **FR-003**: The system MUST include V2B prediction evidence when available, including probability, predicted label, correctness, confidence, risk, and decision-boundary indicators.
- **FR-004**: The system MUST prefer out-of-fold training prediction evidence for full-dataset label-quality scoring when available.
- **FR-005**: When out-of-fold prediction evidence is missing, the system MUST still complete the audit, mark label-quality reliability as limited, and use only the weaker evidence sources already available.
- **FR-005a**: The system MUST consume precomputed out-of-fold prediction evidence when it is provided, but it MUST NOT generate fresh out-of-fold predictions as part of this feature.
- **FR-006**: The system MUST integrate Spec 014 allowed hard-row evidence and blocked-row evidence into the audit table.
- **FR-007**: The system MUST treat suspected-mislabel blocked rows as unsafe for training.
- **FR-008**: The system MUST treat ROI-pipeline-bug and other high-risk blocked rows as requiring manual review unless a future explicit review decision overrides them.
- **FR-009**: The system MUST assess image availability and visual quality risks, including missing image, unreadable image, very small image, mostly dark image, mostly bright image, low contrast image, weak texture, and background-heavy indicators when evidence is available.
- **FR-010**: The system MUST identify exact duplicate images and flag duplicate groups with conflicting labels.
- **FR-011**: The system SHOULD include near-duplicate and embedding-based risk evidence when such evidence is available.
- **FR-012**: The system MUST compute a suspicious-row ranking that prioritizes confident wrong predictions, likely label issues, blocked or ambiguous rows, duplicate conflicts, outliers, ROI/crop issues, and unclear decision-boundary rows.
- **FR-013**: The system MUST create controlled row buckets: clean training rows, hard-valid training rows, excluded rows, and manual-review-required rows.
- **FR-014**: The cleaned training manifest MUST include only clean training rows and hard-valid training rows.
- **FR-015**: The cleaned training manifest MUST exclude suspected mislabeled rows, blocked rows, manual-review-required rows, duplicate-conflict rows, missing or unreadable images, and ROI/crop-risk rows that have not been approved by a future review-lock process.
- **FR-015a**: A row assigned to `manual_review_required` MUST NOT enter `cleaned_training_manifest.csv` during this feature, even if it has partial supporting evidence, unless a later explicit review-lock artifact exists outside the scope of this feature.
- **FR-016**: The system MUST produce a manual review template that lets a future review process record approve, exclude, relabel-needed, or needs-expert-review decisions without changing the original labels.
- **FR-017**: The system MUST produce summary reports with total row count, bucket counts, exclusion reason counts, top suspicious rows, top label issue rows, top confident wrong rows, top anomaly rows, duplicate conflicts, ROI/image-quality issues, and a manual review plan.
- **FR-018**: The system MUST produce input inventory and reliability reporting that identifies present and missing required or optional audit sources.
- **FR-019**: The system MUST include traceability fields sufficient to connect every output row to the audit run and input evidence used.
- **FR-020**: The system MUST produce optional visual review contact sheets for high-priority review buckets when image access is available.
- **FR-021**: The system MUST explicitly report that no training was started, no submission was created, no test labels were used, no leaderboard tuning was performed, and original labels were unchanged.
- **FR-022**: The system MUST fail safely if required training labels or training images are unavailable.
- **FR-023**: The system MUST not require optional third-party visual review or label-quality tools for the core audit to complete.
- **FR-024**: The system MUST preserve the meaning of binary labels as `0 = Reusable` and `1 = Not Reusable` in all audit and manifest outputs.

### Constitution Alignment *(mandatory)*

- **Binary Output**: The audit preserves the project label contract, where `0 = Reusable` and `1 = Not Reusable`; it must not introduce alternate label meanings or change original targets.
- **Accuracy/Speed**: The feature improves future accuracy work by removing or quarantining harmful data before training. It does not change inference speed because it does not train or deploy a model.
- **ROI/Hybrid Flow**: ROI and crop evidence are first-class audit signals. ROI-problem rows are not blindly trained as hard examples; they are excluded or sent to review depending on severity.
- **Annotation/Version Strategy**: The audit consumes existing V2B and Spec 014 evidence and prepares a controlled cleaned manifest for later model work. It does not introduce detector, segmentation, or memory-bank scope.
- **Reproducibility/Leakage**: The audit must record input availability, audit run identity, row-level evidence, and safety confirmations. It must not use test labels, public leaderboard feedback, or submission outcomes.
- **Confidentiality**: All dataset images, labels, predictions, and audit reports remain local project artifacts and must not be exposed publicly.
- **Explainability**: Every bucket decision must include evidence explaining why the row is clean, hard-valid, excluded, or pending review.
- **Bias/Imbalance**: Reports must expose bucket counts and exclusion reasons so class, hard-row, and defect-risk concentration can be inspected before future training.

### Key Entities *(include if feature involves data)*

- **Training Row**: One labeled training image with `image_id`, binary target, and optional bottle, split, annotation, prediction, ROI, duplicate, and hard-row evidence.
- **Prediction Evidence**: Out-of-sample model evidence used to rank likely label issues, confident wrong cases, and decision-boundary uncertainty.
- **Spec 014 Evidence**: Approved hard-row and blocked-row evidence from the previous hard-row visual review workflow.
- **Image Quality Evidence**: Availability, readability, brightness, contrast, texture, size, and background/framing indicators for a training image.
- **Duplicate Evidence**: Exact or near-duplicate grouping information and label-conflict flags.
- **Data Quality Bucket**: The final controlled category assigned to each row: clean training, hard-valid training, excluded from training, or manual review required.
- **Cleaned Training Manifest**: The future-training input package containing only rows that are safe to use without additional review.
- **Manual Review Template**: A reviewer-facing artifact for future human decisions; it records intended review actions but does not modify source labels.
- **Audit Summary Report**: The run-level evidence package describing counts, warnings, input inventory, reliability status, and safety confirmations.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The master audit table contains exactly 100% of rows from the training label file and no duplicate `image_id` entries.
- **SC-002**: The audit completes successfully when required training labels and images are available, even if optional V2B, out-of-fold, embedding, annotation, bottle-type, or historical review inputs are missing.
- **SC-003**: 100% of Spec 014 suspected-mislabel blocked rows are excluded from the cleaned training manifest.
- **SC-004**: 100% of Spec 014 ROI-bug and other high-risk blocked rows are excluded from the cleaned training manifest unless a later explicit review-lock artifact approves them.
- **SC-005**: 100% of duplicate label-conflict rows are excluded from the cleaned training manifest and listed in a duplicate-conflict report.
- **SC-006**: 100% of missing or unreadable image rows are excluded from the cleaned training manifest and listed in an image-quality report.
- **SC-007**: The cleaned training manifest contains only rows assigned to clean training or hard-valid training buckets.
- **SC-008**: The audit summary reports total rows, all bucket counts, top exclusion reasons, missing optional inputs, and safety confirmations in one inspection package.
- **SC-009**: The manual review package ranks at least the top 300 suspicious rows when that many suspicious rows exist, or all suspicious rows when fewer exist.
- **SC-010**: The final reports explicitly confirm no training, no submission creation, no test-label usage, no leaderboard tuning, and no original label modification.

## Assumptions

- The training label file and training image folder are the only required dataset inputs for the core audit.
- Out-of-fold predictions are the preferred evidence for full-dataset label-quality scoring; if they are unavailable, V2B validation predictions and deterministic fallback ranking provide limited but useful evidence.
- Out-of-fold predictions, when used by this feature, are precomputed inputs rather than outputs generated by the audit workflow.
- Optional label-quality and visual-review tooling may improve ranking and review ergonomics, but the core audit must not depend on those tools.
- Human review decisions are out of scope for this spec; they belong in a later review-lock spec.
- Model retraining is out of scope for this spec; cleaned-data training belongs in a later training spec after review evidence is inspected.
- Original labels remain unchanged throughout this feature.
- The cleaned manifest is a conservative training input: rows requiring review are excluded until explicitly approved by a future process.
- No row classified as `manual_review_required` is considered training-safe within this feature.
