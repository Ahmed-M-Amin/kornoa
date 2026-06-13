# Feature Specification: Spec 018 V2B Cleaned Replay

**Feature Branch**: `018-v2b-cleaned-replay`

**Created**: 2026-06-13

**Status**: Draft

**Input**: User description: "Spec 018 V2B-style cleaned-data replay training based on Spec 017 outputs. Train a controlled classifier only on outputs/analysis/data_quality_decision_lock/approved_cleaned_training_manifest.csv using the locked V2B-style split, config-driven paths, saved threshold, saved validation predictions, saved metrics, and full reproducibility artifacts. Compare directly against the locked V2B baseline. No test labels, no leaderboard tuning, no submission during dry-run validation, and no use of deferred or needs-adjudication rows in training."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Validate Cleaned Replay Inputs (Priority: P1)

As the experiment owner, I need a dry-run validation mode that proves the Spec 018 training run will consume only the Spec 017 approved cleaned manifest and will not use deferred, adjudication, test, submission, or leaderboard data.

**Why this priority**: The cleaned replay is only useful if it is controlled. A dry run prevents accidental leakage or accidental training on unresolved rows before GPU time is spent.

**Independent Test**: Run the dry-run validation and confirm it checks the approved manifest, blocked-row exclusions, validation split inputs, baseline artifacts, output directories, and safety flags without starting training or creating a submission.

**Acceptance Scenarios**:

1. **Given** the Spec 017 approved cleaned manifest and decision reports exist, **When** the user runs the dry-run validation, **Then** the system confirms that training input rows come only from `approved_cleaned_training_manifest.csv`.
2. **Given** any row from `needs_adjudication_rows.csv`, `deferred_uncertain_rows.csv`, or `auto_exclude_rows.csv`, **When** the dry run validates the training manifest, **Then** the system fails if that row appears in the planned training input.
3. **Given** baseline checkpoint, threshold, validation predictions, and metrics artifacts are required for comparison, **When** the dry run executes, **Then** the system confirms each comparison artifact exists before training is allowed.

---

### User Story 2 - Train a V2B-Style Cleaned Replay (Priority: P2)

As the model trainer, I need to train a V2B-style classifier using the locked cleaned manifest and the same validation design so the result is comparable to the locked V2B baseline.

**Why this priority**: The main question is whether the cleaned data improves the reliable V2B baseline. The training run must be comparable rather than a new uncontrolled experiment.

**Independent Test**: Execute the cleaned replay training on the configured environment and verify that model weights, validation predictions, threshold, metrics, config snapshot, and runtime metadata are saved.

**Acceptance Scenarios**:

1. **Given** dry-run validation has passed, **When** the cleaned replay training starts, **Then** the system trains only on approved manifest rows and uses the locked validation split for evaluation.
2. **Given** training completes, **When** reports are generated, **Then** the system saves validation predictions, best threshold, metrics, model checkpoint, run config, and runtime metadata.
3. **Given** training input selection is complete, **When** the run records reproducibility artifacts, **Then** it records the manifest path, manifest row count, seed, split reference, training config, and output paths.

---

### User Story 3 - Compare Against Locked V2B Baseline (Priority: P3)

As the model owner, I need a direct comparison report against the locked V2B baseline so the final decision is based on validation evidence rather than public leaderboard feedback.

**Why this priority**: A cleaned-data replay is only valuable if it beats the trusted baseline under the same validation design or clearly explains why it should be rejected.

**Independent Test**: Compare the cleaned replay metrics against the locked V2B baseline metrics and confirm the report identifies whether the cleaned replay is accepted, rejected, or inconclusive.

**Acceptance Scenarios**:

1. **Given** cleaned replay metrics and locked V2B baseline metrics exist, **When** comparison runs, **Then** the system writes a comparison report with F1 delta, threshold, validation row count, and decision.
2. **Given** cleaned replay validation F1 is worse than the locked V2B baseline, **When** the comparison report is written, **Then** the report marks the cleaned replay as rejected for submission use.
3. **Given** cleaned replay validation F1 improves over the locked V2B baseline, **When** the comparison report is written, **Then** the report marks the model as a candidate for later submission generation but does not create a submission in this spec.

---

### Edge Cases

- The approved cleaned manifest is missing, empty, duplicated, or lacks required row identifiers.
- The approved cleaned manifest includes any row from `auto_exclude`, `needs_adjudication`, or `deferred_uncertain` outputs.
- The approved cleaned manifest contains an image_id not present in the original training labels or missing from training images.
- The locked V2B baseline checkpoint, threshold, predictions, metrics, or split reference is missing.
- The validation split cannot be reconstructed or does not align with the locked V2B comparison design.
- Training completes but validation predictions, threshold, metrics, or model checkpoint are not saved.
- Cleaned replay F1 improves while another safety gate fails.
- Dry-run validation accidentally creates training outputs or submission files.
- A training config tries to read test labels, tune from public leaderboard feedback, or include unresolved rows.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a dry-run validation mode for the cleaned replay workflow before training is allowed.
- **FR-002**: System MUST use `outputs/analysis/data_quality_decision_lock/approved_cleaned_training_manifest.csv` as the only training manifest for this spec.
- **FR-003**: System MUST reject any training input that overlaps `auto_exclude_rows.csv`, `needs_adjudication_rows.csv`, or `deferred_uncertain_rows.csv`.
- **FR-004**: System MUST verify every approved training row exists in the original training labels and has an available training image.
- **FR-005**: System MUST use the locked V2B-style validation design so metrics are comparable with the locked V2B baseline.
- **FR-006**: System MUST require locked V2B baseline artifacts for comparison, including checkpoint, best threshold, validation predictions, validation metrics, and split reference when available.
- **FR-007**: System MUST train a controlled classifier only after the dry-run validation passes.
- **FR-008**: System MUST save model checkpoint, validation predictions, best threshold, validation metrics, run config snapshot, manifest snapshot, and runtime metadata for the cleaned replay run.
- **FR-009**: System MUST write a direct comparison report against the locked V2B baseline.
- **FR-010**: System MUST mark the cleaned replay as accepted, rejected, or inconclusive based on validation comparison and safety gates.
- **FR-011**: System MUST preserve original labels and must not relabel training data.
- **FR-012**: System MUST not use test labels, public leaderboard feedback, submission outputs, deferred rows, needs-adjudication rows, or auto-excluded rows for training or threshold selection.
- **FR-013**: System MUST not create a Kaggle submission during dry-run validation.
- **FR-014**: System MUST record safety flags proving no test-label use, no leaderboard tuning, and no submission generation during validation.
- **FR-015**: System MUST provide enough saved artifacts for the run to be reproduced or rejected later without notebook-only state.

### Constitution Alignment *(mandatory)*

- **Binary Output**: The feature preserves `0 = Reusable` and `1 = Not Reusable`; it trains and evaluates a binary classifier without changing class meaning.
- **Accuracy/Speed**: The feature compares validation F1 against locked V2B and records runtime artifacts so accuracy and practicality remain visible.
- **ROI/Hybrid Flow**: The feature consumes cleaned data produced after ROI/crop quality decisions; it does not change ROI preprocessing rules unless a later plan explicitly scopes that work.
- **Annotation/Version Strategy**: The feature is a V2B-style replay using the locked cleaned manifest and baseline comparison artifacts; detector, segmentation, and annotation changes are out of scope.
- **Reproducibility/Leakage**: The feature requires manifest snapshots, config snapshots, validation predictions, threshold files, metrics, split reference, and explicit leakage-prevention flags.
- **Confidentiality**: Dataset images, private labels, model outputs, and derived reports remain inside the project workspace or private training environment.
- **Explainability**: The feature must save validation predictions, threshold, metrics, and comparison reports so the training decision is auditable.
- **Bias/Imbalance**: The feature must report training and validation row counts by target and compare cleaned-manifest composition against the locked baseline context.

### Key Entities *(include if feature involves data)*

- **Cleaned Replay Training Manifest**: The Spec 017 approved manifest used as the only training row source.
- **Locked V2B Baseline Evidence**: Existing baseline checkpoint, threshold, validation predictions, metrics, and split reference used for comparison.
- **Cleaned Replay Run**: One controlled training run with saved config, manifest snapshot, checkpoint, predictions, threshold, metrics, and runtime metadata.
- **Comparison Report**: The decision artifact comparing cleaned replay metrics against locked V2B baseline metrics and declaring accepted, rejected, or inconclusive.
- **Dry-Run Validation Report**: The pre-training safety report proving input paths, row exclusions, baseline artifacts, and leakage controls are valid.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Dry-run validation reports 0 overlaps between the training manifest and auto-excluded, needs-adjudication, or deferred rows.
- **SC-002**: Dry-run validation confirms 100% of planned training rows exist in training labels and have available images.
- **SC-003**: Training starts only after dry-run validation has passed.
- **SC-004**: Completed cleaned replay runs save checkpoint, validation predictions, best threshold, validation metrics, config snapshot, manifest snapshot, and runtime metadata.
- **SC-005**: Comparison report includes locked V2B F1, cleaned replay F1, F1 delta, selected threshold, validation row count, and final decision.
- **SC-006**: Safety report confirms no test labels used, no leaderboard tuning, no submission generation during dry-run validation, and no unresolved rows used in training.
- **SC-007**: The cleaned replay is rejected for submission use if it fails any safety gate or does not beat the locked V2B validation baseline.

## Assumptions

- Spec 017 has completed and produced `approved_cleaned_training_manifest.csv` plus excluded, deferred, and adjudication row files.
- The locked V2B baseline artifacts remain the current trusted baseline for comparison.
- The first Spec 018 implementation should include a dry-run validation command before any training command is used.
- The same validation design used for locked V2B comparison is available or reconstructable from saved artifacts.
- Kaggle submission generation is a later spec and must not be included in Spec 018 dry-run validation.
