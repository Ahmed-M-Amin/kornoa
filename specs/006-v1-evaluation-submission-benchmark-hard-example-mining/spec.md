# Feature Specification: V1 Evaluation, Submission, Benchmark, and Hard-Example Mining

**Feature Branch**: `006-v1-evaluation-submission-benchmark-hard-example-mining`

**Created**: 2026-06-02

**Status**: Draft

**Input**: User description: "From `docs/krones_final_implementation_plan_complete.md`, read phase 6 / the next execution phase after completed V1 training and create the sixth spec. SPEC-001 to SPEC-005 are finished. V1 training results are in `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1`."

## Clarifications

### Session 2026-06-02

- Q: How should high-loss samples be selected when the V1 validation prediction CSV does not contain per-sample loss values? -> A: Rank by wrong/confused probability severity from the existing validation CSV: false positives with high probability, false negatives with low probability, then near-threshold uncertain cases.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate First V1 Submission (Priority: P1)

The practitioner can use the completed V1 training artifacts to predict the test image set and produce a valid first Kaggle submission.

**Why this priority**: V1 training is complete, so the next highest-value outcome is a submission file that can be uploaded and scored.

**Independent Test**: Can be tested with the completed V1 artifact root, a sample submission file, and a test image folder by verifying that every sample row receives exactly one binary target.

**Acceptance Scenarios**:

1. **Given** the completed V1 artifacts, sample submission, and test images, **When** V1 submission generation runs, **Then** a submission file is created with the same image order and label column contract as the sample submission.
2. **Given** a predicted probability for `1 = Not Reusable`, **When** the saved V1 threshold is applied, **Then** the target is `1` at or above the threshold and `0` below it.

---

### User Story 2 - Benchmark V1 Inference Speed (Priority: P2)

The practitioner can measure and save V1 inference speed for the image-flow classifier path.

**Why this priority**: Runtime efficiency is part of the project goal, and V1 needs a baseline speed report before heavier future modules are considered.

**Independent Test**: Can be tested by running V1 prediction over a known image set and verifying that the benchmark report includes image count and timing metrics.

**Acceptance Scenarios**:

1. **Given** the completed V1 artifacts and a test image folder, **When** benchmarking runs, **Then** the benchmark report records total runtime, image count, average time per image, and throughput.

---

### User Story 3 - Mine V1 Hard Examples (Priority: P3)

The practitioner can automatically create offline hard-example files from V1 validation predictions for later V2 analysis.

**Why this priority**: False positives, false negatives, uncertain cases, and high-loss samples are needed to understand V1 mistakes and guide V2 without changing normal inference.

**Independent Test**: Can be tested from the V1 validation prediction CSV alone, without reading test images or producing a submission.

**Acceptance Scenarios**:

1. **Given** V1 validation predictions, **When** hard-example mining runs, **Then** false positive, false negative, uncertain, high-loss sample CSVs, and a summary JSON are created.
2. **Given** normal V1 inference, submission generation, or benchmarking, **When** those workflows run, **Then** hard-example mining does not run unless explicitly requested.

### Edge Cases

- Missing V1 checkpoint, threshold, validation predictions, sample submission, or image files produce clear failures instead of partial outputs.
- The concrete V1 run artifact root is `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1`.
- If a user supplies the parent root `artifacts/kaggle_v1_artifacts/outputs`, the workflow detects its `kaggle_v1` child and uses `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1` as the concrete run root.
- The concrete V1 run artifact root must contain `models/classifier_effnet_b0_best.pth`, `reports/classifier_metrics.json`, `reports/best_threshold.json`, and `predictions/val_classifier_predictions.csv`.
- Existing manual hard-example files may live under `artifacts/kaggle_v1_artifacts/outputs/hard_examples`, but SPEC-006 can regenerate hard examples from `predictions/val_classifier_predictions.csv`.
- The submission preserves the sample submission's required label column while maintaining `0 = Reusable` and `1 = Not Reusable`.
- Hard-example mining uses validation predictions only and never uses test-image predictions or sample submission rows.
- Feature memory bank, detector, segmentation, Grad-CAM, dashboard, V2 training, ensemble, and distillation are out of scope.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST consume the completed V1 classifier checkpoint from the supplied V1 artifact root.
- **FR-002**: System MUST consume the completed V1 best-threshold record from the supplied V1 artifact root.
- **FR-003**: System MUST predict probability for class `1 = Not Reusable` for each requested test image.
- **FR-004**: System MUST convert each probability to a binary target using the saved V1 threshold.
- **FR-005**: System MUST generate a V1 submission file that matches the sample submission row order and label column contract.
- **FR-006**: System MUST save the V1 submission under an ignored output location.
- **FR-007**: System MUST produce a V1 benchmark report with image count, total runtime, average time per image, and throughput.
- **FR-008**: System MUST create hard-example files for false positives, false negatives, uncertain cases, and high-loss proxy samples from V1 validation predictions only.
- **FR-009**: System MUST save a hard-example summary with counts, source prediction reference, threshold reference, and generated artifact paths.
- **FR-010**: System MUST keep hard-example mining offline and disabled during normal inference, submission generation, and benchmarking.
- **FR-011**: System MUST treat `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1` as the default concrete V1 run artifact root unless a later workflow supplies a different artifact root.
- **FR-012**: System MUST NOT implement feature memory bank, detector, segmentation, Grad-CAM, dashboard, V2 training, ensemble, distillation, or hybrid inference behavior in SPEC-006.
- **FR-013**: System MUST define high-loss proxy samples from existing validation CSV fields by ranking wrong or confused predictions: false positives with high probability, false negatives with low probability, then near-threshold uncertain cases.
- **FR-014**: System MUST accept `artifacts/kaggle_v1_artifacts/outputs` as an optional parent-root input by detecting its `kaggle_v1` child and resolving to the concrete V1 run artifact root.
- **FR-015**: System MUST verify the concrete V1 run artifact root contains the classifier checkpoint, classifier metrics JSON, best-threshold JSON, and validation predictions CSV before dependent workflows run.

### Constitution Alignment *(mandatory)*

- **Binary Output**: Preserves `0 = Reusable` and `1 = Not Reusable` for probabilities, decisions, hard-example categories, and submission targets.
- **Accuracy/Speed**: Uses the V1 F1-selected threshold and records inference speed for the V1 baseline.
- **ROI/Hybrid Flow**: Uses the existing V1 classifier image-flow path after ROI/preprocessing; no detector, hybrid fallback, or feature memory lookup is introduced.
- **Annotation/Version Strategy**: This is the SPEC-006 V1 post-training feature. Later V2, detector, hybrid, Grad-CAM, and feature memory-bank specs remain separate.
- **Reproducibility/Leakage**: Uses saved V1 artifacts; hard examples come only from validation predictions and never from test labels or sample submission rows.
- **Confidentiality**: Private data and derived artifacts remain local and under ignored artifact/output locations.
- **Explainability**: Saves probability, target, benchmark, and hard-example analysis artifacts. Visual explanations are deferred.
- **Bias/Imbalance**: Hard-example mining exposes false positives, false negatives, uncertain cases, and high-loss proxy samples for later error and imbalance analysis.

### Key Entities *(include if feature involves data)*

- **V1 Artifact Root**: Concrete completed V1 run output tree containing the classifier checkpoint, metrics, threshold, and validation predictions.
- **V1 Classifier Checkpoint**: Saved model weights used for binary image prediction.
- **Threshold Record**: Saved V1 threshold used to convert probabilities into binary targets.
- **Image Prediction**: Per-image probability and binary target generated by V1 inference.
- **Submission File**: Sample-aligned CSV ready for Kaggle upload.
- **Benchmark Report**: Saved timing summary for V1 inference speed.
- **Hard Example Set**: Offline validation-derived CSVs and summary for V1 mistakes and uncertainty, including a high-loss proxy ranking computed from saved validation probabilities and labels.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The generated submission contains 100% of sample submission rows, with no missing or duplicate image IDs.
- **SC-002**: 100% of submission targets are binary values using the saved V1 threshold.
- **SC-003**: Benchmark output reports image count, total runtime, average time per image, and throughput for every run.
- **SC-004**: Hard-example mining creates all four requested CSV categories and one summary JSON from validation predictions only, with high-loss proxy rows ranked from saved probability severity.
- **SC-005**: Normal inference, submission generation, and benchmarking create zero hard-example files unless hard-example mining is explicitly invoked.

## Assumptions

- SPEC-001 through SPEC-005 are complete and should not be recreated.
- V1 training has completed successfully and reached approximately F1 0.9166.
- The current concrete V1 run artifact root is `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1`.
- The optional parent artifact root is `artifacts/kaggle_v1_artifacts/outputs`; when supplied, workflows should resolve its `kaggle_v1` child as the concrete V1 run artifact root.
- The concrete artifact root includes `models/classifier_effnet_b0_best.pth`, `reports/classifier_metrics.json`, `reports/best_threshold.json`, and `predictions/val_classifier_predictions.csv`.
- Some hard-example CSVs were manually created once under `artifacts/kaggle_v1_artifacts/outputs/hard_examples`; SPEC-006 should formalize automatic, repeatable hard-example generation from validation predictions.
- The existing ROI and preprocessing behavior remains the V1 image preparation path.
- Feature memory bank is later optional research and remains disabled by default.
