# Feature Specification: Binary Classifier Training

**Feature Branch**: `006-binary-classifier-training`

**Spec ID**: `SPEC-005` (`specs/005-binary-classifier-training`; branch numbering is a Spec Kit branch sequence and does not create a separate `006` spec folder)

**Created**: 2026-06-01

**Status**: Draft

**Input**: User description: "Read Phase 4 from docs/krones-final-implementation-plan.md and, according to GitHub Spec Kit best practices, create the next specification for the V1 fast binary classifier. Phase 4 goal: first working end-to-end classifier pipeline using SPEC-001 through SPEC-004 foundations, 384x384 ROI/preprocessed inputs, binary output, stratified validation, weighted binary loss, F1-score, threshold search, saved model weights, validation metrics, best threshold, and validation predictions. Do not implement Kaggle submission, detector training, hybrid inference, Grad-CAM, dashboard, or memory-bank behavior in this spec."

## Clarifications

### Session 2026-06-01

- Q: What validation split policy should the V1 classifier use? -> A: Use an 80/20 stratified train/validation split with a fixed seed.
- Q: How should threshold search break ties when multiple thresholds produce the same best F1-score? -> A: Pick the lowest threshold among tied best-F1 thresholds.
- Q: What imbalance handling should V1 classifier training use? -> A: Use weighted binary loss only for V1 imbalance handling.
- Q: How should the best classifier checkpoint be selected? -> A: Select the best model by validation F1 after threshold search.
- Q: What runtime target should the synthetic training smoke run meet? -> A: Synthetic training smoke run must finish under 1 minute.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train V1 Binary Classifier (Priority: P1)

A project contributor can train the first reusable/not-reusable classifier from
the validated training dataset and SPEC-004 preprocessing outputs, producing a
repeatable baseline that predicts the official binary target.

**Why this priority**: This is the first model-producing V1 feature. Without a
working binary classifier baseline, threshold search, inference, benchmarking,
and submission generation cannot be validated.

**Independent Test**: Can be tested with a tiny synthetic dataset by confirming
that a training run consumes labeled images, produces binary probabilities, and
saves the expected baseline training artifacts without requiring private data.

**Acceptance Scenarios**:

1. **Given** a labeled training dataset with both target classes, **When** a
   contributor starts a classifier training run, **Then** the run completes and
   produces a saved classifier artifact for binary reusable/not-reusable
   prediction.
2. **Given** the same training configuration and seed, **When** the contributor
   repeats the synthetic training run, **Then** the split assignment and reported
   metrics are reproducible within the documented tolerance.

---

### User Story 2 - Measure Validation F1 and Threshold (Priority: P2)

A contributor can evaluate classifier predictions on a validation split and
select a decision threshold that is optimized for F1-score rather than relying
on a fixed default threshold.

**Why this priority**: The challenge ranking depends on F1-score. A model that
trains but does not report validation F1 or a selected threshold is not useful
for the next inference and submission specs.

**Independent Test**: Can be tested by feeding known validation probabilities
and labels into threshold selection and confirming the selected threshold,
F1-score, confusion counts, and metrics output are consistent.

**Acceptance Scenarios**:

1. **Given** validation labels and predicted probabilities, **When** threshold
   search runs, **Then** it reports the threshold with the best validation
   F1-score and saves that threshold for later inference.
2. **Given** class imbalance in the validation labels, **When** metrics are
   reported, **Then** the report includes F1-score and enough supporting counts
   to understand false positives and false negatives.

---

### User Story 3 - Save Reproducible Training Outputs (Priority: P3)

A contributor can review and reuse classifier training outputs, including
metrics, threshold, and validation predictions, while keeping generated model
and prediction artifacts out of source control by default.

**Why this priority**: Later submission, benchmark, and reporting specs depend
on stable classifier artifacts. Generated artifacts may derive from private
training data and must remain local or ignored.

**Independent Test**: Can be tested by running a synthetic training smoke test
and confirming the expected model, metrics, threshold, and validation prediction
outputs are created under approved generated-output locations and remain
untracked.

**Acceptance Scenarios**:

1. **Given** a successful training run, **When** the run finishes, **Then** it
   saves model weights, metrics, best threshold, and validation predictions in
   approved output locations.
2. **Given** generated classifier artifacts, **When** source-control status is
   checked, **Then** generated weights, reports, and prediction files are
   ignored or explicitly excluded unless approved for private team sharing.

### Edge Cases

- If the training labels contain only one binary class, training must stop with
  a readable validation error before producing misleading metrics.
- If a stratified validation split cannot preserve both classes because the
  dataset is too small, the run must report the condition clearly and use only a
  documented synthetic-test fallback where appropriate.
- If any labeled image cannot be loaded or preprocessed, the run must report the
  affected image identifiers instead of silently dropping examples.
- If validation probabilities contain ties across thresholds, threshold
  selection must pick the lowest threshold among tied best-F1 thresholds.
- If class imbalance is present, the training configuration must account for it
  and the metrics report must make the imbalance visible.
- If private Krones data is used locally, model outputs and prediction artifacts
  must remain ignored and untracked by default.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST train a binary classifier for the official
  `0 = Reusable` and `1 = Not Reusable` target using labeled training images.
- **FR-002**: The system MUST consume the validated dataset-loading outputs from
  prior specs rather than hardcoded private dataset paths.
- **FR-003**: The system MUST consume SPEC-004 ROI crops and normalized
  `384x384` preprocessing outputs for classifier inputs.
- **FR-004**: The system MUST create a reproducible 80/20 stratified
  train/validation split with a fixed seed that preserves binary class
  representation when dataset size allows.
- **FR-005**: The system MUST reject training inputs that cannot support a valid
  binary training and validation evaluation.
- **FR-006**: The system MUST account for class imbalance during V1 training
  with weighted binary loss only; weighted sampling and focal loss are deferred
  to later classifier-improvement specs.
- **FR-007**: The system MUST report validation F1-score as the primary model
  selection metric.
- **FR-008**: The system MUST search for and save a best validation decision
  threshold for later inference, using the lowest threshold when multiple
  thresholds tie for best validation F1-score.
- **FR-009**: The system MUST save model weights for the best validation
  F1-score after threshold search, rather than selecting by validation loss or
  final epoch alone.
- **FR-010**: The system MUST save classifier metrics, including validation
  F1-score, selected threshold, and confusion-count information.
- **FR-011**: The system MUST save validation predictions with image
  identifiers, true labels, probabilities, and thresholded predictions.
- **FR-012**: The system MUST make runs reproducible from explicit
  configuration and seed values.
- **FR-013**: The system MUST provide a synthetic training smoke path that
  validates the feature without private Krones data.
- **FR-014**: The system MUST keep generated model, report, and prediction
  artifacts in ignored output locations by default.
- **FR-015**: The system MUST NOT generate Kaggle submissions, infer test
  labels, train detectors, run hybrid inference, produce Grad-CAM outputs, build
  dashboard features, or implement memory-bank behavior in this spec.

### Constitution Alignment *(mandatory)*

- **Binary Output**: This feature creates the first classifier that preserves
  the official `0 = Reusable` and `1 = Not Reusable` target contract.
- **Accuracy/Speed**: Validation F1-score is the primary model-selection
  outcome, and the baseline must remain lightweight enough to support later fast
  inference benchmarking.
- **ROI/Hybrid Flow**: Classifier training depends on SPEC-004 ROI and
  normalized preprocessing outputs; detector fallback and fusion remain later
  specs.
- **Annotation/Version Strategy**: This is `SPEC-005: Binary Classifier
  Training`, the V1 classifier baseline after foundation, dataset validation,
  annotation parsing, and ROI preprocessing.
- **Reproducibility/Leakage**: The train/validation split, seed, threshold, and
  metrics must be saved, and no test labels may be inferred or introduced.
- **Confidentiality**: Private training images and generated private-derived
  weights, reports, and predictions remain local/private and ignored by source
  control by default.
- **Explainability**: This feature saves confidence and validation prediction
  facts needed by later explanation and dashboard specs, but it does not produce
  Grad-CAM or detector visualizations.
- **Bias/Imbalance**: Training must account for binary class imbalance and
  report metrics that make false positives and false negatives visible. V1 uses
  weighted binary loss only for imbalance handling.

### Key Entities *(include if feature involves data)*

- **Classifier Training Run**: A reproducible execution that records dataset
  inputs, split settings, seed, training configuration, selected model artifact,
  and output paths.
- **Training Example**: A labeled image after dataset validation and SPEC-004
  ROI/preprocessing, with image identifier, target label, and normalized image
  input.
- **Validation Prediction**: A validation record containing image identifier,
  true label, predicted probability, thresholded prediction, and split metadata.
- **Classifier Metrics Report**: Summary of validation F1-score, selected
  threshold, class counts, confusion counts, and run metadata.
- **Best Threshold Record**: Saved threshold selection result used by later
  inference and submission specs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A synthetic training smoke run completes in under one minute and
  produces model, metrics, threshold, and validation prediction artifacts
  without private dataset access.
- **SC-002**: Validation predictions include one row for every validation
  example in 100% of covered test runs.
- **SC-003**: The metrics report includes validation F1-score, selected
  threshold, class counts, and confusion counts in 100% of covered test runs.
- **SC-004**: Repeating a synthetic run with the same seed produces the same
  80/20 split assignment, selected threshold, and validation prediction
  ordering.
- **SC-005**: Generated classifier artifacts are written only to approved output
  locations and remain untracked by source control by default.
- **SC-006**: Attempts to train with invalid binary labels, missing images, or
  unusable class distribution produce readable errors naming the problem.
- **SC-007**: The feature produces no Kaggle submission file, detector output,
  Grad-CAM image, dashboard workflow, memory-bank artifact, or test-label
  inference result.

## Assumptions

- SPEC-001 through SPEC-004 are complete enough to provide configuration,
  dataset loading, annotation parsing, ROI cropping, and normalized `384x384`
  preprocessing inputs.
- The first classifier baseline is intended to prove the V1 training pipeline
  before submission generation, detector training, hybrid inference, or
  explainability work begins.
- Private Krones dataset access may be used for local/manual validation, but
  automated tests can rely on synthetic fixtures.
- Generated model weights, metrics, thresholds, and predictions are local
  artifacts and are not intended for public source control.
- The next spec after this one will handle inference/submission behavior rather
  than expanding this training spec into a submission generator.
