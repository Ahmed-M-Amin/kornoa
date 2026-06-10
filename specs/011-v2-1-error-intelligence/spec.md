# Feature Specification: V2.1 Error Intelligence

**Feature Branch**: `011-v2-1-error-intelligence`

**Created**: 2026-06-10

**Status**: Draft

**Input**: User description: "From `docs/new_implinmination.md`, create the first Phase 1 spec as V2.1. Phase 1 is Error Intelligence: build a validation error audit for original V2B before any new training, link errors to baseline facts, detector evidence when available, image-quality diagnostics, and manual review groups."

## Clarifications

### Session 2026-06-10

- Q: How should V2.1 define high-confidence versus near-threshold review groups? -> A: Fixed bands: near-threshold is `abs(probability - threshold) <= 0.05`; high-confidence is `abs(probability - threshold) >= 0.30`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build V2B Error Audit (Priority: P1)

As the model developer, I need one validation error audit for the locked original V2B baseline so I can see exactly which validation images are true positives, true negatives, false positives, and false negatives before changing any model or threshold.

**Why this priority**: This is the foundation for all later F1 improvement work. Without a trusted error audit, new candidates can repeat the V2B-Enhanced failure pattern by improving local F1 while over-rejecting reusable bottles.

**Independent Test**: Can be fully tested by comparing the produced audit row count and confusion categories against the locked V2B validation predictions and labels, then confirming every validation image has exactly one audit row and one error type.

**Acceptance Scenarios**:

1. **Given** the locked V2B validation predictions and validation labels are available, **When** the audit is produced, **Then** every validation image appears once with `image_id`, true label, V2B probability, V2B prediction, error type, and distance from the selected threshold.
2. **Given** the audit has been produced, **When** the error type counts are summarized, **Then** the TP, FP, FN, and TN counts match the V2B baseline metrics used for comparison.
3. **Given** a validation image is near the V2B threshold, **When** the audit is reviewed, **Then** the image is identifiable as near-threshold using the recorded distance from threshold.

---

### User Story 2 - Review High-Risk Error Groups (Priority: P2)

As the model developer, I need the audit grouped into high-confidence and near-threshold false positives and false negatives so I can distinguish calibration problems from model-capacity, preprocessing, crop-quality, or ambiguous-label problems.

**Why this priority**: The roadmap targets hidden-test F1 and runtime together. Grouped error review helps decide whether the next step should be calibration, crop/preprocessing repair, a stronger classifier, detector evidence, or no change.

**Independent Test**: Can be tested by selecting each required review group and verifying that every row in the group satisfies the group definition, while no validation labels outside the validation split are used.

**Acceptance Scenarios**:

1. **Given** the audit includes prediction confidence and threshold distance, **When** review groups are generated, **Then** separate groups exist for high-confidence false positives, high-confidence false negatives, near-threshold false positives, and near-threshold false negatives using the fixed review bands.
2. **Given** reusable bottles are falsely rejected by V2B, **When** the false-positive group is reviewed, **Then** those cases are visible as a specific over-rejection risk group.
3. **Given** the grouped review is complete, **When** a later candidate claims an F1 gain, **Then** the candidate can be compared against the same error groups to show which errors were fixed or newly created.

---

### User Story 3 - Attach Supporting Evidence (Priority: P3)

As the model developer, I need detector and image-quality evidence attached to the V2B audit where available so I can judge whether errors are caused by missing defect evidence, crop quality, blur, brightness, or other non-model factors.

**Why this priority**: Detector and rule changes should only be considered when category-level evidence explains a specific error pattern. Image-quality evidence also helps avoid spending training time on errors caused by bad crops or ambiguous inputs.

**Independent Test**: Can be tested by joining known detector and image-quality records to the audit and verifying that missing evidence is reported explicitly instead of silently dropping audit rows.

**Acceptance Scenarios**:

1. **Given** detector evidence exists for a validation image, **When** the audit is produced, **Then** the image includes detector category, confidence, bounding-box, and area evidence.
2. **Given** detector evidence is missing for a validation image, **When** the audit is produced, **Then** the image remains in the audit and the missing detector evidence is visible.
3. **Given** image-quality diagnostics are available, **When** the audit is produced, **Then** the image includes quality indicators such as brightness, blur, crop size, and crop confidence when available.

### Edge Cases

- If the locked V2B baseline has not been created, the audit must not be accepted as complete.
- If validation labels or prediction rows do not align by `image_id`, mismatches must be reported clearly before any conclusions are drawn.
- If probability values or predictions are missing for any validation image, those rows must be flagged and excluded from metric claims until corrected.
- If detector or image-quality evidence is unavailable, the audit must still preserve every validation image and record the missing evidence status.
- If any test labels, sample-solution labels, or public leaderboard feedback are used to create the audit, the audit is invalid.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST create one V2B validation error audit linked to the locked original V2B baseline.
- **FR-002**: The audit MUST include one row per validation image and MUST preserve each validation `image_id`.
- **FR-003**: Each audit row MUST include the true validation label, V2B probability for `target = 1`, V2B binary prediction, error type, and distance from the selected V2B threshold.
- **FR-004**: The audit MUST summarize TP, FP, FN, TN, precision, recall, F1, validation target distribution, and prediction target distribution.
- **FR-005**: The summarized metrics MUST match the locked V2B baseline before the audit is considered valid.
- **FR-006**: The audit MUST identify high-confidence false positives, high-confidence false negatives, near-threshold false positives, and near-threshold false negatives using fixed probability-distance bands.
- **FR-007**: Near-threshold rows MUST be defined as `abs(probability - threshold) <= 0.05`, and high-confidence rows MUST be defined as `abs(probability - threshold) >= 0.30`.
- **FR-008**: The audit MUST identify reusable bottles over-rejected by V2B as a specific false-positive risk group.
- **FR-009**: The audit MUST attach detector category-level evidence when available, including `image_id`, category, confidence, bounding-box, and area.
- **FR-010**: The audit MUST attach image-quality evidence when available, including brightness, blur, crop size, and crop confidence.
- **FR-011**: Missing detector or image-quality evidence MUST be reported without removing validation rows from the audit.
- **FR-012**: The audit MUST support candidate comparison by preserving enough baseline columns to later count fixed errors, new errors, and changed predictions versus original V2B.
- **FR-013**: The feature MUST produce review-ready outputs for manual analysis of the required error groups.
- **FR-014**: The feature MUST prohibit test labels, sample-solution labels, and public leaderboard results from influencing the audit or its conclusions.
- **FR-015**: The feature MUST record the source paths or identifiers of the baseline predictions, validation labels, threshold record, detector evidence, and image-quality evidence used to produce the audit.
- **FR-016**: The feature MUST not perform new training, create a Kaggle submission, or tune thresholds on public leaderboard feedback.

### Constitution Alignment *(mandatory)*

- **Binary Output**: Preserves `0 = Reusable` and `1 = Not Reusable` as the only target meanings and reports all error groups using those labels.
- **Accuracy/Speed**: Improves decision quality before training by identifying which errors matter most for F1; does not add inference runtime because it is an offline audit feature.
- **ROI/Hybrid Flow**: Uses existing classifier predictions as the decision source and attaches detector evidence only as analysis support, not as an override.
- **Annotation/Version Strategy**: Treats original V2B as the locked baseline for V2.1 analysis and prepares evidence for later V2/V3/V4 candidate decisions.
- **Reproducibility/Leakage**: Requires saved source identifiers, baseline alignment, validation-only labels, and no test-label or public-leaderboard tuning.
- **Confidentiality**: Keeps private dataset references, predictions, and audit artifacts inside local or competition-safe project outputs.
- **Explainability**: Makes each error explainable through probability, threshold distance, detector evidence, image-quality evidence, and manual review group.
- **Bias/Imbalance**: Tracks false-positive over-rejection of reusable bottles, target distribution, and class-level precision/recall behavior before accepting new candidates.

### Key Entities *(include if feature involves data)*

- **Baseline Reference**: The locked original V2B comparison record, including validation metrics, threshold, prediction path, checkpoint path, and submission distribution.
- **Validation Audit Row**: One validation image with identity, true label, V2B probability, V2B prediction, error type, threshold distance, optional detector evidence, and optional image-quality evidence.
- **Error Review Group**: A named subset of audit rows used for manual analysis, such as high-confidence false positives or near-threshold false negatives, based on the fixed probability-distance bands.
- **Detector Evidence Record**: Category-level defect evidence for an image, including category, confidence, bounding-box, and area when available.
- **Image Quality Record**: Non-label diagnostic evidence for an image, including brightness, blur, crop size, and crop confidence when available.
- **Audit Summary**: Aggregate metrics and distributions proving the audit matches the locked V2B baseline and is safe to use for candidate decisions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of validation images from the locked V2B prediction set appear exactly once in the audit.
- **SC-002**: Audit TP, FP, FN, TN, precision, recall, and F1 match the locked V2B baseline within rounding tolerance before the audit is marked valid.
- **SC-003**: 100% of audit rows include `image_id`, true label, V2B probability, V2B prediction, error type, and threshold distance.
- **SC-004**: All four required manual review groups are produced using the fixed probability-distance bands: high-confidence false positives, high-confidence false negatives, near-threshold false positives, and near-threshold false negatives.
- **SC-005**: Missing optional detector or image-quality evidence is explicitly reported for 100% of affected rows.
- **SC-006**: The audit records zero use of test labels, sample-solution labels, or public leaderboard feedback.
- **SC-007**: A reviewer can identify the top false-positive over-rejection cases and top false-negative missed-defect cases from the produced review outputs without rerunning model training.

## Assumptions

- The original V2B baseline lock exists or will be created before this feature is accepted as complete.
- Validation labels are available only for the validation split and are permitted for offline model analysis.
- V2B validation predictions include probabilities or scores that can be compared to the locked threshold.
- Detector and image-quality evidence may be incomplete; incompleteness should be visible rather than blocking the entire audit.
- Version label `2.1` refers to this first roadmap implementation spec, not to a new trained model.
- Training, submission generation, public leaderboard tuning, and V3 work are out of scope for this spec.
