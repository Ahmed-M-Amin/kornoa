# Feature Specification: Hybrid Inference Engine

**Feature Branch**: `009-hybrid-inference-engine`

**Created**: 2026-06-06

**Status**: Draft

**Input**: User description: "Create SPEC-009: Hybrid Inference Engine for the Krones bottle inspection project. Use docs/implementation.md Phase 09 / V4 Hybrid Inference as the authoritative scope. Add classifier confidence gate, detector only for uncertain images, and fusion decision logic. Preserve accepted fast V2 classifier and use SPEC-008 detector artifacts. Do not implement Grad-CAM dashboard, feature memory bank, ensemble, distillation, or final report assets."

## Clarifications

### Session 2026-06-06

- Q: Should detector evidence be allowed to override uncertain classifier decisions in both directions, or only reject bottles? -> A: Rejection-only; detector evidence can change uncertain images to `1 = Not Reusable`, but cannot clear them to `0 = Reusable`.
- Q: What detector usage ceiling should guide hybrid parameter selection? -> A: Prefer configurations with detector usage at or below 30% of validation overlap.
- Q: Should conditional defect area thresholds be global or category-specific? -> A: Use per-category area thresholds with a documented default fallback for unmapped conditional categories.
- V4.1 repair: test-time hybrid inference MUST use exported V2B classifier probabilities from `outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv` instead of target-only `submission_v2b.csv`; the repair MUST also generate `outputs/hybrid/v4/reports/v2b_vs_v4_diff.json`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Keep Fast Classifier Decisions (Priority: P1)

The practitioner can run inference so that the accepted fast classifier remains the default decision source, and images with clear classifier confidence are returned immediately without detector review.

**Why this priority**: Hybrid inference only succeeds if it preserves the speed and public-score baseline established by the accepted V2 classifier instead of turning every image into a detector workload.

**Independent Test**: Can be tested by providing classifier predictions with scores far from the tuned threshold, detector predictions that disagree, and verifying that final decisions, confidence, and decision source stay classifier-based.

**Acceptance Scenarios**:

1. **Given** a bottle image with classifier confidence outside the uncertainty band, **When** hybrid inference runs, **Then** the final binary target equals the classifier decision and the detector is not used for that image.
2. **Given** detector evidence that disagrees with a confident classifier decision, **When** hybrid fusion is applied, **Then** the detector evidence does not override the classifier.
3. **Given** a batch of mostly confident classifier predictions, **When** the batch completes, **Then** the detector usage report shows that only uncertain images were eligible for detector review.

---

### User Story 2 - Review Uncertain Images With Detector Evidence (Priority: P2)

The practitioner can route only classifier-uncertain images through detector evidence, then use defect category and area logic to reject the bottle when the detector identifies an always-faulty defect or a conditional defect above its threshold.

**Why this priority**: The detector adds value on hard cases and supports industrial defect reasoning, but only when classifier uncertainty justifies the runtime cost.

**Independent Test**: Can be tested by creating synthetic predictions for uncertain images with always-faulty defects, conditional defects above and below area thresholds, missing detector evidence, and detector outputs below confidence thresholds.

**Acceptance Scenarios**:

1. **Given** an uncertain classifier prediction and a detected always-faulty defect above the detector confidence threshold, **When** fusion runs, **Then** the final target is `1 = Not Reusable` and the defect reason identifies the always-faulty evidence.
2. **Given** an uncertain classifier prediction and a conditional defect whose area is above its configured threshold, **When** fusion runs, **Then** the final target is `1 = Not Reusable` and the defect reason identifies the conditional-area rule.
3. **Given** an uncertain classifier prediction but no usable detector evidence, **When** fusion runs, **Then** the final target falls back to the classifier threshold decision.

---

### User Story 3 - Tune and Report Hybrid Behavior Safely (Priority: P3)

The practitioner can tune classifier uncertainty and detector decision thresholds on validation overlap only, then save validation reports, final predictions, and strict Kaggle submission output without using test labels.

**Why this priority**: Hybrid inference changes model-selection behavior. It must prove whether the detector improves hard cases without leakage, hidden manual choices, or submission-format drift.

**Independent Test**: Can be tested by joining classifier validation predictions and detector validation predictions by image ID, verifying that only overlap rows with labels are used for parameter search, and checking all saved reports and submission columns.

**Acceptance Scenarios**:

1. **Given** classifier validation predictions, detector validation predictions, and labels with partial overlap, **When** hybrid parameter search runs, **Then** only rows present in both validation prediction sets with labels are used for tuning.
2. **Given** selected hybrid parameters, **When** validation reporting completes, **Then** reports compare classifier baseline, detector baseline, and hybrid performance on the same overlap set.
3. **Given** final test inference outputs, **When** submission export runs, **Then** the submission contains exactly `image_id,target` and no debug, confidence, source, or label columns.

### Edge Cases

- Classifier prediction files use different but recognizable column names for image ID, score, target, or label.
- Detector prediction files contain multiple detections per image and require image-level aggregation.
- Detector prediction files lack confidence-like evidence for threshold tuning.
- Classifier test submission has only `image_id,target` and no available classifier probability artifact for uncertainty gating.
- V4.1 probability export exists but contains missing columns, non-binary predictions, duplicate image IDs, fewer than the expected test rows, or `prob_bad` values outside `[0, 1]`.
- Validation prediction overlap is empty or too small to support trustworthy hybrid parameter search.
- Detector evidence exists for confident classifier predictions but must not be used.
- Detector evidence is missing for classifier-uncertain images.
- Conditional defect categories are detected but area information is missing or invalid.
- Detector usage rises high enough to threaten the intended speed advantage.
- Test labels, public leaderboard feedback, sample-solution labels, and manual test inspection must never influence hybrid parameters.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST preserve `0 = Reusable` and `1 = Not Reusable` for all validation predictions, final predictions, and submission output.
- **FR-002**: System MUST use the accepted fast classifier as the default decision source for every image.
- **FR-003**: System MUST compute classifier uncertainty as a configurable band around the selected classifier decision threshold.
- **FR-004**: System MUST run or apply detector evidence only for images whose classifier score is inside the uncertainty band.
- **FR-005**: System MUST keep the classifier decision for every image whose classifier score is outside the uncertainty band, even when detector evidence disagrees.
- **FR-006**: System MUST keep the classifier decision for uncertain images when detector evidence is missing, invalid, or below usable confidence.
- **FR-007**: System MUST convert detector evidence into image-level defect evidence by selecting the strongest rule-triggering rejection evidence per image when any exists, otherwise selecting the highest-confidence non-rejecting evidence with the relevant defect category or area data.
- **FR-008**: System MUST mark an uncertain image as `1 = Not Reusable` when detector evidence identifies an always-faulty defect above the configured confidence threshold.
- **FR-009**: System MUST mark an uncertain image as `1 = Not Reusable` when detector evidence identifies a conditional defect whose estimated area is above its configured per-category defect-area threshold.
- **FR-010**: System MUST fall back to the classifier threshold decision when detector evidence does not trigger an always-faulty or conditional-area rejection rule.
- **FR-010a**: System MUST NOT allow detector evidence to clear an uncertain classifier rejection to `0 = Reusable`; detector authority is rejection-only in SPEC-009.
- **FR-010b**: System MUST use a documented default conditional-area threshold for unmapped conditional categories and report each fallback use.
- **FR-011**: System MUST tune classifier threshold candidates, uncertainty margins, detector confidence thresholds, and conditional-area thresholds using validation data only.
- **FR-012**: System MUST restrict hybrid parameter search to validation images that have classifier predictions, detector predictions, and ground-truth labels.
- **FR-013**: System MUST compare classifier-only baseline, detector-only baseline where meaningful, and hybrid results on the same validation overlap set.
- **FR-014**: System MUST report detector usage count, detector usage ratio, changed prediction count, changed incorrect-to-correct count, changed correct-to-incorrect count, false-negative fixes, and false-positive fixes.
- **FR-015**: System MUST save validation hybrid predictions with image ID, ground truth, classifier score, classifier target, uncertainty flag, detector evidence, detector-used flag, hybrid target, decision source, and defect reason.
- **FR-016**: System MUST save a hybrid configuration report that records selected thresholds, uncertainty margin, detector usage policy, selected model/artifact identities, input paths, and leakage-control declarations.
- **FR-017**: System MUST save an overlap report covering classifier validation count, detector validation count, overlap count, non-overlap counts, and the statement that parameter search used validation overlap only.
- **FR-018**: System MUST export final submission output with exactly two columns: `image_id` and `target`.
- **FR-019**: System MUST fail with a clear error when test-time classifier confidence or probability is unavailable, because uncertainty gating cannot be inferred from a target-only submission.
- **FR-019a**: System MUST export V2B test classifier probabilities to `outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv` with columns `image_id`, `prob_bad`, `classifier_prediction`, and `target` before V4.1 hybrid submission generation.
- **FR-019b**: System MUST configure hybrid test inference to use `outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv` rather than target-only `submission_v2b.csv`.
- **FR-019c**: System MUST save `outputs/hybrid/v4/reports/v2b_vs_v4_diff.json` comparing V2B classifier targets with V4 hybrid targets, including row count, changed count, changed ratio, target counts, and changed rows.
- **FR-020**: System MUST NOT retrain the accepted classifier, retrain the detector, use test labels, tune on Kaggle public score, or modify accepted V2 or SPEC-008 detector artifacts.
- **FR-021**: System MUST NOT implement Grad-CAM, Streamlit dashboard, feature memory bank, ensemble teacher models, distillation, final Kaggle notebook, or final report assets in SPEC-009.
- **FR-022**: System SHOULD prefer configurations with detector usage at or below 30% of validation overlap when validation F1 remains competitive.
- **FR-023**: System SHOULD prefer the smallest detector usage that preserves or improves validation F1 when candidate hybrid settings are otherwise close.

### Constitution Alignment *(mandatory)*

- **Binary Output**: SPEC-009 preserves the required `0 = Reusable` and `1 = Not Reusable` output while adding decision source and defect-reason diagnostics outside the final submission.
- **Accuracy/Speed**: Hybrid inference optimizes validation F1 and detector usage together; the detector is limited to classifier-uncertain images so runtime remains close to the fast classifier path.
- **ROI/Hybrid Flow**: This feature implements the V4 hybrid flow: ROI-first classifier inference, confidence gate, detector fallback for uncertain cases, and fusion decision logic.
- **Annotation/Version Strategy**: This is the SPEC-009/V4 checkpoint and depends on the accepted V2 classifier and SPEC-008 detector/segmentation artifacts without retraining them.
- **Reproducibility/Leakage**: Selected thresholds, margins, overlap rows, metrics, reports, and input artifact identities are saved; test labels and public leaderboard feedback are excluded from tuning.
- **Confidentiality**: Private images, predictions, detector outputs, reports, and submissions remain in local, Kaggle, or Colab-safe ignored output locations.
- **Explainability**: Hybrid output includes confidence, detector-used flag, decision source, defect category or rule, and inference timing inputs; Grad-CAM and dashboard display remain later specs.
- **Bias/Imbalance**: Reports highlight false-positive and false-negative changes, detector usage by hard cases, and whether hybrid rules improve rare or uncertain defect decisions.

### Key Entities *(include if feature involves data)*

- **Classifier Prediction**: Image-level classifier score, threshold-derived binary target, optional ground-truth label, and uncertainty status.
- **V2B Test Probability Export**: Sample-aligned test classifier probability file with image ID, bad probability, classifier prediction, and target alias used by V4.1 uncertainty gating.
- **Detector Evidence**: Image-level aggregated detector confidence, defect category, optional area estimate, and usability status.
- **Hybrid Rule Configuration**: Selected classifier threshold, uncertainty margin, detector confidence threshold, per-category conditional defect-area thresholds, default conditional-area fallback threshold, and tie-break policy.
- **Validation Overlap Set**: Validation image IDs present in both classifier and detector predictions with labels, used for hybrid tuning and fair comparison.
- **Hybrid Prediction**: Final image-level decision including classifier target, detector-used flag, hybrid target, decision source, defect reason, and timing fields when available.
- **Hybrid Report**: Saved metrics, detector usage statistics, overlap diagnostics, selected configuration, and leakage-control declarations.
- **V2B vs V4 Diff Report**: Saved comparison between V2B classifier targets and V4 hybrid targets, including changed-row counts and changed-row details.
- **Submission File**: Competition-ready image ID and binary target output with no diagnostic columns.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of confident classifier predictions retain the classifier target after hybrid fusion.
- **SC-002**: 100% of detector overrides occur only on classifier-uncertain images with usable rejection evidence.
- **SC-003**: 100% of uncertain images without usable detector evidence fall back to the classifier target.
- **SC-004**: Hybrid parameter search uses 0 test images, 0 test labels, and 0 public-score-derived labels or thresholds.
- **SC-005**: Validation reporting includes classifier baseline, detector baseline where meaningful, and hybrid metrics on exactly the same overlap set.
- **SC-006**: Detector usage ratio, changed predictions, changed incorrect-to-correct, changed correct-to-incorrect, false-negative fixes, and false-positive fixes are all reported for validation.
- **SC-007**: Final submission export contains exactly `image_id,target` with 100% integer binary targets.
- **SC-008**: The workflow fails clearly before submission creation when classifier test confidence is unavailable.
- **SC-009**: Existing classifier-only and detector-preparation regression behavior remains unchanged.
- **SC-010**: For comparable validation F1 candidates within 0.002 absolute F1, the selected hybrid configuration uses less detector review or a narrower uncertainty band.
- **SC-011**: Selected hybrid configuration uses detector review for no more than 30% of validation-overlap images unless every higher-F1 candidate above that limit is explicitly reported as a tradeoff.
- **SC-012**: V4.1 classifier probability export contains 4418 rows, required columns `image_id`, `prob_bad`, `classifier_prediction`, `target`, and only `prob_bad` values in `[0, 1]`.
- **SC-013**: Hybrid configuration records `outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv` as the classifier test prediction input.
- **SC-014**: V4.1 run writes `v2b_vs_v4_diff.json` with row count equal to the final submission row count and without using test labels.

## Assumptions

- `docs/implementation.md` is the repository-local master plan for Phase 09 because `docs/krones-final-implementation-plan.md` is not present in the workspace.
- The external draft `E:\Verschiedenes\SPEC-009-v4-hybrid-inference.md` provides supporting details consistent with Phase 09, while this Spec Kit spec remains stakeholder-focused and technology-agnostic.
- The accepted default classifier is still V2B EfficientNet-B1 with `analysis_only` hard-example strategy and public F1 `0.92121`.
- SPEC-008 produces detector prediction artifacts and defect category information that SPEC-009 can consume without retraining.
- Per-category conditional defect-area thresholds and the default fallback threshold are configured from validation-safe training/validation knowledge, not from test labels or public leaderboard feedback.
- Test-time hybrid inference requires classifier probabilities or confidence scores, not only classifier target labels.
- Detector segmentation masks may be unavailable; bounding-box confidence and estimated area are sufficient for the initial hybrid inference feature.
