# Feature Specification: V5 Strong Classifier

**Feature Branch**: `010-v5-strong-classifier`

**Created**: 2026-06-07

**Status**: Draft

**Input**: User description: "Create SPEC-010 / V5 strong classifier from docs/implementation.md Phase 10 and the attached SPEC-010 V5 explanation. Build on V2B, V4.2 results, and the existing classifier pipeline. The goal is a stronger classifier that beats the current V4.2 public score, exports validation and test probabilities, generates a valid submission, avoids detector retraining and test-label leakage, and prepares the project for V5 strong classifier experiments."

## Clarifications

### Session 2026-06-07

- Q: What candidate scope should SPEC-010 use for the first V5 implementation? -> A: Primary ConvNeXt-Tiny at 512, with EfficientNet-B2 fallback only if resource limits block it.
- Q: How should V5 use hard examples by default? -> A: Keep hard examples analysis-only by default; oversampling requires explicit opt-in.
- Q: What train/validation split should V5 use for model and threshold selection? -> A: Reuse the existing V2B-compatible train/validation split for V5 model and threshold selection.
- Q: What inference-time ceiling should an accepted V5 candidate satisfy? -> A: V5 inference must be no more than 2x slower than V2B for the accepted candidate.
- Q: What schema should V5 validation prediction export require? -> A: Validation predictions require `image_id,true_label,prob_bad,classifier_prediction,target`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Stronger Classifier Candidate (Priority: P1)

The practitioner can run a V5 classifier experiment that uses the existing classifier foundation, stronger image-level learning settings, and validation-safe model selection to produce a candidate stronger than the current V4.2 public baseline.

**Why this priority**: V4.2 hybrid improved V2B only slightly. The next meaningful score gain must come from a stronger classifier rather than more detector-only work.

**Independent Test**: Can be tested by running the V5 experiment on the configured training and validation split, then verifying that validation F1, selected threshold, saved model identity, class distribution, and prediction distribution are reported without using test labels.

**Acceptance Scenarios**:

1. **Given** the existing classified bottle dataset and a V5 experiment configuration, **When** the practitioner runs V5 training, **Then** the workflow saves the best classifier candidate, validation predictions, validation metrics, and selected threshold under the V5 output location.
2. **Given** multiple valid V5 candidate settings, **When** validation metrics are compared, **Then** the selected candidate is chosen by validation F1 after threshold calibration while preserving speed and memory feasibility.
3. **Given** the current public-best V4.2 score of `0.92181`, **When** V5 results are reviewed, **Then** the report clearly states whether V5 is accepted as the new best classifier or kept only as analysis.

---

### User Story 2 - Export Validation and Test Probabilities (Priority: P2)

The practitioner can export validation and test classifier probabilities in a stable schema so V5 can be evaluated directly and can later feed hybrid inference without target-only limitations.

**Why this priority**: V4.1/V4.2 showed that hybrid inference requires classifier probabilities, not only final targets. V5 must preserve probability outputs from the start.

**Independent Test**: Can be tested by generating validation and test prediction files and checking row counts, required columns, binary targets, probability bounds, and duplicate image IDs.

**Acceptance Scenarios**:

1. **Given** a completed V5 classifier candidate, **When** validation prediction export runs, **Then** the validation file includes image ID, true label, bad probability, classifier prediction, and threshold-derived target information needed for validation review.
2. **Given** the V5 classifier candidate and test images, **When** test prediction export runs, **Then** the test probability file contains exactly the required image-level probability and target columns for all expected test images.
3. **Given** any probability export, **When** validation checks run, **Then** all `prob_bad` values are within `[0, 1]`, targets are binary, and duplicate image IDs are rejected.

---

### User Story 3 - Generate Review-Ready Submission Without Leakage (Priority: P3)

The practitioner can generate a strict V5 Kaggle submission and review all required reports before any manual public submission decision.

**Why this priority**: Public-score feedback must not drive hidden training or threshold choices, and generated outputs must remain local/ignored until intentionally reviewed.

**Independent Test**: Can be tested by running submission export from a completed V5 artifact set and validating that the output contains only `image_id,target`, uses the saved threshold, and writes no private artifacts to tracked source locations.

**Acceptance Scenarios**:

1. **Given** a trained V5 classifier and selected threshold, **When** submission export runs, **Then** the generated submission contains exactly `image_id,target` with integer binary targets.
2. **Given** generated V5 reports and predictions, **When** the practitioner reviews outputs, **Then** validation F1, selected threshold, target distribution, prediction distribution, and submission row count are available before public submission.
3. **Given** generated weights, predictions, reports, and submissions, **When** repository status is checked, **Then** those generated artifacts are excluded from tracked source control.

### Edge Cases

- GPU memory is insufficient for the preferred V5 classifier/image-size combination, so the experiment must support a documented fallback candidate without changing the acceptance rule.
- The primary candidate beats V4.2 accuracy but exceeds 2x V2B inference time; the workflow must report the tradeoff and evaluate the fallback candidate before acceptance.
- Validation F1 improves locally but public score does not beat V4.2; V4.2 remains the current best public submission.
- Hard-example oversampling is enabled without explicit opt-in or without split-safety reporting.
- Validation prediction export is missing or invalid for any required column in `image_id,true_label,prob_bad,classifier_prediction,target`.
- Test prediction export contains missing images, duplicate image IDs, non-binary targets, or probabilities outside `[0, 1]`.
- Submission export accidentally includes diagnostic columns beyond `image_id,target`.
- Public leaderboard score is available only after manual submission and must not be used to retrain, tune thresholds, or select hidden settings.
- The V2B-compatible split is unavailable or cannot be reconstructed; the workflow must fail with a clear split-reproducibility error rather than silently selecting a new split.
- Detector-only V3 remains weak and must not be retrained as part of SPEC-010.
- V4.2 hybrid logic may later consume V5 probabilities, but SPEC-010 must not change detector fusion rules except for future-compatible probability output.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST preserve the challenge binary target contract: `0 = Reusable` and `1 = Not Reusable`.
- **FR-002**: System MUST define a V5 strong classifier experiment scope that builds on the existing accepted classifier pipeline and previous V2B/V4.2 lessons.
- **FR-003**: System MUST use ConvNeXt-Tiny at 512 image size as the primary V5 candidate.
- **FR-004**: System MUST use EfficientNet-B2 as the fallback candidate only when resource limits block the primary ConvNeXt-Tiny at 512 candidate.
- **FR-004a**: System MUST NOT expand SPEC-010 into a broad model sweep beyond the primary candidate and resource-limit fallback.
- **FR-005**: System MUST support imbalance-aware learning choices such as weighted loss, focal loss, and class-balanced sampling.
- **FR-005a**: System MUST keep hard examples analysis-only by default; hard-example oversampling requires explicit opt-in and must report split-safety controls.
- **FR-006**: System MUST support stronger safe augmentations that preserve bottle defect evidence and do not remove internal dark defects.
- **FR-007**: System MUST compute validation F1 for every serious V5 candidate.
- **FR-008**: System MUST calibrate the best decision threshold using validation data only.
- **FR-008a**: System MUST reuse the existing V2B-compatible train/validation split for V5 model selection and threshold calibration.
- **FR-009**: System MUST save the selected best V5 classifier artifact under `outputs/kaggle_v5/v5_strong_classifier/models/classifier_best.pth`.
- **FR-010**: System MUST save validation classifier predictions under `outputs/kaggle_v5/v5_strong_classifier/predictions/val_classifier_predictions.csv`.
- **FR-010a**: Validation classifier predictions MUST include `image_id`, `true_label`, `prob_bad`, `classifier_prediction`, and `target`.
- **FR-011**: System MUST save test classifier predictions under `outputs/kaggle_v5/v5_strong_classifier/predictions/test_classifier_predictions.csv`.
- **FR-012**: Test classifier predictions MUST include `image_id`, `prob_bad`, `classifier_prediction`, and `target`.
- **FR-013**: System MUST save the selected threshold report under `outputs/kaggle_v5/v5_strong_classifier/reports/best_threshold.json`.
- **FR-014**: System MUST save classifier metrics under `outputs/kaggle_v5/v5_strong_classifier/reports/classifier_metrics.json`.
- **FR-015**: System MUST generate the V5 submission under `outputs/kaggle_v5/v5_strong_classifier/submissions/submission_v5.csv`.
- **FR-016**: The V5 submission MUST contain exactly two columns: `image_id` and `target`.
- **FR-017**: System MUST report validation target distribution, validation prediction distribution, test prediction distribution, selected threshold, validation F1, submission row count, and inference-time comparison against V2B.
- **FR-018**: System MUST state whether V5 beats the current best V4.2 public score of `0.92181` after manual public-score review.
- **FR-019**: System MUST keep V4.2 as the current best public submission unless V5 achieves a public score greater than `0.92181`.
- **FR-019a**: System MUST require the accepted V5 candidate to run no more than 2x slower than the accepted V2B classifier benchmark; if the primary candidate exceeds this ceiling, the fallback candidate must be evaluated.
- **FR-020**: System MUST NOT retrain the V3 detector as part of SPEC-010.
- **FR-021**: System MUST NOT use test labels, sample-solution labels, public-score-derived thresholds, or manual test inspection for training, threshold search, or model selection.
- **FR-022**: System MUST NOT submit automatically; it must generate submission and reports for review first.
- **FR-023**: System MUST NOT commit generated outputs, artifacts, model weights, predictions, reports, or submissions.
- **FR-024**: System SHOULD keep V5 probability exports compatible with future V4 hybrid inference consumption.

### Constitution Alignment *(mandatory)*

- **Binary Output**: SPEC-010 preserves the required `0 = Reusable` and `1 = Not Reusable` target contract for validation predictions, test predictions, and submission output.
- **Accuracy/Speed**: V5 targets a public-score improvement over V4.2 `0.92181` while keeping accepted inference no more than 2x slower than V2B and feasible for local, Kaggle, and Colab constraints.
- **ROI/Hybrid Flow**: V5 remains an ROI-first classifier feature. It produces classifier probabilities that can later feed hybrid gating, but detector fusion changes are out of scope.
- **Annotation/Version Strategy**: This spec intentionally defines SPEC-010 as V5 Strong Classifier based on the latest V4.2 result and attached brief, even though the older roadmap text labels V5 as dashboard work.
- **Reproducibility/Leakage**: V5 requires saved configuration, selected threshold, metrics, validation predictions, test predictions, model artifact identity, reuse of the V2B-compatible validation split, and explicit no-leakage boundaries.
- **Confidentiality**: Private data, generated weights, predictions, metrics, and submissions stay in ignored local/Kaggle/Colab output locations and are not committed.
- **Explainability**: V5 exports probabilities, confidence-supporting metrics, threshold decisions, and distribution reports. Grad-CAM and dashboard visualization remain out of scope for this classifier spec.
- **Bias/Imbalance**: V5 explicitly addresses imbalance through weighted or focal objectives, balanced sampling options, hard-example-aware sampling, and FP/FN distribution review.

### Key Entities *(include if feature involves data)*

- **V5 Classifier Candidate**: A trained classifier experiment with model identity, image-size setting, imbalance strategy, augmentation policy, validation F1, threshold, and feasibility notes.
- **Validation Prediction Export**: Image-level validation output containing `image_id`, `true_label`, `prob_bad`, `classifier_prediction`, and `target`.
- **Test Prediction Export**: Image-level test output containing `image_id`, `prob_bad`, `classifier_prediction`, and `target` for submission generation and future hybrid consumption.
- **Threshold Report**: Saved threshold-selection result containing selected threshold, validation F1, and supporting metric counts.
- **Classifier Metrics Report**: Saved V5 evaluation summary covering validation F1, target distribution, prediction distribution, confusion counts, and experiment identity.
- **V5 Submission**: Strict public-submission candidate containing only `image_id,target`.
- **Public Score Decision Record**: Review note comparing V5 public score with V4.2 `0.92181` and stating whether V5 becomes the accepted best public submission.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: V5 validation reporting includes validation F1 and a selected threshold for 100% of completed serious candidates.
- **SC-002**: V5 test prediction export contains exactly the required columns `image_id`, `prob_bad`, `classifier_prediction`, and `target`.
- **SC-003**: 100% of exported `prob_bad` values are within `[0, 1]`.
- **SC-004**: 100% of exported validation and test targets are binary integers using `0 = Reusable` and `1 = Not Reusable`.
- **SC-005**: V5 submission export contains exactly `image_id,target` and no diagnostic columns.
- **SC-006**: V5 output files are generated under `outputs/kaggle_v5/v5_strong_classifier/`.
- **SC-007**: Repository checks show 0 generated V5 weights, predictions, reports, submissions, or private artifacts staged for commit.
- **SC-008**: V5 acceptance review compares the V5 public score against V4.2 `0.92181`.
- **SC-009**: V5 becomes the accepted best public submission only when its public score is greater than `0.92181`.
- **SC-010**: The implementation test suite passes before V5 is considered ready for manual submission review.
- **SC-011**: V3 detector-only remains documented as weak at public score `0.74169` and is not retrained in SPEC-010.
- **SC-012**: The result summary reports validation target distribution, validation prediction distribution, test prediction distribution, selected threshold, validation F1, and submission row count.
- **SC-013**: Hard-example oversampling is inactive by default and can only affect training when explicitly enabled with split-safety reporting.
- **SC-014**: V5 model selection and threshold calibration use the same V2B-compatible validation split for 100% of accepted V5 candidates.
- **SC-015**: The accepted V5 candidate reports measured inference time and is no more than 2x slower than the accepted V2B classifier benchmark.
- **SC-016**: V5 validation prediction export contains exactly the required core columns `image_id`, `true_label`, `prob_bad`, `classifier_prediction`, and `target`, with any extra diagnostic columns documented separately.

## Assumptions

- The attached `spec_010_v5_strong_classifier.md` reflects the latest project decision after V4.2 and supersedes older roadmap wording that placed dashboard work at V5/SPEC-010.
- V4.2 hybrid is the current best public submission with score `0.92181`; V2B scored `0.92121`; V3 detector-only scored `0.74169`.
- V4.2 changed 32 rows compared with V2B, all from `0` to `1` using detector decision source, so detector fallback helps slightly but does not close the gap to the final `0.98+` target.
- The preferred V5 candidate is ConvNeXt-Tiny at 512 image size, with EfficientNet-B2 used only as fallback when resource limits block the primary candidate.
- Hard examples are analysis-only by default because V2B-HE oversampling overfit and did not generalize publicly.
- V5 comparisons are most useful when the validation split remains compatible with V2B/V4.2 rather than changing split composition.
- The accepted V2B classifier benchmark remains the runtime baseline for V5 speed acceptance.
- The existing classifier pipeline, V2 configuration conventions, threshold search, submission generation, and V4.2 probability-export lessons are available for the later planning and implementation phases.
- Generated outputs remain ignored and local under `outputs/kaggle_v5/v5_strong_classifier/`.
