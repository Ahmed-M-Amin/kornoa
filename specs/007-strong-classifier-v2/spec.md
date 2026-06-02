# Feature Specification: Strong Classifier V2

**Feature Branch**: `007-strong-classifier-v2`

**Created**: 2026-06-02

**Status**: Draft

**Input**: User description: "Read phase 7 from docs/implementation.md and create SPEC-007: Strong Classifier V2. SPEC-001 to SPEC-006 are complete. V1 EfficientNet-B0 at 384x384 reached F1 0.91656 locally, Kaggle public score 0.91693, best threshold 0.48, with V1 hard-example files available. Create only spec.md and checklists/requirements.md. Do not implement code or tests."

## Clarifications

### Session 2026-06-02

- Q: What minimum validation F1 improvement should V2 target over V1? -> A: Target at least +0.02 validation F1 over V1 while reporting speed cost. Treat +0.02 as the minimum target, not the final ambition; V2 should still attempt stronger results toward 0.95+ if speed remains acceptable.
- Q: What inference-speed ceiling should define acceptable V2 speed? -> A: V2 may be up to 2x slower than V1 at maximum; 2x is the maximum allowed speed cost, not the target speed. Prefer the fastest model when F1 is close.
- Q: What does close validation F1 mean for candidate tie-breaking? -> A: Close validation F1 means absolute validation F1 difference <= 0.002. When candidate models are within 0.002 validation F1, choose the faster model.
- Q: What is the default hard-example strategy? -> A: Allowed strategies are `none`, `analysis_only`, and `oversample`. The default is `analysis_only`; oversampling must be explicitly enabled with `hard_example_strategy=oversample`.
- Q: How must V2 avoid leakage when using V1 hard-example rows for training? -> A: Hard-example rows may be used for V2 training only if those image IDs are excluded from the current V2 validation split. V2 must verify disjoint train/validation image IDs before any oversampling.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train a Stronger Single Classifier (Priority: P1)

The practitioner can produce a V2 classifier candidate that improves meaningfully over the completed V1 EfficientNet-B0 baseline while preserving the official binary target and the fast classifier-only inference path.

**Why this priority**: The current V1 score around 0.9166 F1 and Kaggle public score around 0.91693 are far below the top public leaderboard score around 0.97324, so the next project value is a stronger classifier recipe rather than a new system component.

**Independent Test**: Can be tested by comparing the selected V2 validation F1 after threshold search against the saved V1 baseline metrics and by confirming that normal inference remains a single classifier decision.

**Acceptance Scenarios**:

1. **Given** completed V1 artifacts and a configured V2 classifier experiment, **When** V2 training completes, **Then** the selected V2 candidate has saved metrics, validation predictions, best threshold, and a model checkpoint under the V2 output root.
2. **Given** a V2 validation prediction table, **When** threshold search is performed, **Then** the selected model is chosen by validation F1 after threshold search rather than by raw loss alone.
3. **Given** normal V2 inference, **When** images are evaluated, **Then** each image follows `image -> ROI crop/preprocess -> classifier -> threshold -> target` with no detector, feature memory bank, ensemble, or dashboard dependency.

---

### User Story 2 - Reduce V1 Mistakes With Hard Examples (Priority: P2)

The practitioner can use V1 false positives, false negatives, uncertain cases, and high-loss samples as offline training and analysis inputs to improve the V2 classifier.

**Why this priority**: V1 produced hundreds of false positives and false negatives, and the existing hard-example files identify the highest-value mistakes to target before heavier future modules are considered.

**Independent Test**: Can be tested by verifying that V2 reports loaded hard-example counts, uses only validation-derived hard-example files, and never reads test images for training, threshold tuning, or model selection.

**Acceptance Scenarios**:

1. **Given** V1 hard-example files, **When** V2 training input preparation runs, **Then** false-positive, false-negative, uncertain, and high-loss groups are available for oversampling or analysis.
2. **Given** V2 submission generation or benchmark inference, **When** the workflow runs, **Then** hard-example memory does not run and no hard-example lookup affects predictions.
3. **Given** expected V1 hard-example counts and loaded hard-example files, **When** counts differ between files or summaries, **Then** the discrepancy is reported in V2 comparison artifacts.

---

### User Story 3 - Compare V2 Against V1 Transparently (Priority: P3)

The practitioner can evaluate whether V2 is a true improvement by comparing accuracy, threshold behavior, error counts, speed, model size, backbone, and image size against the saved V1 baseline.

**Why this priority**: A larger or slower classifier is only acceptable if the F1 gain justifies the inference cost, and V2 must remain fast enough for the project's image-flow inference goal.

**Independent Test**: Can be tested by producing a V1-vs-V2 comparison report from completed V1 and V2 artifacts and verifying that all required comparison fields are present.

**Acceptance Scenarios**:

1. **Given** V1 and V2 metrics, **When** the comparison report is generated, **Then** it includes validation F1, Kaggle public score when available, threshold, false positives, false negatives, uncertain samples, inference speed, model size or backbone name, and image size.
2. **Given** V2 benchmark results, **When** V2 uses a larger backbone or image size, **Then** the report states the F1 gain and the inference-speed cost relative to V1.
3. **Given** no Kaggle V2 score yet, **When** the comparison report is generated, **Then** the Kaggle field is explicitly marked unavailable rather than fabricated.

### Edge Cases

- V1 baseline artifacts may exist under the concrete artifact root `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1` or under earlier documented `outputs/kaggle_v1` references; V2 comparison must identify which baseline root was used.
- V1 hard-example files may have manually supplied counts or regenerated summary counts; V2 must report the loaded source counts and not silently overwrite the baseline story.
- Hard-example CSV rows that no longer map to the training split are excluded from training input and reported.
- Hard-example rows are derived from V1 validation predictions. They may be used for V2 training only if those image IDs are excluded from the current V2 validation split.
- V2 must create or verify split separation before any hard-example oversampling. Any image ID used for hard-example oversampling must not appear in V2 validation.
- Validation and test preprocessing remain deterministic even when training augmentation is strengthened.
- Augmentations must not remove visible defects, crop away the relevant bottle region, or delete internal dark regions that may indicate defects.
- Class imbalance may vary by split; V2 must report class counts and selected imbalance strategy for every run.
- Optional 448x448 image-size experiments are evaluated only after 384x384 experiments and must include speed comparison.
- Optional very light test-time augmentation is allowed only as later experimentation when speed remains acceptable; it is not the default V2 path.
- Test images, sample submission rows, or Kaggle public feedback are never used for training, threshold tuning, or model selection.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST define V2 as a stronger binary classifier feature, not as a detector, dashboard, hybrid engine, feature memory bank, or ensemble-first system.
- **FR-002**: System MUST preserve the official target mapping `0 = Reusable` and `1 = Not Reusable`.
- **FR-003**: System MUST keep default V2 inference as a single lightweight classifier path: image, ROI crop/preprocess, classifier probability, saved threshold, binary target.
- **FR-004**: System MUST compare V2 against V1 using the completed V1 baseline: validation F1 around 0.91656, Kaggle public score around 0.91693 when available, best threshold 0.48, and the saved V1 confusion counts.
- **FR-005**: System MUST report class counts for training and validation splits before V2 training decisions are made.
- **FR-006**: System MUST support better imbalance handling through focal loss and weighted random sampling as candidate V2 strategies.
- **FR-007**: System MUST report false-positive and false-negative counts for each V2 candidate after validation threshold search.
- **FR-008**: System MUST support stronger safe training augmentation, including brightness and contrast variation, gamma correction, light blur or noise, and small rotation, shift, and scale variation.
- **FR-009**: System MUST prohibit dangerous augmentation or cropping that removes defects, removes the useful bottle region, or deletes internal dark regions.
- **FR-010**: System MUST keep validation and test preprocessing deterministic and comparable to the V1 evaluation path.
- **FR-011**: System MUST use V1 hard-example files as offline training or analysis inputs only: `false_positives.csv`, `false_negatives.csv`, `uncertain.csv`, and `high_loss_samples.csv`.
- **FR-012**: System MUST NOT use hard-example memory during normal V2 inference, submission generation, or benchmarking.
- **FR-013**: System MUST evaluate EfficientNet-B0 with an improved V2 recipe as V2A.
- **FR-014**: System MUST evaluate EfficientNet-B1 as V2B.
- **FR-015**: System MUST evaluate EfficientNet-B2 as V2C.
- **FR-016**: System MAY evaluate ConvNeXt-Tiny only as a speed-checked lightweight experiment.
- **FR-017**: System MUST NOT use a large heavy model as the default V2 solution.
- **FR-018**: System MUST keep 384x384 as the baseline V2 image size.
- **FR-019**: System MAY evaluate 448x448 only after 384x384 experiments and only with F1 gain versus speed-cost comparison.
- **FR-020**: System MUST save a best threshold for the selected V2 candidate.
- **FR-021**: System MUST compare the V1 threshold against the selected V2 threshold.
- **FR-022**: System MUST select the V2 model by validation F1 after threshold search.
- **FR-023**: System MUST save confusion counts for selected V2 validation predictions.
- **FR-024**: System MUST generate the expected V2 outputs later under `outputs/kaggle_v2/`: model checkpoint, classifier metrics, best threshold, V1-vs-V2 comparison, validation predictions, submission, and inference benchmark.
- **FR-025**: System MUST preserve data confidentiality by keeping private datasets, model weights, predictions, submissions, hard examples, and benchmarks in ignored local output locations.
- **FR-026**: System MUST NOT implement detector, segmentation, Grad-CAM, Streamlit dashboard, feature memory bank, default ensemble, distillation, hybrid inference engine, target remapping, or any test-data training behavior in SPEC-007.
- **FR-027**: System MUST treat 2x V1 inference time as the maximum allowed speed cost for V2, not the target speed.
- **FR-028**: System MUST define close validation F1 as absolute validation F1 difference `<= 0.002` and prefer the faster candidate when competing V2 candidates are within that tolerance.
- **FR-029**: System MUST allow hard-example strategies `none`, `analysis_only`, and `oversample`, with `analysis_only` as the default.
- **FR-030**: System MUST NOT change V2 training composition from hard examples unless `hard_example_strategy=oversample` is explicitly enabled.
- **FR-031**: System MUST verify that any image ID used for hard-example oversampling is excluded from the current V2 validation split before oversampling begins.
- **FR-032**: System MUST include hard-example split-leakage reporting in the V2 training report: hard-example rows loaded, eligible for training, excluded because they are in V2 validation, used for oversampling, and confirmation that training and validation image IDs are disjoint.

### Constitution Alignment *(mandatory)*

- **Binary Output**: V2 preserves `0 = Reusable` and `1 = Not Reusable` for labels, probabilities, thresholds, confusion counts, predictions, and submissions.
- **Accuracy/Speed**: V2 targets a strong F1 improvement over V1 while keeping single-classifier inference fast and benchmarking every selected candidate.
- **ROI/Hybrid Flow**: V2 reuses the existing ROI crop and preprocessing flow and does not introduce detector, segmentation, feature memory bank, or hybrid fallback behavior.
- **Annotation/Version Strategy**: SPEC-007 is the V2 classifier stage. Detector, Grad-CAM, dashboard, feature memory bank, ensemble, and distillation remain later specifications or optional future research.
- **Reproducibility/Leakage**: V2 uses saved V1 artifacts, saved hard-example files, deterministic validation/test preprocessing, saved thresholds, and validation-only model selection. Test images are excluded from training, threshold tuning, and model selection.
- **Confidentiality**: Krones data, checkpoints, predictions, hard examples, benchmarks, and submissions remain local and ignored; no private images or labels are exposed in specification artifacts.
- **Explainability**: V2 provides comparison metrics, confusion counts, and hard-example analysis inputs. Visual explainability is intentionally deferred.
- **Bias/Imbalance**: V2 explicitly addresses class imbalance, reports class counts, tracks false positives and false negatives, and uses validation-derived hard examples to reduce mistake patterns.

### Key Entities *(include if feature involves data)*

- **V1 Baseline Record**: Completed V1 metrics, threshold, benchmark, submission, and hard-example counts used as the required comparison baseline.
- **V2 Candidate**: A single classifier experiment defined by backbone family, image size, imbalance strategy, augmentation recipe, hard-example usage, threshold, and validation results.
- **Hard-Example Source Set**: Validation-derived V1 false positives, false negatives, uncertain samples, and high-loss samples used only for training analysis or oversampling.
- **Threshold Record**: Saved threshold selected from validation predictions and used to convert class-1 probability into the binary target.
- **Comparison Report**: Required V1-vs-V2 artifact containing accuracy, error counts, threshold, speed, backbone, model size, image size, and Kaggle public score when available.
- **V2 Output Set**: Local ignored artifacts expected after implementation: model checkpoint, classifier metrics, best threshold, validation predictions, submission, benchmark, and comparison report.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Selected V2 validation F1 targets at least +0.02 over the V1 validation F1 baseline of 0.91656, with the exact delta reported; +0.02 is the minimum target, while stronger results toward 0.95+ remain desirable when inference speed stays acceptable.
- **SC-002**: Selected V2 does not regress both false positives and false negatives at the same time compared with the V1 confusion counts of 344 false positives and 344 false negatives.
- **SC-003**: V2 comparison report includes 100% of required V1-vs-V2 fields: validation F1, Kaggle public score when available, threshold, false positives, false negatives, uncertain samples, inference speed, model size or backbone name, and image size.
- **SC-004**: V2 benchmark reports inference speed for the selected model, and any larger backbone, image-size, or optional inference experiment stays no more than 2x slower than V1 speed cost while stating its speed cost relative to V1.
- **SC-005**: V2 produces all expected later output categories under `outputs/kaggle_v2/` without exposing private datasets or derived artifacts outside ignored local paths.
- **SC-006**: Normal V2 inference performs zero detector, segmentation, Grad-CAM, dashboard, feature-memory-bank, ensemble, distillation, or hard-example-memory steps by default.
- **SC-007**: V2 training and threshold selection use zero test images and zero sample-submission-derived pseudo-labels.
- **SC-008**: When hard-example oversampling is enabled, 100% of oversampled hard-example image IDs are absent from the current V2 validation split and the training report records loaded, eligible, excluded, and used counts.
- **SC-009**: When V2 candidates differ by `<= 0.002` validation F1, the selected candidate is the faster model.

## Assumptions

- SPEC-001 through SPEC-006 are complete and should only be referenced by SPEC-007.
- V1 completed with EfficientNet-B0 at 384x384, validation F1 approximately 0.91656, Kaggle public score approximately 0.91693, and best threshold 0.48.
- The local V1 metrics artifact currently records class counts, confusion counts, runtime, device, weights, and threshold metadata.
- The V1 hard-example baseline is expected to include false positives, false negatives, uncertain samples, and high-loss samples; loaded counts must be reported because manually supplied and regenerated hard-example summaries may differ.
- V2's default deliverable is one fast classifier, not a large model, ensemble, detector, or hybrid system.
- The default hard-example strategy is `analysis_only`, so V2 may inspect and report hard examples without changing training composition.
- A 384x384 V2 candidate must be evaluated before any optional 448x448 experiment.
- Kaggle public score comparison is included only when a V2 submission has actually been uploaded and scored.
