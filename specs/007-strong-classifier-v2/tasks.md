# Tasks: Strong Classifier V2

**Input**: Design documents from `/specs/007-strong-classifier-v2/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/strong-classifier-v2-contract.md](./contracts/strong-classifier-v2-contract.md), [quickstart.md](./quickstart.md)

**Tests**: Required by the plan and constitution gates. Write tests first and verify they fail before implementation.

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish SPEC-007 configuration, test scaffolds, and V2 output conventions without changing V1 behavior.

- [X] T001 Create V2 classifier config scaffold in `configs/classifier_v2.yaml`
- [X] T002 [P] Create V2 classifier test scaffold in `tests/test_classifier_v2.py`
- [X] T003 [P] Create V2 training test scaffold in `tests/test_training_v2.py`
- [X] T004 [P] Create V2 transform test scaffold in `tests/test_transforms_v2.py`
- [X] T005 [P] Create V2 comparison test scaffold in `tests/test_v2_comparison.py`
- [X] T006 [P] Create V2 inference test scaffold in `tests/test_inference_v2.py`
- [X] T007 [P] Add synthetic V2 hard-example fixture builders in `tests/test_training_v2.py`
- [X] T008 [P] Add synthetic V1 and V2 metrics fixture builders in `tests/test_v2_comparison.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared V2 configuration, experiment metadata, output paths, and safety guards required before user stories.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T009 Add V2 output path constants for `outputs/kaggle_v2/` artifacts in `src/training/train_classifier.py`
- [X] T010 Define V2 experiment configuration parsing and validation in `src/training/train_classifier.py`
- [X] T011 Add V2 experiment selection fields for experiment name, backbone, image size, imbalance strategy, augmentation recipe, hard-example strategy, and speed ceiling in `configs/classifier_v2.yaml`
- [X] T012 [P] Add tests for V2 config validation and forbidden feature flags in `tests/test_training_v2.py`
- [X] T013 [P] Add tests that generated V2 artifacts remain ignored under `outputs/kaggle_v2/` in `tests/test_training_v2.py`
- [X] T014 [P] Add tests that default hard-example strategy is `analysis_only` and allowed values are `none`, `analysis_only`, and `oversample` in `tests/test_training_v2.py`
- [X] T015 Add explicit scope guards rejecting detector, segmentation, Grad-CAM, dashboard, feature memory bank, default ensemble, distillation, and hybrid inference options in `src/training/train_classifier.py`
- [X] T016 Add V2 run metadata capture for seed, config snapshot, device, backbone, image size, speed ceiling, close-F1 tolerance, hard-example strategy, and experiment name in `src/training/train_classifier.py`
- [X] T017 Add V2-compatible artifact root detection for `outputs/kaggle_v2/` in `src/inference/predict.py`
- [X] T018 Add tests for V2 artifact root detection without breaking SPEC-006 V1 artifact loading in `tests/test_inference_v2.py`

**Checkpoint**: V2 foundation ready. User story implementation can now begin.

---

## Phase 3: User Story 1 - Train a Stronger Single Classifier (Priority: P1) MVP

**Goal**: Train and select a stronger single V2 classifier candidate by validation F1 after threshold search while preserving fast classifier-only inference.

**Independent Test**: Use synthetic data and tiny checkpoints to verify V2A/B/C candidate configuration, focal loss, weighted sampling, safe augmentation, threshold-search selection, saved V2 artifacts, and classifier-only inference scope.

### Tests for User Story 1

- [X] T019 [P] [US1] Add focal loss behavior tests in `tests/test_classifier_v2.py`
- [X] T020 [P] [US1] Add weighted random sampler construction tests in `tests/test_training_v2.py`
- [X] T021 [P] [US1] Add class-count reporting tests for train and validation splits in `tests/test_training_v2.py`
- [X] T022 [P] [US1] Add safe augmentation boundary tests for brightness, contrast, gamma, blur, noise, rotation, shift, and scale in `tests/test_transforms_v2.py`
- [X] T023 [P] [US1] Add deterministic validation and test preprocessing tests in `tests/test_transforms_v2.py`
- [X] T024 [P] [US1] Add V2 candidate backbone validation tests for V2A EfficientNet-B0, V2B EfficientNet-B1, V2C EfficientNet-B2, and optional ConvNeXt-Tiny in `tests/test_classifier_v2.py`
- [X] T025 [P] [US1] Add image-size ordering tests proving 448x448 candidates cannot be selected before 384x384 candidates in `tests/test_training_v2.py`
- [X] T026 [P] [US1] Add threshold-search selection tests proving V2 selects by validation F1 after threshold search in `tests/test_training_v2.py`
- [X] T027 [P] [US1] Add tests that normal V2 inference uses only classifier probability, saved threshold, and binary target in `tests/test_inference_v2.py`

### Implementation for User Story 1

- [X] T028 [US1] Implement focal loss option with binary target compatibility in `src/training/losses.py`
- [X] T029 [US1] Implement weighted random sampler construction from training labels in `src/training/train_classifier.py`
- [X] T030 [US1] Implement train and validation class-count reporting for V2 runs in `src/training/train_classifier.py`
- [X] T031 [US1] Implement V2 safe training augmentation recipe in `src/data/transforms.py`
- [X] T032 [US1] Preserve deterministic validation and test preprocessing for V2 in `src/data/transforms.py`
- [X] T033 [US1] Add EfficientNet-B1, EfficientNet-B2, and optional ConvNeXt-Tiny candidate support in `src/models/classifier.py`
- [X] T034 [US1] Add V2 candidate experiment ordering for V2A, V2B, V2C, optional ConvNeXt-Tiny, and optional 448x448 in `src/training/train_classifier.py`
- [X] T035 [US1] Extend V2 training loop to save selected checkpoint to `outputs/kaggle_v2/models/classifier_best.pth` in `src/training/train_classifier.py`
- [X] T036 [US1] Extend V2 metrics saving to `outputs/kaggle_v2/reports/classifier_metrics.json` in `src/training/train_classifier.py`
- [X] T037 [US1] Extend V2 threshold saving to `outputs/kaggle_v2/reports/best_threshold.json` in `src/training/train_classifier.py`
- [X] T038 [US1] Extend V2 validation prediction saving to `outputs/kaggle_v2/predictions/val_classifier_predictions.csv` in `src/training/train_classifier.py`
- [X] T039 [US1] Ensure V2 selected model target is at least +0.02 F1 over V1 when reporting run status in `src/training/train_classifier.py`
- [X] T040 [US1] Ensure selected V2 inference remains classifier-only in `src/inference/predict.py`

**Checkpoint**: User Story 1 delivers the V2 classifier MVP and is independently testable.

---

## Phase 4: User Story 2 - Reduce V1 Mistakes With Hard Examples (Priority: P2)

**Goal**: Use V1 false positives, false negatives, uncertain samples, and high-loss samples as offline training and analysis inputs for V2 without running memory during inference.

**Independent Test**: Use synthetic hard-example CSVs to verify loading, count reporting, row exclusion, oversampling weights, and no inference-time hard-example lookup.

### Tests for User Story 2

- [X] T041 [P] [US2] Add hard-example source loading tests for false positives, false negatives, uncertain samples, and high-loss samples in `tests/test_training_v2.py`
- [X] T042 [P] [US2] Add tests for reporting loaded hard-example counts and mismatched summary counts in `tests/test_v2_comparison.py`
- [X] T043 [P] [US2] Add tests excluding hard-example rows that do not map to training image IDs in `tests/test_training_v2.py`
- [X] T044 [P] [US2] Add tests that hard-example image IDs in current V2 validation are excluded before oversampling in `tests/test_training_v2.py`
- [X] T045 [P] [US2] Add tests for split disjointness report fields and train/validation image ID confirmation in `tests/test_training_v2.py`
- [X] T046 [P] [US2] Add tests for hard-example oversampling weights without duplicating validation/test leakage in `tests/test_training_v2.py`
- [X] T047 [P] [US2] Add tests proving oversampling only happens when `hard_example_strategy=oversample` is explicitly enabled in `tests/test_training_v2.py`
- [X] T048 [P] [US2] Add tests proving hard-example memory is not used during V2 prediction, submission, or benchmark workflows in `tests/test_inference_v2.py`

### Implementation for User Story 2

- [X] T049 [US2] Implement hard-example source-set loading from `outputs/hard_examples/*.csv` and `outputs/reports/hard_example_summary.json` in `src/training/hard_example_mining.py`
- [X] T050 [US2] Implement loaded hard-example count and mismatch reporting in `src/training/hard_example_mining.py`
- [X] T051 [US2] Implement training-ID validation and excluded-row reporting for hard examples in `src/training/train_classifier.py`
- [X] T052 [US2] Implement V2 validation image ID exclusion before hard-example oversampling in `src/training/train_classifier.py`
- [X] T053 [US2] Implement split disjointness reporting with loaded, eligible, validation-excluded, and oversampled hard-example counts in `src/training/train_classifier.py`
- [X] T054 [US2] Implement hard-example oversampling weights for false positives, false negatives, uncertain samples, and high-loss samples in `src/training/train_classifier.py`
- [X] T055 [US2] Integrate hard-example oversampling into V2 training only when `hard_example_strategy=oversample` is enabled in `configs/classifier_v2.yaml`
- [X] T056 [US2] Add runtime guards preventing hard-example memory lookup in `src/inference/predict.py`
- [X] T057 [US2] Add runtime guards preventing hard-example memory lookup in `src/inference/submission.py`
- [X] T058 [US2] Add runtime guards preventing hard-example memory lookup in `src/inference/benchmark.py`

**Checkpoint**: User Story 2 is functional and independently testable without test-image access.

---

## Phase 5: User Story 3 - Compare V2 Against V1 Transparently (Priority: P3)

**Goal**: Produce mandatory V1-vs-V2 comparison artifacts covering accuracy, threshold, errors, uncertainty, speed, backbone, model size, image size, and Kaggle score when available.

**Independent Test**: Use synthetic V1 and V2 artifacts to verify comparison JSON fields, F1 delta, speed ceiling, close-F1 fastest-model preference, and unavailable Kaggle score handling.

### Tests for User Story 3

- [X] T059 [P] [US3] Add V1 baseline loading tests for metrics, threshold, benchmark, hard-example counts, backbone, and image size in `tests/test_v2_comparison.py`
- [X] T060 [P] [US3] Add V2 candidate result tests for F1 delta, threshold delta, FP/FN delta, uncertain delta, speed delta, model size, and image-size fields in `tests/test_v2_comparison.py`
- [X] T061 [P] [US3] Add tests for 2x V1 speed ceiling and non-selectable slow candidates in `tests/test_v2_comparison.py`
- [X] T062 [P] [US3] Add tests that close-F1 tolerance is absolute validation F1 difference `<= 0.002` in `tests/test_v2_comparison.py`
- [X] T063 [P] [US3] Add tests that close-F1 candidates prefer the faster model in `tests/test_v2_comparison.py`
- [X] T064 [P] [US3] Add tests that missing V2 Kaggle public score is marked unavailable in `tests/test_v2_comparison.py`
- [X] T065 [P] [US3] Add V2 submission output contract tests for `outputs/kaggle_v2/submissions/submission_v2.csv` in `tests/test_inference_v2.py`
- [X] T066 [P] [US3] Add V2 benchmark report field tests for speed multiplier and speed ceiling in `tests/test_inference_v2.py`

### Implementation for User Story 3

- [X] T067 [US3] Implement V1 baseline record loading in `src/training/metrics.py`
- [X] T068 [US3] Implement V2 candidate result serialization in `src/training/metrics.py`
- [X] T069 [US3] Implement V1-vs-V2 comparison report generation in `src/training/metrics.py`
- [X] T070 [US3] Save V1-vs-V2 comparison report to `outputs/kaggle_v2/reports/v1_vs_v2_comparison.json` in `src/training/train_classifier.py`
- [X] T071 [US3] Implement selected-candidate speed ceiling validation in `src/training/metrics.py`
- [X] T072 [US3] Implement close-F1 tolerance `<= 0.002` and fastest-candidate preference in `src/training/metrics.py`
- [X] T073 [US3] Extend V2 submission generation to write `outputs/kaggle_v2/submissions/submission_v2.csv` in `src/inference/submission.py`
- [X] T074 [US3] Extend V2 benchmark reporting to write `outputs/kaggle_v2/benchmarks/v2_inference_benchmark.json` with speed multiplier fields in `src/inference/benchmark.py`

**Checkpoint**: User Story 3 produces complete V1-vs-V2 comparison and V2 inference artifacts.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate integration, leakage controls, confidentiality, quickstart commands, and no-regression behavior.

- [X] T075 [P] Add generated-output ignore validation for all `outputs/kaggle_v2/` artifacts in `tests/test_training_v2.py`
- [X] T076 [P] Add scope-guard tests proving detector, segmentation, Grad-CAM, dashboard, feature memory bank, default ensemble, distillation, and hybrid inference outputs are not created by SPEC-007 workflows in `tests/test_inference_v2.py`
- [X] T077 [P] Add leakage tests proving test images and sample-submission rows are not used for V2 training, threshold tuning, or model selection in `tests/test_training_v2.py`
- [X] T078 [P] Add Kaggle and Colab path compatibility tests for `configs/classifier_v2.yaml` path overrides in `tests/test_training_v2.py`
- [X] T079 Run SPEC-007 targeted tests `pytest tests/test_classifier_v2.py tests/test_training_v2.py tests/test_transforms_v2.py tests/test_v2_comparison.py tests/test_inference_v2.py -q`
- [X] T080 Run existing V1 classifier and inference regression tests `pytest tests/test_classifier.py tests/test_training.py tests/test_inference.py tests/test_submission.py tests/test_benchmark.py tests/test_hard_example_mining.py -q`
- [X] T081 Run full project regression command `pytest`
- [X] T082 Review `git status --short` to confirm only SPEC-007 code, tests, docs, config, and intended agent context changes are present

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational; delivers the V2 classifier MVP.
- **User Story 2 (Phase 4)**: Depends on Foundational and can integrate after US1 training inputs exist.
- **User Story 3 (Phase 5)**: Depends on Foundational and uses outputs from US1 plus benchmark/submission extensions.
- **Polish (Phase 6)**: Depends on all selected user stories being complete.

### User Story Dependencies

- **US1 Train a Stronger Single Classifier**: MVP and first implementation target.
- **US2 Reduce V1 Mistakes With Hard Examples**: Can be developed after foundational V2 config and data interfaces; final integration depends on US1 training input handling.
- **US3 Compare V2 Against V1 Transparently**: Can develop comparison helpers from synthetic artifacts in parallel, but final acceptance depends on US1 selected-candidate outputs and V2 benchmark artifacts.

### Parallel Opportunities

- T002-T008 can run in parallel after T001.
- T012-T013 can run in parallel with foundational implementation after T009-T011 are drafted.
- T019-T027 can be written in parallel before US1 implementation.
- T041-T048 can be written in parallel before US2 implementation.
- T059-T066 can be written in parallel before US3 implementation.
- T075-T078 can run in parallel during polish.

---

## Parallel Example: User Story 1

```text
Task: "Add focal loss behavior tests in tests/test_classifier_v2.py"
Task: "Add weighted random sampler construction tests in tests/test_training_v2.py"
Task: "Add safe augmentation boundary tests for brightness, contrast, gamma, blur, noise, rotation, shift, and scale in tests/test_transforms_v2.py"
Task: "Add threshold-search selection tests proving V2 selects by validation F1 after threshold search in tests/test_training_v2.py"
Task: "Add tests that normal V2 inference uses only classifier probability, saved threshold, and binary target in tests/test_inference_v2.py"
```

---

## Parallel Example: User Story 3

```text
Task: "Add V1 baseline loading tests for metrics, threshold, benchmark, hard-example counts, backbone, and image size in tests/test_v2_comparison.py"
Task: "Add tests for 2x V1 speed ceiling and non-selectable slow candidates in tests/test_v2_comparison.py"
Task: "Add tests that close-F1 tolerance is absolute validation F1 difference <= 0.002 in tests/test_v2_comparison.py"
Task: "Add tests that close-F1 candidates prefer the faster model in tests/test_v2_comparison.py"
Task: "Add V2 benchmark report field tests for speed multiplier and speed ceiling in tests/test_inference_v2.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational V2 config, output paths, metadata, and scope guards.
3. Write and fail US1 tests.
4. Implement V2A/B/C classifier recipe, focal loss, weighted sampling, safe augmentation, threshold selection, and V2 artifact saving.
5. Validate US1 independently with `pytest tests/test_classifier_v2.py tests/test_training_v2.py tests/test_transforms_v2.py tests/test_inference_v2.py -q`.

### Incremental Delivery

1. Add US1 for the V2 classifier MVP.
2. Add US2 for offline V1 hard-example oversampling and reporting.
3. Add US3 for V1-vs-V2 comparison, V2 submission, and V2 benchmark.
4. Run SPEC-007 targeted tests, V1 regression tests, and full project regression.

### Scope Discipline

- Do not modify SPEC-001 through SPEC-006 except for tiny compatibility fixes if implementation reveals a blocker.
- Do not implement detector, segmentation, Grad-CAM, dashboard, feature memory bank, default ensemble, distillation, or hybrid inference.
- Do not use test images, sample submission rows, or Kaggle public feedback for V2 training, threshold tuning, or model selection.
- Keep hard-example memory offline and disabled during normal inference, submission, and benchmark workflows.

