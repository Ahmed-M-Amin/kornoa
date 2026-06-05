# Tasks: Detector / Segmentation Training

**Input**: Design documents from `/specs/008-detector-segmentation-training/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/detector-training-contract.md](./contracts/detector-training-contract.md), [quickstart.md](./quickstart.md)

**Tests**: Required by the feature specification and constitution gates. Write tests first and verify they fail before implementation.

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish SPEC-008 documentation, config, output paths, and test scaffolds without changing classifier behavior.

- [X] T001 Create SPEC-008 documentation set in `specs/008-detector-segmentation-training/`
- [X] T002 Update active Spec Kit feature pointer in `.specify/feature.json`
- [X] T003 Update agent context plan reference in `AGENTS.md`
- [X] T004 Extend detector configuration scaffold in `configs/detector.yaml`
- [X] T005 [P] Create detector conversion test scaffold in `tests/test_detector_yolo_conversion.py`
- [X] T006 [P] Add synthetic detector fixture helpers in `tests/test_detector_yolo_conversion.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared COCO audit, category mapping, output path, and leakage guard behavior needed before user stories.

**CRITICAL**: No detector training command work can begin until this phase is complete.

- [X] T007 Add detector output path constants and config loading in `src/training/train_detector.py`
- [X] T008 [P] Add category mapping helpers in `src/data/yolo_converter.py`
- [X] T009 [P] Add invalid bbox filtering helpers in `src/data/yolo_converter.py`
- [X] T010 Add detector scope guards preventing test-image, hybrid, dashboard, Grad-CAM, memory-bank, ensemble, and distillation behavior in `src/training/train_detector.py`
- [X] T011 Add generated-output ignore validation for `outputs/detector/` artifacts in `tests/test_detector_yolo_conversion.py`

---

## Phase 3: User Story 1 - Audit Defect Annotations (Priority: P1) MVP

**Goal**: Load COCO annotations and produce a detector audit plus defect category mapping.

**Independent Test**: Synthetic COCO annotations produce category mapping and audit reports with missing references, invalid boxes, no-annotation images, and multi-defect images.

### Tests for User Story 1

- [X] T012 [P] [US1] Add COCO audit report tests in `tests/test_detector_yolo_conversion.py`
- [X] T013 [P] [US1] Add category mapping report tests in `tests/test_detector_yolo_conversion.py`
- [X] T014 [P] [US1] Add missing image reference and invalid bbox audit tests in `tests/test_detector_yolo_conversion.py`

### Implementation for User Story 1

- [X] T015 [US1] Implement detector category mapping in `src/data/yolo_converter.py`
- [X] T016 [US1] Implement detector annotation audit report generation in `src/data/yolo_converter.py`
- [X] T017 [US1] Add `audit` command support in `src/training/train_detector.py`

**Checkpoint**: User Story 1 can audit annotations independently.

---

## Phase 4: User Story 2 - Generate Detector Dataset (Priority: P2)

**Goal**: Convert valid COCO annotations to YOLO dataset folders, labels, split reports, and `data.yaml`.

**Independent Test**: Synthetic annotations convert into deterministic disjoint train/validation folders with normalized labels and valid `data.yaml`.

### Tests for User Story 2

- [X] T018 [P] [US2] Add YOLO label conversion tests in `tests/test_detector_yolo_conversion.py`
- [X] T019 [P] [US2] Add deterministic split disjointness tests in `tests/test_detector_yolo_conversion.py`
- [X] T020 [P] [US2] Add `data.yaml` creation tests in `tests/test_detector_yolo_conversion.py`
- [X] T037 [P] [US2] Add explicit FR-008a/SC-004a no-annotation conversion tests in `tests/test_detector_yolo_conversion.py` verifying no-annotation train/validation images are included, every no-annotation image has a same-stem empty YOLO `.txt` label file, and conversion reports include `no_annotation_train_count`, `no_annotation_val_count`, and `total_empty_label_files`
- [X] T039 [P] [US2] Add repeated-conversion regression tests in `tests/test_detector_yolo_conversion.py` proving stale generated train/validation image and label files are removed before new conversion output is written
- [X] T040 [P] [US2] Add primary-defect-category stratification and deterministic fallback report tests in `tests/test_detector_yolo_conversion.py`

### Implementation for User Story 2

- [X] T021 [US2] Implement deterministic detector split assignment in `src/data/yolo_converter.py`
- [X] T022 [US2] Implement YOLO label writing and image copying in `src/data/yolo_converter.py`
- [X] T023 [US2] Implement `data.yaml`, split report, and conversion report writing in `src/data/yolo_converter.py`
- [X] T024 [US2] Add `convert` command support in `src/training/train_detector.py`
- [X] T038 [US2] Implement FR-008a/SC-004a no-annotation conversion report fields in `src/data/yolo_converter.py`: `no_annotation_train_count`, `no_annotation_val_count`, and `total_empty_label_files`
- [X] T041 [US2] Implement generated split-folder cleanup in `src/data/yolo_converter.py` and report `output_dataset_cleaned`, `cleaned_paths`, `split_strategy_used`, and stratification fallback reason
- [X] T042 [US2] Implement practical primary-defect-category stratification in `src/data/yolo_converter.py`, including `no_annotation`, with deterministic random fallback for rare strata

**Checkpoint**: User Story 2 can generate a detector dataset independently.

---

## Phase 5: User Story 3 - Visualize and Train Detector Candidates (Priority: P3)

**Goal**: Save sample defect overlays and expose documented manual training/evaluation commands.

**Independent Test**: Synthetic images produce visualization files; train/evaluate commands validate arguments without running training in tests.

### Tests for User Story 3

- [X] T025 [P] [US3] Add sample box visualization creation tests in `tests/test_detector_yolo_conversion.py`
- [X] T026 [P] [US3] Add train/evaluate command smoke tests that do not start training in `tests/test_detector_yolo_conversion.py`
- [X] T043 [P] [US3] Add dry-run evaluation report metadata tests for `evaluation_mode`, `per_category_metrics_available`, and `validation_prediction_examples_available` in `tests/test_detector_yolo_conversion.py`

### Implementation for User Story 3

- [X] T027 [US3] Implement category-colored box visualization in `src/data/yolo_converter.py`
- [X] T028 [US3] Implement `visualize` command support in `src/training/train_detector.py`
- [X] T029 [US3] Implement manual YOLO training command wrapper in `src/training/train_detector.py`
- [X] T030 [US3] Implement detector evaluation report writer in `src/training/train_detector.py`
- [X] T044 [US3] Update dry-run detector evaluation reports in `src/training/train_detector.py` with explicit unavailable per-category metrics and validation-example metadata

**Checkpoint**: User Story 3 can create visualizations and document runnable detector commands.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate integration, leakage controls, confidentiality, docs, and no-regression behavior.

- [X] T031 [P] Validate no accepted V2B classifier files or configs were changed
- [X] T032 [P] Run SPEC-008 targeted tests `pytest tests/test_coco_parser.py tests/test_detector_yolo_conversion.py -q`
- [X] T033 Run SPEC-005/SPEC-006 regression tests `pytest tests/test_classifier.py tests/test_training.py tests/test_inference.py tests/test_submission.py tests/test_benchmark.py tests/test_hard_example_mining.py -q`
- [X] T034 Run SPEC-007 targeted tests `pytest tests/test_classifier_v2.py tests/test_training_v2.py tests/test_transforms_v2.py tests/test_v2_comparison.py tests/test_inference_v2.py -q`
- [X] T035 Run full project regression command `pytest -q`
- [X] T036 Review `git status --short` to confirm intended SPEC-008 scope only

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational and delivers the MVP audit.
- **User Story 2 (Phase 4)**: Depends on Foundational and uses US1 category/audit helpers.
- **User Story 3 (Phase 5)**: Depends on Foundational and benefits from US2 converted dataset outputs.
- **Polish (Phase 6)**: Depends on all selected user stories.

### User Story Dependencies

- **US1 Audit Defect Annotations**: MVP and first implementation target.
- **US2 Generate Detector Dataset**: Can start after foundational helpers but should reuse category mapping from US1.
- **US3 Visualize and Train Detector Candidates**: Can start after visualization inputs exist; training/evaluation wrappers do not depend on actual detector training during tests.

### Parallel Opportunities

- T005-T006 can run in parallel after T004.
- T008-T009 can run in parallel after T007.
- T012-T014 can run in parallel before US1 implementation.
- T018-T020 can run in parallel before US2 implementation.
- T025-T026 can run in parallel before US3 implementation.
- T031-T032 can run in parallel during polish.

---

## Parallel Example: User Story 2

```text
Task: "Add YOLO label conversion tests in tests/test_detector_yolo_conversion.py"
Task: "Add deterministic split disjointness tests in tests/test_detector_yolo_conversion.py"
Task: "Add data.yaml creation tests in tests/test_detector_yolo_conversion.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational detector config, output paths, category mapping, invalid bbox filtering, and scope guards.
3. Write and fail US1 tests.
4. Implement annotation audit and category mapping.
5. Validate US1 independently with `pytest tests/test_coco_parser.py tests/test_detector_yolo_conversion.py -q`.

### Incremental Delivery

1. Add US1 for annotation audit and category mapping.
2. Add US2 for deterministic YOLO dataset conversion.
3. Add US3 for visualizations and manual training/evaluation wrappers.
4. Run SPEC-008 targeted tests, SPEC-005/SPEC-006 regressions, SPEC-007 targeted tests, and full project regression.

### Scope Discipline

- Do not implement SPEC-009 hybrid inference or classifier-detector fusion.
- Do not modify accepted V2B EfficientNet-B1 `analysis_only` artifacts or model selection.
- Do not train detectors automatically in tests.
- Do not use test images, sample submission rows, Kaggle public feedback, hard-example mining outputs, pseudo-labels, dashboard, Grad-CAM, feature memory bank, ensemble, or distillation.
