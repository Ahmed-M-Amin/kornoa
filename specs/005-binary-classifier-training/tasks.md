---
description: "Task list for SPEC-005 Binary Classifier Training"
---

# Tasks: Binary Classifier Training

**Input**: Design documents from `/specs/005-binary-classifier-training/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/classifier-training-contract.md, quickstart.md

**Tests**: Synthetic-fixture pytest coverage is required so classifier training behavior can be validated without private Krones data.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish SPEC-005 module and test locations while keeping inference, detector, submission, dashboard, Grad-CAM, and memory-bank work out of scope.

- [X] T001 Create classifier model module scaffold in `src/models/classifier.py`
- [X] T002 Create classifier training orchestration scaffold in `src/training/train_classifier.py`
- [X] T003 Create weighted loss helper scaffold in `src/training/losses.py`
- [X] T004 Create classifier metric helper scaffold in `src/training/metrics.py`
- [X] T005 Create threshold search helper scaffold in `src/training/threshold_search.py`
- [X] T006 Create classifier unit test scaffold in `tests/test_classifier.py`
- [X] T007 Create training metrics test scaffold in `tests/test_training.py`
- [X] T008 Create training pipeline test scaffold in `tests/test_training.py`
- [X] T009 Verify `configs/classifier.yaml` uses `image_size: 384` and binary `num_classes: 2`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define shared records, config defaults, reproducibility helpers, synthetic training fixtures, and artifact confidentiality checks before user-story behavior.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T010 Define `TrainingExample`, `SplitAssignment`, `TrainingRunConfig`, `TrainingRunResult`, `ValidationPrediction`, `ClassifierMetricsReport`, and `BestThresholdRecord` structures in `src/training/train_classifier.py`
- [X] T011 Define SPEC-005 output path constants for `outputs/models/classifier_effnet_b0_best.pth`, `outputs/reports/classifier_metrics.json`, `outputs/reports/best_threshold.json`, and `outputs/predictions/val_classifier_predictions.csv` in `src/training/train_classifier.py`
- [X] T012 Implement config loading and validation for classifier training settings in `src/training/train_classifier.py`
- [X] T013 Implement reproducibility seed setup for Python, NumPy, and PyTorch in `src/training/train_classifier.py`
- [X] T014 Implement generated classifier artifact ignore/tracking validation helper in `src/training/train_classifier.py`
- [X] T015 [P] Create synthetic classifier training fixture metadata in `tests/fixtures/synthetic_dataset/classifier_cases/classifier_cases.json`
- [X] T016 [P] Create synthetic classifier positive-class image fixtures in `tests/fixtures/synthetic_dataset/classifier_cases/positive/`
- [X] T017 [P] Create synthetic classifier negative-class image fixtures in `tests/fixtures/synthetic_dataset/classifier_cases/negative/`
- [X] T018 [P] Create synthetic classifier `train.csv` fixture with at least five examples per class in `tests/fixtures/synthetic_dataset/classifier_cases/train.csv`
- [X] T019 [P] Create synthetic classifier image directory fixture in `tests/fixtures/synthetic_dataset/classifier_cases/train_images/`

**Checkpoint**: Shared records, config validation, reproducibility, fixtures, and artifact helpers are ready for user-story tests.

---

## Phase 3: User Story 1 - Train V1 Binary Classifier (Priority: P1) MVP

**Goal**: A contributor can train the first reusable/not-reusable classifier from labeled images and SPEC-004 normalized preprocessing outputs, then persist minimal end-to-end training artifacts.

**Independent Test**: Run a synthetic training smoke test that consumes labeled images, performs an 80/20 stratified split, trains a tiny smoke classifier path, emits binary probabilities, saves best weights/basic metrics/validation predictions, and completes under one minute.

### Tests for User Story 1

- [X] T020 [P] [US1] Write failing binary label validation test in `tests/test_training.py`
- [X] T021 [P] [US1] Write failing missing/unreadable image reporting test in `tests/test_training.py`
- [X] T022 [P] [US1] Write failing one-class training rejection test in `tests/test_training.py`
- [X] T023 [P] [US1] Write failing 80/20 stratified split reproducibility test in `tests/test_training.py`
- [X] T024 [P] [US1] Write failing normalized SPEC-004 preprocessing integration test in `tests/test_training.py`
- [X] T025 [P] [US1] Write failing classifier forward-output shape and probability-range test in `tests/test_classifier.py`
- [X] T026 [P] [US1] Write failing weighted binary loss construction test in `tests/test_classifier.py`
- [X] T027 [P] [US1] Write failing synthetic training smoke runtime test under one minute in `tests/test_training.py`
- [X] T028 [P] [US1] Write failing minimal best model artifact output test after synthetic smoke training in `tests/test_training.py`
- [X] T029 [P] [US1] Write failing basic training/validation metrics JSON output test after synthetic smoke training in `tests/test_training.py`
- [X] T030 [P] [US1] Write failing validation probabilities/predictions CSV output test after synthetic smoke training in `tests/test_training.py`

### Implementation for User Story 1

- [X] T031 [US1] Implement binary label loading and validation from configured `train.csv` in `src/training/train_classifier.py`
- [X] T032 [US1] Implement readable missing/unreadable image diagnostics in `src/training/train_classifier.py`
- [X] T033 [US1] Implement one-class and insufficient split validation in `src/training/train_classifier.py`
- [X] T034 [US1] Implement reproducible 80/20 stratified train/validation split in `src/training/train_classifier.py`
- [X] T035 [US1] Implement SPEC-004 ROI/preprocessing integration for normalized `3x384x384` classifier inputs in `src/training/train_classifier.py`
- [X] T036 [US1] Implement EfficientNet-B0-compatible classifier factory plus tiny synthetic-smoke variant in `src/models/classifier.py`
- [X] T037 [US1] Implement weighted binary loss helper in `src/training/losses.py`
- [X] T038 [US1] Implement minimal validation F1 calculation for checkpoint selection in `src/training/metrics.py`
- [X] T039 [US1] Implement V1 training loop with seeded synthetic-smoke mode in `src/training/train_classifier.py`
- [X] T040 [US1] Implement best model weight saving by validation F1 in `src/training/train_classifier.py`
- [X] T041 [US1] Implement basic training/validation metrics JSON saving in `src/training/train_classifier.py`
- [X] T042 [US1] Implement validation probabilities/predictions CSV saving in `src/training/train_classifier.py`
- [X] T043 [US1] Run User Story 1 checks with `pytest tests/test_classifier.py tests/test_training.py -q`

**Checkpoint**: User Story 1 is complete when synthetic classifier training runs independently, produces binary validation probabilities, and persists minimal model, metrics, and validation prediction outputs.

---

## Phase 4: User Story 2 - Measure Validation F1 and Threshold (Priority: P2)

**Goal**: A contributor can evaluate validation probabilities, calculate full F1-score metrics, and select the best decision threshold with deterministic tie-breaking.

**Independent Test**: Run metric and threshold tests that feed known labels/probabilities and verify F1-score, confusion counts, class counts, threshold selection, and lowest-threshold tie-break behavior.

### Tests for User Story 2

- [X] T044 [P] [US2] Write failing F1-score and confusion-count test in `tests/test_training.py`
- [X] T045 [P] [US2] Write failing class-count metrics test in `tests/test_training.py`
- [X] T046 [P] [US2] Write failing best-threshold selection test in `tests/test_training.py`
- [X] T047 [P] [US2] Write failing lowest-threshold tie-break test in `tests/test_training.py`
- [X] T048 [P] [US2] Write failing checkpoint-selection-by-validation-F1-after-threshold-search test in `tests/test_training.py`

### Implementation for User Story 2

- [X] T049 [US2] Strengthen validation F1-score calculation in `src/training/metrics.py`
- [X] T050 [US2] Implement confusion-count and class-count reporting in `src/training/metrics.py`
- [X] T051 [US2] Implement threshold candidate evaluation in `src/training/threshold_search.py`
- [X] T052 [US2] Implement lowest-threshold tie-break for tied best F1 in `src/training/threshold_search.py`
- [X] T053 [US2] Integrate validation metric reporting into `src/training/train_classifier.py`
- [X] T054 [US2] Upgrade best-checkpoint selection to use validation F1 after threshold search in `src/training/train_classifier.py`
- [X] T055 [US2] Run User Story 2 checks with `pytest tests/test_training.py tests/test_training.py -q`

**Checkpoint**: User Story 2 is complete when threshold and metrics behavior works independently with deterministic synthetic inputs.

---

## Phase 5: User Story 3 - Save Reproducible Training Outputs (Priority: P3)

**Goal**: A contributor can review and reuse the finalized classifier training outputs while generated model, report, threshold, and prediction artifacts remain ignored and untracked.

**Independent Test**: Run a synthetic training smoke test and confirm finalized model, rich metrics, best threshold, and validation prediction files are written under approved ignored output locations.

### Tests for User Story 3

- [X] T056 [P] [US3] Write failing finalized model artifact output test in `tests/test_training.py`
- [X] T057 [P] [US3] Write failing rich classifier metrics JSON output test in `tests/test_training.py`
- [X] T058 [P] [US3] Write failing best-threshold JSON output test in `tests/test_training.py`
- [X] T059 [P] [US3] Write failing finalized validation predictions CSV output test in `tests/test_training.py`
- [X] T060 [P] [US3] Write failing generated classifier artifact confidentiality test in `tests/test_training.py`
- [X] T061 [P] [US3] Write failing training CLI synthetic-smoke test in `tests/test_training.py`

### Implementation for User Story 3

- [X] T062 [US3] Harden finalized best model weight saving in `src/training/train_classifier.py`
- [X] T063 [US3] Implement rich classifier metrics JSON saving with threshold, class counts, confusion counts, runtime, and artifact paths in `src/training/train_classifier.py`
- [X] T064 [US3] Implement best-threshold JSON saving in `src/training/train_classifier.py`
- [X] T065 [US3] Harden validation predictions CSV saving with image IDs, true labels, probabilities, threshold, and predicted labels in `src/training/train_classifier.py`
- [X] T066 [US3] Implement generated artifact confidentiality checks for classifier outputs in `src/training/train_classifier.py`
- [X] T067 [US3] Implement `python -m src.training.train_classifier --config configs/classifier.yaml --dataset-root tests/fixtures/synthetic_dataset --synthetic-smoke` CLI in `src/training/train_classifier.py`
- [X] T068 [US3] Run User Story 3 checks with `pytest tests/test_training.py -q`

**Checkpoint**: User Story 3 is complete when all finalized classifier training artifacts are generated, reviewable, reproducible, ignored, and untracked.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the full feature against the contract, quickstart, and constitution boundaries.

- [X] T069 Validate implemented behavior against `specs/005-binary-classifier-training/contracts/classifier-training-contract.md`
- [X] T070 Run quickstart automated tests with `pytest tests/test_classifier.py tests/test_training.py tests/test_training.py -q`
- [X] T071 Run manual synthetic smoke command from `specs/005-binary-classifier-training/quickstart.md`
- [X] T072 Confirm synthetic training smoke completes under one minute in `tests/test_training.py`
- [X] T073 Confirm generated classifier artifacts under `outputs/models/`, `outputs/reports/`, and `outputs/predictions/` are ignored and untracked using `git status --short --ignored` and `git ls-files`
- [X] T074 Confirm SPEC-005 does not implement Kaggle submission generation, test-image inference, detector training, hybrid inference, Grad-CAM, dashboard, memory bank, ensemble teacher, distillation, or hard-example mining in `src/models/classifier.py`, `src/training/train_classifier.py`, `src/training/losses.py`, `src/training/metrics.py`, and `src/training/threshold_search.py`
- [X] T075 Run prior-spec regression gate with `pytest tests/test_project_foundation.py tests/test_dataset.py tests/test_dataset_audit.py tests/test_coco_parser.py tests/test_roi.py tests/test_preprocessing.py -q`
- [X] T076 Update `configs/classifier.yaml` only for SPEC-005 training defaults and avoid private dataset paths
- [X] T077 Leave `notebooks/02_train_classifier.ipynb` unchanged unless SPEC-005 quickstart or documentation explicitly requires a synthetic classifier workflow link

---

## Phase 7: Post-Review Remediation

**Purpose**: Close pre-real-training blockers found in SPEC-005 review while staying inside classifier-training scope.

- [X] T078 Add real EfficientNet-B0 factory coverage that runs when `torchvision` is available in `tests/test_classifier.py`
- [X] T079 Replace full-split tensor stacking with per-sample `Dataset` and `DataLoader` mini-batch training in `src/training/train_classifier.py`
- [X] T080 Add configurable `device`, `num_workers`, and `pin_memory` training settings in `configs/classifier.yaml` and `src/training/train_classifier.py`
- [X] T081 Move model, inputs, labels, and weighted BCE `pos_weight` onto the selected training device in `src/training/train_classifier.py` and `src/training/losses.py`
- [X] T082 Reuse SPEC-002 dataset audit helpers for training label/image validation in `src/training/train_classifier.py`
- [X] T083 Add tests for DataLoader batch sizing, CPU device path, train-split-only class weights, validation-only threshold search, and saved device metrics in `tests/test_training.py`
- [X] T084 Reconcile active SPEC-005 directory naming in `specs/005-binary-classifier-training/spec.md` and `specs/005-binary-classifier-training/plan.md` without creating a conflicting `006` spec folder

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories because records, config validation, reproducibility, fixtures, and artifact helpers are shared.
- **User Story 1 (Phase 3)**: Depends on Foundational; delivers MVP training plus minimal persisted output behavior.
- **User Story 2 (Phase 4)**: Depends on User Story 1 because threshold and upgraded checkpoint selection consume validation probabilities and minimal artifacts.
- **User Story 3 (Phase 5)**: Depends on User Stories 1 and 2 because finalized outputs include trained weights, rich metrics, threshold, and validation predictions.
- **Polish (Phase 6)**: Depends on all desired user stories.
- **Post-Review Remediation (Phase 7)**: Depends on Polish and closes the real-training readiness blockers.

### User Story Dependencies

- **US1 Train V1 Binary Classifier**: Required MVP and prerequisite for richer metrics, threshold search, and finalized artifact outputs.
- **US2 Measure Validation F1 and Threshold**: Builds on US1 validation probabilities and minimal F1 selection.
- **US3 Save Reproducible Training Outputs**: Builds on US1 minimal artifacts and US2 metric/threshold results.

### Within Each User Story

- Tests are written before implementation.
- Shared data structures and fixtures are completed before behavior using them.
- Story checkpoint command runs before moving to the next story.

---

## Parallel Opportunities

- T015, T016, T017, T018, and T019 can run in parallel after T010 and T011.
- T020 through T030 can run in parallel after foundational fixtures exist.
- T044 through T048 can run in parallel after US1 validation probability and minimal artifact structures exist.
- T056 through T061 can run in parallel after US1 and US2 output contracts exist.
- T036 and T037 can be implemented in parallel because they touch separate files.
- T049, T050, T051, and T052 can be implemented in parallel only if coordinated across `metrics.py` and `threshold_search.py`.

---

## Parallel Example: User Story 1

```text
Task: "Write failing binary label validation test in tests/test_training.py"
Task: "Write failing classifier forward-output shape and probability-range test in tests/test_classifier.py"
Task: "Write failing weighted binary loss construction test in tests/test_classifier.py"
Task: "Write failing minimal best model artifact output test after synthetic smoke training in tests/test_training.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational records, config validation, reproducibility, fixtures, and artifact helpers.
3. Complete Phase 3: User Story 1.
4. Stop and validate with `pytest tests/test_classifier.py tests/test_training.py -q`.

### Incremental Delivery

1. Add classifier/training/loss/metric/threshold scaffolds and synthetic classifier fixtures.
2. Add binary label validation, stratified split, SPEC-004 preprocessing integration, classifier factory, weighted loss, synthetic training smoke path, and minimal persisted outputs.
3. Add full validation F1, confusion/class counts, threshold search, lowest-threshold tie-break, and upgraded best-checkpoint selection.
4. Add finalized model, rich metrics, threshold, and validation prediction outputs plus confidentiality validation.
5. Run contract, quickstart, prior-spec regression, and out-of-scope validation.

### Final Validation

1. Run `pytest tests/test_classifier.py tests/test_training.py tests/test_training.py -q`.
2. Run the synthetic quickstart command from `quickstart.md`.
3. Confirm no private Krones dataset access is required for automated tests.
4. Confirm generated classifier artifacts are ignored and untracked.
5. Confirm Kaggle submission, test-image inference, detector training, hybrid inference, Grad-CAM, dashboard, memory bank, ensemble teacher, distillation, and hard-example mining are not implemented in SPEC-005.

---

## Notes

- [P] tasks use different files and can run in parallel after prerequisites.
- Story labels map directly to the three user stories in `spec.md`.
- SPEC-005 trains the V1 classifier only; first Kaggle submission starts in a later spec.
- US1 must persist minimal end-to-end outputs; US3 hardens and completes the richer final artifact/report behavior.
- Validation F1 and threshold behavior must be deterministic and reproducible from seed/config.
