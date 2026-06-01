---
description: "Task list for SPEC-002 Dataset Loading and Validation"
---

# Tasks: Dataset Loading and Validation

**Input**: Design documents from `/specs/002-dataset-loading-validation/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/dataset-audit-contract.md, quickstart.md

**Tests**: Synthetic-fixture pytest coverage is required so dataset behavior can be validated without private Krones data.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish shared package/config/test files needed by every dataset loading story.

- [ ] T001 Create data package marker in `src/data/__init__.py`
- [ ] T002 Create utility package marker in `src/utils/__init__.py`
- [ ] T003 Create initial configurable dataset paths template in `configs/paths.yaml`
- [ ] T004 Create synthetic dataset fixture directory structure in `tests/fixtures/synthetic_dataset/train_images/` and `tests/fixtures/synthetic_dataset/test_images/`
- [ ] T005 Update `requirements.txt` with `matplotlib` and `Pillow`, keeping existing `opencv-python` if already present

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Provide reusable configuration and synthetic fixture data before story work begins.

**CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T006 Write failing config loader tests for valid, missing, and malformed path configs in `tests/test_dataset.py`
- [ ] T007 Implement YAML config loading and readable config errors in `src/utils/config.py`
- [ ] T008 [P] Create synthetic `train.csv` with reusable and not-reusable rows in `tests/fixtures/synthetic_dataset/train.csv`
- [ ] T009 [P] Create synthetic `sample_submission.csv` in `tests/fixtures/synthetic_dataset/sample_submission.csv`
- [ ] T010 [P] Create synthetic audit-only COCO metadata in `tests/fixtures/synthetic_dataset/train_annotations.json`
- [ ] T011 [P] Create valid small PNG or JPEG train/test image files with known dimensions in `tests/fixtures/synthetic_dataset/train_images/` and `tests/fixtures/synthetic_dataset/test_images/`

**Checkpoint**: Config loading and synthetic test data are ready for user-story tests.

---

## Phase 3: User Story 1 - Load Competition Dataset (Priority: P1) MVP

**Goal**: A contributor can point the project at the configured dataset root and verify required dataset files and image folders are present.

**Independent Test**: Use the synthetic dataset root to confirm required files/folders are discovered and missing items produce readable errors.

### Tests for User Story 1

- [ ] T012 [US1] Write failing required file/folder presence tests in `tests/test_dataset.py`
- [ ] T013 [US1] Write failing missing required item diagnostic tests in `tests/test_dataset.py`

### Implementation for User Story 1

- [ ] T014 [US1] Implement dataset root and required path resolution in `src/data/dataset.py`
- [ ] T015 [US1] Implement required file/folder validation with readable missing-item diagnostics in `src/data/dataset.py`
- [ ] T016 [US1] Add CLI entry behavior for dataset presence checks in `src/data/audit.py`
- [ ] T017 [US1] Run User Story 1 checks with `pytest tests/test_dataset.py -q`

**Checkpoint**: User Story 1 is complete when required dataset files/folders are verified from config and missing items fail loudly.

---

## Phase 4: User Story 2 - Validate Dataset Integrity (Priority: P2)

**Goal**: A contributor can audit train/test images, label rows, audit-only annotation metadata, image IDs, missing files, and image sizes.

**Independent Test**: Run synthetic fixture tests that verify label-to-image matching, present-but-unreferenced images, orphaned annotations, and image-size summaries.

### Tests for User Story 2

- [ ] T018 [US2] Write failing training label parsing and binary target validation tests in `tests/test_dataset.py`
- [ ] T019 [US2] Write failing train/test image discovery tests in `tests/test_dataset.py`
- [ ] T020 [US2] Write failing label-to-image matching and present-unreferenced image tests in `tests/test_dataset.py`
- [ ] T021 [US2] Write failing audit-only annotation coverage and orphaned annotation tests in `tests/test_dataset_audit.py`
- [ ] T022 [US2] Write failing image-size summary tests using known-dimension synthetic images in `tests/test_dataset_audit.py`

### Implementation for User Story 2

- [ ] T023 [US2] Implement training label loading and binary target validation in `src/data/dataset.py`
- [ ] T024 [US2] Implement train/test image discovery in `src/data/dataset.py`
- [ ] T025 [US2] Implement label-to-image matching and present-unreferenced image reporting in `src/data/dataset.py`
- [ ] T026 [US2] Implement audit-only COCO metadata summary for coverage, categories, ROI availability, and orphaned annotations in `src/data/audit.py`
- [ ] T027 [US2] Implement image metadata reading and image-size summary in `src/data/audit.py`
- [ ] T028 [US2] Run User Story 2 checks with `pytest tests/test_dataset.py tests/test_dataset_audit.py -q`

**Checkpoint**: User Story 2 is complete when dataset integrity issues are counted, summarized, and reported without creating test labels.

---

## Phase 5: User Story 3 - Produce Dataset Audit Outputs (Priority: P3)

**Goal**: A contributor can generate report-ready dataset audit CSV outputs and figures in approved ignored output locations.

**Independent Test**: Run the audit on synthetic data and confirm expected reports/figures are produced under `outputs/reports/` and `outputs/figures/`.

### Tests for User Story 3

- [ ] T029 [US3] Write failing dataset summary report generation tests in `tests/test_dataset_audit.py`
- [ ] T030 [US3] Write failing class and defect distribution output tests in `tests/test_dataset_audit.py`
- [ ] T031 [US3] Write failing audit figure output tests for class distribution, defect distribution, and sample grid in `tests/test_dataset_audit.py`
- [ ] T032 [US3] Write failing generated artifact confidentiality tests that verify generated files in `outputs/reports/` and `outputs/figures/` are ignored and not tracked in `tests/test_dataset_audit.py`

### Implementation for User Story 3

- [ ] T033 [US3] Implement CSV report generation for dataset summary, missing files, orphaned annotations, class distribution, defect distribution, and image-size summary in `src/data/audit.py`
- [ ] T034 [US3] Implement class distribution, defect distribution, and sample grid figure generation in `src/data/audit.py`
- [ ] T035 [US3] Implement audit output directory creation under `outputs/reports/` and `outputs/figures/` in `src/data/audit.py`
- [ ] T036 [US3] Implement generated-output ignore validation helpers for `outputs/reports/` and `outputs/figures/` in `src/data/audit.py`
- [ ] T037 [US3] Update dataset audit notebook to call reusable audit behavior in `notebooks/01_dataset_audit.ipynb`
- [ ] T038 [US3] Run User Story 3 checks with `pytest tests/test_dataset_audit.py -q`

**Checkpoint**: User Story 3 is complete when report and figure artifacts are generated from synthetic data in approved output locations.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the full feature against the contract, quickstart, and constitution.

- [ ] T039 Validate implemented behavior against `specs/002-dataset-loading-validation/contracts/dataset-audit-contract.md`
- [ ] T040 Run quickstart automated tests with `pytest tests/test_dataset.py tests/test_dataset_audit.py -q`
- [ ] T041 When valid private dataset access is available, run configured private dataset audit command from quickstart with `python -m src.data.audit --config configs/paths.yaml`; otherwise run synthetic fixture validation only
- [ ] T042 When valid private dataset access is available, record dataset audit runtime and whether it is under the five-minute target in `outputs/reports/dataset_summary.csv`; otherwise document that only synthetic fixture runtime was validated
- [ ] T043 Confirm private dataset files and generated audit artifacts are not tracked using `git ls-files`, and confirm files generated under `outputs/reports/` and `outputs/figures/` are ignored using `git status --short --ignored`
- [ ] T044 Update `docs/krones-final-implementation-plan.md` only if SPEC-002 implementation changes the approved Phase 2 behavior

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories because config loading and synthetic fixtures are shared.
- **User Story 1 (Phase 3)**: Depends on Foundational; delivers MVP dataset discovery.
- **User Story 2 (Phase 4)**: Depends on User Story 1 because integrity checks need resolved dataset paths and required-file validation.
- **User Story 3 (Phase 5)**: Depends on User Story 2 because audit outputs require integrity summaries.
- **Polish (Phase 6)**: Depends on all desired user stories.

### User Story Dependencies

- **US1 Load Competition Dataset**: MVP and prerequisite for all later dataset checks.
- **US2 Validate Dataset Integrity**: Builds on US1 discovery and validation.
- **US3 Produce Dataset Audit Outputs**: Builds on US2 summaries and emits reports/figures.

### Within Each User Story

- Tests are written before implementation.
- Implementation files are created after failing test intent is captured.
- Story checkpoint command runs before moving to the next story.

---

## Parallel Opportunities

- T001 and T002 can run in parallel.
- T008, T009, T010, and T011 can run in parallel after T007.
- T012 and T013 can run in parallel.
- T018, T019, T020, T021, and T022 can run in parallel after US1.
- T029, T030, T031, and T032 can run in parallel after US2.

---

## Parallel Example: User Story 2

```text
Task: "Write failing training label parsing and binary target validation tests in tests/test_dataset.py"
Task: "Write failing train/test image discovery tests in tests/test_dataset.py"
Task: "Write failing label-to-image matching and present-unreferenced image tests in tests/test_dataset.py"
Task: "Write failing audit-only annotation coverage and orphaned annotation tests in tests/test_dataset_audit.py"
Task: "Write failing image-size summary tests in tests/test_dataset_audit.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational config and synthetic fixtures.
3. Complete Phase 3: User Story 1.
4. Stop and validate with `pytest tests/test_dataset.py -q`.

### Incremental Delivery

1. Add config loading and synthetic fixture data.
2. Add dataset required-file discovery for US1.
3. Add integrity summaries for US2.
4. Add report/figure outputs for US3.
5. Run contract and quickstart validation.

### Final Validation

1. Run `pytest tests/test_dataset.py tests/test_dataset_audit.py -q`.
2. Run `python -m src.data.audit --config configs/paths.yaml` only when private dataset access is valid; otherwise run synthetic fixture validation only.
3. Record audit runtime and whether it is under the five-minute target when private dataset access is valid.
4. Run `git ls-files`.
5. Run `git status --short --ignored`.
6. Verify private dataset files and generated audit artifacts under `outputs/reports/` and `outputs/figures/` are not tracked or staged.

---

## Notes

- [P] tasks use different files and can run in parallel after prerequisites.
- Story labels map directly to the three user stories in `spec.md`.
- SPEC-002 performs audit-only annotation metadata handling; reusable COCO parser APIs remain in SPEC-003.
- This task list intentionally does not create real training labels for test images, model weights, submissions, or public dataset artifacts.
