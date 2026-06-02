---
description: "Task list for SPEC-004 ROI Cropping and Preprocessing"
---

# Tasks: ROI Cropping and Preprocessing

**Input**: Design documents from `/specs/004-roi-cropping-preprocessing/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/roi-preprocessing-contract.md, quickstart.md

**Tests**: Synthetic-fixture pytest coverage is required so ROI crop and preprocessing behavior can be validated without private Krones data.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish ROI/preprocessing module and test locations while keeping training, detector, submission, and dashboard work out of scope.

- [X] T001 Create ROI crop module scaffold in `src/data/roi.py`
- [X] T002 Create preprocessing orchestration scaffold in `src/data/preprocessing.py`
- [X] T003 Create transform profile scaffold in `src/data/transforms.py`
- [X] T004 Create SPEC-004 test module scaffold in `tests/test_roi.py`
- [X] T005 Confirm existing `requirements.txt` entries cover `Pillow`, `numpy`, `opencv-python`, `albumentations`, and `pytest` for SPEC-004

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define shared data records, defaults, synthetic fixtures, and confidentiality helpers before user-story behavior.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T006 Define `RoiCropRequest`, `RoiCropResult`, `PreprocessingProfile`, and `RoiSampleReport` structures in `src/data/roi.py`
- [X] T007 Define SPEC-004 constants for default `384x384` size, 10% ROI margin, 10% minimum ROI area, and `outputs/figures/roi_samples/` in `src/data/roi.py`
- [X] T008 [P] Create synthetic dark-region ROI image fixture in `tests/fixtures/synthetic_dataset/roi_cases/dark_region.jpg`
- [X] T009 [P] Create synthetic small-ROI image fixture in `tests/fixtures/synthetic_dataset/roi_cases/small_roi.jpg`
- [X] T010 [P] Create synthetic unusual-aspect image fixture in `tests/fixtures/synthetic_dataset/roi_cases/wide_image.jpg`
- [X] T011 [P] Create synthetic ROI metadata fixture in `tests/fixtures/synthetic_dataset/roi_cases/roi_cases.json`
- [X] T012 Implement image loading, RGB conversion, and readable image errors in `src/data/roi.py`
- [X] T013 Implement generated ROI sample ignore/tracking validation helper in `src/data/roi.py`

**Checkpoint**: Shared records, defaults, fixtures, and helpers are ready for user-story tests.

---

## Phase 3: User Story 1 - Produce Safe ROI Crops (Priority: P1) MVP

**Goal**: A contributor can produce safe `384x384` ROI crops from annotation metadata or deterministic fallbacks while preserving internal dark defect evidence.

**Independent Test**: Run synthetic crop tests that verify annotation ROI expansion, clipped bounds, square center fallback, full-resize fallback, output size, and dark-region preservation.

### Tests for User Story 1

- [X] T014 [P] [US1] Write failing annotation ROI crop expansion and clipped-bounds test in `tests/test_roi.py`
- [X] T015 [P] [US1] Write failing unsafe small-ROI fallback test in `tests/test_roi.py`
- [X] T016 [P] [US1] Write failing missing ROI square-center fallback test in `tests/test_roi.py`
- [X] T017 [P] [US1] Write failing full-image resize fallback test for unsafe center crop in `tests/test_roi.py`
- [X] T018 [P] [US1] Write failing internal dark-region preservation test in `tests/test_roi.py`

### Implementation for User Story 1

- [X] T019 [US1] Implement annotation bbox validation and image-bound clipping in `src/data/roi.py`
- [X] T020 [US1] Implement 10% annotation ROI margin expansion in `src/data/roi.py`
- [X] T021 [US1] Implement 10% minimum ROI area safety check in `src/data/roi.py`
- [X] T022 [US1] Implement square center fallback crop in `src/data/roi.py`
- [X] T023 [US1] Implement full-image resize fallback in `src/data/roi.py`
- [X] T024 [US1] Implement final `384x384` crop resizing and RGB output normalization in `src/data/roi.py`
- [X] T025 [US1] Preserve internal dark pixels by avoiding dark-pixel removal logic in `src/data/roi.py`
- [X] T026 [US1] Run User Story 1 checks with `pytest tests/test_roi.py -q`

**Checkpoint**: User Story 1 is complete when safe ROI crops and fallbacks work independently on synthetic fixtures.

---

## Phase 4: User Story 2 - Apply Split-Specific Preprocessing (Priority: P2)

**Goal**: A contributor can apply bounded training preprocessing and deterministic validation/test preprocessing without leaking random augmentation into evaluation.

**Independent Test**: Run synthetic preprocessing tests that confirm training transforms keep dimensions and bounds, while validation/test outputs are identical across repeated runs.

### Tests for User Story 2

- [X] T027 [P] [US2] Write failing preprocessing profile validation test in `tests/test_roi.py`
- [X] T028 [P] [US2] Write failing training preprocessing output-size and bounded-transform test in `tests/test_roi.py`
- [X] T029 [P] [US2] Write failing validation preprocessing determinism test in `tests/test_roi.py`
- [X] T030 [P] [US2] Write failing test preprocessing determinism test in `tests/test_roi.py`
- [X] T031 [P] [US2] Write failing reproducible seeded training preprocessing test in `tests/test_roi.py`

### Implementation for User Story 2

- [X] T032 [US2] Implement preprocessing profile factory for `train`, `validation`, and `test` splits in `src/data/transforms.py`
- [X] T033 [US2] Implement bounded training transform composition for resize, normalize, brightness/contrast, rotation, shift/scale, blur, and noise in `src/data/transforms.py`
- [X] T034 [US2] Implement deterministic validation/test transform composition for resize and normalize only in `src/data/transforms.py`
- [X] T035 [US2] Implement split-specific preprocessing entry point that combines ROI crop results with transform profiles in `src/data/preprocessing.py`
- [X] T036 [US2] Implement seed handling for reproducible training preprocessing in `src/data/preprocessing.py`
- [X] T037 [US2] Run User Story 2 checks with `pytest tests/test_roi.py -q`

**Checkpoint**: User Story 2 is complete when split-specific preprocessing behaves reproducibly and does not augment validation/test data.

---

## Phase 5: User Story 3 - Generate Reviewable ROI Samples (Priority: P3)

**Goal**: A contributor can generate reviewable ROI sample images under ignored output locations and confirm generated artifacts are not tracked.

**Independent Test**: Run synthetic sample-generation tests that save annotation, fallback, and deterministic preprocessed validation/test samples under ignored output locations and verify generated files are ignored and untracked.

### Tests for User Story 3

- [X] T038 [P] [US3] Write failing ROI sample generation test for annotation and fallback crops in `tests/test_roi.py`
- [X] T039 [P] [US3] Write failing ROI sample output location test for `outputs/figures/roi_samples/` in `tests/test_roi.py`
- [X] T040 [P] [US3] Write failing generated ROI sample confidentiality test in `tests/test_roi.py`
- [X] T041 [P] [US3] Write failing ROI quickstart CLI smoke test using synthetic fixtures in `tests/test_roi.py`
- [X] T042 [P] [US3] Write failing preprocessed validation/test sample output test in `tests/test_roi.py`

### Implementation for User Story 3

- [X] T043 [US3] Implement ROI sample directory creation under `outputs/figures/roi_samples/` in `src/data/roi.py`
- [X] T044 [US3] Implement synthetic ROI sample image saving with crop method metadata in `src/data/roi.py`
- [X] T045 [US3] Implement preprocessed validation/test sample saving under `outputs/figures/roi_samples/` in `src/data/preprocessing.py`
- [X] T046 [US3] Implement generated ROI sample confidentiality checks for ignored and untracked files in `src/data/roi.py`
- [X] T047 [US3] Add optional quickstart/manual CLI for synthetic ROI sample generation in `src/data/roi.py`
- [X] T048 [US3] Run User Story 3 checks with `pytest tests/test_roi.py -q`

**Checkpoint**: User Story 3 is complete when ROI samples are reviewable, correctly located, and protected from source control.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the full feature against the contract, quickstart, and constitution boundaries.

- [X] T049 Validate implemented behavior against `specs/004-roi-cropping-preprocessing/contracts/roi-preprocessing-contract.md`
- [X] T050 Run quickstart automated tests with `pytest tests/test_roi.py -q`
- [X] T051 Run synthetic batch performance smoke test for ROI crop plus deterministic preprocessing under one second in `tests/test_roi.py`
- [X] T052 Run manual synthetic smoke command from `specs/004-roi-cropping-preprocessing/quickstart.md`
- [X] T053 Confirm generated ROI samples under `outputs/figures/roi_samples/` are ignored and untracked using `git status --short --ignored` and `git ls-files`
- [X] T054 Confirm SPEC-004 does not implement classifier training, detector export, Kaggle submission generation, Grad-CAM, dashboard code, or test-label inference in `src/data/roi.py`, `src/data/preprocessing.py`, and `src/data/transforms.py`
- [X] T055 Leave `notebooks/01_dataset_audit.ipynb` unchanged unless SPEC-004 quickstart or documentation needs explicitly require a synthetic ROI workflow link
- [X] T056 Update `docs/krones-final-implementation-plan.md` only if SPEC-004 implementation changes the approved Phase 3 behavior

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories because shared records, constants, fixtures, and helpers are required.
- **User Story 1 (Phase 3)**: Depends on Foundational; delivers MVP ROI crop behavior.
- **User Story 2 (Phase 4)**: Depends on User Story 1 because preprocessing consumes ROI crop outputs.
- **User Story 3 (Phase 5)**: Depends on User Story 1 and can run after or alongside User Story 2 once crop outputs exist.
- **Polish (Phase 6)**: Depends on all desired user stories.

### User Story Dependencies

- **US1 Produce Safe ROI Crops**: Required MVP and prerequisite for preprocessing and sample generation.
- **US2 Apply Split-Specific Preprocessing**: Builds on US1 crop outputs.
- **US3 Generate Reviewable ROI Samples**: Builds on US1 crop outputs and validates confidentiality for generated sample images.

### Within Each User Story

- Tests are written before implementation.
- Shared data structures and constants are completed before behavior using them.
- Story checkpoint command runs before moving to the next story.

---

## Parallel Opportunities

- T008, T009, T010, and T011 can run in parallel after T006 and T007.
- T014, T015, T016, T017, and T018 can run in parallel after foundational fixtures exist.
- T027, T028, T029, T030, and T031 can run in parallel after US1 crop outputs exist.
- T038, T039, T040, T041, and T042 can run in parallel after US1 crop outputs exist.
- T032 and T034 can be implemented in parallel in `src/data/transforms.py` only if coordinated to avoid same-file conflicts; otherwise keep sequential.

---

## Parallel Example: User Story 1

```text
Task: "Write failing annotation ROI crop expansion and clipped-bounds test in tests/test_roi.py"
Task: "Write failing unsafe small-ROI fallback test in tests/test_roi.py"
Task: "Write failing missing ROI square-center fallback test in tests/test_roi.py"
Task: "Write failing full-image resize fallback test for unsafe center crop in tests/test_roi.py"
Task: "Write failing internal dark-region preservation test in tests/test_roi.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational records, constants, fixtures, and helpers.
3. Complete Phase 3: User Story 1.
4. Stop and validate with `pytest tests/test_roi.py -q`.

### Incremental Delivery

1. Add ROI module scaffolds and synthetic ROI fixtures.
2. Add safe ROI crop and fallback behavior for US1.
3. Add split-specific preprocessing profiles for US2.
4. Add reviewable ROI crop and preprocessed sample generation plus confidentiality validation for US3.
5. Run contract, quickstart, and out-of-scope validation.

### Final Validation

1. Run `pytest tests/test_roi.py -q`.
2. Run the synthetic quickstart command from `quickstart.md`.
3. Confirm no private Krones dataset access is required.
4. Confirm generated ROI samples are ignored and untracked.
5. Confirm classifier training, detector export, Kaggle submission generation, Grad-CAM, dashboard code, and test-label inference are not implemented in SPEC-004.

---

## Notes

- [P] tasks use different files and can run in parallel after prerequisites.
- Story labels map directly to the three user stories in `spec.md`.
- SPEC-004 prepares images only; binary classifier training starts in SPEC-005.
- ROI cropping must preserve internal dark defect evidence and must not remove dark pixels blindly.
