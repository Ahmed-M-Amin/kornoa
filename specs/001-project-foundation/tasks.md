---
description: "Task list for SPEC-001 Project Foundation"
---

# Tasks: Project Foundation

**Input**: Design documents from `/specs/001-project-foundation/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/foundation-structure.md, quickstart.md

**Tests**: Foundation smoke tests are required by the specification and plan. Write the tests before implementing the safe foundation files they validate.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish root documentation and dependency entry points shared by every story.

- [X] T001 Create project README with `.specify/memory/constitution.md` reference and `SPEC-001` through `SPEC-014` future specification list in `README.md`
- [X] T002 Create Python dependency entry point in `requirements.txt` with `torch`, `torchvision`, `timm`, `ultralytics`, `opencv-python`, `albumentations`, `pandas`, `numpy`, `scikit-learn`, `pycocotools`, `streamlit`, `grad-cam`, `onnxruntime`, `pytest`, and `PyYAML`
- [X] T003 [P] Create root Python package marker in `src/__init__.py`
- [X] T004 [P] Create test package marker in `tests/__init__.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define confidentiality and generated-artifact safeguards before creating foundation areas.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Create private dataset and generated artifact ignore rules in `.gitignore`
- [X] T006 [P] Add output directory tracking exceptions for `.gitkeep` files in `.gitignore`
- [X] T007 [P] Add Python cache, virtual environment, and notebook checkpoint ignore rules in `.gitignore`

**Checkpoint**: Private data and generated artifact protections are defined.

---

## Phase 3: User Story 1 - Initialize Project Workspace (Priority: P1) MVP

**Goal**: A contributor can open the repository and identify actual safe foundation directories/files for configuration, source modules, tests, notebooks, outputs, documentation, and Spec Kit governance.

**Independent Test**: Inspect a fresh checkout and verify required foundation directories/files exist and match the planned layout.

### Tests for User Story 1

- [X] T008 [US1] Write failing workspace structure test for root files and directories in `tests/test_project_foundation.py`
- [X] T009 [US1] Write failing source package import test for `src/` package markers in `tests/test_project_foundation.py`

### Implementation for User Story 1

- [X] T010 [P] [US1] Create valid minimal notebook JSON placeholder files in `notebooks/01_dataset_audit.ipynb`, `notebooks/02_train_classifier.ipynb`, `notebooks/03_train_detector.ipynb`, `notebooks/04_distillation_memory.ipynb`, and `notebooks/05_kaggle_submission.ipynb`
- [X] T011 [P] [US1] Create source package markers in `src/data/__init__.py`, `src/models/__init__.py`, `src/training/__init__.py`, `src/inference/__init__.py`, `src/explainability/__init__.py`, `src/dashboard/__init__.py`, and `src/utils/__init__.py`
- [X] T012 [P] [US1] Create future test placeholder files in `tests/test_dataset.py`, `tests/test_coco_parser.py`, `tests/test_roi.py`, `tests/test_inference.py`, and `tests/test_submission.py`
- [X] T013 [US1] Run workspace structure and import checks with `pytest tests/test_project_foundation.py -q`

**Checkpoint**: User Story 1 is complete when the repository foundation layout exists and package import checks pass.

---

## Phase 4: User Story 2 - Configure Dataset Paths Safely (Priority: P2)

**Goal**: A contributor can define local Windows, Kaggle, and Colab dataset paths through configuration without hardcoding private paths inside core modules.

**Independent Test**: Review and parse configuration files to confirm dataset paths are configurable and private data is not embedded as tracked source content beyond the documented local example.

### Tests for User Story 2

- [X] T014 [US2] Write failing configuration readability test for all YAML config files in `tests/test_project_foundation.py`
- [X] T015 [US2] Write failing paths configuration test for local, Kaggle, and Colab keys in `tests/test_project_foundation.py`

### Implementation for User Story 2

- [X] T016 [P] [US2] Create dataset path configuration template in `configs/paths.yaml`
- [X] T017 [P] [US2] Create classifier configuration placeholder in `configs/classifier.yaml`
- [X] T018 [P] [US2] Create detector configuration placeholder in `configs/detector.yaml`
- [X] T019 [P] [US2] Create inference configuration placeholder in `configs/inference.yaml`
- [X] T020 [P] [US2] Create dashboard configuration placeholder in `configs/dashboard.yaml`
- [X] T021 [US2] Run configuration readability checks with `pytest tests/test_project_foundation.py -q`

**Checkpoint**: User Story 2 is complete when config files parse successfully and dataset roots are configurable for local Windows, Kaggle, and Colab.

---

## Phase 5: User Story 3 - Prepare Reproducible Experiment Outputs (Priority: P3)

**Goal**: A contributor can place future model weights, predictions, metrics, figures, submissions, and hard-example artifacts in predictable generated-output locations without tracking generated private contents.

**Independent Test**: Confirm output category directories exist, contain only safe tracking placeholders, and generated artifact patterns are ignored.

### Tests for User Story 3

- [X] T022 [US3] Write failing output category structure test in `tests/test_project_foundation.py`
- [X] T023 [US3] Write failing ignore-rule coverage test for generated artifact categories in `tests/test_project_foundation.py`

### Implementation for User Story 3

- [X] T024 [P] [US3] Create output tracking placeholder in `outputs/.gitkeep`
- [X] T025 [P] [US3] Create model output tracking placeholder in `outputs/models/.gitkeep`
- [X] T026 [P] [US3] Create prediction output tracking placeholder in `outputs/predictions/.gitkeep`
- [X] T027 [P] [US3] Create figure output tracking placeholder in `outputs/figures/.gitkeep`
- [X] T028 [P] [US3] Create report output tracking placeholder in `outputs/reports/.gitkeep`
- [X] T029 [P] [US3] Create submission output tracking placeholder in `outputs/submissions/.gitkeep`
- [X] T030 [P] [US3] Create hard-example output tracking placeholder in `outputs/hard_examples/.gitkeep`
- [X] T031 [US3] Run output structure and ignore-rule checks with `pytest tests/test_project_foundation.py -q`

**Checkpoint**: User Story 3 is complete when generated-output categories exist and generated contents remain protected by `.gitignore`.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the full foundation against the Spec Kit contract and quickstart.

- [X] T032 Validate required root files, configs, source packages, outputs, and tests against `specs/001-project-foundation/contracts/foundation-structure.md`
- [X] T033 Run the quickstart validation commands documented in `specs/001-project-foundation/quickstart.md`
- [X] T034 Confirm no private dataset files from `1st-krones-vision-ai-challenge/` or generated artifact contents are tracked using `git ls-files`, and confirm ignore behavior using `git status --short --ignored`
- [X] T035 Update `docs/krones-final-implementation-plan.md` only if SPEC-001 implementation changes the approved foundation structure

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories because ignore rules must exist first.
- **User Story 1 (Phase 3)**: Depends on Foundational; delivers MVP workspace structure.
- **User Story 2 (Phase 4)**: Depends on Foundational and can run after or alongside US1 once shared test file conventions are established.
- **User Story 3 (Phase 5)**: Depends on Foundational and can run after or alongside US1 once shared test file conventions are established.
- **Polish (Phase 6)**: Depends on all desired user stories.

### User Story Dependencies

- **US1 Initialize Project Workspace**: Required MVP; establishes source/test/notebook layout.
- **US2 Configure Dataset Paths Safely**: Independent after Foundational; uses config directory conventions.
- **US3 Prepare Reproducible Experiment Outputs**: Independent after Foundational; uses output ignore conventions.

### Within Each User Story

- Tests are written before implementation.
- Implementation files are created after failing test intent is captured.
- Story checkpoint command runs before moving to the next story.

---

## Parallel Opportunities

- T003 and T004 can run in parallel.
- T006 and T007 can run in parallel after T005.
- T010, T011, and T012 can run in parallel after T008-T009.
- T016, T017, T018, T019, and T020 can run in parallel after T014-T015.
- T024, T025, T026, T027, T028, T029, and T030 can run in parallel after T022-T023.

---

## Parallel Example: User Story 2

```text
Task: "Create dataset path configuration template in configs/paths.yaml"
Task: "Create classifier configuration placeholder in configs/classifier.yaml"
Task: "Create detector configuration placeholder in configs/detector.yaml"
Task: "Create inference configuration placeholder in configs/inference.yaml"
Task: "Create dashboard configuration placeholder in configs/dashboard.yaml"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational ignore rules.
3. Complete Phase 3: User Story 1.
4. Stop and validate with `pytest tests/test_project_foundation.py -q`.

### Incremental Delivery

1. Add safe root and ignore foundation.
2. Add workspace layout for US1.
3. Add config templates for US2.
4. Add output placeholders for US3.
5. Run contract and quickstart validation.

### Final Validation

1. Run `pytest tests/test_project_foundation.py -q`.
2. Run `git ls-files`.
3. Run `git status --short --ignored`.
4. Verify private dataset and generated artifacts are not tracked or staged.

---

## Notes

- [P] tasks use different files and can run in parallel after their prerequisites.
- Story labels map directly to the three user stories in `spec.md`.
- This task list intentionally does not create real training data, predictions, model weights, submissions, or report artifacts.
