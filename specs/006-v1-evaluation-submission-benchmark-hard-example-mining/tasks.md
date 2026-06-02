# Tasks: V1 Evaluation, Submission, Benchmark, and Hard-Example Mining

**Input**: Design documents from `/specs/006-v1-evaluation-submission-benchmark-hard-example-mining/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/v1-evaluation-contract.md](./contracts/v1-evaluation-contract.md), [quickstart.md](./quickstart.md)

**Tests**: Required by the plan and constitution gates. Write tests first and verify they fail before implementation.

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish SPEC-006 module files and shared test utilities without implementing story behavior.

- [X] T001 Create empty SPEC-006 module files in `src/inference/predict.py`, `src/inference/submission.py`, `src/inference/benchmark.py`, and `src/training/hard_example_mining.py`
- [X] T002 [P] Create test file scaffolds in `tests/test_inference.py`, `tests/test_submission.py`, `tests/test_benchmark.py`, and `tests/test_hard_example_mining.py`
- [X] T003 [P] Add synthetic checkpoint/image helper utilities for SPEC-006 tests in `tests/test_inference.py`
- [X] T004 [P] Add synthetic sample-submission and validation-prediction fixture helpers in `tests/test_submission.py` and `tests/test_hard_example_mining.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared artifact, threshold, prediction, and error contracts that all user stories depend on.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Define `V1ArtifactPaths` and `V1Prediction` data structures in `src/inference/predict.py`
- [X] T006 Implement V1 artifact-root path resolution for concrete root `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1`, optional parent root `artifacts/kaggle_v1_artifacts/outputs`, and required concrete-run-file validation in `src/inference/predict.py`
- [X] T007 Implement saved threshold JSON loading and validation in `src/inference/predict.py`
- [X] T008 Implement training-device selection reuse or equivalent device resolution for inference in `src/inference/predict.py`
- [X] T009 Implement shared image discovery and sample-image path validation helpers in `src/inference/predict.py`
- [X] T010 Add readable exceptions for missing checkpoint, threshold, sample submission, validation predictions, and image files in `src/inference/predict.py`
- [X] T011 [P] Add tests for concrete artifact-root path resolution, parent-root `kaggle_v1` child detection, and missing artifact errors in `tests/test_inference.py`
- [X] T012 [P] Add tests for saved threshold loading, invalid threshold records, and no hardcoded threshold fallback in `tests/test_inference.py`

**Checkpoint**: Foundation ready. User story implementation can now begin.

---

## Phase 3: User Story 1 - Generate First V1 Submission (Priority: P1) MVP

**Goal**: Generate a valid V1 Kaggle submission from completed V1 artifacts and sample submission rows.

**Independent Test**: Use synthetic test images, a saved tiny classifier checkpoint, a saved threshold JSON, and a sample submission file to verify a sample-aligned binary output CSV with no hard-example side effects.

### Tests for User Story 1

- [X] T013 [P] [US1] Add prediction test for loading a classifier checkpoint and returning class-1 probabilities in `tests/test_inference.py`
- [X] T014 [P] [US1] Add prediction test for threshold-to-target conversion using saved threshold in `tests/test_inference.py`
- [X] T015 [P] [US1] Add submission test for preserving sample submission row order and label column contract in `tests/test_submission.py`
- [X] T016 [P] [US1] Add submission test for failing clearly when a sample image is missing in `tests/test_submission.py`
- [X] T017 [P] [US1] Add submission test proving hard-example files are not created during submission generation in `tests/test_submission.py`

### Implementation for User Story 1

- [X] T018 [US1] Implement classifier checkpoint loading with EfficientNet-B0 default and tiny synthetic test mode in `src/inference/predict.py`
- [X] T019 [US1] Implement lazy ROI/preprocessing image prediction for one or more images in `src/inference/predict.py`
- [X] T020 [US1] Implement probability-to-target conversion using saved threshold in `src/inference/predict.py`
- [X] T021 [US1] Implement ordered sample submission loading and label column detection in `src/inference/submission.py`
- [X] T022 [US1] Implement sample-aligned V1 submission CSV writing to `outputs/submissions/submission_v1.csv` by default in `src/inference/submission.py`
- [X] T023 [US1] Implement submission CLI arguments for dataset root, artifact root, explicit paths, device, batch size, and output path in `src/inference/submission.py`
- [X] T024 [US1] Ensure submission generation creates no `outputs/hard_examples` files or hard-example summary side effects in `src/inference/submission.py`

**Checkpoint**: User Story 1 is functional and independently testable.

---

## Phase 4: User Story 2 - Benchmark V1 Inference Speed (Priority: P2)

**Goal**: Measure and save V1 fast classifier inference speed without running hard-example mining or later heavy modules.

**Independent Test**: Run benchmark over synthetic images and verify JSON timing fields, image count, artifact references, and no hard-example side effects.

### Tests for User Story 2

- [X] T025 [P] [US2] Add benchmark test for image count, total runtime, average milliseconds per image, and throughput fields in `tests/test_benchmark.py`
- [X] T026 [P] [US2] Add benchmark test for artifact path references in `tests/test_benchmark.py`
- [X] T027 [P] [US2] Add benchmark test proving no hard-example files are generated in `tests/test_benchmark.py`

### Implementation for User Story 2

- [X] T028 [US2] Define benchmark report data structure in `src/inference/benchmark.py`
- [X] T029 [US2] Implement V1 inference timing around the fast classifier prediction path in `src/inference/benchmark.py`
- [X] T030 [US2] Implement benchmark JSON writing to `outputs/benchmarks/v1_inference_benchmark.json` by default in `src/inference/benchmark.py`
- [X] T031 [US2] Implement benchmark CLI arguments for image directory, sample submission mode, artifact root, explicit paths, device, batch size, and output path in `src/inference/benchmark.py`
- [X] T032 [US2] Ensure benchmark excludes submission scoring, detector, Grad-CAM, feature memory bank, and hard-example mining behavior in `src/inference/benchmark.py`

**Checkpoint**: User Story 2 is functional and independently testable.

---

## Phase 5: User Story 3 - Mine V1 Hard Examples (Priority: P3)

**Goal**: Regenerate offline V1 hard-example files from validation predictions only.

**Independent Test**: Use a synthetic validation prediction CSV and saved threshold to verify false positives, false negatives, uncertain rows, high-loss proxy ranking, and summary JSON without reading test images.

### Tests for User Story 3

- [X] T033 [P] [US3] Add hard-example test for false positive and false negative CSV generation in `tests/test_hard_example_mining.py`
- [X] T034 [P] [US3] Add hard-example test for uncertain row selection near the saved threshold in `tests/test_hard_example_mining.py`
- [X] T035 [P] [US3] Add hard-example test for high-loss proxy ranking from probability severity in `tests/test_hard_example_mining.py`
- [X] T036 [P] [US3] Add hard-example test for summary JSON counts, source paths, threshold reference, and artifact paths in `tests/test_hard_example_mining.py`
- [X] T037 [P] [US3] Add hard-example test proving mining does not require model checkpoint or test image access in `tests/test_hard_example_mining.py`

### Implementation for User Story 3

- [X] T038 [US3] Define hard-example row and mining result data structures in `src/training/hard_example_mining.py`
- [X] T039 [US3] Implement validation prediction CSV loading and binary prediction recomputation fallback in `src/training/hard_example_mining.py`
- [X] T040 [US3] Implement false positive and false negative filtering in `src/training/hard_example_mining.py`
- [X] T041 [US3] Implement uncertain-case filtering with configurable uncertainty margin in `src/training/hard_example_mining.py`
- [X] T042 [US3] Implement high-loss proxy ranking from false positives with high probability, false negatives with low probability, then near-threshold uncertain cases in `src/training/hard_example_mining.py`
- [X] T043 [US3] Implement hard-example CSV writing to `outputs/hard_examples/false_positives.csv`, `outputs/hard_examples/false_negatives.csv`, `outputs/hard_examples/uncertain.csv`, and `outputs/hard_examples/high_loss_samples.csv` in `src/training/hard_example_mining.py`
- [X] T044 [US3] Implement hard-example summary JSON writing to `outputs/reports/hard_example_summary.json` in `src/training/hard_example_mining.py`
- [X] T045 [US3] Implement hard-example mining CLI arguments for concrete artifact root, optional parent artifact root, predictions path, threshold path, output directory, summary path, uncertainty margin, and high-loss limit in `src/training/hard_example_mining.py`

**Checkpoint**: User Story 3 is functional and independently testable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate integration, scope boundaries, generated artifact confidentiality, and quickstart commands.

- [X] T046 [P] Add generated-output ignore validation for submission, benchmark, hard-example, and report outputs in `tests/test_submission.py` and `tests/test_hard_example_mining.py`
- [X] T047 [P] Add CLI smoke tests for `python -m src.inference.submission`, `python -m src.inference.benchmark`, and `python -m src.training.hard_example_mining` in `tests/test_submission.py`, `tests/test_benchmark.py`, and `tests/test_hard_example_mining.py`
- [X] T048 [P] Add scope-guard tests proving detector, segmentation, Grad-CAM, dashboard, V2 training, feature memory bank, ensemble, distillation, and hybrid inference outputs are not created by SPEC-006 workflows in `tests/test_inference.py`
- [X] T049 Run SPEC-006 quickstart validation command `pytest tests/test_inference.py tests/test_submission.py tests/test_benchmark.py tests/test_hard_example_mining.py -q`
- [X] T050 Run full project regression command `pytest`
- [X] T051 Review `git status --short` to confirm only SPEC-006 code, tests, docs, and intended `AGENTS.md` context changes are present

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational; delivers MVP submission generation.
- **User Story 2 (Phase 4)**: Depends on Foundational and prediction helpers from US1 tasks T018-T020.
- **User Story 3 (Phase 5)**: Depends on Foundational only; independent of US1 and US2.
- **Polish (Phase 6)**: Depends on all selected user stories being complete.

### User Story Dependencies

- **US1 Generate First V1 Submission**: MVP. Must be completed before claiming V1 submission readiness.
- **US2 Benchmark V1 Inference Speed**: Can start after foundational prediction helpers exist; does not depend on submission CSV writing.
- **US3 Mine V1 Hard Examples**: Can start after foundational threshold/artifact helpers; does not depend on inference, submission, or benchmark implementations.

### Parallel Opportunities

- T002-T004 can run in parallel after T001.
- T011-T012 can run in parallel with foundational implementation review after T005-T010 are drafted.
- T013-T017 can be written in parallel before US1 implementation.
- T025-T027 can be written in parallel before US2 implementation.
- T033-T037 can be written in parallel before US3 implementation.
- US2 and US3 can proceed in parallel after foundational tasks and the shared prediction interfaces are stable.
- T046-T048 can run in parallel during polish.

---

## Parallel Example: User Story 1

```text
Task: "Add prediction test for loading a classifier checkpoint and returning class-1 probabilities in tests/test_inference.py"
Task: "Add prediction test for threshold-to-target conversion using saved threshold in tests/test_inference.py"
Task: "Add submission test for preserving sample submission row order and label column contract in tests/test_submission.py"
Task: "Add submission test for failing clearly when a sample image is missing in tests/test_submission.py"
Task: "Add submission test proving hard-example files are not created during submission generation in tests/test_submission.py"
```

---

## Parallel Example: User Story 3

```text
Task: "Add hard-example test for false positive and false negative CSV generation in tests/test_hard_example_mining.py"
Task: "Add hard-example test for uncertain row selection near the saved threshold in tests/test_hard_example_mining.py"
Task: "Add hard-example test for high-loss proxy ranking from probability severity in tests/test_hard_example_mining.py"
Task: "Add hard-example test for summary JSON counts, source paths, threshold reference, and artifact paths in tests/test_hard_example_mining.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational artifact, threshold, image, and error helpers.
3. Write and fail US1 tests.
4. Implement prediction and submission generation.
5. Validate US1 independently with `pytest tests/test_inference.py tests/test_submission.py -q`.

### Incremental Delivery

1. Add US1 for first V1 submission.
2. Add US2 for benchmark reporting without changing submission behavior.
3. Add US3 for offline hard-example mining without changing normal inference.
4. Run SPEC-006 quickstart tests and full project regression.

### Scope Discipline

- Do not modify SPEC-001 through SPEC-005 except for tiny compatibility fixes if implementation reveals a blocker.
- Do not implement detector, segmentation, Grad-CAM, dashboard, V2 training, feature memory bank, ensemble, distillation, or hybrid inference.
- Do not run hard-example mining from normal prediction, submission, or benchmark workflows.
