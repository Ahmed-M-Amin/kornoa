# Tasks: V5 Strong Classifier

**Input**: Design documents from `/specs/010-v5-strong-classifier/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/v5-strong-classifier-contract.md`, `quickstart.md`

**Tests**: Pytest tasks are included because SPEC-010 changes classifier configuration, training selection, prediction export schemas, submission generation, benchmark acceptance, and leakage safeguards.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add the V5 configuration and test placeholders needed by all stories.

- [X] T001 Create V5 strong classifier config skeleton in `configs/v5_strong_classifier.yaml`
- [X] T002 [P] Add V5 classifier test module placeholder and fixtures in `tests/test_classifier_v5.py`
- [X] T003 [P] Add V5 training test module placeholder and fixtures in `tests/test_training_v5.py`
- [X] T004 [P] Add V5 inference test module placeholder and fixtures in `tests/test_inference_v5.py`
- [X] T005 [P] Add V5 artifact/schema helper placeholders in `src/training/train_classifier.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared config parsing, split safety, schema validation, and output helpers required before any user story can be completed.

**Critical**: No user story work should begin until this phase is complete.

- [X] T006 Implement V5 config parsing for experiment, model, training, data, benchmark, and output fields in `src/training/train_classifier.py`
- [X] T007 Add config validation that requires primary `convnext_tiny`, image size `512`, fallback `efficientnet_b2`, and no broad model sweep in `src/training/train_classifier.py`
- [X] T008 Add V5 output directory resolution for `outputs/kaggle_v5/v5_strong_classifier/` in `src/training/train_classifier.py`
- [X] T009 Implement reusable JSON/CSV schema validation helpers for V5 prediction, threshold, metrics, and submission artifacts in `src/training/train_classifier.py`
- [X] T010 Implement V2B-compatible split source validation and clear failure when the split cannot be reconstructed in `src/training/train_classifier.py`
- [X] T011 Implement hard-example strategy validation with `analysis_only` default and explicit `oversample` opt-in in `src/training/hard_example_mining.py`
- [X] T012 Implement V5 model factory validation for `convnext_tiny` primary and `efficientnet_b2` fallback in `src/models/classifier.py`
- [X] T013 Implement image-size `512` transform acceptance while preserving ROI/internal dark defects in `src/data/transforms.py`
- [X] T014 [P] Add tests for V5 config default values and output root validation in `tests/test_classifier_v5.py`
- [X] T015 [P] Add tests for model candidate scope and fallback-only gating in `tests/test_classifier_v5.py`
- [X] T016 [P] Add tests for V2B-compatible split reuse and missing-split failure in `tests/test_training_v5.py`
- [X] T017 [P] Add tests for hard-example `analysis_only` default and oversampling opt-in in `tests/test_training_v5.py`
- [X] T018 [P] Add tests for image size `512` transform acceptance and dark-defect preservation boundaries in `tests/test_classifier_v5.py`
- [X] T019 [P] Add tests for stronger safe V5 augmentation policy in `tests/test_classifier_v5.py`
- [X] T020 Implement stronger safe V5 augmentation policy while preserving bottle defect evidence in `src/data/transforms.py`

**Checkpoint**: Foundation ready; user story implementation can now proceed.

---

## Phase 3: User Story 1 - Train Stronger Classifier Candidate (Priority: P1) MVP

**Goal**: Run a V5 classifier experiment that produces a validation-selected candidate using ConvNeXt-Tiny 512 primary behavior and EfficientNet-B2 only as resource fallback.

**Independent Test**: Run the V5 training workflow on synthetic classifier fixtures and verify the saved model identity, validation F1, selected threshold, distributions, split source, hard-example strategy, and runtime comparison are reported without test labels.

### Tests for User Story 1

- [X] T021 [P] [US1] Add test that V5 training selects ConvNeXt-Tiny 512 as the primary candidate in `tests/test_training_v5.py`
- [X] T022 [P] [US1] Add test that EfficientNet-B2 fallback is evaluated only after primary resource or speed rejection in `tests/test_training_v5.py`
- [X] T023 [P] [US1] Add test that validation threshold search uses validation rows only in `tests/test_training_v5.py`
- [X] T024 [P] [US1] Add test that threshold report contains selected threshold, validation F1, confusion counts, split source, and validation-only selection source in `tests/test_training_v5.py`
- [X] T025 [P] [US1] Add test that classifier metrics include target distribution, prediction distribution, submission row count, V2B benchmark reference, speed ratio, hard-example strategy, and public-score decision fields in `tests/test_training_v5.py`
- [X] T026 [P] [US1] Add test that public score is rejected as a training or threshold input in `tests/test_training_v5.py`
- [X] T027 [P] [US1] Add test that explicit hard-example oversampling records split-safety report fields in `tests/test_training_v5.py`

### Implementation for User Story 1

- [X] T028 [US1] Extend V5 training setup to instantiate ConvNeXt-Tiny 512 primary candidate from `configs/v5_strong_classifier.yaml` in `src/training/train_classifier.py`
- [X] T029 [US1] Implement EfficientNet-B2 fallback path only for resource-limit or speed/memory rejection cases in `src/training/train_classifier.py`
- [X] T030 [US1] Wire weighted BCE or focal loss and class-balanced sampling choices for V5 in `src/training/train_classifier.py`
- [X] T031 [US1] Ensure hard-example reports remain analysis-only by default and oversampling requires explicit opt-in in `src/training/train_classifier.py`
- [X] T032 [US1] Add hard-example oversampling split-safety report fields when oversampling is explicitly enabled in `src/training/train_classifier.py`
- [X] T033 [US1] Reuse V2B-compatible train/validation split during V5 training and threshold search in `src/training/train_classifier.py`
- [X] T034 [US1] Save best V5 model artifact to `outputs/kaggle_v5/v5_strong_classifier/models/classifier_best.pth` from `src/training/train_classifier.py`
- [X] T035 [US1] Save V5 threshold report to `outputs/kaggle_v5/v5_strong_classifier/reports/best_threshold.json` from `src/training/train_classifier.py`
- [X] T036 [US1] Save V5 classifier metrics report with submission row count to `outputs/kaggle_v5/v5_strong_classifier/reports/classifier_metrics.json` from `src/training/train_classifier.py`
- [X] T037 [US1] Add V2B benchmark comparison and 2x speed-ceiling decision fields in `src/inference/benchmark.py`
- [X] T038 [US1] Run `python -m pytest tests/test_classifier_v5.py tests/test_training_v5.py -q` and fix US1 regressions

**Checkpoint**: User Story 1 is independently functional and testable as the MVP.

---

## Phase 4: User Story 2 - Export Validation and Test Probabilities (Priority: P2)

**Goal**: Export validation and test classifier probabilities in stable V5 schemas for evaluation, submission generation, and later hybrid consumption.

**Independent Test**: Generate synthetic validation and test prediction files and verify required columns, row uniqueness, probability bounds, binary target values, no test labels, and V2B-compatible validation split membership.

### Tests for User Story 2

- [X] T039 [P] [US2] Add test that validation prediction export requires `image_id,true_label,prob_bad,classifier_prediction,target` in `tests/test_inference_v5.py`
- [X] T040 [P] [US2] Add test that test prediction export requires `image_id,prob_bad,classifier_prediction,target` and forbids label columns in `tests/test_inference_v5.py`
- [X] T041 [P] [US2] Add test that probability exports reject duplicate image IDs and non-binary targets in `tests/test_inference_v5.py`
- [X] T042 [P] [US2] Add test that probability exports reject `prob_bad` values outside `[0, 1]` in `tests/test_inference_v5.py`
- [X] T043 [P] [US2] Add test that validation prediction rows are restricted to the V2B-compatible validation split in `tests/test_inference_v5.py`

### Implementation for User Story 2

- [X] T044 [US2] Implement validation prediction export with required core columns in `src/training/train_classifier.py`
- [X] T045 [US2] Implement validation prediction schema validation for labels, probabilities, binary targets, duplicate IDs, and split membership in `src/training/train_classifier.py`
- [X] T046 [US2] Extend classifier inference submission flow to save V5 test probabilities in `src/inference/submission.py`
- [X] T047 [US2] Implement test prediction schema validation for columns, probability bounds, binary targets, duplicate IDs, and no label columns in `src/inference/submission.py`
- [X] T048 [US2] Ensure V5 test probability exports remain compatible with V4 hybrid probability consumption in `src/inference/submission.py`
- [X] T049 [US2] Run `python -m pytest tests/test_inference_v5.py -q` and fix US2 regressions

**Checkpoint**: User Stories 1 and 2 both work independently.

---

## Phase 5: User Story 3 - Generate Review-Ready Submission Without Leakage (Priority: P3)

**Goal**: Generate strict V5 submission output and review-ready reports without automatic Kaggle submission or leakage from test labels/public score feedback.

**Independent Test**: Run V5 submission export from synthetic V5 artifacts and verify strict `image_id,target` output, saved test probabilities, metrics review fields, no automatic submission, and no generated artifacts staged for commit.

### Tests for User Story 3

- [X] T050 [P] [US3] Add test that V5 submission contains exactly `image_id,target` in `tests/test_inference_v5.py`
- [X] T051 [P] [US3] Add test that V5 submission uses saved V5 threshold and test probabilities in `tests/test_inference_v5.py`
- [X] T052 [P] [US3] Add test that V5 submission export rejects sample-solution labels, test labels, public-score inputs, and manual test-inspection fields in `tests/test_inference_v5.py`
- [X] T053 [P] [US3] Add test that generated V5 metrics include public-score comparison baseline `0.92181`, V3 detector-only score `0.74169`, submission row count, and analysis-only fallback status in `tests/test_inference_v5.py`
- [X] T054 [P] [US3] Add test that generated V5 output paths are ignored and not staged by default in `tests/test_inference_v5.py`
- [X] T055 [P] [US3] Add scope-guard test proving V5 workflows do not retrain V3 detector or modify V4.2 fusion behavior in `tests/test_inference_v5.py`

### Implementation for User Story 3

- [X] T056 [US3] Implement V5 strict submission generation to `outputs/kaggle_v5/v5_strong_classifier/submissions/submission_v5.csv` in `src/inference/submission.py`
- [X] T057 [US3] Add V5 leakage guards for test labels, sample-solution labels, public-score-derived thresholds, and automatic submission in `src/inference/submission.py`
- [X] T058 [US3] Add V5 acceptance review fields for V4.2 public baseline, V3 detector-only weak baseline, submission row count, public-score decision, and analysis-only status in `src/training/train_classifier.py`
- [X] T059 [US3] Add generated-output ignore validation for `outputs/kaggle_v5/v5_strong_classifier/` in `tests/test_inference_v5.py`
- [X] T060 [US3] Run `python -m pytest tests/test_inference_v5.py tests/test_submission.py -q` and fix US3 regressions

**Checkpoint**: All user stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate full SPEC-010 behavior, preserve existing V2/V4 behavior, and update supporting documentation.

- [X] T061 [P] Update `specs/010-v5-strong-classifier/quickstart.md` if implemented commands or outputs differ from the planned contract
- [X] T062 [P] Update `docs/file-path-index.md` by running `python scripts/update_file_path_index.py`
- [X] T063 [P] Add V5 closeout placeholder or result note section if V5 public score is available in `docs/v5_strong_classifier_closeout.md`
- [X] T064 Run `python -m pytest tests/test_classifier_v5.py tests/test_training_v5.py tests/test_inference_v5.py -q` and fix focused V5 regressions
- [X] T065 Run `python -m pytest tests/test_classifier_v2.py tests/test_training_v2.py tests/test_inference_v2.py tests/test_v2_comparison.py -q` and fix V2 regressions
- [X] T066 Run `python -m pytest tests/test_hybrid_fusion.py tests/test_hybrid_submission.py -q` and fix V4 probability-consumption regressions
- [X] T067 Run `python -m pytest tests -q` and fix full-suite regressions
- [X] T068 Run a placeholder scan for unresolved clarification text, template markers, and unfinished notes in `specs/010-v5-strong-classifier/` and `configs/v5_strong_classifier.yaml`
- [X] T069 Review `git diff --stat` and verify no generated outputs, private datasets, model weights, predictions, reports, benchmarks, or submissions are tracked

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational; MVP.
- **User Story 2 (Phase 4)**: Depends on Foundational and can use synthetic artifacts, but real end-to-end export depends on US1 training outputs.
- **User Story 3 (Phase 5)**: Depends on Foundational and can use synthetic V5 artifacts, but real end-to-end submission depends on US1 and US2 outputs.
- **Polish (Phase 6)**: Depends on selected user stories being complete.

### User Story Dependencies

- **US1 Train Stronger Classifier Candidate**: MVP; no dependency on US2 or US3 after foundation.
- **US2 Export Validation and Test Probabilities**: Independently testable with synthetic artifacts; integrates with US1 outputs for real runs.
- **US3 Generate Review-Ready Submission Without Leakage**: Independently testable with synthetic artifacts; integrates with US1/US2 outputs for real submission generation.

### Parallel Opportunities

- T002-T005 can run in parallel after T001.
- T014-T019 can run in parallel after T006-T013 are stubbed.
- T021-T027 can run in parallel for US1 test coverage.
- T039-T043 can run in parallel for US2 test coverage.
- T050-T055 can run in parallel for US3 test coverage.
- T061-T063 can run in parallel during polish.

---

## Parallel Example: User Story 1

```text
Task: "T021 [P] [US1] Add test that V5 training selects ConvNeXt-Tiny 512 as the primary candidate in tests/test_training_v5.py"
Task: "T024 [P] [US1] Add test that threshold report contains selected threshold, validation F1, confusion counts, split source, and validation-only selection source in tests/test_training_v5.py"
Task: "T027 [P] [US1] Add test that explicit hard-example oversampling records split-safety report fields in tests/test_training_v5.py"
```

## Parallel Example: User Story 2

```text
Task: "T039 [P] [US2] Add test that validation prediction export requires image_id,true_label,prob_bad,classifier_prediction,target in tests/test_inference_v5.py"
Task: "T040 [P] [US2] Add test that test prediction export requires image_id,prob_bad,classifier_prediction,target and forbids label columns in tests/test_inference_v5.py"
Task: "T042 [P] [US2] Add test that probability exports reject prob_bad values outside [0, 1] in tests/test_inference_v5.py"
```

## Parallel Example: User Story 3

```text
Task: "T050 [P] [US3] Add test that V5 submission contains exactly image_id,target in tests/test_inference_v5.py"
Task: "T052 [P] [US3] Add test that V5 submission export rejects sample-solution labels, test labels, public-score inputs, and manual test-inspection fields in tests/test_inference_v5.py"
Task: "T055 [P] [US3] Add scope-guard test proving V5 workflows do not retrain V3 detector or modify V4.2 fusion behavior in tests/test_inference_v5.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational V5 config, model scope, split, schema, and hard-example guards.
3. Complete Phase 3 US1 training candidate workflow.
4. Run the US1 focused pytest command and inspect threshold/metrics artifacts.
5. Stop and validate that ConvNeXt-Tiny 512 is primary, EfficientNet-B2 is fallback-only, and no test/public-score leakage exists.

### Incremental Delivery

1. Complete Setup + Foundation.
2. Deliver US1 so V5 can train and produce validation-selected candidate artifacts.
3. Add US2 so validation/test probability exports are schema-stable.
4. Add US3 so strict submission and review-ready acceptance reporting are reproducible.
5. Run polish regression checks to confirm V2B classifier and V4 hybrid behavior remain unchanged.

### TDD Guidance

- Write the tests in each user-story phase before implementing the corresponding behavior.
- Confirm each new test fails for the expected missing behavior before adding implementation.
- Re-run the story-specific pytest command at each checkpoint.

## Notes

- `[P]` tasks are parallelizable because they touch different files or independent test cases.
- `[US1]`, `[US2]`, and `[US3]` labels map directly to the prioritized user stories in `spec.md`.
- No task should retrain the V3 detector or change V4.2 fusion logic.
- No task should read test labels, sample-solution labels, manually inspected test labels, or public-score-derived thresholds.
- Generated artifacts belong under ignored `outputs/kaggle_v5/v5_strong_classifier/` paths.
