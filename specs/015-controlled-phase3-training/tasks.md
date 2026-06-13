# Tasks: Controlled Phase 3 Hard-Example Training

**Input**: Design documents from `/specs/015-controlled-phase3-training/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Include pytest coverage for baseline locking, governed Phase 3 eligibility enforcement, candidate artifact validation, runtime evidence, submission exports, and comparison/report contracts.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the controlled Phase 3 config surface, documentation, and test entrypoints.

- [ ] T001 Create the controlled Phase 3 configuration scaffold in `configs/phase3_controlled_training.yaml`
- [ ] T002 Update the Spec 015 execution flow, inputs, outputs, and verification steps in `specs/015-controlled-phase3-training/quickstart.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared comparison, artifact, and safety infrastructure that MUST be complete before any user story work begins.

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T003 Extend config parsing and validation for controlled Phase 3 inputs, outputs, and safety rules in `src/training/train_classifier.py`
- [ ] T004 Add shared baseline-lock, candidate-governance, and artifact-package validation helpers in `src/training/train_classifier.py`
- [ ] T005 [P] Add reusable synthetic fixtures for locked baselines, governed Phase 3 candidate packages, and controlled candidate outputs in `tests/test_phase3_controlled_training.py`
- [ ] T006 [P] Add regression coverage for no-test-label, no-baseline-overwrite, and blocked-row exclusion rules in `tests/test_phase3_controlled_training.py`

**Checkpoint**: Foundation ready; controlled Phase 3 work can proceed story by story on a stable config and safety surface.

---

## Phase 3: User Story 1 - Lock Training Baseline And Candidate Inputs (Priority: P1) MVP

**Goal**: Create one locked comparison anchor and one governed Phase 3 eligibility set derived only from approved Spec 014 outputs.

**Independent Test**: Run the baseline-lock and governed-input preparation flow and confirm that only `phase3_use_allowed = true` rows enter the training set, blocked rows are excluded, and all baseline references resolve to one stable anchor.

### Tests for User Story 1

- [ ] T007 [P] [US1] Add locked-baseline report contract tests in `tests/test_phase3_controlled_training.py`
- [ ] T008 [P] [US1] Add governed Phase 3 eligibility and blocked-row exclusion tests in `tests/test_phase3_controlled_training.py`

### Implementation for User Story 1

- [ ] T009 [US1] Implement locked baseline report loading, validation, and serialization in `src/training/train_classifier.py`
- [ ] T010 [US1] Implement governed Phase 3 training input resolution from approved candidate-package rows in `src/training/train_classifier.py`
- [ ] T011 [US1] Implement row-traceability, single-baseline-anchor, and blocked-row rejection checks in `src/training/train_classifier.py`
- [ ] T012 [US1] Implement baseline-lock and governed-input reporting outputs under `outputs/analysis/phase3_controlled_training/` in `src/training/train_classifier.py`

**Checkpoint**: User Story 1 should now create a valid baseline anchor and governed training input set that can be tested independently.

---

## Phase 4: User Story 2 - Run Controlled Phase 3 Candidate Training (Priority: P2)

**Goal**: Train and evaluate serious Phase 3 candidates under one controlled workflow with complete artifacts, threshold evidence, and governed hard-example comparisons.

**Independent Test**: Run one controlled candidate through training and evaluation, then verify the output package includes full validation metrics, hard-example-only metrics, excluded-hard-example metrics, threshold evidence, target distribution, changed-row evidence, and required saved artifacts.

### Tests for User Story 2

- [ ] T013 [P] [US2] Add controlled candidate artifact-package and threshold-selection tests in `tests/test_phase3_controlled_training.py`
- [ ] T014 [P] [US2] Add changed-row comparison, hard-example acceptance-gate, and ablation-status tests in `tests/test_phase3_controlled_training.py`
- [ ] T015 [P] [US2] Extend runtime benchmark and submission contract tests for controlled Phase 3 artifact roots in `tests/test_benchmark.py`
- [ ] T016 [P] [US2] Extend controlled submission export and metrics update tests in `tests/test_submission.py`

### Implementation for User Story 2

- [ ] T017 [US2] Implement candidate hypothesis capture, governed candidate orchestration, and artifact-root routing in `src/training/train_classifier.py`
- [ ] T018 [US2] Implement full validation, hard-example-only, excluded-hard-example, and changed-row comparison reporting in `src/training/train_classifier.py`
- [ ] T019 [US2] Implement controlled Phase 3 threshold-stability, target-distribution, and artifact-completeness outputs in `src/training/train_classifier.py`
- [ ] T020 [US2] Extend repeated runtime evidence support and controlled benchmark metadata in `src/inference/benchmark.py`
- [ ] T021 [US2] Extend submission-aligned export and controlled artifact metrics updates in `src/inference/submission.py`

**Checkpoint**: User Stories 1 and 2 should both work independently, with one serious candidate producing a complete controlled Phase 3 output package.

---

## Phase 5: User Story 3 - Select Competition-Ready Finalists With Evidence (Priority: P3)

**Goal**: Produce one rolling comparison table, runtime evidence pack, and final insight/recommendation package for competition-aware candidate selection.

**Independent Test**: Generate the comparison and final-decision outputs, then confirm that every serious candidate appears once with a decision, runtime evidence, artifact completeness, and final recommendation context.

### Tests for User Story 3

- [ ] T022 [P] [US3] Add comparison-table, finalist-readiness, and decision-reason contract tests in `tests/test_phase3_controlled_training.py`
- [ ] T023 [P] [US3] Add final insight evidence pack and runtime-pack aggregation tests in `tests/test_phase3_controlled_training.py`

### Implementation for User Story 3

- [ ] T024 [US3] Implement the rolling candidate comparison table and per-candidate decision status outputs in `src/training/train_classifier.py`
- [ ] T025 [US3] Implement repeated runtime evidence aggregation and finalist-readiness gating in `src/training/train_classifier.py`
- [ ] T026 [US3] Implement the final insight evidence pack and competition-aware recommendation summary in `src/training/train_classifier.py`

**Checkpoint**: All user stories should now be independently functional, with candidate selection evidence ready for review.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final consistency, documentation, and verification across the full Spec 015 workflow.

- [ ] T027 [P] Update the stable workflow and output guarantees in `specs/015-controlled-phase3-training/contracts/controlled-phase3-training-contract.md`
- [ ] T028 [P] Align Spec 015 entities, validation rules, and evidence-pack semantics in `specs/015-controlled-phase3-training/data-model.md`
- [ ] T029 [P] Refresh the active path index so Spec 015 artifacts remain discoverable in `docs/file-path-index.md`
- [ ] T030 Run `python -m pytest tests/test_phase3_controlled_training.py tests/test_training_v2.py tests/test_training_v5.py tests/test_benchmark.py tests/test_submission.py -q` and `python -m py_compile src/training/train_classifier.py src/inference/benchmark.py src/inference/submission.py src/data/hard_examples.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion; delivers the MVP
- **User Story 2 (Phase 4)**: Depends on User Story 1 outputs because controlled candidate training requires the locked baseline and governed training set
- **User Story 3 (Phase 5)**: Depends on User Story 2 outputs because comparison and final recommendation require candidate artifact packages and runtime evidence
- **Polish (Phase 6)**: Depends on the desired user stories being complete

### User Story Dependencies

- **US1**: Starts after Foundational and has no dependency on later stories
- **US2**: Requires the locked baseline report and governed Phase 3 training inputs produced by US1
- **US3**: Requires the candidate evaluation packages produced by US2

### Within Each User Story

- Tests should be written before implementation tasks in the same story
- Shared config, safety, and artifact helpers land before story-specific orchestration
- Baseline locking and governed input construction precede candidate training
- Candidate training precedes runtime aggregation and final comparison

### Parallel Opportunities

- `T005` and `T006` can proceed in parallel because they cover shared fixtures and safety regressions in `tests/test_phase3_controlled_training.py`
- `T007` and `T008` can be written in parallel because they cover different US1 behaviors
- `T013`, `T014`, `T015`, and `T016` can proceed in parallel because they target distinct controlled-candidate and export surfaces
- `T022` and `T023` can be written in parallel because they validate different US3 reporting outputs
- `T027`, `T028`, and `T029` can be completed in parallel during polish

---

## Parallel Example: User Story 1

```bash
# Launch US1 test work together:
Task: "Add locked-baseline report contract tests in tests/test_phase3_controlled_training.py"
Task: "Add governed Phase 3 eligibility and blocked-row exclusion tests in tests/test_phase3_controlled_training.py"
```

---

## Parallel Example: User Story 2

```bash
# Launch US2 test work together:
Task: "Add controlled candidate artifact-package and threshold-selection tests in tests/test_phase3_controlled_training.py"
Task: "Add changed-row comparison, hard-example acceptance-gate, and ablation-status tests in tests/test_phase3_controlled_training.py"
Task: "Extend runtime benchmark and submission contract tests for controlled Phase 3 artifact roots in tests/test_benchmark.py"
Task: "Extend controlled submission export and metrics update tests in tests/test_submission.py"
```

---

## Parallel Example: User Story 3

```bash
# Launch US3 test work together:
Task: "Add comparison-table, finalist-readiness, and decision-reason contract tests in tests/test_phase3_controlled_training.py"
Task: "Add final insight evidence pack and runtime-pack aggregation tests in tests/test_phase3_controlled_training.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Validate the locked baseline and governed Phase 3 eligibility set before moving to candidate training

### Incremental Delivery

1. Finish Setup + Foundational to stabilize config, safety, and artifact validation
2. Deliver US1 to lock the comparison anchor and governed training inputs
3. Deliver US2 to produce complete controlled candidate packages
4. Deliver US3 to produce comparison, runtime, and insight evidence for final selection
5. Finish Polish to align docs and run verification commands

### Suggested MVP Scope

1. Phase 1: Setup
2. Phase 2: Foundational
3. Phase 3: User Story 1

---

## Notes

- Every task follows the required checklist format with a task ID and exact file path
- Governed Phase 3 training eligibility must remain downstream of Spec 014 `phase3_use_allowed = true` outputs
- Hard-example improvement and runtime evidence are separate acceptance gates
- The workflow must preserve no test-label use, no public-score-only threshold tuning, and no overwriting of locked baseline artifacts
