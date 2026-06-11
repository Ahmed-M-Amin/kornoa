# Tasks: Locked Same-Split Error Intelligence

**Input**: Design documents from `/specs/012-locked-same-split-intelligence/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/locked-same-split-contract.md, quickstart.md

**Tests**: Required. This feature is an analysis/reporting quality gate and must include pytest coverage for locked-row alignment, section metrics, rolling comparison-table updates, candidate decision status, and leakage safeguards.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the feature-specific analysis output and test scaffolding for locked same-split intelligence.

- [X] T001 Add same-split analysis config keys for locked-row sources, rolling comparison output, and candidate identity handling in `configs/v2_2_hard_examples.yaml`
- [X] T002 Create or update the locked same-split analysis module entrypoint and constants in `src/analysis/v2_2_same_split_eval.py`
- [X] T003 Create the locked same-split test module scaffolding with synthetic fixture helpers in `tests/test_v2_2_same_split_eval.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared safety, normalization, sectioning, and decision helpers that all user stories depend on.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 [P] Add failing safety tests that reject test labels, sample-submission labels, training actions, submission creation, and unsafe analysis-side-effect paths in `tests/test_v2_2_same_split_eval.py`
- [X] T005 [P] Add failing alignment tests for binary locked-row validation, duplicate `image_id` rejection, normalized `image_id` joins, and row-count mismatch handling in `tests/test_v2_2_same_split_eval.py`
- [X] T006 [P] Add failing candidate-selection tests for one-candidate-per-run enforcement, required candidate provenance fields, and V2.2-as-first-example defaults in `tests/test_v2_2_same_split_eval.py`
- [X] T007 Implement shared analysis config parsing, path resolution, run-manifest capture, and safety validation in `src/analysis/v2_2_same_split_eval.py`
- [X] T008 Implement locked V2B source loading, candidate source loading, `image_id` normalization, and deterministic same-row alignment helpers in `src/analysis/v2_2_same_split_eval.py`
- [X] T009 Implement shared hard-example registry, section membership helpers, and comparison-bucket enumeration used by all downstream metrics and reports in `src/analysis/v2_2_same_split_eval.py`
- [X] T010 Implement candidate gate evaluation and decision-payload helpers for `accepted`, `rejected`, and `manual_review`, including the `0.01` hard-example regression tolerance and explicit decision reasons, in `src/analysis/v2_2_same_split_eval.py`

**Checkpoint**: Foundation ready; user story implementation can now begin.

---

## Phase 3: User Story 1 - Compare Locked Validation Behavior (Priority: P1)

**Goal**: Evaluate one selected V2.x candidate against the original locked V2B validation rows and prove both models are scored on the same row set.

**Independent Test**: Run the same-split evaluator on synthetic V2B and candidate predictions, then verify both models report metrics on the exact same locked `image_id` set with matching row counts and a valid summary.

### Tests for User Story 1

- [X] T011 [P] [US1] Add a failing test that verifies V2B and the selected candidate are evaluated on exactly the same locked validation rows in `tests/test_v2_2_same_split_eval.py`
- [X] T012 [P] [US1] Add a failing test that verifies full locked-row metrics include F1, precision, recall, TP, FP, TN, FN, prediction distribution, and target distribution for both models in `tests/test_v2_2_same_split_eval.py`
- [X] T013 [P] [US1] Add a failing test that verifies the summary JSON records candidate provenance, safety flags, and one decision status in `tests/test_v2_2_same_split_eval.py`

### Implementation for User Story 1

- [X] T014 [US1] Implement locked same-split metric computation for V2B and the selected candidate in `src/analysis/v2_2_same_split_eval.py`
- [X] T015 [US1] Implement summary JSON generation with candidate provenance, safety flags, and locked-row section output in `src/analysis/v2_2_same_split_eval.py`
- [X] T016 [US1] Implement CLI run orchestration and expected-analysis-only error handling in `src/analysis/v2_2_same_split_eval.py`

**Checkpoint**: User Story 1 is functional and independently testable.

---

## Phase 4: User Story 2 - Expose Hard-Example Impact (Priority: P2)

**Goal**: Show whether candidate gains come from true hard-row improvement or only from easier rows by splitting the locked evaluation into the three required sections.

**Independent Test**: Run the evaluator with synthetic hard-negative, hard-positive, and uncertain files and verify that section row counts, hard-example types, and section metrics are correct.

### Tests for User Story 2

- [X] T017 [P] [US2] Add a failing test that verifies the evaluator produces `all_original_v2b_validation_rows`, `hard_example_rows_only`, and `original_v2b_validation_excluding_hard_examples` in `tests/test_v2_2_same_split_eval.py`
- [X] T018 [P] [US2] Add a failing test that verifies rows are tagged with the correct hard-example type, including overlapping hard-example sources, in `tests/test_v2_2_same_split_eval.py`
- [X] T019 [P] [US2] Add a failing test that verifies `accepted` requires full locked-row improvement plus no hard-example-only F1 regression beyond `0.01` in `tests/test_v2_2_same_split_eval.py`
- [X] T020 [P] [US2] Add a failing test that verifies `rejected` requires non-improved full locked-row F1 plus hard-example-only F1 regression beyond `0.01`, and that mixed outcomes become `manual_review`, in `tests/test_v2_2_same_split_eval.py`

### Implementation for User Story 2

- [X] T021 [US2] Implement sectioned same-split metric generation for all rows, hard-example-only rows, and excluding-hard-example rows in `src/analysis/v2_2_same_split_eval.py`
- [X] T022 [US2] Implement hard-example type preservation and overlap-safe tagging in `src/analysis/v2_2_same_split_eval.py`
- [X] T023 [US2] Implement explicit `accepted`, `rejected`, and `manual_review` decision rule application across section metrics in `src/analysis/v2_2_same_split_eval.py`

**Checkpoint**: User Stories 1 and 2 both work independently and expose the hard-row bottleneck clearly.

---

## Phase 5: User Story 3 - Produce Review-Ready Error Intelligence (Priority: P3)

**Goal**: Emit row-level comparison buckets and maintain a rolling comparison table across evaluated candidates for later failure-mode review and candidate selection.

**Independent Test**: Run the evaluator repeatedly with synthetic candidate identities and verify row-level buckets are exhaustive and the rolling comparison table updates consistently across runs.

### Tests for User Story 3

- [X] T024 [P] [US3] Add a failing test that verifies every evaluated row lands in exactly one of the four comparison buckets in `tests/test_v2_2_same_split_eval.py`
- [X] T025 [P] [US3] Add a failing test that verifies row-level error outputs contain the required comparison columns in `tests/test_v2_2_same_split_eval.py`
- [X] T026 [P] [US3] Add a failing test that verifies the rolling candidate comparison table adds or updates one row per candidate with decision status, decision reason, and section metrics in `tests/test_v2_2_same_split_eval.py`
- [X] T027 [P] [US3] Add a failing test that verifies optional detector and image-quality evidence can remain absent without invalidating the comparison contract in `tests/test_v2_2_same_split_eval.py`

### Implementation for User Story 3

- [X] T028 [US3] Implement row-level comparison bucket generation and CSV writing in `src/analysis/v2_2_same_split_eval.py`
- [X] T029 [US3] Implement rolling candidate comparison table persistence with decision reason fields in `src/analysis/v2_2_same_split_eval.py`
- [X] T030 [US3] Implement optional-evidence-compatible output handling without changing the locked same-split contract in `src/analysis/v2_2_same_split_eval.py`

**Checkpoint**: All user stories are independently functional and the feature produces review-ready locked same-split intelligence.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, documentation alignment, and full-suite verification.

- [X] T031 [P] Add quickstart and output-path references for the rolling comparison table to `docs/file-path-index.md` by running `python scripts/update_file_path_index.py`
- [X] T032 Validate the quickstart command and expected output paths in `specs/012-locked-same-split-intelligence/quickstart.md`
- [X] T033 Run `python -m py_compile src/analysis/v2_2_same_split_eval.py`
- [X] T034 Run `python -m pytest tests/test_v2_2_same_split_eval.py -q`
- [X] T035 Run `python -m pytest tests -q`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational completion; this is the MVP.
- **User Story 2 (Phase 4)**: Depends on the shared locked-row loading and section helpers from Foundational and on US1 metrics scaffolding.
- **User Story 3 (Phase 5)**: Depends on locked-row metrics and decision helpers from US1 and US2.
- **Polish (Phase 6)**: Depends on selected user stories being complete.

### User Story Dependencies

- **US1 Compare Locked Validation Behavior**: No dependency on other stories after Foundation.
- **US2 Expose Hard-Example Impact**: Requires locked-row comparison from US1.
- **US3 Produce Review-Ready Error Intelligence**: Requires US1 and US2 outputs because the rolling comparison table depends on section metrics and decision status.

### Within Each User Story

- Write story tests before implementation.
- Implement shared calculations before report writing.
- Implement report writing before CLI or persistence integration.
- Verify each checkpoint before moving to the next priority.

### Parallel Opportunities

- T004, T005, and T006 can be written in parallel once the test module exists.
- T011, T012, and T013 are independent US1 tests.
- T017, T018, T019, and T020 are independent US2 tests.
- T024, T025, T026, and T027 are independent US3 tests.
- T031 and T032 can run in parallel during polish.

---

## Parallel Example: User Story 1

```bash
Task: "Add a failing test that verifies V2B and the selected candidate are evaluated on exactly the same locked validation rows in tests/test_v2_2_same_split_eval.py"
Task: "Add a failing test that verifies full locked-row metrics include F1, precision, recall, TP, FP, TN, FN, prediction distribution, and target distribution for both models in tests/test_v2_2_same_split_eval.py"
Task: "Add a failing test that verifies the summary JSON records candidate provenance, safety flags, and one decision status in tests/test_v2_2_same_split_eval.py"
```

## Parallel Example: User Story 2

```bash
Task: "Add a failing test that verifies the evaluator produces all three required sections in tests/test_v2_2_same_split_eval.py"
Task: "Add a failing test that verifies rows are tagged with the correct hard-example type, including overlapping hard-example sources, in tests/test_v2_2_same_split_eval.py"
Task: "Add a failing test that verifies accepted requires full locked-row improvement plus no hard-example-only F1 regression beyond 0.01 in tests/test_v2_2_same_split_eval.py"
Task: "Add a failing test that verifies rejected requires non-improved full locked-row F1 plus hard-example-only regression beyond 0.01, and mixed outcomes become manual_review, in tests/test_v2_2_same_split_eval.py"
```

## Parallel Example: User Story 3

```bash
Task: "Add a failing test that verifies every evaluated row lands in exactly one of the four comparison buckets in tests/test_v2_2_same_split_eval.py"
Task: "Add a failing test that verifies row-level error outputs contain the required comparison columns in tests/test_v2_2_same_split_eval.py"
Task: "Add a failing test that verifies the rolling candidate comparison table adds or updates one row per candidate with decision status, decision reason, and section metrics in tests/test_v2_2_same_split_eval.py"
Task: "Add a failing test that verifies optional detector and image-quality evidence can remain absent without invalidating the comparison contract in tests/test_v2_2_same_split_eval.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational helpers.
3. Complete Phase 3 US1 locked same-row comparison.
4. Stop and validate US1 with `python -m pytest tests/test_v2_2_same_split_eval.py -q`.

### Incremental Delivery

1. US1 delivers the valid locked-row comparison and summary.
2. US2 adds hard-example sectioning and the explicit accepted/rejected/manual-review rules.
3. US3 adds row-level review outputs and the rolling candidate comparison table.
4. Polish adds quickstart alignment, file index refresh, and full-suite verification.

### Safety Rules

- Do not train.
- Do not generate a Kaggle submission.
- Do not use test labels or sample-submission labels.
- Do not tune thresholds on public leaderboard feedback.
- Do not compare multiple candidates in one evaluation run.
- Do not let optional evidence absence block the core locked same-split comparison.
