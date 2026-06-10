# Tasks: V2.1 Error Intelligence

**Input**: Design documents from `/specs/011-v2-1-error-intelligence/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/error-intelligence-contract.md, quickstart.md

**Tests**: Required. This feature is an audit/reporting quality gate and must include synthetic pytest coverage for baseline matching, review-group bands, optional evidence joins, and leakage safeguards.

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files or independent sections
- **[Story]**: Maps the task to a spec user story, such as `[US1]`
- Every task includes an exact repository path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the config and module/test placeholders for the V2.1 offline audit lane.

- [X] T001 Add V2.1 audit config skeleton with required baseline, evidence, analysis, and output keys in `configs/v2_1_error_intelligence.yaml`
- [X] T002 Create the V2.1 analysis module with constants, error class, CLI stub, and no training/submission behavior in `src/analysis/v2_1_error_intelligence.py`
- [X] T003 Create the V2.1 test module with synthetic fixture helpers in `tests/test_v2_1_error_intelligence.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared parsing, validation, metrics, and safety helpers that all user stories depend on.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 [P] Add config loading tests for required keys, default fixed bands, positive class `1`, and forbidden training/submission flags in `tests/test_v2_1_error_intelligence.py`
- [X] T005 [P] Add leakage-safety tests rejecting test labels, sample-solution labels, public leaderboard fields, and non-validation label paths in `tests/test_v2_1_error_intelligence.py`
- [X] T006 Implement config dataclasses or normalized config dictionaries and required-key validation in `src/analysis/v2_1_error_intelligence.py`
- [X] T007 Implement path safety checks for missing required baseline inputs and forbidden leakage inputs in `src/analysis/v2_1_error_intelligence.py`
- [X] T008 Implement shared image-id normalization, binary label validation, threshold loading, and metric extraction helpers in `src/analysis/v2_1_error_intelligence.py`
- [X] T009 Implement binary F1, precision, recall, and TP/FP/FN/TN computation helpers using positive class `1` in `src/analysis/v2_1_error_intelligence.py`

**Checkpoint**: Foundation ready; user story implementation can now begin.

---

## Phase 3: User Story 1 - Build V2B Error Audit (Priority: P1) MVP

**Goal**: Produce one validated V2B audit row per validation image and prove computed metrics match the locked original V2B baseline.

**Independent Test**: Run the US1 tests with synthetic predictions, labels, threshold, and baseline metrics; verify the audit CSV has one row per validation image and the summary marks `baseline_match` true.

### Tests for User Story 1

- [X] T010 [P] [US1] Add a failing test that builds a synthetic V2B audit and verifies required audit columns, one row per image, error types, and threshold distance in `tests/test_v2_1_error_intelligence.py`
- [X] T011 [P] [US1] Add a failing test that verifies computed TP/FP/FN/TN, precision, recall, and F1 match the locked baseline metrics in `tests/test_v2_1_error_intelligence.py`
- [X] T012 [P] [US1] Add a failing test that metric mismatch raises a clear V2.1 audit error before reports are accepted in `tests/test_v2_1_error_intelligence.py`

### Implementation for User Story 1

- [X] T013 [US1] Implement validation label and V2B prediction loading with strict `image_id`, probability, and binary target normalization in `src/analysis/v2_1_error_intelligence.py`
- [X] T014 [US1] Implement audit-row construction with `true_label`, `v2b_probability`, `v2b_prediction`, `error_type`, `threshold`, and `threshold_distance` in `src/analysis/v2_1_error_intelligence.py`
- [X] T015 [US1] Implement baseline metric recomputation and locked-baseline comparison with a configurable rounding tolerance in `src/analysis/v2_1_error_intelligence.py`
- [X] T016 [US1] Implement writing `audit/v2b_validation_error_audit.csv` and `reports/v2b_error_intelligence_summary.json` in `src/analysis/v2_1_error_intelligence.py`
- [X] T017 [US1] Implement `run_error_intelligence(config_path)` orchestration and CLI exit behavior without tracebacks for expected input errors in `src/analysis/v2_1_error_intelligence.py`

**Checkpoint**: User Story 1 is functional as the MVP and independently testable.

---

## Phase 4: User Story 2 - Review High-Risk Error Groups (Priority: P2)

**Goal**: Split the validated audit into fixed-band review groups for false positives, false negatives, and over-rejected reusable bottles.

**Independent Test**: Run the US2 tests against the synthetic audit and verify every review-group row satisfies the fixed `0.05` near-threshold or `0.30` high-confidence rule.

### Tests for User Story 2

- [X] T018 [P] [US2] Add a failing test for near-threshold false-positive and false-negative CSVs using `abs(probability - threshold) <= 0.05` in `tests/test_v2_1_error_intelligence.py`
- [X] T019 [P] [US2] Add a failing test for high-confidence false-positive and false-negative CSVs using `abs(probability - threshold) >= 0.30` in `tests/test_v2_1_error_intelligence.py`
- [X] T020 [P] [US2] Add a failing test for `over_rejected_reusable.csv` and `reports/v2b_error_group_counts.csv` in `tests/test_v2_1_error_intelligence.py`

### Implementation for User Story 2

- [X] T021 [US2] Implement fixed-band flags `is_near_threshold`, `is_high_confidence`, and `is_over_rejected_reusable` in `src/analysis/v2_1_error_intelligence.py`
- [X] T022 [US2] Implement review group filtering for high-confidence FP/FN and near-threshold FP/FN in `src/analysis/v2_1_error_intelligence.py`
- [X] T023 [US2] Implement over-rejected reusable filtering where `true_label = 0` and `v2b_prediction = 1` in `src/analysis/v2_1_error_intelligence.py`
- [X] T024 [US2] Implement writing all review-group CSVs under `review_groups/` and group-count report under `reports/` in `src/analysis/v2_1_error_intelligence.py`

**Checkpoint**: User Stories 1 and 2 both work independently and produce review-ready error groups.

---

## Phase 5: User Story 3 - Attach Supporting Evidence (Priority: P3)

**Goal**: Attach optional detector and image-quality evidence while preserving all validation audit rows and explicitly reporting missing evidence.

**Independent Test**: Run the US3 tests with partial evidence inputs and verify no validation rows are dropped, present evidence is attached, and missing evidence is counted.

### Tests for User Story 3

- [X] T025 [P] [US3] Add a failing test that category-level detector evidence joins into audit rows with category, confidence, bbox, area, and status fields in `tests/test_v2_1_error_intelligence.py`
- [X] T026 [P] [US3] Add a failing test that image-quality evidence joins into audit rows with brightness, blur, crop size, crop confidence, and status fields in `tests/test_v2_1_error_intelligence.py`
- [X] T027 [P] [US3] Add a failing test that missing optional detector or image-quality evidence preserves all audit rows and writes missing counts in `tests/test_v2_1_error_intelligence.py`

### Implementation for User Story 3

- [X] T028 [US3] Implement optional detector evidence loading and strongest-per-image summarization in `src/analysis/v2_1_error_intelligence.py`
- [X] T029 [US3] Implement optional image-quality evidence loading and normalization in `src/analysis/v2_1_error_intelligence.py`
- [X] T030 [US3] Implement left-join evidence attachment with `present` and `missing` status fields in `src/analysis/v2_1_error_intelligence.py`
- [X] T031 [US3] Implement `reports/v2b_optional_evidence_report.json` and update summary missing-evidence counts in `src/analysis/v2_1_error_intelligence.py`

**Checkpoint**: All user stories are independently functional and the audit contains supporting evidence where available.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, provenance, documentation alignment, and project index updates.

- [X] T032 [P] Add provenance report assertions for all required and optional source identifiers in `tests/test_v2_1_error_intelligence.py`
- [X] T033 Implement `reports/v2b_target_distribution_report.json` and `reports/v2b_audit_provenance.json` in `src/analysis/v2_1_error_intelligence.py`
- [X] T034 [P] Add quickstart command and output-path references to `docs/file-path-index.md` by running `python scripts/update_file_path_index.py`
- [X] T035 Run `python -m py_compile src/analysis/v2_1_error_intelligence.py`
- [X] T036 Run `python -m pytest tests/test_v2_1_error_intelligence.py -v`
- [X] T037 Run `python -m pytest tests -q`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational completion; this is the MVP.
- **User Story 2 (Phase 4)**: Depends on the validated audit rows from US1.
- **User Story 3 (Phase 5)**: Depends on the validated audit rows from US1 and can proceed in parallel with US2 after US1.
- **Polish (Phase 6)**: Depends on selected user stories being complete.

### User Story Dependencies

- **US1 Build V2B Error Audit**: No dependency on other stories after Foundation.
- **US2 Review High-Risk Error Groups**: Requires US1 audit rows; no dependency on US3.
- **US3 Attach Supporting Evidence**: Requires US1 audit rows; no dependency on US2.

### Within Each User Story

- Write story tests before implementation.
- Implement loaders before audit transformations.
- Implement transformations before report writing.
- Verify each checkpoint before moving to the next priority.

## Parallel Opportunities

- T004 and T005 can be written in parallel with T001-T003 after the test file exists.
- T010, T011, and T012 are independent tests for US1.
- T018, T019, and T020 are independent tests for US2.
- T025, T026, and T027 are independent tests for US3.
- US2 and US3 can be implemented in parallel after US1 is complete because they touch independent transformation/reporting paths inside the same module and separate test cases.
- T032 and T034 can run in parallel during polish.

## Parallel Example: User Story 1

```bash
Task: "Add a failing test that builds a synthetic V2B audit and verifies required audit columns, one row per image, error types, and threshold distance in tests/test_v2_1_error_intelligence.py"
Task: "Add a failing test that verifies computed TP/FP/FN/TN, precision, recall, and F1 match the locked baseline metrics in tests/test_v2_1_error_intelligence.py"
Task: "Add a failing test that metric mismatch raises a clear V2.1 audit error before reports are accepted in tests/test_v2_1_error_intelligence.py"
```

## Parallel Example: User Story 2

```bash
Task: "Add a failing test for near-threshold false-positive and false-negative CSVs using abs(probability - threshold) <= 0.05 in tests/test_v2_1_error_intelligence.py"
Task: "Add a failing test for high-confidence false-positive and false-negative CSVs using abs(probability - threshold) >= 0.30 in tests/test_v2_1_error_intelligence.py"
Task: "Add a failing test for over_rejected_reusable.csv and reports/v2b_error_group_counts.csv in tests/test_v2_1_error_intelligence.py"
```

## Parallel Example: User Story 3

```bash
Task: "Add a failing test that category-level detector evidence joins into audit rows with category, confidence, bbox, area, and status fields in tests/test_v2_1_error_intelligence.py"
Task: "Add a failing test that image-quality evidence joins into audit rows with brightness, blur, crop size, crop confidence, and status fields in tests/test_v2_1_error_intelligence.py"
Task: "Add a failing test that missing optional detector or image-quality evidence preserves all audit rows and writes missing counts in tests/test_v2_1_error_intelligence.py"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational validation helpers.
3. Complete Phase 3 US1 audit generation and baseline matching.
4. Stop and validate US1 with `python -m pytest tests/test_v2_1_error_intelligence.py -v`.

### Incremental Delivery

1. US1 delivers the validated baseline audit.
2. US2 adds review-ready fixed-band error groups.
3. US3 adds optional detector and image-quality evidence.
4. Polish adds provenance, target-distribution reports, file index refresh, and full-suite verification.

### Safety Rules

- Do not train.
- Do not generate a Kaggle submission.
- Do not tune thresholds.
- Do not use test labels, sample-solution labels, or public leaderboard scores.
- Do not drop validation rows because optional evidence is missing.
