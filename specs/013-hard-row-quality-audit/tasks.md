# Tasks: Hard Row Quality Audit

**Input**: Design documents from `/specs/013-hard-row-quality-audit/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/hard-row-quality-audit-contract.md, quickstart.md

**Tests**: Required. This feature is an analysis/reporting quality gate and must include pytest coverage for hard-row identity, exact-one category assignment, optional-evidence handling, safety safeguards, and summary aggregation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the feature-specific config, module, and test scaffolding for the hard-row audit workflow.

- [X] T001 Add hard-row audit config skeleton for governed row sources, optional evidence sources, and output paths in `configs/hard_row_quality_audit.yaml`
- [X] T002 Create the hard-row audit module entrypoint, constants, and error class in `src/analysis/hard_row_quality_audit.py`
- [X] T003 Create the hard-row audit test module scaffolding with synthetic fixture helpers in `tests/test_hard_row_quality_audit.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared safety, normalization, category validation, and summary helpers that all user stories depend on.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 [P] Add failing safety tests that reject test labels, public leaderboard inputs, training actions, submission creation, and relabel application in `tests/test_hard_row_quality_audit.py`
- [X] T005 [P] Add failing identity tests for duplicate hard-row rejection, normalized `image_id` handling, and exact governed-row coverage in `tests/test_hard_row_quality_audit.py`
- [X] T006 [P] Add failing category tests that enforce exactly one primary audit category and non-empty audit rationale per row in `tests/test_hard_row_quality_audit.py`
- [X] T007 Implement shared config loading, path resolution, run-manifest capture, and safety validation in `src/analysis/hard_row_quality_audit.py`
- [X] T008 Implement governed hard-row loading, row-context loading, and shared `image_id` normalization helpers in `src/analysis/hard_row_quality_audit.py`
- [X] T009 Implement primary-category validation, optional-evidence status normalization, and row-completeness checks in `src/analysis/hard_row_quality_audit.py`
- [X] T010 Implement summary aggregation helpers for category counts and ranked failure-mode records in `src/analysis/hard_row_quality_audit.py`

**Checkpoint**: Foundation ready; user story implementation can now begin.

---

## Phase 3: User Story 1 - Audit The Hardest Rows (Priority: P1)

**Goal**: Produce one governed row-level audit table over the hardest validation-derived rows with stable identity and exactly one primary category per row.

**Independent Test**: Run the hard-row audit on synthetic governed inputs and verify that every input row appears exactly once in the output with one valid primary category and rationale.

### Tests for User Story 1

- [X] T011 [P] [US1] Add a failing test that verifies every governed hard row appears exactly once in `audit/hard_row_audit.csv` in `tests/test_hard_row_quality_audit.py`
- [X] T012 [P] [US1] Add a failing test that verifies the row-level audit output includes required comparison context and hard-example type fields in `tests/test_hard_row_quality_audit.py`
- [X] T013 [P] [US1] Add a failing test that verifies missing governed input files fail clearly before audit outputs are written in `tests/test_hard_row_quality_audit.py`

### Implementation for User Story 1

- [X] T014 [US1] Implement row-level audit table construction with stable identity and required context fields in `src/analysis/hard_row_quality_audit.py`
- [X] T015 [US1] Implement writing `audit/hard_row_audit.csv` with exact-one primary category and rationale per row in `src/analysis/hard_row_quality_audit.py`
- [X] T016 [US1] Implement `run_hard_row_quality_audit(config_path)` orchestration and CLI exit behavior without tracebacks for expected audit input errors in `src/analysis/hard_row_quality_audit.py`

**Checkpoint**: User Story 1 is functional and independently testable.

---

## Phase 4: User Story 2 - Capture Evidence For Each Audit Decision (Priority: P2)

**Goal**: Enrich each audit row with optional detector and image-quality evidence while keeping the audit valid when those sources are absent.

**Independent Test**: Run the audit with synthetic optional evidence sources and verify that evidence fields are attached when present and explicitly marked missing when absent.

### Tests for User Story 2

- [X] T017 [P] [US2] Add a failing test that verifies detector evidence joins into audit rows with explicit `present` or `missing` status in `tests/test_hard_row_quality_audit.py`
- [X] T018 [P] [US2] Add a failing test that verifies crop-quality or image-quality evidence joins into audit rows with explicit `present` or `missing` status in `tests/test_hard_row_quality_audit.py`
- [X] T019 [P] [US2] Add a failing test that verifies optional evidence absence preserves all governed rows and does not invalidate the audit in `tests/test_hard_row_quality_audit.py`

### Implementation for User Story 2

- [X] T020 [US2] Implement optional detector evidence loading and normalization in `src/analysis/hard_row_quality_audit.py`
- [X] T021 [US2] Implement optional crop-quality or image-quality evidence loading and normalization in `src/analysis/hard_row_quality_audit.py`
- [X] T022 [US2] Implement row-level evidence attachment with explicit status fields and secondary notes support in `src/analysis/hard_row_quality_audit.py`

**Checkpoint**: User Stories 1 and 2 both work independently and preserve evidence quality.

---

## Phase 5: User Story 3 - Produce A Decision-Ready Audit Summary (Priority: P3)

**Goal**: Generate aggregated category counts and a ranked failure-mode summary that guides the next classifier, preprocessing, or detector branch.

**Independent Test**: Run the audit summary on synthetic audited rows and verify that category totals equal the audited row count and that failure-mode groups are ranked consistently.

### Tests for User Story 3

- [X] T023 [P] [US3] Add a failing test that verifies `reports/hard_row_audit_category_counts.csv` totals equal the full audited row count in `tests/test_hard_row_quality_audit.py`
- [X] T024 [P] [US3] Add a failing test that verifies `reports/hard_row_failure_mode_summary.json` contains ranked action-oriented failure-mode groups in `tests/test_hard_row_quality_audit.py`
- [X] T025 [P] [US3] Add a failing test that verifies likely label issues remain distinguishable from likely-correct-but-hard rows in the summary outputs in `tests/test_hard_row_quality_audit.py`

### Implementation for User Story 3

- [X] T026 [US3] Implement aggregated category-count report generation in `src/analysis/hard_row_quality_audit.py`
- [X] T027 [US3] Implement ranked failure-mode summary generation with recommended next-action fields in `src/analysis/hard_row_quality_audit.py`
- [X] T028 [US3] Implement summary output persistence under `reports/` in `src/analysis/hard_row_quality_audit.py`

**Checkpoint**: All user stories are independently functional and the feature produces a decision-ready hard-row audit.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, documentation alignment, and full-suite verification.

- [X] T029 [P] Add hard-row audit paths to `docs/file-path-index.md` by running `python scripts/update_file_path_index.py`
- [X] T030 Validate the quickstart command and expected output paths in `specs/013-hard-row-quality-audit/quickstart.md`
- [X] T031 Run `python -m py_compile src/analysis/hard_row_quality_audit.py`
- [X] T032 Run `python -m pytest tests/test_hard_row_quality_audit.py -q`
- [X] T033 Run `python -m pytest tests -q`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational completion; this is the MVP.
- **User Story 2 (Phase 4)**: Depends on governed-row audit records from US1.
- **User Story 3 (Phase 5)**: Depends on completed row-level audit records from US1 and evidence handling from US2.
- **Polish (Phase 6)**: Depends on selected user stories being complete.

### User Story Dependencies

- **US1 Audit The Hardest Rows**: No dependency on other stories after Foundation.
- **US2 Capture Evidence For Each Audit Decision**: Requires US1 row-level audit structure.
- **US3 Produce A Decision-Ready Audit Summary**: Requires US1 row outputs and benefits from US2 evidence fields.

### Within Each User Story

- Write story tests before implementation.
- Implement row loading and normalization before output writing.
- Implement output writing before CLI or persistence integration.
- Verify each checkpoint before moving to the next priority.

### Parallel Opportunities

- T004, T005, and T006 can be written in parallel once the test module exists.
- T011, T012, and T013 are independent US1 tests.
- T017, T018, and T019 are independent US2 tests.
- T023, T024, and T025 are independent US3 tests.
- T029 and T030 can run in parallel during polish.

---

## Parallel Example: User Story 1

```bash
Task: "Add a failing test that verifies every governed hard row appears exactly once in audit/hard_row_audit.csv in tests/test_hard_row_quality_audit.py"
Task: "Add a failing test that verifies the row-level audit output includes required comparison context and hard-example type fields in tests/test_hard_row_quality_audit.py"
Task: "Add a failing test that verifies missing governed input files fail clearly before audit outputs are written in tests/test_hard_row_quality_audit.py"
```

## Parallel Example: User Story 2

```bash
Task: "Add a failing test that verifies detector evidence joins into audit rows with explicit present or missing status in tests/test_hard_row_quality_audit.py"
Task: "Add a failing test that verifies crop-quality or image-quality evidence joins into audit rows with explicit present or missing status in tests/test_hard_row_quality_audit.py"
Task: "Add a failing test that verifies optional evidence absence preserves all governed rows and does not invalidate the audit in tests/test_hard_row_quality_audit.py"
```

## Parallel Example: User Story 3

```bash
Task: "Add a failing test that verifies reports/hard_row_audit_category_counts.csv totals equal the full audited row count in tests/test_hard_row_quality_audit.py"
Task: "Add a failing test that verifies reports/hard_row_failure_mode_summary.json contains ranked action-oriented failure-mode groups in tests/test_hard_row_quality_audit.py"
Task: "Add a failing test that verifies likely label issues remain distinguishable from likely-correct-but-hard rows in the summary outputs in tests/test_hard_row_quality_audit.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational helpers.
3. Complete Phase 3 US1 row-level audit workflow.
4. Stop and validate US1 with `python -m pytest tests/test_hard_row_quality_audit.py -q`.

### Incremental Delivery

1. US1 delivers the governed hard-row audit table.
2. US2 adds optional evidence attachment and explicit missing-evidence handling.
3. US3 adds category summaries and ranked failure-mode outputs.
4. Polish adds quickstart alignment, file index refresh, and full-suite verification.

### Safety Rules

- Do not train.
- Do not generate a Kaggle submission.
- Do not use test labels or public leaderboard feedback.
- Do not auto-apply relabeling.
- Do not let optional evidence absence block the core audit.
