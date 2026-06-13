# Tasks: Spec 017 Automated Decision Lock

**Input**: Design documents from `specs/017-automated-decision-lock/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/decision-lock-contract.md`, `quickstart.md`

**Tests**: Required. The feature is a safety gate for later training, so each user story includes tests before implementation.

**Organization**: Tasks are grouped by user story so the locked manifest MVP can be implemented and verified before adjudication and optional-evidence enhancements.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files or only reads existing context.
- **[Story]**: User story label from `spec.md`.
- Every task references an exact repository path.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add the new config, module shell, and test file needed by all stories.

- [X] T001 Create default decision-lock config in `configs/data_quality_decision_lock.yaml`
- [X] T002 Create importable module and CLI skeleton in `src/analysis/data_quality_decision_lock.py`
- [X] T003 Create test module with temporary fixture builders in `tests/test_data_quality_decision_lock.py`
- [X] T004 [P] Review existing Spec 016 audit helpers for reusable patterns in `src/analysis/data_quality_audit.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement shared loading, validation, writing, and safety controls that every story depends on.

**Critical**: No user story work should be considered complete until these foundation tasks pass.

- [X] T005 Write failing config contract tests for required paths and safety flags in `tests/test_data_quality_decision_lock.py`
- [X] T006 Implement config loading and required path validation in `src/analysis/data_quality_decision_lock.py`
- [X] T007 Implement safety flag validation that rejects training, submissions, test labels, leaderboard input, and relabeling in `src/analysis/data_quality_decision_lock.py`
- [X] T008 Write failing tests for required output directory creation in `tests/test_data_quality_decision_lock.py`
- [X] T009 Implement output directory creation for root, reports, and contact sheets in `src/analysis/data_quality_decision_lock.py`
- [X] T010 Implement CSV and JSON writer helpers with stable column ordering in `src/analysis/data_quality_decision_lock.py`
- [X] T011 Run foundational validation with `python -m pytest tests/test_data_quality_decision_lock.py -q`

**Checkpoint**: Foundation ready. User story implementation can start.

---

## Phase 3: User Story 1 - Lock Safe Training Rows (Priority: P1) MVP

**Goal**: Produce `approved_cleaned_training_manifest.csv` containing only rows with an explicit approved keep decision and no blocked, unresolved, conflicting, or missing-image rows.

**Independent Test**: Run the workflow against synthetic Spec 016 and Spec 014 fixtures and verify one decision per master row, blocked rows excluded, approved manifest contains only `auto_keep`, and safety summary flags remain true.

### Tests for User Story 1

- [X] T012 [US1] Write failing test for one decision record per Spec 016 master row in `tests/test_data_quality_decision_lock.py`
- [X] T013 [US1] Write failing test that Spec 014 blocked rows cannot enter the approved manifest in `tests/test_data_quality_decision_lock.py`
- [X] T014 [US1] Write failing test that unresolved manual-review rows cannot enter the approved manifest in `tests/test_data_quality_decision_lock.py`
- [X] T015 [US1] Write failing test that approved rows must exist in train labels and train images in `tests/test_data_quality_decision_lock.py`
- [X] T016 [US1] Write failing test for required summary safety fields in `tests/test_data_quality_decision_lock.py`

### Implementation for User Story 1

- [X] T017 [US1] Implement Spec 016 input loading and schema normalization in `src/analysis/data_quality_decision_lock.py`
- [X] T018 [US1] Implement Spec 014 allowed and blocked row loading in `src/analysis/data_quality_decision_lock.py`
- [X] T019 [US1] Implement train label and train image availability validation in `src/analysis/data_quality_decision_lock.py`
- [X] T020 [US1] Implement base decision assignment for clean, hard-valid, manual-review, blocked, and invalid rows in `src/analysis/data_quality_decision_lock.py`
- [X] T021 [US1] Implement approved manifest generation from `auto_keep` decisions only in `src/analysis/data_quality_decision_lock.py`
- [X] T022 [US1] Implement `decision_master.csv`, `auto_keep_rows.csv`, `auto_exclude_rows.csv`, and `deferred_uncertain_rows.csv` outputs in `src/analysis/data_quality_decision_lock.py`
- [X] T023 [US1] Implement `decision_lock_summary.json` with blocked overlap, unresolved review, missing image, and safety counts in `src/analysis/data_quality_decision_lock.py`
- [X] T024 [US1] Run MVP validation with `python -m pytest tests/test_data_quality_decision_lock.py -q`

**Checkpoint**: User Story 1 is independently functional and can create the locked approved manifest MVP.

---

## Phase 4: User Story 2 - Reduce Manual Review to High-Impact Adjudication (Priority: P2)

**Goal**: Convert broad manual-review rows into ranked `needs_adjudication` and `defer` outputs with ROI/crop group summaries and representative contact sheets.

**Independent Test**: Run the workflow against fixtures containing ROI/crop review rows and verify grouped ROI summaries, ranked adjudication queue, representative rows first, and unresolved adjudication rows excluded from the approved manifest.

### Tests for User Story 2

- [X] T025 [US2] Write failing test for ROI/crop review row grouping and proposed group actions in `tests/test_data_quality_decision_lock.py`
- [X] T026 [US2] Write failing test for adjudication queue columns and risk ranking in `tests/test_data_quality_decision_lock.py`
- [X] T027 [US2] Write failing test that adjudication and defer rows are excluded from the approved manifest in `tests/test_data_quality_decision_lock.py`
- [X] T028 [US2] Write failing test for contact sheet file creation from representative rows in `tests/test_data_quality_decision_lock.py`

### Implementation for User Story 2

- [X] T029 [US2] Implement ROI/crop severity grouping and group risk scoring in `src/analysis/data_quality_decision_lock.py`
- [X] T030 [US2] Implement over-broad ROI rule downgrading when corroborating evidence is absent in `src/analysis/data_quality_decision_lock.py`
- [X] T031 [US2] Implement adjudication queue ranking, risk buckets, group ids, and suggested decisions in `src/analysis/data_quality_decision_lock.py`
- [X] T032 [US2] Implement `needs_adjudication_rows.csv` and `adjudication_queue.csv` outputs in `src/analysis/data_quality_decision_lock.py`
- [X] T033 [US2] Implement `roi_recalibration_summary.csv` and ROI group representative selection in `src/analysis/data_quality_decision_lock.py`
- [X] T034 [US2] Implement contact sheet generation for top adjudication, auto-exclude, and ROI representatives in `src/analysis/data_quality_decision_lock.py`
- [X] T035 [US2] Run adjudication validation with `python -m pytest tests/test_data_quality_decision_lock.py -q`

**Checkpoint**: User Story 2 is independently functional and reduces broad review to a ranked adjudication queue.

---

## Phase 5: User Story 3 - Upgrade Evidence Before Decisions (Priority: P3)

**Goal**: Use optional prediction, embedding, cluster, duplicate, outlier, and Cleanlab-style evidence when present, while remaining conservative and explicit when evidence is missing.

**Independent Test**: Run the workflow with complete, partial, and missing optional evidence fixtures and verify evidence inventory status, duplicate-conflict blocking, confidence downgrades, and optional evidence contributions to decision reasons.

### Tests for User Story 3

- [X] T036 [US3] Write failing test that missing optional evidence is reported without failing the run in `tests/test_data_quality_decision_lock.py`
- [X] T037 [US3] Write failing test that duplicate-conflict rows are excluded from the approved manifest in `tests/test_data_quality_decision_lock.py`
- [X] T038 [US3] Write failing test for partial optional evidence warnings in `tests/test_data_quality_decision_lock.py`
- [X] T039 [US3] Write failing test that optional prediction and label-quality evidence affects risk level and decision confidence in `tests/test_data_quality_decision_lock.py`

### Implementation for User Story 3

- [X] T040 [US3] Implement optional evidence loading for predictions, embeddings, Cleanlab-style scores, and cluster assignments in `src/analysis/data_quality_decision_lock.py`
- [X] T041 [US3] Implement evidence availability classification as missing, partial, or available in `src/analysis/data_quality_decision_lock.py`
- [X] T042 [US3] Implement exact and near-duplicate group conflict handling from available evidence in `src/analysis/data_quality_decision_lock.py`
- [X] T043 [US3] Implement prediction risk and label-quality risk integration into decision confidence in `src/analysis/data_quality_decision_lock.py`
- [X] T044 [US3] Implement cluster and outlier evidence columns in decision records when available in `src/analysis/data_quality_decision_lock.py`
- [X] T045 [US3] Implement `evidence_inventory.json` and missing evidence warnings in `src/analysis/data_quality_decision_lock.py`
- [X] T046 [US3] Implement `decision_rule_audit.csv` with one traceability row per decision record in `src/analysis/data_quality_decision_lock.py`
- [X] T047 [US3] Run optional-evidence validation with `python -m pytest tests/test_data_quality_decision_lock.py -q`

**Checkpoint**: User Story 3 is independently functional and upgrades decisions when optional evidence exists.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final safety validation, documentation, and path index updates.

- [X] T048 Update quickstart validation details if implementation paths differ in `specs/017-automated-decision-lock/quickstart.md`
- [X] T049 Update repository path index entries for Spec 017 files in `docs/file-path-index.md`
- [X] T050 Add Spec 017 implementation notes or generated output descriptions to `docs/file-path-index.md`
- [X] T051 Run full Spec 017 test suite with `python -m pytest tests/test_data_quality_decision_lock.py -q`
- [X] T052 Run compile check with `python -m py_compile src/analysis/data_quality_decision_lock.py`
- [X] T053 Run real workflow dry validation with `python -m src.analysis.data_quality_decision_lock run --config configs/data_quality_decision_lock.yaml`
- [X] T054 Inspect `outputs/analysis/data_quality_decision_lock/reports/decision_lock_summary.json` for safety flags and blocked overlap counts
- [X] T055 Confirm no training artifacts or submission files were created by Spec 017 outputs under `outputs/analysis/data_quality_decision_lock/`
- [X] T056 Record final implementation results in `specs/017-automated-decision-lock/tasks.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies.
- **Phase 2 Foundational**: Depends on Phase 1 and blocks all user stories.
- **Phase 3 User Story 1**: Depends on Phase 2 and is the MVP.
- **Phase 4 User Story 2**: Depends on Phase 2 and can run after or alongside US1 once shared output contracts are stable.
- **Phase 5 User Story 3**: Depends on Phase 2 and can run after or alongside US1 once base decision records are stable.
- **Phase 6 Polish**: Depends on selected user stories being complete.

### User Story Dependencies

- **US1 Lock Safe Training Rows**: Required MVP and safest first implementation target.
- **US2 Reduce Manual Review**: Builds on decision records and ROI evidence; should not change US1 safety gates.
- **US3 Upgrade Evidence**: Adds optional evidence and duplicate handling; should not weaken US1 or US2 gates.

### Parallel Opportunities

- T001, T002, and T003 can be prepared in parallel if separate workers coordinate file ownership.
- T004 can run in parallel with T001 through T003 because it only reviews existing code.
- US2 tests T025 through T028 can be drafted while US1 implementation is underway after foundation is stable.
- US3 tests T036 through T039 can be drafted while US1 implementation is underway after foundation is stable.
- Documentation tasks T048 through T050 can run after output names stabilize.

## Parallel Example: User Story 2

```text
Task: "Write failing test for ROI/crop review row grouping and proposed group actions in tests/test_data_quality_decision_lock.py"
Task: "Write failing test for adjudication queue columns and risk ranking in tests/test_data_quality_decision_lock.py"
Task: "Write failing test that adjudication and defer rows are excluded from the approved manifest in tests/test_data_quality_decision_lock.py"
```

## Parallel Example: User Story 3

```text
Task: "Write failing test that missing optional evidence is reported without failing the run in tests/test_data_quality_decision_lock.py"
Task: "Write failing test that duplicate-conflict rows are excluded from the approved manifest in tests/test_data_quality_decision_lock.py"
Task: "Write failing test for partial optional evidence warnings in tests/test_data_quality_decision_lock.py"
```

## Implementation Strategy

### MVP First

1. Complete Phase 1 setup.
2. Complete Phase 2 foundation.
3. Complete Phase 3 US1.
4. Stop and validate the approved manifest safety contract before adding adjudication or optional evidence.

### Incremental Delivery

1. US1 delivers the locked approved manifest MVP.
2. US2 adds ranked adjudication and ROI/crop recalibration without changing the manifest safety contract.
3. US3 adds optional evidence and duplicate conflict handling without weakening safety gates.
4. Polish validates tests, compile check, real workflow output, docs, and path index.

### Final Validation Commands

```powershell
python -m pytest tests/test_data_quality_decision_lock.py -q
python -m py_compile src/analysis/data_quality_decision_lock.py
python -m src.analysis.data_quality_decision_lock run --config configs/data_quality_decision_lock.yaml
```

## Implementation Results

- Real workflow validation completed on 2026-06-13 with `approved_manifest_count = 35211`, `auto_exclude_count = 20`, and `needs_adjudication_count = 111`.
- Safety summary confirmed `no_training_started = true`, `no_submission_created = true`, `no_test_labels_used = true`, and `no_leaderboard_tuning = true`.
