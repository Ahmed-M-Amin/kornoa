# Tasks: Controlled Data Quality Audit Pipeline

**Input**: Design documents from `/specs/016-data-quality-audit/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Include tests and validation tasks because the feature specification, contract, and constitution require leakage-safe audit behavior, deterministic outputs, and explicit safety confirmations.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the new Spec 016 config, module, and test file scaffolding in the existing analysis-oriented project layout.

- [X] T001 Create Spec 016 audit config scaffold in `configs/data_quality_audit.yaml`
- [X] T002 Create Spec 016 audit module scaffold in `src/analysis/data_quality_audit.py`
- [X] T003 [P] Create Spec 016 test scaffold in `tests/test_data_quality_audit.py`
- [X] T004 [P] Add initial Spec 016 entries to `docs/file-path-index.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the shared audit foundations that all user stories depend on.

**Critical**: No user story work can begin until this phase is complete.

- [X] T005 Implement config loading, path resolution, and safety validation in `src/analysis/data_quality_audit.py`
- [X] T006 [P] Add config and safety validation tests in `tests/test_data_quality_audit.py`
- [X] T007 Implement core data structures, required column definitions, and audit-run traceability helpers in `src/analysis/data_quality_audit.py`
- [X] T008 Implement training label ingestion and base master-table creation in `src/analysis/data_quality_audit.py`
- [X] T009 [P] Add master-table construction tests in `tests/test_data_quality_audit.py`
- [X] T010 Implement CLI argument parsing for `run` and `build-contact-sheets` commands in `src/analysis/data_quality_audit.py`
- [X] T011 [P] Add CLI smoke tests or command-path validation tests in `tests/test_data_quality_audit.py`

**Checkpoint**: Foundation ready. User-story work can now proceed.

---

## Phase 3: User Story 1 - Build Full Data Quality Evidence (Priority: P1)

**Goal**: Produce one complete master audit row per training image with merged prediction, Spec 014, ROI/crop, duplicate, and traceability evidence.

**Independent Test**: Run the audit on a small synthetic dataset and confirm `data_quality_master.csv` contains every source row exactly once with the expected evidence columns, prediction reliability behavior, and no training/submission activity.

- [X] T012 [P] [US1] Add tests for V2B and precomputed OOF prediction ingestion in `tests/test_data_quality_audit.py`
- [X] T013 [US1] Implement V2B and OOF prediction normalization and merge logic in `src/analysis/data_quality_audit.py`
- [X] T014 [P] [US1] Add tests for Spec 014 allowed and blocked row integration in `tests/test_data_quality_audit.py`
- [X] T015 [US1] Implement Spec 014 governance evidence loading and merge logic in `src/analysis/data_quality_audit.py`
- [X] T016 [P] [US1] Add tests for missing optional inputs and reliability downgrades in `tests/test_data_quality_audit.py`
- [X] T017 [US1] Implement optional-input warnings, input inventory collection, and reliability status handling in `src/analysis/data_quality_audit.py`
- [X] T018 [P] [US1] Add tests for exact duplicate detection and label-conflict flagging in `tests/test_data_quality_audit.py`
- [X] T019 [US1] Implement exact duplicate detection, duplicate grouping, and duplicate-conflict evidence in `src/analysis/data_quality_audit.py`
- [X] T020 [P] [US1] Add tests for image availability and ROI/crop quality evidence in `tests/test_data_quality_audit.py`
- [X] T021 [US1] Implement image availability checks and ROI/crop quality metrics in `src/analysis/data_quality_audit.py`
- [X] T022 [P] [US1] Add tests for deterministic label-issue and suspicious-row fallback scoring in `tests/test_data_quality_audit.py`
- [X] T023 [US1] Implement deterministic suspicious-row ranking, boundary distance logic, and fallback label-issue scoring in `src/analysis/data_quality_audit.py`
- [X] T024 [US1] Integrate the full master-table assembly flow in `src/analysis/data_quality_audit.py`

**Checkpoint**: User Story 1 should now produce a complete, deterministic `data_quality_master.csv` without requiring any later bucketing or review outputs.

---

## Phase 4: User Story 2 - Separate Safe Training Rows From Risky Rows (Priority: P1)

**Goal**: Classify rows into controlled buckets and create a conservative cleaned training manifest that contains only training-safe rows.

**Independent Test**: Run the bucketing logic on synthetic rows covering allowed hard rows, blocked rows, duplicate conflicts, ROI-problem rows, missing-image rows, and manual-review rows; confirm only clean and hard-valid rows enter `cleaned_training_manifest.csv`.

- [X] T025 [P] [US2] Add tests for controlled bucket assignment precedence in `tests/test_data_quality_audit.py`
- [X] T026 [US2] Implement bucket assignment rules for `clean_train`, `hard_valid_train`, `exclude_from_training`, and `manual_review_required` in `src/analysis/data_quality_audit.py`
- [X] T027 [P] [US2] Add tests ensuring `manual_review_required` rows never enter the manifest in `tests/test_data_quality_audit.py`
- [X] T028 [US2] Implement `cleaned_training_manifest.csv` generation in `src/analysis/data_quality_audit.py`
- [X] T029 [P] [US2] Add tests for blocked-row overlap and stricter-governance resolution in `tests/test_data_quality_audit.py`
- [X] T030 [US2] Implement blocked-row precedence and final exclusion-reason handling in `src/analysis/data_quality_audit.py`
- [X] T031 [US2] Implement per-bucket CSV output writing in `src/analysis/data_quality_audit.py`

**Checkpoint**: User Story 2 should now produce conservative bucket outputs and a training-safe cleaned manifest with no review-only rows leaking into training.

---

## Phase 5: User Story 3 - Prepare Manual Review Evidence (Priority: P2)

**Goal**: Produce reviewer-facing artifacts that prioritize the most suspicious rows and make visual inspection practical without changing labels.

**Independent Test**: Run the audit and confirm ranked suspicious outputs, review template, duplicate-conflict report, ROI issue report, and optional contact sheets are generated from the same row evidence.

- [X] T032 [P] [US3] Add tests for review template and ranked suspicious-row report generation in `tests/test_data_quality_audit.py`
- [X] T033 [US3] Implement `review_decision_template.csv`, `top_suspicious_rows.csv`, `top_label_issue_rows.csv`, and `top_confident_wrong_rows.csv` generation in `src/analysis/data_quality_audit.py`
- [X] T034 [P] [US3] Add tests for duplicate-conflict and ROI-issue report generation in `tests/test_data_quality_audit.py`
- [X] T035 [US3] Implement `duplicate_conflicts.csv`, `roi_quality_issues.csv`, and related review reports in `src/analysis/data_quality_audit.py`
- [X] T036 [P] [US3] Add tests for contact-sheet generation from existing outputs in `tests/test_data_quality_audit.py`
- [X] T037 [US3] Implement optional contact-sheet generation and the `build-contact-sheets` command in `src/analysis/data_quality_audit.py`

**Checkpoint**: User Story 3 should now support a human review pass without requiring any label mutation or training approval logic.

---

## Phase 6: User Story 4 - Produce Safety and Traceability Reports (Priority: P2)

**Goal**: Produce the summary and inventory outputs that explain exactly what the audit used, what it excluded, and which safety rules remained enforced.

**Independent Test**: Run the audit and verify the summary JSON, bucket counts, exclusion reasons, input inventory, validation alignment summary, and safety confirmations match the generated outputs.

- [X] T038 [P] [US4] Add tests for summary-report counts and safety confirmations in `tests/test_data_quality_audit.py`
- [X] T039 [US4] Implement `data_quality_summary.json`, `bucket_counts.csv`, and `exclusion_reason_counts.csv` generation in `src/analysis/data_quality_audit.py`
- [X] T040 [P] [US4] Add tests for input inventory and validation-alignment reporting in `tests/test_data_quality_audit.py`
- [X] T041 [US4] Implement `input_file_inventory.csv` and `validation_alignment_summary.json` generation in `src/analysis/data_quality_audit.py`
- [X] T042 [US4] Implement the end-to-end `run` orchestration that writes all core outputs in `src/analysis/data_quality_audit.py`

**Checkpoint**: User Story 4 should now produce a full audit evidence package that can be reviewed before any later cleaned-data training spec begins.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Finish repository integration, verification, and documentation updates that apply across all stories.

- [X] T043 [P] Update `docs/file-path-index.md` with final Spec 016 config, source, test, and output paths
- [X] T044 [P] Refine inline module comments and user-facing error messages in `src/analysis/data_quality_audit.py`
- [X] T045 Run `python -m pytest tests/test_data_quality_audit.py -q`
- [X] T046 Run `python -m py_compile src/analysis/data_quality_audit.py`
- [ ] T047 Run `python -m src.analysis.data_quality_audit run --config configs/data_quality_audit.yaml` against available local artifacts and inspect output paths
- [ ] T048 Run `python -m src.analysis.data_quality_audit build-contact-sheets --config configs/data_quality_audit.yaml` if image artifacts are available locally
- [X] T049 Verify that no training, submission generation, test-label use, leaderboard tuning, or original label modification occurs anywhere in `src/analysis/data_quality_audit.py`, `configs/data_quality_audit.yaml`, and `tests/test_data_quality_audit.py`
- [X] T050 Validate `specs/016-data-quality-audit/quickstart.md` against the implemented commands and output paths

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Setup**: No dependencies; can start immediately.
- **Phase 2: Foundational**: Depends on Phase 1; blocks all user stories.
- **Phase 3: User Story 1**: Depends on Phase 2.
- **Phase 4: User Story 2**: Depends on Phase 3 because bucketing requires the assembled master evidence.
- **Phase 5: User Story 3**: Depends on Phase 4 because review artifacts are built from final bucketed rows.
- **Phase 6: User Story 4**: Depends on Phases 3-5 because summary outputs reconcile all prior outputs.
- **Phase 7: Polish**: Depends on completion of all desired stories.

### User Story Dependencies

- **US1**: Can begin after foundational work; no dependency on other stories.
- **US2**: Depends on US1 because safe/unsafe bucketing needs the full merged evidence table.
- **US3**: Depends on US2 because review artifacts are built from bucketed rows and exclusion decisions.
- **US4**: Depends on US1-US3 because summary outputs and safety reporting reconcile the final audit package.

### Within Each User Story

- Tests should be written before the corresponding implementation logic.
- Data ingestion and normalization come before orchestration.
- Bucket assignment comes before manifest and review artifacts.
- Core implementation should land before end-to-end orchestration and quickstart validation.

### Parallel Opportunities

- Setup tasks `T003` and `T004` can run in parallel with `T001`/`T002` once the feature paths are known.
- Foundational test tasks `T006` and `T009`/`T011` can run in parallel with their paired implementation tasks.
- In US1, tasks `T012`, `T014`, `T016`, `T018`, `T020`, and `T022` are parallelizable test tasks for different evidence types.
- In US3, report and contact-sheet test tasks `T032`, `T034`, and `T036` can run in parallel.
- In Phase 7, documentation and verification tasks `T043`, `T044`, and `T049` can run in parallel before the final command validations.

---

## Parallel Example: User Story 1

```bash
# Launch US1 evidence tests together:
Task: "Add tests for V2B and precomputed OOF prediction ingestion in tests/test_data_quality_audit.py"
Task: "Add tests for Spec 014 allowed and blocked row integration in tests/test_data_quality_audit.py"
Task: "Add tests for missing optional inputs and reliability downgrades in tests/test_data_quality_audit.py"
Task: "Add tests for exact duplicate detection and label-conflict flagging in tests/test_data_quality_audit.py"
Task: "Add tests for image availability and ROI/crop quality evidence in tests/test_data_quality_audit.py"
Task: "Add tests for deterministic label-issue and suspicious-row fallback scoring in tests/test_data_quality_audit.py"
```

---

## Parallel Example: User Story 3

```bash
# Launch US3 review-output tests together:
Task: "Add tests for review template and ranked suspicious-row report generation in tests/test_data_quality_audit.py"
Task: "Add tests for duplicate-conflict and ROI-issue report generation in tests/test_data_quality_audit.py"
Task: "Add tests for contact-sheet generation from existing outputs in tests/test_data_quality_audit.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: User Story 1.
4. Stop and validate `data_quality_master.csv` generation independently.

### Incremental Delivery

1. Add US1 to build the full evidence table.
2. Add US2 to make the audit training-safe.
3. Add US3 to support human review.
4. Add US4 to finalize traceability and safety reporting.
5. Finish with Phase 7 verification and quickstart alignment.

### Recommended Practical Scope

The minimal useful delivery is **US1 + US2**. That produces a complete evidence table plus a conservative cleaned manifest, which is enough to block unsafe rows before later review-lock and retraining work.

---

## Notes

- All tasks use the required checklist format with task ID, optional `[P]`, optional `[US#]`, and explicit file paths.
- This feature is intentionally audit-only; no task in this list introduces training, submission generation, or test-label usage.
- `manual_review_required` rows remain excluded from `cleaned_training_manifest.csv` throughout this feature.
