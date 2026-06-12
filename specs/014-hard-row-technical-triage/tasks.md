# Tasks: Hard-Row Technical Triage

**Input**: Design documents from `/specs/014-hard-row-technical-triage/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Include pytest coverage for governed review generation, evidence completion, candidate-package validation, and CLI flows in `tests/test_hard_row_visual_review.py`.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the existing hard-row review lane and feature docs for Spec 014 execution.

- [X] T001 Update Spec 014 CLI commands, expected outputs, and acceptance checks in `specs/014-hard-row-technical-triage/quickstart.md`
- [X] T002 Update governed hard-row triage defaults, output paths, and evidence thresholds in `configs/hard_row_visual_review.yaml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core workflow and safety scaffolding that MUST be complete before any user story work can be implemented safely.

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 Extend config resolution for governed review, evidence completion, and Phase 3 export paths in `src/analysis/hard_row_visual_review.py`
- [X] T004 Add shared row-validation and analysis-only safety guards for validation-only hard rows in `src/analysis/hard_row_visual_review.py`
- [X] T005 [P] Add reusable synthetic fixture builders for governed hard-row audits, annotations, and package inputs in `tests/test_hard_row_visual_review.py`
- [X] T006 Add regression coverage for config-path validation and safety-gate enforcement in `tests/test_hard_row_visual_review.py`

**Checkpoint**: Foundation ready; governed review, evidence completion, and package export work can now proceed story by story.

---

## Phase 3: User Story 1 - Build Hard-Row Review Set (Priority: P1) MVP

**Goal**: Produce one governed review manifest and manual-review template for the validation-derived hard-row pool with stable row identity and reproducible triage outputs.

**Independent Test**: Run the default workflow and confirm the generated review manifest contains exactly the governed rows, unique `image_id` values, prediction context, hard-example classifications, and fallback outputs when optional tooling is unavailable.

### Tests for User Story 1

- [X] T007 [P] [US1] Add governed review output and manual-review template contract tests in `tests/test_hard_row_visual_review.py`
- [X] T008 [P] [US1] Add duplicate-`image_id`, missing-audit-input, and optional-tooling fallback tests in `tests/test_hard_row_visual_review.py`

### Implementation for User Story 1

- [X] T009 [US1] Implement governed hard-row audit loading and unique `image_id` enforcement in `src/analysis/hard_row_visual_review.py`
- [X] T010 [US1] Implement review-manifest and manual-review-template generation with baseline and candidate prediction context in `src/analysis/hard_row_visual_review.py`
- [X] T011 [US1] Implement clustering, anomaly-scoring, representative triage, and fallback report writers in `src/analysis/hard_row_visual_review.py`
- [X] T012 [US1] Implement default CLI workflow output wiring for review manifests, summary reports, and contact sheets in `src/analysis/hard_row_visual_review.py`

**Checkpoint**: User Story 1 should now generate a governed hard-row review set that is independently testable and ready for evidence completion.

---

## Phase 4: User Story 2 - Enrich Hard Rows With Technical Evidence (Priority: P2)

**Goal**: Complete annotation, ROI, crop-quality, and evidence-overlay enrichment for every governed hard row while preserving explicit missing-evidence states.

**Independent Test**: Run the `--complete-evidence` workflow and confirm each governed row gets annotation evidence or an explicit missing status, crop-quality metrics or an explicit missing status, detector fallback behavior, and evidence asset paths.

### Tests for User Story 2

- [X] T013 [P] [US2] Add annotation-evidence and detector-fallback coverage for governed hard rows in `tests/test_hard_row_visual_review.py`
- [X] T014 [P] [US2] Add ROI, crop-quality, missing-image, and derived-evidence edge-case tests in `tests/test_hard_row_visual_review.py`

### Implementation for User Story 2

- [X] T015 [US2] Implement annotation-evidence extraction and explicit missing-evidence statuses in `src/analysis/hard_row_visual_review.py`
- [X] T016 [US2] Implement ROI and crop-quality evidence metrics, derived ROI fallback, and image-missing handling in `src/analysis/hard_row_visual_review.py`
- [X] T017 [US2] Implement evidence-asset rendering and evidence-completed manifest assembly in `src/analysis/hard_row_visual_review.py`
- [X] T018 [US2] Implement evidence-aware action-plan generation with separate `phase3_use_allowed` gating in `src/analysis/hard_row_visual_review.py`

**Checkpoint**: User Stories 1 and 2 should both work independently, with evidence-aware outputs ready for packaging.

---

## Phase 5: User Story 3 - Export Safe Phase 3 Candidate Package (Priority: P3)

**Goal**: Export validated allowed and blocked Phase 3 candidate files from the evidence-aware action plan without leaking blocked rows into allowed outputs.

**Independent Test**: Run the `--export-phase3-candidates` workflow and confirm input-row totals and allowed/blocked counts are validated before writing files, allowed rows are split by action type, blocked rows are split by risk class, and no blocked row appears in any allowed package output.

### Tests for User Story 3

- [X] T019 [P] [US3] Add Phase 3 package export success-path coverage for allowed and blocked output splits in `tests/test_hard_row_visual_review.py`
- [X] T020 [P] [US3] Add action-plan count-validation and blocked-row exclusion failure tests in `tests/test_hard_row_visual_review.py`

### Implementation for User Story 3

- [X] T021 [US3] Implement evidence-aware action-plan validation for total, allowed, and blocked row counts in `src/analysis/hard_row_visual_review.py`
- [X] T022 [US3] Implement Phase 3 candidate package writers for allowed general, artifact-focus, tiny-defect, and blocked risk-class CSV outputs in `src/analysis/hard_row_visual_review.py`
- [X] T023 [US3] Implement Phase 3 package summary serialization and `--export-phase3-candidates` CLI flow in `src/analysis/hard_row_visual_review.py`

**Checkpoint**: All three user stories should now be independently functional and export-safe.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final consistency, documentation, and verification across the full Spec 014 workflow.

- [X] T024 [P] Update the Spec 014 stable workflow contract and output guarantees in `specs/014-hard-row-technical-triage/contracts/hard-row-technical-triage-contract.md`
- [X] T025 [P] Align Spec 014 entities, safety-gate language, and package classes in `specs/014-hard-row-technical-triage/data-model.md`
- [X] T026 Run `python -m py_compile src/analysis/hard_row_visual_review.py` and `python -m pytest tests/test_hard_row_visual_review.py -q` for `src/analysis/hard_row_visual_review.py` and `tests/test_hard_row_visual_review.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion; delivers the MVP
- **User Story 2 (Phase 4)**: Depends on User Story 1 outputs because evidence completion consumes the review manifest
- **User Story 3 (Phase 5)**: Depends on User Story 2 outputs because package export consumes the evidence-aware action plan
- **Polish (Phase 6)**: Depends on the desired user stories being complete

### User Story Dependencies

- **US1**: Starts after Foundational and has no dependency on later stories
- **US2**: Requires the governed review manifest and manual-review template produced by US1
- **US3**: Requires the evidence-aware action plan produced by US2

### Within Each User Story

- Tests should be written before implementation tasks in the same story
- Shared validation and safety helpers land before story-specific workflow logic
- Review-manifest generation precedes evidence completion
- Evidence completion precedes candidate-package export

### Parallel Opportunities

- `T005` and `T006` can proceed in parallel once the foundational code shape is settled
- `T007` and `T008` can be written in parallel because they cover different US1 behaviors in `tests/test_hard_row_visual_review.py`
- `T013` and `T014` can be written in parallel because they cover different US2 evidence scenarios in `tests/test_hard_row_visual_review.py`
- `T019` and `T020` can be written in parallel because they cover different US3 export behaviors in `tests/test_hard_row_visual_review.py`
- `T024` and `T025` can be completed in parallel during polish because they touch different design artifacts

---

## Parallel Example: User Story 1

```bash
# Launch US1 test work together:
Task: "Add governed review output and manual-review template contract tests in tests/test_hard_row_visual_review.py"
Task: "Add duplicate-image_id, missing-audit-input, and optional-tooling fallback tests in tests/test_hard_row_visual_review.py"
```

---

## Parallel Example: User Story 2

```bash
# Launch US2 test work together:
Task: "Add annotation-evidence and detector-fallback coverage for governed hard rows in tests/test_hard_row_visual_review.py"
Task: "Add ROI, crop-quality, missing-image, and derived-evidence edge-case tests in tests/test_hard_row_visual_review.py"
```

---

## Parallel Example: User Story 3

```bash
# Launch US3 test work together:
Task: "Add Phase 3 package export success-path coverage for allowed and blocked output splits in tests/test_hard_row_visual_review.py"
Task: "Add action-plan count-validation and blocked-row exclusion failure tests in tests/test_hard_row_visual_review.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Validate the governed review manifest and manual-review template before moving on

### Incremental Delivery

1. Finish Setup + Foundational to stabilize the workflow surface
2. Deliver US1 to lock the governed review set and triage reports
3. Deliver US2 to enrich all governed rows with evidence and gating decisions
4. Deliver US3 to export the validated candidate package
5. Finish Polish to align docs and run verification commands

### Suggested MVP Scope

1. Phase 1: Setup
2. Phase 2: Foundational
3. Phase 3: User Story 1

---

## Notes

- Every task follows the required checklist format with a task ID and exact file path
- `phase3_use_allowed` remains a separate safety gate from `recommended_action`
- The workflow must remain analysis-only: no training, no submission creation, no test-label use, and no label overwriting
