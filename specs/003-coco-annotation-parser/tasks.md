---
description: "Task list for SPEC-003 COCO Annotation Parser"
---

# Tasks: COCO Annotation Parser

**Input**: Design documents from `/specs/003-coco-annotation-parser/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/coco-parser-contract.md, quickstart.md

**Tests**: Synthetic-fixture pytest coverage is required so parser behavior can be validated without private Krones data.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish parser module and test file locations without touching private data.

- [X] T001 Create COCO parser module scaffold in `src/data/coco_parser.py`
- [X] T002 Create parser test module scaffold in `tests/test_coco_parser.py`
- [X] T003 Create malformed annotation fixture directory in `tests/fixtures/synthetic_dataset/coco_cases/`
- [X] T004 Confirm `requirements.txt` needs no new runtime dependency for SPEC-003 core parsing

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Provide shared parser records, errors, fixture cases, and contract helpers before user-story work begins.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Define parser error and diagnostic record structures in `src/data/coco_parser.py`
- [X] T006 Define normalized image, category, annotation, and summary record structures in `src/data/coco_parser.py`
- [X] T007 [P] Create synthetic missing-collection annotation fixture in `tests/fixtures/synthetic_dataset/coco_cases/missing_collections.json`
- [X] T008 [P] Create synthetic malformed JSON fixture in `tests/fixtures/synthetic_dataset/coco_cases/malformed.json`
- [X] T009 [P] Create synthetic relationship-error annotation fixture in `tests/fixtures/synthetic_dataset/coco_cases/relationship_errors.json`
- [X] T010 [P] Create synthetic malformed-box annotation fixture in `tests/fixtures/synthetic_dataset/coco_cases/malformed_boxes.json`

**Checkpoint**: Parser contracts and synthetic COCO cases are ready for user-story tests.

---

## Phase 3: User Story 1 - Parse COCO Annotation File (Priority: P1) MVP

**Goal**: A contributor can load a valid COCO-style annotation file and receive normalized records plus a validation summary.

**Independent Test**: Run parser tests against `tests/fixtures/synthetic_dataset/train_annotations.json` and confirm images, annotations, categories, mappings, and top-level collection validation work without private data.

### Tests for User Story 1

- [X] T011 [US1] Write failing valid COCO parsing test for image, annotation, category, and summary counts in `tests/test_coco_parser.py`
- [X] T012 [US1] Write failing missing required collection diagnostic test using `tests/fixtures/synthetic_dataset/coco_cases/missing_collections.json` in `tests/test_coco_parser.py`
- [X] T013 [US1] Write failing malformed JSON diagnostic test using `tests/fixtures/synthetic_dataset/coco_cases/malformed.json` in `tests/test_coco_parser.py`
- [X] T014 [US1] Write failing missing `train_annotations.json` diagnostic test that verifies a readable diagnostic names the missing file in `tests/test_coco_parser.py`

### Implementation for User Story 1

- [X] T015 [US1] Implement configured COCO JSON file loading and readable missing-file errors in `src/data/coco_parser.py`
- [X] T016 [US1] Implement required `images`, `annotations`, and `categories` collection validation in `src/data/coco_parser.py`
- [X] T017 [US1] Implement normalized image, annotation, and category record creation in `src/data/coco_parser.py`
- [X] T018 [US1] Implement basic image-to-annotation mapping summary in `src/data/coco_parser.py`
- [X] T019 [US1] Run User Story 1 checks with `pytest tests/test_coco_parser.py -q`

**Checkpoint**: User Story 1 is complete when valid COCO metadata parses into normalized records and missing or malformed input fails with readable diagnostics.

---

## Phase 4: User Story 2 - Validate Annotation Relationships (Priority: P2)

**Goal**: A contributor can detect orphaned annotations, unknown categories, malformed boxes, and duplicate identifiers before downstream features consume annotations.

**Independent Test**: Run parser tests against controlled synthetic invalid records and confirm each relationship issue is reported without blocking valid records.

### Tests for User Story 2

- [X] T020 [US2] Write failing orphaned annotation diagnostic test in `tests/test_coco_parser.py`
- [X] T021 [US2] Write failing unknown category diagnostic test in `tests/test_coco_parser.py`
- [X] T022 [US2] Write failing malformed bounding-box diagnostic test in `tests/test_coco_parser.py`
- [X] T023 [US2] Write failing duplicate image, annotation, and category identifier tests in `tests/test_coco_parser.py`
- [X] T024 [US2] Write failing no-test-label-inference regression test in `tests/test_coco_parser.py`

### Implementation for User Story 2

- [X] T025 [US2] Implement orphaned annotation detection and diagnostics in `src/data/coco_parser.py`
- [X] T026 [US2] Implement unknown category detection and diagnostics in `src/data/coco_parser.py`
- [X] T027 [US2] Implement malformed bounding-box validation for missing values, negative coordinates, and non-positive dimensions in `src/data/coco_parser.py`
- [X] T028 [US2] Implement duplicate image, annotation, and category identifier detection in `src/data/coco_parser.py`
- [X] T029 [US2] Ensure relationship diagnostics preserve valid normalized records in `src/data/coco_parser.py`
- [X] T030 [US2] Run User Story 2 checks with `pytest tests/test_coco_parser.py -q`

**Checkpoint**: User Story 2 is complete when parser diagnostics report relationship and box quality problems without creating test labels or discarding unrelated valid records.

---

## Phase 5: User Story 3 - Provide Annotation-Derived Summaries (Priority: P3)

**Goal**: A contributor can review category, ROI, box, segmentation, and coverage summaries for later ROI, detector, explainability, and reporting specs.

**Independent Test**: Run parser summary tests and confirm coverage, category distribution, ROI availability, bounding-box availability, and segmentation availability are reported from synthetic COCO metadata.

### Tests for User Story 3

- [X] T031 [US3] Write failing annotation coverage summary test for annotated and unannotated images in `tests/test_coco_parser.py`
- [X] T032 [US3] Write failing category and defect-label distribution summary test in `tests/test_coco_parser.py`
- [X] T033 [US3] Write failing ROI availability summary test based on bounding boxes or segmentation in `tests/test_coco_parser.py`
- [X] T034 [US3] Write failing segmentation availability summary test without mask rasterization in `tests/test_coco_parser.py`
- [X] T035 [US3] Write failing parser performance smoke test for the synthetic fixture under one second in `tests/test_coco_parser.py`

### Implementation for User Story 3

- [X] T036 [US3] Implement coverage, annotated-image, and unannotated-image summaries in `src/data/coco_parser.py`
- [X] T037 [US3] Implement category or defect-label distribution summaries in `src/data/coco_parser.py`
- [X] T038 [US3] Implement ROI availability summary from bounding-box or segmentation metadata in `src/data/coco_parser.py`
- [X] T039 [US3] Implement segmentation availability summary without segmentation mask rasterization in `src/data/coco_parser.py`
- [X] T040 [US3] Add optional quickstart/manual CLI convenience for local/private validation in `src/data/coco_parser.py`
- [X] T041 [US3] Run User Story 3 checks with `pytest tests/test_coco_parser.py -q`

**Checkpoint**: User Story 3 is complete when parser summaries cover all annotation gate outputs needed by later specs while staying out of detector export and ROI cropping.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the full parser against the contract, quickstart, and constitution.

- [X] T042 Validate parser behavior against `specs/003-coco-annotation-parser/contracts/coco-parser-contract.md`
- [X] T043 Run quickstart automated tests with `pytest tests/test_coco_parser.py -q`
- [X] T044 Confirm parser tests use only `tests/fixtures/synthetic_dataset/` and do not require private Krones dataset access
- [X] T045 When valid private dataset access is available, measure parsing runtime for the real private competition `train_annotations.json` and report whether it is under the one-minute target; otherwise document synthetic-only runtime validation
- [X] T046 Confirm generated diagnostics, if any, remain ignored and untracked using `git ls-files` and `git status --short --ignored`
- [X] T047 Confirm SPEC-003 does not implement detector-format export, ROI cropping, segmentation mask rasterization, model training, or test-label inference in `src/data/coco_parser.py`
- [X] T048 Update `docs/krones-final-implementation-plan.md` only if SPEC-003 implementation changes the approved roadmap behavior

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories because parser records and fixtures are shared.
- **User Story 1 (Phase 3)**: Depends on Foundational; delivers MVP COCO loading and normalized records.
- **User Story 2 (Phase 4)**: Depends on User Story 1 because relationship validation needs parsed records.
- **User Story 3 (Phase 5)**: Depends on User Story 2 because summaries depend on validated records and diagnostics.
- **Polish (Phase 6)**: Depends on all desired user stories.

### User Story Dependencies

- **US1 Parse COCO Annotation File**: MVP and prerequisite for relationship validation and summaries.
- **US2 Validate Annotation Relationships**: Builds on US1 normalized records.
- **US3 Provide Annotation-Derived Summaries**: Builds on US1 records and US2 validation outcomes.

### Within Each User Story

- Tests are written before implementation.
- Shared record structures are completed before parser behavior that returns them.
- Story checkpoint command runs before moving to the next story.

---

## Parallel Opportunities

- T001 and T002 can run in parallel.
- T007, T008, T009, and T010 can run in parallel after T005 and T006.
- T011, T012, T013, and T014 can run in parallel after foundational fixtures exist.
- T020, T021, T022, T023, and T024 can run in parallel after US1.
- T031, T032, T033, T034, and T035 can run in parallel after US2.

---

## Parallel Example: User Story 2

```text
Task: "Write failing orphaned annotation diagnostic test in tests/test_coco_parser.py"
Task: "Write failing unknown category diagnostic test in tests/test_coco_parser.py"
Task: "Write failing malformed bounding-box diagnostic test in tests/test_coco_parser.py"
Task: "Write failing duplicate image, annotation, and category identifier tests in tests/test_coco_parser.py"
Task: "Write failing no-test-label-inference regression test in tests/test_coco_parser.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational parser records and synthetic cases.
3. Complete Phase 3: User Story 1.
4. Stop and validate with `pytest tests/test_coco_parser.py -q`.

### Incremental Delivery

1. Add parser scaffold and synthetic COCO cases.
2. Add valid COCO loading and normalized record output for US1.
3. Add relationship and bounding-box diagnostics for US2.
4. Add derived annotation summaries for US3.
5. Run contract and quickstart validation.

### Final Validation

1. Run `pytest tests/test_coco_parser.py -q`.
2. Confirm no private Krones dataset access is required.
3. Confirm detector-format export, ROI cropping, segmentation mask rasterization, model training, and test-label inference are not implemented in SPEC-003.
4. Run `git ls-files`.
5. Run `git status --short --ignored`.
6. Verify private annotation files and generated diagnostics are not tracked or staged.

---

## Notes

- [P] tasks use different files and can run in parallel after prerequisites.
- Story labels map directly to the three user stories in `spec.md`.
- SPEC-003 returns normalized annotation records plus validation summaries.
- Detector-format export remains out of scope for SPEC-003.
- ROI cropping and preprocessing remain in SPEC-004.
