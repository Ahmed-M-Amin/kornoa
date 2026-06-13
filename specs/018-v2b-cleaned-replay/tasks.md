# Tasks: Spec 018 V2B Cleaned Replay

**Input**: Design documents from `specs/018-v2b-cleaned-replay/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/v2b-cleaned-replay-contract.md`, `quickstart.md`

**Tests**: Required. This feature gates cleaned-data training and must include dry-run, overlap, no-training, artifact, and comparison tests.

**Organization**: Tasks are grouped by user story so dry-run validation can ship as the MVP before training and comparison wiring.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files or only reads context.
- **[Story]**: User story label from `spec.md`.
- Every task references an exact repository path.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add the cleaned replay config and test harness scaffolding.

- [X] T001 Create cleaned replay config scaffold in `configs/v2b_cleaned_replay_training.yaml`
- [X] T002 Create cleaned replay pytest module and fixture builders in `tests/test_v2b_cleaned_replay_training.py`
- [X] T003 [P] Review existing Phase 3 dry-run helpers in `src/training/train_classifier.py`
- [X] T004 [P] Review Spec 017 output contract in `src/analysis/data_quality_decision_lock.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Extend config parsing and shared validation structures before user stories are implemented.

**Critical**: User story work depends on this phase.

- [X] T005 Write failing config-load test for `configs/v2b_cleaned_replay_training.yaml` in `tests/test_v2b_cleaned_replay_training.py`
- [X] T006 Add cleaned replay fields to `TrainingRunConfig` in `src/training/train_classifier.py`
- [X] T007 Extend `load_classifier_config` to parse `cleaned_replay`, `baseline`, and safety fields from `configs/v2b_cleaned_replay_training.yaml`
- [X] T008 Add reusable CSV image_id loading and overlap validation helpers in `src/training/train_classifier.py`
- [X] T009 Add reusable baseline artifact validation helper for cleaned replay in `src/training/train_classifier.py`
- [X] T010 Run foundational tests with `python -m pytest tests/test_v2b_cleaned_replay_training.py -q`

**Checkpoint**: Config and shared validation are ready.

---

## Phase 3: User Story 1 - Validate Cleaned Replay Inputs (Priority: P1) MVP

**Goal**: Provide dry-run validation that proves only the Spec 017 approved manifest can feed training and no unresolved rows, test labels, submissions, or leaderboard data are used.

**Independent Test**: Run `python -m src.training.train_classifier --config configs/v2b_cleaned_replay_training.yaml --dry-run-cleaned-replay` against fixtures and verify the dry-run report contains zero overlaps and no training side effects.

### Tests for User Story 1

- [X] T011 [US1] Write failing dry-run pass test with valid cleaned replay config in `tests/test_v2b_cleaned_replay_training.py`
- [X] T012 [US1] Write failing dry-run failure test for approved manifest overlapping auto-excluded rows in `tests/test_v2b_cleaned_replay_training.py`
- [X] T013 [US1] Write failing dry-run failure test for approved manifest overlapping needs-adjudication rows in `tests/test_v2b_cleaned_replay_training.py`
- [X] T014 [US1] Write failing dry-run failure test for approved manifest overlapping deferred rows in `tests/test_v2b_cleaned_replay_training.py`
- [X] T015 [US1] Write failing dry-run failure test for missing train image or label in `tests/test_v2b_cleaned_replay_training.py`
- [X] T016 [US1] Write failing dry-run no-training side-effect test by monkeypatching `run_training` in `tests/test_v2b_cleaned_replay_training.py`

### Implementation for User Story 1

- [X] T017 [US1] Implement `run_cleaned_replay_dry_run_validation` in `src/training/train_classifier.py`
- [X] T018 [US1] Write `cleaned_replay_dry_run_validation.json` under the configured reports directory in `src/training/train_classifier.py`
- [X] T019 [US1] Add `--dry-run-cleaned-replay` CLI argument in `src/training/train_classifier.py`
- [X] T020 [US1] Wire `--dry-run-cleaned-replay` into `main` without calling `run_training` in `src/training/train_classifier.py`
- [X] T021 [US1] Run dry-run tests with `python -m pytest tests/test_v2b_cleaned_replay_training.py -q`

**Checkpoint**: User Story 1 is independently functional and is the MVP.

---

## Phase 4: User Story 2 - Train a V2B-Style Cleaned Replay (Priority: P2)

**Goal**: Train using only the approved cleaned manifest and save full reproducibility artifacts.

**Independent Test**: Run a synthetic-smoke cleaned replay training test with monkeypatched training internals and verify model, predictions, threshold, metrics, config snapshot, manifest snapshot, and runtime metadata are written.

### Tests for User Story 2

- [X] T022 [US2] Write failing test that cleaned replay training filters examples to approved manifest rows in `tests/test_v2b_cleaned_replay_training.py`
- [X] T023 [US2] Write failing synthetic-smoke training artifact test for cleaned replay outputs in `tests/test_v2b_cleaned_replay_training.py`
- [X] T024 [US2] Write failing test that training refuses to start when dry-run safety validation fails in `tests/test_v2b_cleaned_replay_training.py`
- [X] T025 [US2] Write failing test that manifest snapshot and config snapshot are persisted in `tests/test_v2b_cleaned_replay_training.py`

### Implementation for User Story 2

- [X] T026 [US2] Implement approved-manifest filtering for cleaned replay training examples in `src/training/train_classifier.py`
- [X] T027 [US2] Ensure cleaned replay training excludes auto-excluded, adjudication, and deferred rows before split creation in `src/training/train_classifier.py`
- [X] T028 [US2] Persist approved manifest snapshot to `reports/approved_manifest_snapshot.csv` in `src/training/train_classifier.py`
- [X] T029 [US2] Persist cleaned replay run manifest to `reports/cleaned_replay_run_manifest.json` in `src/training/train_classifier.py`
- [X] T030 [US2] Persist runtime metadata to `benchmarks/runtime_metadata.json` in `src/training/train_classifier.py`
- [X] T031 [US2] Run training artifact tests with `python -m pytest tests/test_v2b_cleaned_replay_training.py -q`

**Checkpoint**: User Story 2 is independently functional and can produce a reproducible cleaned replay training run.

---

## Phase 5: User Story 3 - Compare Against Locked V2B Baseline (Priority: P3)

**Goal**: Produce a direct comparison report and accept/reject/inconclusive decision against the locked V2B baseline.

**Independent Test**: Use fixture metrics and baseline reports to verify the comparison report includes F1 delta, thresholds, validation row count, safety status, artifact paths, and correct decision.

### Tests for User Story 3

- [X] T032 [US3] Write failing comparison test that rejects cleaned replay when F1 is below locked V2B in `tests/test_v2b_cleaned_replay_training.py`
- [X] T033 [US3] Write failing comparison test that accepts cleaned replay only when F1 beats V2B and safety gates pass in `tests/test_v2b_cleaned_replay_training.py`
- [X] T034 [US3] Write failing comparison test for required report fields in `tests/test_v2b_cleaned_replay_training.py`

### Implementation for User Story 3

- [X] T035 [US3] Implement cleaned replay comparison record builder in `src/training/train_classifier.py`
- [X] T036 [US3] Implement `cleaned_replay_comparison.json` writer in `src/training/train_classifier.py`
- [X] T037 [US3] Integrate comparison report generation after successful cleaned replay training in `src/training/train_classifier.py`
- [X] T038 [US3] Run comparison tests with `python -m pytest tests/test_v2b_cleaned_replay_training.py -q`

**Checkpoint**: User Story 3 is independently functional and produces the promotion decision report.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, path index, and final verification.

- [X] T039 Update quickstart if implementation command or output names differ in `specs/018-v2b-cleaned-replay/quickstart.md`
- [X] T040 Update path index descriptions for Spec 018 config/test artifacts in `scripts/update_file_path_index.py`
- [X] T041 Refresh `docs/file-path-index.md` with `python scripts/update_file_path_index.py`
- [X] T042 Run Spec 018 tests with `python -m pytest tests/test_v2b_cleaned_replay_training.py -q`
- [X] T043 Run related Phase 3 regression tests with `python -m pytest tests/test_phase3_controlled_training.py -q`
- [X] T044 Run compile check with `python -m py_compile src/training/train_classifier.py`
- [X] T045 Run real dry-run validation with `python -m src.training.train_classifier --config configs/v2b_cleaned_replay_training.yaml --dry-run-cleaned-replay`
- [X] T046 Inspect `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/cleaned_replay_dry_run_validation.json` for safety fields
- [X] T047 Confirm no submission file was created by dry-run validation under `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/`
- [X] T048 Record implementation results in `specs/018-v2b-cleaned-replay/tasks.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies.
- **Phase 2 Foundational**: Depends on Phase 1 and blocks user stories.
- **Phase 3 US1**: Depends on Phase 2 and is the MVP.
- **Phase 4 US2**: Depends on US1 because training must be gated by dry-run validation.
- **Phase 5 US3**: Depends on US2 because comparison requires cleaned replay artifacts.
- **Phase 6 Polish**: Depends on selected user stories being complete.

### User Story Dependencies

- **US1 Validate Cleaned Replay Inputs**: Required first. It is independently testable.
- **US2 Train V2B-Style Cleaned Replay**: Requires the US1 dry-run gate.
- **US3 Compare Against Locked V2B Baseline**: Requires US2 metrics and artifact outputs.

### Parallel Opportunities

- T003 and T004 can run in parallel because they only review existing code.
- T011 through T016 can be drafted together after foundational config parsing is stable.
- T032 through T034 can be drafted while US2 implementation proceeds because they use fixture metrics.
- Documentation tasks T039 through T041 can run after command/output names stabilize.

## Parallel Example: User Story 1

```text
Task: "Write failing dry-run pass test with valid cleaned replay config in tests/test_v2b_cleaned_replay_training.py"
Task: "Write failing dry-run failure test for approved manifest overlapping auto-excluded rows in tests/test_v2b_cleaned_replay_training.py"
Task: "Write failing dry-run no-training side-effect test by monkeypatching run_training in tests/test_v2b_cleaned_replay_training.py"
```

## Parallel Example: User Story 3

```text
Task: "Write failing comparison test that rejects cleaned replay when F1 is below locked V2B in tests/test_v2b_cleaned_replay_training.py"
Task: "Write failing comparison test that accepts cleaned replay only when F1 beats V2B and safety gates pass in tests/test_v2b_cleaned_replay_training.py"
Task: "Write failing comparison test for required report fields in tests/test_v2b_cleaned_replay_training.py"
```

## Implementation Strategy

### MVP First

1. Complete Phase 1 setup.
2. Complete Phase 2 foundation.
3. Complete Phase 3 dry-run validation.
4. Stop and validate the real dry-run before training support is trusted.

### Incremental Delivery

1. US1 proves the cleaned manifest and safety gates are valid.
2. US2 adds controlled training and reproducibility artifacts.
3. US3 adds V2B comparison and promotion decision logic.
4. Polish verifies tests, regression tests, compile check, real dry-run, and docs.

### Final Validation Commands

```powershell
python -m pytest tests/test_v2b_cleaned_replay_training.py -q
python -m pytest tests/test_phase3_controlled_training.py -q
python -m py_compile src/training/train_classifier.py
python -m src.training.train_classifier --config configs/v2b_cleaned_replay_training.yaml --dry-run-cleaned-replay
```

## Implementation Results

- 2026-06-13: Implemented Spec 018 cleaned replay config, trainer mode, dry-run validation, approved-manifest filtering, runtime/config/manifest snapshots, and direct V2B baseline comparison.
- Validation results:
  `python -m pytest tests/test_v2b_cleaned_replay_training.py -q` -> 14 passed
  `python -m pytest tests/test_phase3_controlled_training.py -q` -> 16 passed
  `python -m py_compile src/training/train_classifier.py` -> passed
  `python -m src.training.train_classifier --config configs/v2b_cleaned_replay_training.yaml --dry-run-cleaned-replay` -> passed
- Real dry-run report:
  `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/cleaned_replay_dry_run_validation.json`
- Real dry-run safety summary:
  `approved_manifest_count=35211`, `auto_exclude_overlap_count=0`, `needs_adjudication_overlap_count=0`, `deferred_overlap_count=0`, `missing_train_label_count=0`, `missing_train_image_count=0`, `training_started=false`, `no_test_labels_used=true`, `no_submission_created=true`, `no_leaderboard_tuning=true`
- Follow-up note:
  Empty blocked-side CSVs from Spec 017 are now accepted when the file exists and still exposes the `image_id` column.
