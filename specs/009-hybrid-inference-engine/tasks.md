# Tasks: Hybrid Inference Engine

**Input**: Design documents from `/specs/009-hybrid-inference-engine/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/hybrid-inference-contract.md`, `quickstart.md`

**Tests**: Pytest tasks are included because SPEC-009 changes inference, fusion, submission, reporting, leakage safeguards, and the project constitution requires applicable quality-gate checks.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add the hybrid configuration and file placeholders needed by all stories.

- [X] T001 Create hybrid inference config skeleton with classifier, detector, hybrid, and output path groups in `configs/hybrid_inference.yaml`
- [X] T002 [P] Create hybrid fusion module placeholder with public dataclass/function stubs in `src/inference/fusion.py`
- [X] T003 [P] Create hybrid submission CLI module placeholder with `search`, `submit`, and `run` command stubs in `src/inference/hybrid_submission.py`
- [X] T004 [P] Create hybrid fusion test module with shared synthetic dataframe builders in `tests/test_hybrid_fusion.py`
- [X] T005 [P] Create hybrid submission test module with temporary config/output fixtures in `tests/test_hybrid_submission.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core parsing, normalization, metrics, and config behavior required before any user story can be completed.

**Critical**: No user story work should begin until this phase is complete.

- [X] T006 Implement `HybridFusionConfig` and bounded threshold/margin validation in `src/inference/fusion.py`
- [X] T007 Implement stable `normalize_image_id` handling filenames and paths in `src/inference/fusion.py`
- [X] T008 Implement `load_best_threshold` for classifier threshold reports in `src/inference/fusion.py`
- [X] T009 Implement classifier prediction column normalization with alias support in `src/inference/fusion.py`
- [X] T010 Implement detector prediction normalization and image-level aggregation by strongest evidence in `src/inference/fusion.py`
- [X] T011 Implement binary metrics helper with accuracy, precision, recall, F1, TP, FP, TN, FN, and support counts in `src/inference/fusion.py`
- [X] T012 Implement YAML config loading and default hybrid output path resolution in `src/inference/hybrid_submission.py`
- [X] T013 [P] Add tests for classifier schema normalization and threshold-derived targets in `tests/test_hybrid_fusion.py`
- [X] T014 [P] Add tests for detector schema normalization and multi-row aggregation in `tests/test_hybrid_fusion.py`
- [X] T015 [P] Add tests for metric calculations including zero-division cases in `tests/test_hybrid_fusion.py`

**Checkpoint**: Foundation ready; user story implementation can now proceed.

---

## Phase 3: User Story 1 - Keep Fast Classifier Decisions (Priority: P1) MVP

**Goal**: Confident classifier predictions remain unchanged, and detector evidence is not used for confident rows.

**Independent Test**: Provide classifier scores outside the uncertainty band and detector rows that disagree; final hybrid predictions must equal classifier targets with `detector_used=false`.

### Tests for User Story 1

- [X] T016 [P] [US1] Add test that confident classifier positives and negatives remain unchanged in `tests/test_hybrid_fusion.py`
- [X] T017 [P] [US1] Add test that detector disagreement cannot override confident classifier rows in `tests/test_hybrid_fusion.py`
- [X] T018 [P] [US1] Add test that detector usage report counts only uncertain eligible rows in `tests/test_hybrid_fusion.py`

### Implementation for User Story 1

- [X] T019 [US1] Implement classifier uncertainty mask and confident-row preservation in `src/inference/fusion.py`
- [X] T020 [US1] Implement initial `apply_hybrid_fusion` output columns for classifier-only decisions in `src/inference/fusion.py`
- [X] T021 [US1] Implement detector-used count and ratio calculation for fused prediction frames in `src/inference/fusion.py`
- [X] T022 [US1] Run `pytest tests/test_hybrid_fusion.py -k "confident or usage"` and fix US1 regressions

**Checkpoint**: User Story 1 is independently functional and testable.

---

## Phase 4: User Story 2 - Review Uncertain Images With Detector Evidence (Priority: P2)

**Goal**: Uncertain classifier predictions can be rejected by detector evidence using always-faulty and per-category conditional-area rules, while missing or weak evidence falls back to classifier decisions.

**Independent Test**: Provide uncertain predictions with always-faulty defects, conditional defects above and below thresholds, missing detector evidence, and weak detector confidence; only usable rejection evidence may set `hybrid_target=1`.

### Tests for User Story 2

- [X] T023 [P] [US2] Add test for always-faulty defect rejection on uncertain rows in `tests/test_hybrid_fusion.py`
- [X] T024 [P] [US2] Add test for conditional defect rejection using per-category area thresholds in `tests/test_hybrid_fusion.py`
- [X] T025 [P] [US2] Add test for default conditional-area fallback threshold reporting in `tests/test_hybrid_fusion.py`
- [X] T026 [P] [US2] Add test that missing, invalid, or below-threshold detector evidence falls back to classifier in `tests/test_hybrid_fusion.py`
- [X] T027 [P] [US2] Add test that detector evidence cannot clear uncertain classifier rejections to `0 = Reusable` in `tests/test_hybrid_fusion.py`

### Implementation for User Story 2

- [X] T028 [US2] Implement detector rejection reason evaluation for always-faulty categories in `src/inference/fusion.py`
- [X] T029 [US2] Implement per-category conditional-area threshold evaluation in `src/inference/fusion.py`
- [X] T030 [US2] Implement default conditional-area threshold fallback tracking in `src/inference/fusion.py`
- [X] T031 [US2] Extend `apply_hybrid_fusion` to apply rejection-only detector authority on uncertain rows in `src/inference/fusion.py`
- [X] T032 [US2] Run `pytest tests/test_hybrid_fusion.py -k "faulty or conditional or fallback or clear"` and fix US2 regressions

**Checkpoint**: User Stories 1 and 2 both work independently.

---

## Phase 5: User Story 3 - Tune and Report Hybrid Behavior Safely (Priority: P3)

**Goal**: Tune hybrid parameters on validation overlap only, save reports and validation predictions, and generate strict Kaggle submission output without test-label leakage.

**Independent Test**: Provide partial-overlap validation predictions and test predictions with classifier confidence; search must use overlap rows only, reports must compare baselines on the same rows, and submission must contain exactly `image_id,target`.

### Tests for User Story 3

- [X] T033 [P] [US3] Add test that overlap report excludes classifier-only and detector-only validation rows from search in `tests/test_hybrid_fusion.py`
- [X] T034 [P] [US3] Add test that parameter search prefers <=30% detector usage for close-F1 candidates in `tests/test_hybrid_fusion.py`
- [X] T035 [P] [US3] Add test that parameter search evaluates per-category and default conditional-area threshold candidates in `tests/test_hybrid_fusion.py`
- [X] T036 [P] [US3] Add test that search reports classifier, detector, and hybrid metrics on the same overlap set in `tests/test_hybrid_fusion.py`
- [X] T037 [P] [US3] Add test that `hybrid_config.json`, `hybrid_metrics.json`, and `overlap_report.json` contain required contract fields in `tests/test_hybrid_submission.py`
- [X] T038 [P] [US3] Add test that `search` command writes config, metrics, overlap, and validation prediction artifacts in `tests/test_hybrid_submission.py`
- [X] T039 [P] [US3] Add test that `run` command writes both validation reports and strict submission artifacts in `tests/test_hybrid_submission.py`
- [X] T040 [P] [US3] Add test that final submission schema is exactly `image_id,target` in `tests/test_hybrid_submission.py`
- [X] T041 [P] [US3] Add test that missing classifier test confidence fails before submission creation in `tests/test_hybrid_submission.py`
- [X] T042 [P] [US3] Add test that CLI rejects sample-solution or test-label paths in `tests/test_hybrid_submission.py`
- [X] T062 [P] [US3] Add test that V2B test probability export writes 4418-compatible rows with `image_id`, `prob_bad`, `classifier_prediction`, and `target` columns in `tests/test_inference_v2.py`
- [X] T063 [P] [US3] Add test that hybrid config and reports use `outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv` instead of target-only `submission_v2b.csv` in `tests/test_hybrid_submission.py`
- [X] T064 [P] [US3] Add test that `v2b_vs_v4_diff.json` is generated and compares V2B classifier targets with V4 hybrid targets in `tests/test_hybrid_submission.py`

### Implementation for User Story 3

- [X] T043 [US3] Implement validation overlap report construction in `src/inference/fusion.py`
- [X] T044 [US3] Implement deterministic hybrid parameter search with classifier threshold, uncertainty margin, detector confidence, per-category conditional-area threshold, default area threshold, F1, recall, precision, detector usage, margin, and confidence tie-breaks in `src/inference/fusion.py`
- [X] T045 [US3] Implement classifier, detector, and hybrid metrics report generation on the overlap set in `src/inference/fusion.py`
- [X] T046 [US3] Implement `search` command to load inputs, run parameter search, and write reports/predictions under `outputs/hybrid/v4/` in `src/inference/hybrid_submission.py`
- [X] T047 [US3] Implement report schema validation for `hybrid_config.json`, `hybrid_metrics.json`, and `overlap_report.json` before writing in `src/inference/hybrid_submission.py`
- [X] T048 [US3] Implement classifier test confidence discovery and explicit missing-confidence error in `src/inference/hybrid_submission.py`
- [X] T049 [US3] Implement `submit` command to apply selected validation parameters to test predictions and write strict submission CSV in `src/inference/hybrid_submission.py`
- [X] T050 [US3] Implement `run` command to execute search then submission with one config in `src/inference/hybrid_submission.py`
- [X] T051 [US3] Add leakage guard checks for test labels, sample solution paths, and public-score-derived inputs in `src/inference/hybrid_submission.py`
- [X] T052 [US3] Implement optional hybrid benchmark/timing report generation under `outputs/hybrid/v4/benchmarks/` when timing inputs are available in `src/inference/hybrid_submission.py`
- [X] T053 [US3] Run `pytest tests/test_hybrid_fusion.py tests/test_hybrid_submission.py` and fix US3 regressions
- [X] T065 [US3] Implement V2B test probability export command writing `outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv` in `src/inference/submission.py`
- [X] T066 [US3] Extend classifier prediction normalization to accept `prob_bad` and `classifier_prediction` columns in `src/inference/fusion.py`
- [X] T067 [US3] Update `configs/hybrid_inference.yaml` to use `outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv` for classifier test predictions
- [X] T068 [US3] Implement `v2b_vs_v4_diff.json` report generation under `outputs/hybrid/v4/reports/` in `src/inference/hybrid_submission.py`
- [X] T069 [US3] Run V2B probability export and V4.1 hybrid inference without retraining V2B, retraining V3, using test labels, or submitting automatically

**Checkpoint**: All user stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the full feature, preserve existing behavior, and update supporting documentation.

- [X] T054 [P] Update `configs/hybrid_inference.yaml` comments/defaults with accepted V2B and SPEC-008 artifact path examples
- [X] T055 [P] Update `specs/009-hybrid-inference-engine/quickstart.md` if implemented CLI flags or outputs differ from the planned contract
- [X] T056 [P] Add shared output-directory creation and JSON serialization helpers for hybrid reports in `src/inference/hybrid_submission.py`
- [X] T057 Validate final submission output rejects extra diagnostic columns in `src/inference/hybrid_submission.py`
- [X] T058 Run `pytest tests/test_inference.py tests/test_submission.py tests/test_hybrid_fusion.py tests/test_hybrid_submission.py`
- [X] T059 Run `pytest tests/test_detector_yolo_conversion.py tests/test_detector_cli_entrypoints.py` to confirm SPEC-008 detector behavior remains unchanged
- [X] T060 Run a placeholder scan for unresolved clarification text, template markers, and unfinished notes in `specs/009-hybrid-inference-engine/` and `configs/hybrid_inference.yaml`
- [X] T061 Review `git diff --stat` and verify no private dataset files, generated outputs, model weights, or submission artifacts are tracked

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational; MVP.
- **User Story 2 (Phase 4)**: Depends on Foundational and reuses US1 fusion output columns.
- **User Story 3 (Phase 5)**: Depends on Foundational and uses US1/US2 fusion behavior for search/submission.
- **Polish (Phase 6)**: Depends on selected user stories being complete.

### User Story Dependencies

- **US1 Keep Fast Classifier Decisions**: MVP, no dependency on US2 or US3 after foundation.
- **US2 Review Uncertain Images With Detector Evidence**: Builds on US1 preservation behavior but remains testable with synthetic frames.
- **US3 Tune and Report Hybrid Behavior Safely**: Depends on fusion behavior from US1 and US2 for end-to-end search and submission.

### Parallel Opportunities

- T002-T005 can run in parallel after T001.
- T013-T015 can run in parallel after T006-T011.
- T016-T018 can run in parallel.
- T023-T027 can run in parallel.
- T033-T042 can run in parallel.
- T054-T056 can run in parallel during polish.

---

## Parallel Example: User Story 1

```text
Task: "T016 [P] [US1] Add test that confident classifier positives and negatives remain unchanged in tests/test_hybrid_fusion.py"
Task: "T017 [P] [US1] Add test that detector disagreement cannot override confident classifier rows in tests/test_hybrid_fusion.py"
Task: "T018 [P] [US1] Add test that detector usage report counts only uncertain eligible rows in tests/test_hybrid_fusion.py"
```

## Parallel Example: User Story 2

```text
Task: "T023 [P] [US2] Add test for always-faulty defect rejection on uncertain rows in tests/test_hybrid_fusion.py"
Task: "T024 [P] [US2] Add test for conditional defect rejection using per-category area thresholds in tests/test_hybrid_fusion.py"
Task: "T026 [P] [US2] Add test for missing, invalid, or below-threshold detector evidence fallback in tests/test_hybrid_fusion.py"
```

## Parallel Example: User Story 3

```text
Task: "T033 [P] [US3] Add test that overlap report excludes classifier-only and detector-only validation rows from search in tests/test_hybrid_fusion.py"
Task: "T035 [P] [US3] Add test that parameter search evaluates per-category and default conditional-area threshold candidates in tests/test_hybrid_fusion.py"
Task: "T038 [P] [US3] Add test that search command writes config, metrics, overlap, and validation prediction artifacts in tests/test_hybrid_submission.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational normalization/config/metrics work.
3. Complete Phase 3 US1 confident-classifier preservation.
4. Run the US1 pytest subset and inspect fused prediction outputs.
5. Stop and validate that detector disagreement cannot affect confident classifier rows.

### Incremental Delivery

1. Complete Setup + Foundation.
2. Deliver US1 so classifier-first behavior is protected.
3. Add US2 so uncertain rows can be rejected by detector evidence.
4. Add US3 so validation search, reports, and final submission are reproducible.
5. Run polish regression checks to confirm classifier-only and detector-preparation behavior remains unchanged.

### TDD Guidance

- Write the tests in each user-story phase before implementing the corresponding fusion or CLI behavior.
- Confirm each new test fails for the expected missing behavior before adding implementation.
- Re-run the story-specific pytest command at each checkpoint.

## Notes

- `[P]` tasks are parallelizable because they touch different files or independent test cases.
- `[US1]`, `[US2]`, and `[US3]` labels map directly to the prioritized user stories in `spec.md`.
- No task should retrain classifier or detector models.
- No task should read test labels, sample-solution labels, or public leaderboard feedback.
- Generated artifacts belong under ignored `outputs/hybrid/v4/` paths.

## V4.2 Result Notes

- V4.2 hybrid public score: `0.92181`.
- V2B classifier public score: `0.92121`.
- Public-score improvement: `+0.00060`.
- V4.2 changed 32 rows compared with V2B.
- All changed rows were `0 -> 1` and used the detector decision source.
- V4.2 is now the current best public submission.
- V3 detector-only remains weak at `0.74169`.
- Conclusion: hybrid detector fallback helps slightly, but the final `0.98` target requires V5 strong classifier work.

