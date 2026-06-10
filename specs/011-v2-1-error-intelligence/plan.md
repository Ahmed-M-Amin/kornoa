# Implementation Plan: V2.1 Error Intelligence

**Branch**: `011-v2-1-error-intelligence` | **Date**: 2026-06-10 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/011-v2-1-error-intelligence/spec.md`

## Summary

Implement V2.1 Error Intelligence as an offline analysis feature for the locked original V2B baseline. The plan builds one validation audit row per V2B validation image, validates that audit metrics match the locked baseline, attaches detector and image-quality evidence when available, splits false positives and false negatives into fixed review bands, and writes review-ready artifacts before any new training or Kaggle submission work.

The technical approach extends the existing `src/analysis/` lane rather than changing training or inference. It will add a config-driven V2B audit workflow, reusable validation helpers, synthetic pytest coverage, and CSV/JSON outputs under ignored analysis output paths.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Python test environment.

**Primary Dependencies**: Existing `pandas`, `PyYAML`, `scikit-learn`, `pytest`, and optional `Pillow`/image utilities already used by analysis review tooling.

**Storage**: Filesystem only. Inputs are configured CSV/JSON/image artifacts; outputs are CSV and JSON reports under `outputs/analysis/v2_1_error_intelligence/` or a configured equivalent.

**Testing**: Pytest with synthetic validation predictions, labels, baseline metrics, detector evidence, and image-quality evidence.

**Target Platform**: Local Windows development first, with config/path behavior compatible with Kaggle and Google Colab when artifacts are copied there.

**Project Type**: Single Python computer-vision analysis and reporting project.

**Performance Goals**: Offline audit should process the validation prediction set in seconds for normal project-scale CSV inputs; it must not add inference runtime because it is not part of final prediction.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; use validation labels only; do not use test labels, sample-solution labels, or public leaderboard feedback; do not train; do not generate Kaggle submissions; do not tune thresholds; high-confidence and near-threshold groups use fixed probability-distance bands from the clarified spec.

**Scale/Scope**: One V2B validation audit, aggregate summary, fixed error review groups, detector-evidence joins, image-quality joins, provenance report, and tests. Out of scope: model training, threshold optimization beyond using the locked threshold, public leaderboard analysis, detector rule fusion, V3/V4 changes, dashboard UI, Grad-CAM generation, and final submission generation.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The audit preserves `0 = Reusable` and `1 = Not Reusable` in labels, predictions, confusion counts, and target-distribution reports.
- Accuracy and speed: PASS. V2.1 improves model-selection quality before training and has no final inference runtime cost.
- Hybrid ROI-first architecture: PASS. This feature consumes classifier predictions and optional detector evidence for analysis only; it does not alter ROI, detector execution, or fusion behavior.
- Reproducibility and leakage control: PASS. Inputs, thresholds, metric checks, evidence availability, and output paths are recorded; test labels and public leaderboard tuning are explicitly forbidden.
- Confidentiality: PASS. Private labels, predictions, detector evidence, image diagnostics, and generated reports remain local or in private competition environments.
- Explainability and reporting: PASS. Audit rows explain errors through probability, threshold distance, error type, detector evidence, image-quality evidence, and review group.
- Bias and imbalance: PASS. Reports include FP/FN groups, over-rejected reusable bottles, class distributions, and prediction distributions.
- Annotation and model-version strategy: PASS. V2.1 is an analysis checkpoint for the original V2B baseline, not a new trained model.
- Modularity and compatibility: PASS. Analysis code, config, report writing, and tests stay separate from training and inference modules.

## Project Structure

### Documentation (this feature)

```text
specs/011-v2-1-error-intelligence/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- error-intelligence-contract.md
|-- checklists/
|   `-- requirements.md
`-- tasks.md
```

### Source Code (repository root)

```text
configs/
`-- v2_1_error_intelligence.yaml

src/
`-- analysis/
    |-- v2_1_error_intelligence.py
    |-- artifact_audit.py
    `-- v2b_error_review.py

tests/
`-- test_v2_1_error_intelligence.py

outputs/
`-- analysis/
    `-- v2_1_error_intelligence/
        |-- audit/
        |-- reports/
        `-- review_groups/
```

**Structure Decision**: Extend `src/analysis/` because V2.1 is an offline audit/reporting feature. Keep it separate from `src/training/` and `src/inference/` so it cannot accidentally train, submit, or change prediction behavior.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/error-intelligence-contract.md](./contracts/error-intelligence-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: Data model preserves binary target semantics and distinguishes labels, V2B probabilities, V2B predictions, and audit-only groups.
- PASS: Contract requires validation-only inputs and includes leakage safeguards for test labels, sample submissions, and public leaderboard feedback.
- PASS: Fixed near-threshold and high-confidence bands are encoded in the contract and quickstart acceptance checks.
- PASS: Missing optional detector or image-quality evidence is represented explicitly instead of dropping rows.
- PASS: Generated artifacts stay under ignored analysis output paths.
- PASS: Design does not introduce model training, detector fusion, Kaggle submission generation, public-score tuning, V3 work, dashboard UI, or Grad-CAM generation.
