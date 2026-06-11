# Implementation Plan: Locked Same-Split Error Intelligence

**Branch**: `012-locked-same-split-intelligence` | **Date**: 2026-06-11 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/012-locked-same-split-intelligence/spec.md`

## Summary

Implement Phase 1 Locked Same-Split Error Intelligence as an offline comparison workflow that evaluates one selected V2.x candidate, with V2.2 as the first required example, against the locked original V2B validation rows. The workflow must preserve row identity, produce sectioned same-split metrics, emit row-level comparison buckets, maintain a rolling multi-candidate comparison table, and stay strictly analysis-only.

The technical approach extends the existing `src/analysis/v2_2_same_split_eval.py` lane instead of creating a parallel workflow. It will add config normalization, rolling comparison-table support, a clarified candidate decision rule, reusable reporting helpers, and synthetic pytest coverage while keeping all outputs under ignored analysis paths.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Python test environment.

**Primary Dependencies**: Existing `pandas`, `PyYAML`, `scikit-learn`, `pytest`, and current classifier inference utilities in `src.inference.predict`.

**Storage**: Filesystem only. Inputs are CSV/JSON/model artifacts; outputs are CSV and JSON reports under `outputs/analysis/v2_2_same_split_eval/` or a configured equivalent.

**Testing**: Pytest with synthetic locked-row predictions, candidate inference stubs, hard-example inputs, and rolling comparison-table assertions.

**Target Platform**: Local Windows development first, with config-driven paths compatible with Kaggle and Google Colab when artifacts are copied there.

**Project Type**: Single Python computer-vision analysis and reporting project.

**Performance Goals**: Offline same-split comparison should finish in seconds for normal validation CSV sizes and must add zero runtime to final inference because it is not part of the deployed prediction path.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; use validation-only locked rows; do not use test labels, sample-submission labels, or public leaderboard tuning; do not train; do not generate Kaggle submissions; evaluate one selected V2.x candidate at a time; mark a candidate `accepted` only when full locked-row performance improves and hard-example-only F1 does not regress by more than an absolute `0.01` tolerance from V2B.

**Scale/Scope**: One locked-row comparison workflow, three evaluation sections, four row-level comparison buckets, one rolling candidate comparison table, one threshold sweep, one candidate decision rule, and tests. Out of scope: new training, detector/rule fusion changes, leaderboard submission logic, dashboard UI, Grad-CAM generation, and multi-candidate inference in one run.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The workflow preserves `0 = Reusable` and `1 = Not Reusable` in targets, predictions, confusion counts, review outputs, and candidate decisions.
- Accuracy and speed: PASS. This feature improves model-selection quality and final evidence without adding production inference cost.
- Hybrid ROI-first architecture: PASS. The workflow consumes existing classifier outputs and optional later evidence only for analysis; it does not force detector execution or alter ROI/hybrid inference behavior.
- Reproducibility and leakage control: PASS. Inputs, thresholds, section rules, comparison-table updates, and safety checks are recorded; test-label use and public-score tuning are forbidden.
- Confidentiality: PASS. Predictions, checkpoints, and same-split reports remain local or in private competition-safe environments.
- Explainability and reporting: PASS. Outputs explain improvements and regressions through row-level buckets, sectioned metrics, and candidate comparison records.
- Bias and imbalance: PASS. Hard-example concentration, target distributions, and changed FP/FN balance are visible by section and by candidate.
- Annotation and model-version strategy: PASS. This is a V2.x analysis checkpoint, with V2.2 as the first required use case, not a new trained model.
- Modularity and compatibility: PASS. Analysis code, reporting, config handling, and tests stay separate from training and final inference; paths remain config-driven for local, Kaggle, and Colab use.

## Project Structure

### Documentation (this feature)

```text
specs/012-locked-same-split-intelligence/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- locked-same-split-contract.md
|-- checklists/
|   `-- requirements.md
`-- tasks.md
```

### Source Code (repository root)

```text
configs/
`-- v2_2_hard_examples.yaml

src/
`-- analysis/
    `-- v2_2_same_split_eval.py

tests/
`-- test_v2_2_same_split_eval.py

outputs/
`-- analysis/
    `-- v2_2_same_split_eval/
        |-- reports/
        `-- errors/
```

**Structure Decision**: Extend `src/analysis/v2_2_same_split_eval.py` because the repository already has the locked-row comparison lane there. Keep all new behavior in analysis/reporting so it cannot accidentally train, submit, or alter deployed prediction behavior.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/locked-same-split-contract.md](./contracts/locked-same-split-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: Data model preserves binary target semantics and distinguishes locked rows, candidate rows, section metrics, comparison buckets, and rolling candidate comparison records.
- PASS: Contract keeps the workflow validation-only, analysis-only, and one-candidate-at-a-time while supporting later optional evidence attachment.
- PASS: The `accepted` decision gate is explicit: full locked-row improvement plus no hard-example-only F1 regression beyond `0.01`.
- PASS: Rolling comparison-table behavior is documented without introducing leaderboard tuning, training, or submission generation.
- PASS: Outputs stay under ignored analysis paths and remain compatible with local, Kaggle, and Colab artifact movement.
