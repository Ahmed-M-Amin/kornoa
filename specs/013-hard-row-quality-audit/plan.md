# Implementation Plan: Hard Row Quality Audit

**Branch**: `013-hard-row-quality-audit` | **Date**: 2026-06-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/013-hard-row-quality-audit/spec.md`

## Summary

Implement roadmap Phase 2 as an offline hard-row audit workflow over the governed hard-example pool, expected to be 442 rows after normalization. The feature will create one reproducible row-level audit table, preserve review context from prior locked same-split analysis, assign exactly one primary audit category per row, record concise audit rationale and optional evidence, and produce an aggregated summary that ranks the most fixable failure modes for the next training branch.

The technical approach should add a dedicated analysis lane for the audit rather than modifying training or submission code. It should reuse existing hard-example loading, row normalization, V2.1/V2.2 analysis helpers where practical, and write all audit outputs under ignored analysis paths.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Python test environment.

**Primary Dependencies**: Existing `pandas`, `PyYAML`, `pytest`, and current analysis helpers in `src.data.hard_examples`, `src.analysis.v2_1_error_intelligence`, and `src.analysis.v2_2_same_split_eval`.

**Storage**: Filesystem only. Inputs are CSV and JSON analysis artifacts; outputs are CSV and JSON audit reports under `outputs/analysis/hard_row_quality_audit/` or a configured equivalent.

**Testing**: Pytest with synthetic hard-row source inputs, category assignments, duplicate-row checks, optional-evidence absence cases, and summary aggregation assertions.

**Target Platform**: Local Windows development first, with config-driven paths compatible with Kaggle and Google Colab when artifacts are copied there.

**Project Type**: Single Python computer-vision analysis and reporting project.

**Performance Goals**: The audit should complete in seconds to low minutes for the governed hard-row pool and must add zero runtime to final inference because it is offline analysis only.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; operate on validation-derived hard rows only; do not use test labels or public leaderboard tuning; do not train; do not create submissions; do not auto-apply relabeling; assign exactly one primary category per audited row; preserve optional detector or image-quality evidence without making them mandatory.

**Scale/Scope**: One governed hard-row audit pool, five primary audit categories, one row-level audit table, one aggregate category summary, one ranked failure-mode summary, optional evidence attachment, and tests. Out of scope: classifier retraining, relabel application, detector-rule changes, Kaggle submission logic, dashboard UI, and hidden-test evaluation.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The audit explains why binary `0/1` decisions are unreliable on the hardest rows without changing the binary contract itself.
- Accuracy and speed: PASS. The feature improves downstream model-selection quality without adding inference cost.
- Hybrid ROI-first architecture: PASS. The audit may inspect ROI or detector-evidence quality as failure sources, but it does not change production ROI or hybrid execution logic.
- Reproducibility and leakage control: PASS. Governed hard-row sources, audit categories, rationale, and summary outputs can be saved and reviewed without test-label or public-score usage.
- Confidentiality: PASS. The feature remains local and competition-safe, with analysis artifacts written under ignored paths.
- Explainability and reporting: PASS. The core output is an explainable row-level rationale plus an aggregated failure-mode summary.
- Bias and imbalance: PASS. The feature focuses on the highest-risk subset where ambiguity, mislabels, and rare failure modes can distort later training.
- Annotation and model-version strategy: PASS. The audit is a V2.x analysis follow-up and does not imply a new model or annotation rewrite by default.
- Modularity and compatibility: PASS. Audit logic, evidence loading, and summary reporting remain separate from training, inference, and dashboard code; paths remain config-driven for local, Kaggle, and Colab use.

## Project Structure

### Documentation (this feature)

```text
specs/013-hard-row-quality-audit/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- hard-row-quality-audit-contract.md
|-- checklists/
|   `-- requirements.md
`-- tasks.md
```

### Source Code (repository root)

```text
configs/
`-- hard_row_quality_audit.yaml

src/
|-- analysis/
|   `-- hard_row_quality_audit.py
`-- data/
    `-- hard_examples.py

tests/
`-- test_hard_row_quality_audit.py

outputs/
`-- analysis/
    `-- hard_row_quality_audit/
        |-- audit/
        `-- reports/
```

**Structure Decision**: Add a dedicated `src/analysis/hard_row_quality_audit.py` workflow because the feature is a new analysis stage with its own contract and outputs. Reuse existing normalization and hard-example helpers instead of merging this logic into training or submission modules.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/hard-row-quality-audit-contract.md](./contracts/hard-row-quality-audit-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: The data model preserves stable hard-row identity, exact-one primary category assignment, optional evidence fields, and summary aggregation.
- PASS: The contract keeps the workflow validation-derived, analysis-only, and row-complete while allowing optional evidence absence.
- PASS: The design explicitly separates likely label issues, preprocessing issues, detector-evidence issues, ambiguous rows, and likely-correct-but-hard rows.
- PASS: Outputs remain under ignored analysis paths and guide future classifier, preprocessing, or detector decisions without introducing training or submission behavior.
