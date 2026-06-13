# Implementation Plan: Controlled Data Quality Audit Pipeline

**Branch**: `016-data-quality-audit` | **Date**: 2026-06-13 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/016-data-quality-audit/spec.md`

## Summary

Implement Spec 016 as a conservative, data-first audit workflow that evaluates every training row before any new classifier training is allowed. The workflow must merge training labels, available V2B and precomputed OOF prediction evidence, Spec 014 hard-row governance, image quality signals, duplicate/conflict evidence, and ranked suspicion signals into one master table, then split rows into clean, hard-valid, excluded, and manual-review buckets.

The technical approach extends the existing analysis stack rather than creating a notebook-only data review path. It will add a dedicated `src/analysis/data_quality_audit.py` CLI, a config at `configs/data_quality_audit.yaml`, audit reports under `outputs/analysis/data_quality_audit/`, and focused tests in `tests/test_data_quality_audit.py`. The workflow remains audit-only: no model training, no submission generation, no test-label use, and no leaderboard tuning.

## Technical Context

**Language/Version**: Python 3.11 target, consistent with the current local training and analysis environment.

**Primary Dependencies**: Existing `pandas`, `numpy`, `PyYAML`, `pytest`, `Pillow`, and current analysis/reporting patterns in `src/analysis`; optional `cleanlab`, optional `FiftyOne`, and optional image-hash/embedding support when available.

**Storage**: Filesystem only. Inputs are CSV, JSON, and image files from the local dataset, V2B artifacts, and Spec 014 outputs; outputs are CSV, JSON, and optional contact-sheet images under `outputs/analysis/data_quality_audit/`.

**Testing**: Pytest coverage centered on `tests/test_data_quality_audit.py`, plus compile validation for the new audit module.

**Target Platform**: Local Windows development first, with config-driven paths that remain compatible with Kaggle and Google Colab artifact layouts when those artifacts are copied locally.

**Project Type**: Single Python computer-vision analysis and reporting feature inside the existing repository.

**Performance Goals**: The audit should complete in practical local time on the training set, produce one row per training image, rank suspicious rows for manual review, and write all required outputs in one deterministic run without starting model training.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; consume precomputed OOF predictions when available but do not generate them in this feature; exclude all `manual_review_required` rows from `cleaned_training_manifest.csv`; no test labels; no public-leaderboard tuning; no original label edits; no submission creation; core workflow must not require optional third-party tools.

**Scale/Scope**: One master audit table for the full training set, four controlled bucket outputs, one cleaned training manifest, one review template, one summary-report package, one input inventory, and optional contact sheets. Out of scope: training, relabel execution, detector retraining, hybrid-inference redesign, and reviewer approval locking.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The plan preserves `0 = Reusable` and `1 = Not Reusable` in all audit and manifest outputs and does not alter source labels.
- Accuracy and speed: PASS. The feature improves future training signal quality without adding inference-time behavior; audit runtime remains bounded to offline analysis and reporting.
- Hybrid ROI-first architecture: PASS. ROI and crop quality are treated as first-class evidence, and ROI-risk rows are quarantined instead of being blindly reused as hard examples.
- Reproducibility and leakage control: PASS. The workflow is config-driven, records input inventory and audit-run traceability, consumes only precomputed prediction evidence, and forbids training, submission generation, test-label use, or leaderboard-driven tuning.
- Confidentiality: PASS. All dataset images, labels, predictions, and audit outputs remain local/private artifacts.
- Explainability and reporting: PASS. Every controlled bucket and suspicious-row output is backed by row-level evidence, ranked reports, and review artifacts.
- Bias and imbalance: PASS. The audit exposes bucket counts and exclusion reasons, which enables later inspection of class and hard-row concentration before retraining.
- Annotation and model-version strategy: PASS. This is a V2B/Spec 014-era data-governance feature and does not expand into detector, segmentation, or memory-bank scope.
- Modularity and compatibility: PASS. The plan adds a dedicated analysis CLI, config, tests, and outputs while staying compatible with the repo’s existing analysis structure and config-driven paths.

## Project Structure

### Documentation (this feature)

```text
specs/016-data-quality-audit/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- data-quality-audit-contract.md
`-- tasks.md
```

### Source Code (repository root)

```text
configs/
|-- hard_row_quality_audit.yaml
|-- hard_row_visual_review.yaml
`-- data_quality_audit.yaml

src/
`-- analysis/
    |-- hard_row_quality_audit.py
    |-- hard_row_visual_review.py
    |-- v2_1_fiftyone_review.py
    `-- data_quality_audit.py

tests/
|-- test_hard_row_quality_audit.py
|-- test_hard_row_visual_review.py
|-- test_v2_1_fiftyone_review.py
`-- test_data_quality_audit.py

outputs/
`-- analysis/
    |-- hard_row_quality_audit/
    |-- hard_row_visual_review/
    `-- data_quality_audit/
        |-- reports/
        `-- contact_sheets/
```

**Structure Decision**: Extend the existing `src/analysis` and `tests` audit patterns because the repository already contains row-audit, visual-review, and report-generation workflows. Spec 016 should add a dedicated data-quality audit layer that consumes V2B and Spec 014 evidence and emits training-safe manifests rather than inventing a separate notebook or training pipeline.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/data-quality-audit-contract.md](./contracts/data-quality-audit-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: The data model separates immutable source labels, optional prediction evidence, Spec 014 governance evidence, row-level risk signals, bucket decisions, and the cleaned manifest without mixing review-only rows into training-ready outputs.
- PASS: The contract enforces audit-only behavior, conservative manifest rules, and optional-tool fallback behavior in a way that matches the constitution’s leakage, confidentiality, and reproducibility rules.
- PASS: The design keeps ROI/crop risk and duplicate/conflict evidence explicit rather than hiding them inside a generic score, which aligns with the repository’s ROI-first and explainability requirements.
- PASS: Outputs remain config-driven, local/private, and modular, so the feature fits the repository’s current analysis workflows and Windows/Kaggle/Colab compatibility expectations.
