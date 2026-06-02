# Implementation Plan: Dataset Loading and Validation

**Branch**: `002-dataset-loading-validation` | **Date**: 2026-06-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-dataset-loading-validation/spec.md`

## Summary

Implement dataset loading and validation for the Krones competition dataset.
The feature will read dataset paths from configuration, verify required files,
load training labels, discover train/test images, perform audit-only annotation
metadata checks, and generate dataset audit reports and figures without
committing private data or generated artifacts.

## Technical Context

**Language/Version**: Python 3.11.

**Primary Dependencies**: `pandas`, `PyYAML`, `Pillow` and existing
`opencv-python` for image metadata, `matplotlib` for audit figures, `pytest` for
validation.

**Storage**: Filesystem only. Inputs are private dataset files under configured
dataset roots. Outputs go to `outputs/reports/` and `outputs/figures/`, which
must remain ignored except safe placeholders.

**Testing**: Pytest unit and integration-style tests using temporary synthetic
dataset fixtures; no private dataset files are required for automated tests.

**Target Platform**: Local Windows development first, with path configuration
compatible with Kaggle and Google Colab.

**Project Type**: Single Python computer-vision project with reusable modules
and a dataset audit notebook.

**Performance Goals**: Audit a normal competition dataset in under five minutes
on local development hardware when private dataset access is valid, and record
the measured runtime in the generated report outputs.

**Constraints**: Do not hardcode the local private dataset path in core modules.
Do not infer or create test labels. Do not commit private images, annotations,
labels, sample visualizations, or generated audit artifacts.

**Scale/Scope**: SPEC-002 covers dataset loading, validation, audit reports, and
audit-only COCO metadata summaries. Reusable COCO parser APIs remain in
SPEC-003.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The feature validates labels for later
  `0 = Reusable` and `1 = Not Reusable` decisions without changing the target
  contract.
- Accuracy and speed: PASS. Audit outputs provide class/defect imbalance and
  dataset quality facts needed for later F1 and speed-aware decisions.
- Hybrid ROI-first architecture: PASS. The feature reports ROI availability but
  does not implement ROI cropping.
- Reproducibility and leakage control: PASS. Dataset paths are config-driven;
  test images are discovered only and no test labels are inferred.
- Confidentiality: PASS. Private dataset inputs and generated audit artifacts
  remain local/private and ignored by source control.
- Explainability and reporting: PASS. Audit figures support later reporting and
  review while respecting confidentiality.
- Bias and imbalance: PASS. Class distribution, defect-label distribution, and
  missing data summaries are required outputs.
- Annotation and model-version strategy: PASS. This is SPEC-002 in V1; it reads
  enough annotation metadata for audit summaries and leaves reusable parser APIs
  to SPEC-003.
- Modularity and compatibility: PASS. Planned modules separate configuration,
  dataset loading, audit logic, reporting, notebook usage, and tests.

## Project Structure

### Documentation (this feature)

```text
specs/002-dataset-loading-validation/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- dataset-audit-contract.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
configs/
`-- paths.yaml

src/
|-- data/
|   |-- dataset.py
|   `-- audit.py
`-- utils/
    `-- config.py

notebooks/
`-- 01_dataset_audit.ipynb

outputs/
|-- reports/
`-- figures/

tests/
|-- fixtures/
|   `-- synthetic_dataset/
|-- test_dataset.py
`-- test_dataset_audit.py
```

**Structure Decision**: Keep dataset discovery and audit behavior in
`src/data/`, configuration loading in `src/utils/`, and notebook usage as a thin
consumer of the reusable audit functions.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [quickstart.md](./quickstart.md), and
[contracts/dataset-audit-contract.md](./contracts/dataset-audit-contract.md).

## Post-Design Constitution Check

- PASS: Dataset loading remains config-driven and avoids hardcoded private paths.
- PASS: Audit-only annotation metadata boundary is explicit; SPEC-003 owns
  reusable parser APIs.
- PASS: Generated reports and figures are directed to ignored output locations.
- PASS: Synthetic fixtures allow automated tests without private dataset access.
- PASS: Test labels are not inferred or manually assigned.
