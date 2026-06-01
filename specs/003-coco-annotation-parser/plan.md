# Implementation Plan: COCO Annotation Parser

**Branch**: `003-coco-annotation-parser` | **Date**: 2026-06-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-coco-annotation-parser/spec.md`

## Summary

Implement reusable COCO annotation parsing for the Krones bottle inspection
project. The feature will load configured COCO-style annotation metadata,
produce normalized image, annotation, category, and validation summary records,
and report relationship and metadata quality issues without requiring private
dataset access for automated tests. Detector-format export and ROI cropping are
explicitly out of scope.

## Technical Context

**Language/Version**: Python 3.11.

**Primary Dependencies**: Python standard library `json`, `dataclasses`, and
`pathlib`; existing `pytest` for validation. No new runtime dependency is
required for core parser behavior.

**Storage**: Filesystem only. Inputs are configured COCO JSON files. Optional
diagnostic outputs, if implemented, must remain under ignored generated-output
locations.

**Testing**: Pytest unit and integration-style tests using synthetic COCO
fixtures only; no private Krones data is required.

**Target Platform**: Local Windows development first, with path handling
compatible with Kaggle and Google Colab.

**Project Type**: Single Python computer-vision project with reusable data
modules.

**Performance Goals**: Parse and validate the synthetic fixture in under one
second and parse a normal competition annotation file in under one minute on
local development hardware when private dataset access is available.

**Constraints**: Do not hardcode private annotation paths. Do not infer test
labels. Do not export detector-format annotations in SPEC-003. Do not perform
ROI cropping in SPEC-003. Keep private annotation contents and derived
diagnostics local/private.

**Scale/Scope**: SPEC-003 covers normalized COCO image, annotation, category,
and validation summary records. It supports relationship validation, malformed
box detection, category distribution, ROI availability, and segmentation
availability. Detector conversion belongs to detector specs; ROI crop behavior
belongs to SPEC-004.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The parser does not change binary target
  semantics and does not create bottle predictions or test labels.
- Accuracy and speed: PASS. Valid annotation records reduce downstream training,
  ROI, and explainability errors; parser performance targets are bounded.
- Hybrid ROI-first architecture: PASS. The parser exposes ROI availability and
  region metadata for later ROI-first work but does not implement crop logic.
- Reproducibility and leakage control: PASS. Parser behavior is config-driven
  and testable from synthetic fixtures; test labels are not inferred.
- Confidentiality: PASS. Private annotation files and derived diagnostics remain
  local/private and are not committed publicly.
- Explainability and reporting: PASS. Parsed boxes, masks, categories, and ROI
  availability support later visual explanations and report assets.
- Bias and imbalance: PASS. Category and defect-label summaries support later
  rare-defect and imbalance review.
- Annotation and model-version strategy: PASS. This is SPEC-003 in V1 and owns
  reusable COCO parsing; detector export and ROI cropping are deferred.
- Modularity and compatibility: PASS. Parser behavior is isolated in data
  modules, avoids hardcoded paths, and supports local, Kaggle, and Colab path
  conventions.

## Project Structure

### Documentation (this feature)

```text
specs/003-coco-annotation-parser/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- coco-parser-contract.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
src/
`-- data/
    `-- coco_parser.py

tests/
|-- fixtures/
|   `-- synthetic_dataset/
|       `-- train_annotations.json
`-- test_coco_parser.py
```

**Structure Decision**: Keep reusable COCO parsing in `src/data/coco_parser.py`
so SPEC-002 audit behavior and future SPEC-004/SPEC-006 work can consume the
same normalized records without duplicating parser logic.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [quickstart.md](./quickstart.md), and
[contracts/coco-parser-contract.md](./contracts/coco-parser-contract.md).

## Post-Design Constitution Check

- PASS: Parser outputs normalized records and validation summaries only; no
  detector-format export or ROI cropping is included.
- PASS: Synthetic fixture validation supports automated tests without private
  Krones dataset access.
- PASS: Private annotation paths are configuration inputs, not hardcoded module
  constants.
- PASS: Parser diagnostics report relationship problems without turning
  annotations into test labels.
- PASS: Parser records support future ROI, detector, explainability, and
  imbalance analysis while preserving confidentiality.
