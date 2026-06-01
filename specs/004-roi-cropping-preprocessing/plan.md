# Implementation Plan: ROI Cropping and Preprocessing

**Branch**: `005-roi-cropping-preprocessing` | **Date**: 2026-06-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/004-roi-cropping-preprocessing/spec.md`

## Summary

Implement reusable ROI cropping and split-specific preprocessing for the Krones
bottle inspection pipeline. Training images use annotation-derived ROI metadata
when available, expanded by 10% and clipped to image bounds; otherwise they use
a safe square center crop and finally full-image resize if a crop is unsafe.
Validation and test preprocessing remain deterministic. Generated ROI review
samples are written under ignored output locations, and internal dark bottle
regions must be preserved.

## Technical Context

**Language/Version**: Python 3.11.

**Primary Dependencies**: Existing `Pillow`, `numpy`, `opencv-python`, and
`albumentations` entries in `requirements.txt`; existing SPEC-003
`src/data/coco_parser.py` records for annotation metadata. Core crop geometry
should remain dependency-light and testable with `Pillow`/`numpy`.

**Storage**: Filesystem only. Inputs are images and optional annotation records.
Generated review samples are stored under `outputs/figures/roi_samples/`, which
must remain ignored by source control.

**Testing**: Pytest with synthetic images and synthetic annotation metadata only.
No private Krones dataset access is required for automated tests.

**Target Platform**: Local Windows development first, with path behavior and
dependencies compatible with Kaggle and Google Colab.

**Project Type**: Single Python computer-vision project with reusable data
modules.

**Performance Goals**: ROI crop and deterministic preprocessing for synthetic
fixtures complete well under one second per test batch; per-image preprocessing
is lightweight enough for later fast classifier inference.

**Constraints**: Default output size is `384x384`. Annotation ROI crops expand
by 10% on each side and clip to image bounds. ROIs under 10% of image area are
unsafe and must fall back. Fallback is square center crop, then full-image
resize if unsafe. Do not remove dark pixels blindly. Do not train classifiers,
export detector data, generate submissions, or infer test labels in SPEC-004.

**Scale/Scope**: SPEC-004 covers ROI crop requests/results, split-specific
preprocessing profiles, deterministic validation/test preprocessing, bounded
training augmentations, generated ROI samples, and confidentiality checks for
generated artifacts.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. This feature does not classify images or alter
  binary label semantics; it prepares image inputs for later classifier specs.
- Accuracy and speed: PASS. ROI preprocessing reduces irrelevant border pixels
  and keeps transforms lightweight for later fast inference.
- Hybrid ROI-first architecture: PASS. This feature owns ROI preprocessing and
  explicitly preserves internal dark defects while deferring classifier,
  detector fallback, and fusion logic.
- Reproducibility and leakage control: PASS. Validation/test preprocessing is
  deterministic and no test labels are inferred.
- Confidentiality: PASS. Private images and private-derived ROI samples remain
  local/private and generated outputs are ignored by default.
- Explainability and reporting: PASS. Consistent ROI crops support later
  Grad-CAM, detector visualization, and report figures.
- Bias and imbalance: PASS. Dark and rare defect evidence is preserved so crop
  logic does not erase minority defect patterns.
- Annotation and model-version strategy: PASS. This is SPEC-004 in the V1
  pipeline after reusable COCO annotation parsing; detector export and classifier
  training are out of scope.
- Modularity and compatibility: PASS. ROI/preprocessing behavior is isolated in
  data modules and remains compatible with local Windows, Kaggle, and Colab.

## Project Structure

### Documentation (this feature)

```text
specs/004-roi-cropping-preprocessing/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- roi-preprocessing-contract.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
src/
`-- data/
    |-- roi.py
    |-- preprocessing.py
    `-- transforms.py

tests/
|-- fixtures/
|   `-- synthetic_dataset/
|       |-- roi_cases/
|       |-- train_images/
|       `-- train_annotations.json
`-- test_roi.py
```

**Structure Decision**: Keep crop geometry in `src/data/roi.py`, split-specific
preprocessing orchestration in `src/data/preprocessing.py`, and augmentation
profile construction in `src/data/transforms.py`. This follows the existing
data-module boundary and keeps training/model code out of SPEC-004.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [quickstart.md](./quickstart.md), and
[contracts/roi-preprocessing-contract.md](./contracts/roi-preprocessing-contract.md).

## Post-Design Constitution Check

- PASS: Design artifacts keep classifier training, detector export, submission
  generation, and test-label inference out of scope.
- PASS: ROI crop rules preserve internal dark image regions and avoid blind
  dark-pixel removal.
- PASS: Validation/test preprocessing is deterministic and reproducible.
- PASS: Generated ROI samples are under ignored output locations.
- PASS: Automated validation uses synthetic fixtures only.
