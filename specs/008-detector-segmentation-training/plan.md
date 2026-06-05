# Implementation Plan: Detector / Segmentation Training

**Branch**: `008-detector-segmentation-training` | **Date**: 2026-06-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/008-detector-segmentation-training/spec.md`

## Summary

Build the SPEC-008 detector/segmentation foundation from `train_annotations.json`: parse and audit COCO annotations, produce a stable defect-category mapping, convert valid bounding boxes to YOLO-format detector labels first, include no-annotation training images as empty-label background examples, generate deterministic disjoint detector splits and `data.yaml`, save visual audit figures, and document lightweight YOLOv8n/YOLO11n bbox training/evaluation commands. SPEC-008 does not implement hybrid inference, classifier-detector fusion, dashboard, Grad-CAM, feature memory bank, ensemble, distillation, YOLO segmentation training, or any change to the accepted V2B EfficientNet-B1 `analysis_only` classifier.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Python test environment.

**Primary Dependencies**: Existing `PyYAML`, `Pillow`, `pytest`, and standard library modules; optional `ultralytics` dependency already declared for manual YOLO training/evaluation command support.

**Storage**: Filesystem only. Generated detector artifacts are written under ignored `outputs/detector/dataset/`, `outputs/detector/reports/`, `outputs/detector/figures/`, and `outputs/detector/models/`.

**Testing**: Pytest with synthetic COCO annotations and synthetic images. Tests cover COCO audit extension, category mapping, YOLO label conversion, empty label files for no-annotation images, invalid bbox filtering, split disjointness, `data.yaml` creation, and sample visualization creation. Tests must not train a detector.

**Target Platform**: Local Windows development first, with path/config behavior compatible with Kaggle and Google Colab free-tier execution.

**Project Type**: Single Python computer-vision training and inference project.

**Performance Goals**: Detector preparation should be lightweight and deterministic. Detector runtime is not added to default inference; later SPEC-009 may use detector only for uncertain images.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; keep current accepted fast classifier as V2B EfficientNet-B1 `analysis_only` with public F1 `0.92121`; use only training annotations/images for detector dataset generation; include no-annotation training images with empty YOLO labels and report their count; do not use test images, sample-submission rows, Kaggle public feedback, hard-example mining outputs, or pseudo-labels; do not implement hybrid inference or fusion; do not train a detector automatically in tests.

**Scale/Scope**: SPEC-008 covers bbox-detector data preparation first, audit reporting, visualization, command wrappers, and evaluation report parsing/writing. Mask validation and visualization are supported when clean segmentation masks exist, but YOLO segmentation training is deferred until mask quality and coverage are proven. It prepares artifacts for SPEC-009 but does not change inference behavior.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. Detector categories explain defects but do not replace `0 = Reusable` and `1 = Not Reusable`.
- Accuracy and speed: PASS. Detector artifacts support future accuracy work, while default classifier inference remains unchanged and fast.
- Hybrid ROI-first architecture: PASS. SPEC-008 prepares detector/segmentation outputs for later conditional use but does not implement fusion.
- Reproducibility and leakage control: PASS. Detector split seed, category mapping, audit reports, data.yaml, and generated labels are saved; test images and public feedback are excluded.
- Confidentiality: PASS. Private annotations, labels, visualizations, reports, and model artifacts remain in ignored output paths.
- Explainability and reporting: PASS. Defect boxes/masks and category-colored overlays provide explainability inputs for later specs.
- Bias and imbalance: PASS. Category distribution, sparse categories, images without annotations, no-annotation empty-label counts, and multi-defect images are reported.
- Annotation and model-version strategy: PASS. This is SPEC-008 Detector / Segmentation Training using COCO annotations; SPEC-009 owns hybrid inference.
- Modularity and compatibility: PASS. COCO parsing, YOLO conversion, detector command support, visualization, and tests remain separate modules.

## Project Structure

### Documentation (this feature)

```text
specs/008-detector-segmentation-training/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- tasks.md
|-- contracts/
|   `-- detector-training-contract.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
configs/
`-- detector.yaml

src/
|-- data/
|   |-- coco_parser.py
|   `-- yolo_converter.py
`-- training/
    `-- train_detector.py

tests/
|-- test_coco_parser.py
`-- test_detector_yolo_conversion.py

outputs/
`-- detector/
    |-- dataset/
    |   |-- images/
    |   |   |-- train/
    |   |   `-- val/
    |   |-- labels/
    |   |   |-- train/
    |   |   `-- val/
    |   `-- data.yaml
    |-- reports/
    |-- figures/
    `-- models/
```

**Structure Decision**: Extend the existing COCO parser and add a focused YOLO conversion module plus detector command wrappers. Do not create a parallel detector application or modify classifier inference.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/detector-training-contract.md](./contracts/detector-training-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: Design preserves the binary target contract and keeps detector outputs explanatory.
- PASS: Default inference remains the accepted fast classifier path; detector use is deferred to SPEC-009.
- PASS: Dataset conversion uses deterministic disjoint splits, includes no-annotation training images as empty-label background examples, and excludes test images.
- PASS: Generated detector artifacts stay under ignored `outputs/detector/` paths.
- PASS: Detector training/evaluation commands are documented but not executed by tests.
