# Implementation Plan: Project Foundation

**Branch**: `001-project-foundation` | **Date**: 2026-05-31 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-project-foundation/spec.md`

## Summary

Create the safe project foundation for the Krones Industrial Bottle Inspection AI
repository. The implementation will add the root structure, safe placeholder
files, configuration templates, dependency entry point, output placeholders, and
ignore rules needed before dataset loading, model training, inference,
dashboarding, and reporting specs can proceed.

## Technical Context

**Language/Version**: Python 3.11 target for later project modules; this feature
creates repository structure and text/config files only.

**Primary Dependencies**: `torch`, `torchvision`, `timm`, `ultralytics`,
`opencv-python`, `albumentations`, `pandas`, `numpy`, `scikit-learn`,
`pycocotools`, `streamlit`, `grad-cam`, `onnxruntime`, `pytest`, `PyYAML`.

**Storage**: Filesystem only. Private dataset files and generated artifacts stay
outside source control through `.gitignore`.

**Testing**: Repository inspection plus pytest smoke tests for package imports
and config file readability.

**Target Platform**: Local Windows development, Kaggle notebooks, and Google
Colab free-tier execution.

**Project Type**: Single Python computer-vision project with notebooks,
training/inference modules, and Streamlit dashboard in later specs.

**Performance Goals**: No runtime model performance in this feature. Foundation
must support later F1-score, inference-time, and dashboard-responsiveness work.

**Constraints**: Create only safe foundation files. Do not commit private dataset
files, model weights, predictions, reports, figures, submissions, or hard-example
artifacts. Paths must be config-driven.

**Scale/Scope**: SPEC-001 only. Later specs handle dataset audit, COCO parsing,
ROI, classifier, detector, hybrid inference, memory bank, dashboard, submission,
and reporting.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. This foundation does not implement prediction,
  but reserves source/test areas for later `0 = Reusable` and `1 = Not Reusable`
  flows.
- Accuracy and speed: PASS. Config and output areas support later F1,
  inference-time, benchmark, and dashboard responsiveness work.
- Hybrid ROI-first architecture: PASS. Structure reserves modules for ROI,
  classifier, detector/segmentation, memory bank, fusion, and explainability.
- Reproducibility and leakage control: PASS. Planned configs, output categories,
  and ignore rules support reproducible runs without private artifact leakage.
- Confidentiality: PASS. `.gitignore` will exclude private dataset directories
  and generated private artifacts.
- Explainability and reporting: PASS. Structure reserves explainability,
  visualization, figures, and reports areas.
- Bias and imbalance: PASS. Structure reserves dataset audit and reporting areas
  for later class/defect distribution analysis.
- Annotation and model-version strategy: PASS. This is SPEC-001 and aligns with
  V1 foundation; later specs cover COCO annotation and model versions.
- Modularity and compatibility: PASS. Single-project structure separates data,
  models, training, inference, explainability, dashboard, utilities, tests,
  configs, notebooks, and outputs.

## Project Structure

### Documentation (this feature)

```text
specs/001-project-foundation/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- foundation-structure.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
configs/
|-- paths.yaml
|-- classifier.yaml
|-- detector.yaml
|-- inference.yaml
`-- dashboard.yaml

notebooks/
|-- 01_dataset_audit.ipynb
|-- 02_train_classifier.ipynb
|-- 03_train_detector.ipynb
|-- 04_distillation_memory.ipynb
`-- 05_kaggle_submission.ipynb

src/
|-- __init__.py
|-- data/
|   `-- __init__.py
|-- models/
|   `-- __init__.py
|-- training/
|   `-- __init__.py
|-- inference/
|   `-- __init__.py
|-- explainability/
|   `-- __init__.py
|-- dashboard/
|   `-- __init__.py
`-- utils/
    `-- __init__.py

outputs/
|-- .gitkeep
|-- models/
|   `-- .gitkeep
|-- predictions/
|   `-- .gitkeep
|-- figures/
|   `-- .gitkeep
|-- reports/
|   `-- .gitkeep
|-- submissions/
|   `-- .gitkeep
`-- hard_examples/
    `-- .gitkeep

tests/
|-- __init__.py
|-- test_project_foundation.py
|-- test_dataset.py
|-- test_coco_parser.py
|-- test_roi.py
|-- test_inference.py
`-- test_submission.py

.gitignore
README.md
requirements.txt
```

**Structure Decision**: Use a single Python project layout with domain-separated
modules under `src/`. SPEC-001 creates safe placeholders only; later specs fill
the module implementation files.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [quickstart.md](./quickstart.md), and
[contracts/foundation-structure.md](./contracts/foundation-structure.md).

## Post-Design Constitution Check

- PASS: No private dataset files or generated artifacts are introduced.
- PASS: `.gitignore` contract protects dataset and generated artifact categories.
- PASS: Config files support local Windows, Kaggle, and Colab path separation.
- PASS: Source/test/output structure supports later constitution-required
  modules, quality gates, reproducibility, and reporting.
