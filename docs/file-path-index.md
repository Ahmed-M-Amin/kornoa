# Project File Path Index

Generated: 2026-06-07 21:47:48 UTC

Refresh command:

```powershell
python scripts/update_file_path_index.py
```

Excluded by design: private datasets, `artifacts/`, `outputs/`, model weights, caches, virtual environments, secrets, and local agent/tool state.

## Configuration

- `configs/classifier.yaml`
- `configs/classifier_v2.yaml`
- `configs/dashboard.yaml`
- `configs/detector.yaml`
- `configs/detector_v3b_audit.yaml`
- `configs/hybrid_inference.yaml`
- `configs/inference.yaml`
- `configs/paths.yaml`
- `configs/paths_local_real.yaml`
- `configs/v5_strong_classifier.yaml`

## Documentation

- `docs/file-path-index.md`
- `docs/implementation.md`
- `docs/v4_hybrid_closeout.md`
- `docs/v5_strong_classifier_closeout.md`

## Maintenance Scripts

- `scripts/update_file_path_index.py`

## Notebooks

- `korona-v2b.ipynb`
- `korona.ipynb`
- `krones_v3_yolov8n_50epoch_clean.ipynb`
- `notebooks/01_dataset_audit.ipynb`
- `notebooks/02_train_classifier.ipynb`
- `notebooks/03_train_detector.ipynb`
- `notebooks/04_distillation_memory.ipynb`
- `notebooks/05_kaggle_submission.ipynb`

## Repository Root

- `.gitignore`
- `AGENTS.md`
- `GEMINI.md`
- `README.md`
- `requirements.txt`

## Source Code

- `src/__init__.py`
- `src/dashboard/__init__.py`
- `src/data/__init__.py`
- `src/data/audit.py`
- `src/data/coco_parser.py`
- `src/data/dataset.py`
- `src/data/preprocessing.py`
- `src/data/roi.py`
- `src/data/transforms.py`
- `src/data/yolo_converter.py`
- `src/explainability/__init__.py`
- `src/inference/__init__.py`
- `src/inference/benchmark.py`
- `src/inference/detector_audit.py`
- `src/inference/fusion.py`
- `src/inference/hybrid_submission.py`
- `src/inference/predict.py`
- `src/inference/submission.py`
- `src/models/__init__.py`
- `src/models/classifier.py`
- `src/training/__init__.py`
- `src/training/hard_example_mining.py`
- `src/training/losses.py`
- `src/training/metrics.py`
- `src/training/threshold_search.py`
- `src/training/train_classifier.py`
- `src/training/train_detector.py`
- `src/utils/__init__.py`
- `src/utils/config.py`

## Spec Kit Artifacts

- `specs/001-project-foundation/checklists/requirements.md`
- `specs/001-project-foundation/contracts/foundation-structure.md`
- `specs/001-project-foundation/data-model.md`
- `specs/001-project-foundation/plan.md`
- `specs/001-project-foundation/quickstart.md`
- `specs/001-project-foundation/research.md`
- `specs/001-project-foundation/spec.md`
- `specs/001-project-foundation/tasks.md`
- `specs/002-dataset-loading-validation/checklists/requirements.md`
- `specs/002-dataset-loading-validation/contracts/dataset-audit-contract.md`
- `specs/002-dataset-loading-validation/data-model.md`
- `specs/002-dataset-loading-validation/plan.md`
- `specs/002-dataset-loading-validation/quickstart.md`
- `specs/002-dataset-loading-validation/research.md`
- `specs/002-dataset-loading-validation/spec.md`
- `specs/002-dataset-loading-validation/tasks.md`
- `specs/003-coco-annotation-parser/checklists/requirements.md`
- `specs/003-coco-annotation-parser/contracts/coco-parser-contract.md`
- `specs/003-coco-annotation-parser/data-model.md`
- `specs/003-coco-annotation-parser/plan.md`
- `specs/003-coco-annotation-parser/quickstart.md`
- `specs/003-coco-annotation-parser/research.md`
- `specs/003-coco-annotation-parser/spec.md`
- `specs/003-coco-annotation-parser/tasks.md`
- `specs/004-roi-cropping-preprocessing/checklists/requirements.md`
- `specs/004-roi-cropping-preprocessing/contracts/roi-preprocessing-contract.md`
- `specs/004-roi-cropping-preprocessing/data-model.md`
- `specs/004-roi-cropping-preprocessing/plan.md`
- `specs/004-roi-cropping-preprocessing/quickstart.md`
- `specs/004-roi-cropping-preprocessing/research.md`
- `specs/004-roi-cropping-preprocessing/spec.md`
- `specs/004-roi-cropping-preprocessing/tasks.md`
- `specs/005-binary-classifier-training/checklists/requirements.md`
- `specs/005-binary-classifier-training/contracts/classifier-training-contract.md`
- `specs/005-binary-classifier-training/data-model.md`
- `specs/005-binary-classifier-training/plan.md`
- `specs/005-binary-classifier-training/quickstart.md`
- `specs/005-binary-classifier-training/research.md`
- `specs/005-binary-classifier-training/spec.md`
- `specs/005-binary-classifier-training/tasks.md`
- `specs/006-v1-evaluation-submission-benchmark-hard-example-mining/checklists/requirements.md`
- `specs/006-v1-evaluation-submission-benchmark-hard-example-mining/contracts/v1-evaluation-contract.md`
- `specs/006-v1-evaluation-submission-benchmark-hard-example-mining/data-model.md`
- `specs/006-v1-evaluation-submission-benchmark-hard-example-mining/plan.md`
- `specs/006-v1-evaluation-submission-benchmark-hard-example-mining/quickstart.md`
- `specs/006-v1-evaluation-submission-benchmark-hard-example-mining/research.md`
- `specs/006-v1-evaluation-submission-benchmark-hard-example-mining/spec.md`
- `specs/006-v1-evaluation-submission-benchmark-hard-example-mining/tasks.md`
- `specs/007-strong-classifier-v2/checklists/requirements.md`
- `specs/007-strong-classifier-v2/contracts/strong-classifier-v2-contract.md`
- `specs/007-strong-classifier-v2/data-model.md`
- `specs/007-strong-classifier-v2/plan.md`
- `specs/007-strong-classifier-v2/quickstart.md`
- `specs/007-strong-classifier-v2/research.md`
- `specs/007-strong-classifier-v2/spec.md`
- `specs/007-strong-classifier-v2/tasks.md`
- `specs/008-detector-segmentation-training/checklists/requirements.md`
- `specs/008-detector-segmentation-training/contracts/detector-training-contract.md`
- `specs/008-detector-segmentation-training/data-model.md`
- `specs/008-detector-segmentation-training/plan.md`
- `specs/008-detector-segmentation-training/quickstart.md`
- `specs/008-detector-segmentation-training/research.md`
- `specs/008-detector-segmentation-training/spec.md`
- `specs/008-detector-segmentation-training/tasks.md`
- `specs/009-hybrid-inference-engine/checklists/requirements.md`
- `specs/009-hybrid-inference-engine/contracts/hybrid-inference-contract.md`
- `specs/009-hybrid-inference-engine/data-model.md`
- `specs/009-hybrid-inference-engine/plan.md`
- `specs/009-hybrid-inference-engine/quickstart.md`
- `specs/009-hybrid-inference-engine/research.md`
- `specs/009-hybrid-inference-engine/spec.md`
- `specs/009-hybrid-inference-engine/tasks.md`
- `specs/010-v5-strong-classifier/checklists/requirements.md`
- `specs/010-v5-strong-classifier/contracts/v5-strong-classifier-contract.md`
- `specs/010-v5-strong-classifier/data-model.md`
- `specs/010-v5-strong-classifier/plan.md`
- `specs/010-v5-strong-classifier/quickstart.md`
- `specs/010-v5-strong-classifier/research.md`
- `specs/010-v5-strong-classifier/spec.md`
- `specs/010-v5-strong-classifier/tasks.md`

## Spec Kit Tooling

- `.specify/extensions.yml`
- `.specify/extensions/.registry`
- `.specify/extensions/git/commands/speckit.git.commit.md`
- `.specify/extensions/git/commands/speckit.git.feature.md`
- `.specify/extensions/git/commands/speckit.git.initialize.md`
- `.specify/extensions/git/commands/speckit.git.remote.md`
- `.specify/extensions/git/commands/speckit.git.validate.md`
- `.specify/extensions/git/config-template.yml`
- `.specify/extensions/git/extension.yml`
- `.specify/extensions/git/git-config.yml`
- `.specify/extensions/git/README.md`
- `.specify/extensions/git/scripts/bash/auto-commit.sh`
- `.specify/extensions/git/scripts/bash/create-new-feature.sh`
- `.specify/extensions/git/scripts/bash/git-common.sh`
- `.specify/extensions/git/scripts/bash/initialize-repo.sh`
- `.specify/extensions/git/scripts/powershell/auto-commit.ps1`
- `.specify/extensions/git/scripts/powershell/create-new-feature.ps1`
- `.specify/extensions/git/scripts/powershell/git-common.ps1`
- `.specify/extensions/git/scripts/powershell/initialize-repo.ps1`
- `.specify/feature.json`
- `.specify/init-options.json`
- `.specify/integration.json`
- `.specify/integrations/codex.manifest.json`
- `.specify/integrations/cursor-agent.manifest.json`
- `.specify/integrations/gemini.manifest.json`
- `.specify/integrations/opencode.manifest.json`
- `.specify/integrations/speckit.manifest.json`
- `.specify/memory/constitution.md`
- `.specify/scripts/powershell/check-prerequisites.ps1`
- `.specify/scripts/powershell/common.ps1`
- `.specify/scripts/powershell/create-new-feature.ps1`
- `.specify/scripts/powershell/setup-plan.ps1`
- `.specify/scripts/powershell/setup-tasks.ps1`
- `.specify/templates/checklist-template.md`
- `.specify/templates/constitution-template.md`
- `.specify/templates/plan-template.md`
- `.specify/templates/spec-template.md`
- `.specify/templates/tasks-template.md`
- `.specify/workflows/speckit/workflow.yml`
- `.specify/workflows/workflow-registry.json`

## Tests

- `tests/__init__.py`
- `tests/fixtures/synthetic_dataset/classifier_cases/classifier_cases.json`
- `tests/fixtures/synthetic_dataset/classifier_cases/negative/neg_000.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/negative/neg_001.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/negative/neg_002.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/negative/neg_003.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/negative/neg_004.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/positive/pos_000.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/positive/pos_001.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/positive/pos_002.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/positive/pos_003.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/positive/pos_004.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/sample_submission.csv`
- `tests/fixtures/synthetic_dataset/classifier_cases/test_images/.gitkeep`
- `tests/fixtures/synthetic_dataset/classifier_cases/train.csv`
- `tests/fixtures/synthetic_dataset/classifier_cases/train_images/neg_000.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/train_images/neg_001.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/train_images/neg_002.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/train_images/neg_003.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/train_images/neg_004.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/train_images/pos_000.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/train_images/pos_001.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/train_images/pos_002.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/train_images/pos_003.jpg`
- `tests/fixtures/synthetic_dataset/classifier_cases/train_images/pos_004.jpg`
- `tests/fixtures/synthetic_dataset/coco_cases/malformed.json`
- `tests/fixtures/synthetic_dataset/coco_cases/malformed_boxes.json`
- `tests/fixtures/synthetic_dataset/coco_cases/missing_collections.json`
- `tests/fixtures/synthetic_dataset/coco_cases/relationship_errors.json`
- `tests/fixtures/synthetic_dataset/roi_cases/dark_region.jpg`
- `tests/fixtures/synthetic_dataset/roi_cases/roi_cases.json`
- `tests/fixtures/synthetic_dataset/roi_cases/small_roi.jpg`
- `tests/fixtures/synthetic_dataset/roi_cases/wide_image.jpg`
- `tests/fixtures/synthetic_dataset/sample_submission.csv`
- `tests/fixtures/synthetic_dataset/test_images/img_005.jpg`
- `tests/fixtures/synthetic_dataset/test_images/img_006.jpg`
- `tests/fixtures/synthetic_dataset/train.csv`
- `tests/fixtures/synthetic_dataset/train_annotations.json`
- `tests/fixtures/synthetic_dataset/train_images/img_001.jpg`
- `tests/fixtures/synthetic_dataset/train_images/img_002.jpg`
- `tests/fixtures/synthetic_dataset/train_images/img_003.jpg`
- `tests/fixtures/synthetic_dataset/train_images/img_004.jpg`
- `tests/test_benchmark.py`
- `tests/test_classifier.py`
- `tests/test_classifier_v2.py`
- `tests/test_classifier_v5.py`
- `tests/test_coco_parser.py`
- `tests/test_dataset.py`
- `tests/test_dataset_audit.py`
- `tests/test_detector_audit.py`
- `tests/test_detector_cli_entrypoints.py`
- `tests/test_detector_yolo_conversion.py`
- `tests/test_file_path_index.py`
- `tests/test_hard_example_mining.py`
- `tests/test_hybrid_fusion.py`
- `tests/test_hybrid_submission.py`
- `tests/test_inference.py`
- `tests/test_inference_v2.py`
- `tests/test_inference_v5.py`
- `tests/test_preprocessing.py`
- `tests/test_project_foundation.py`
- `tests/test_roi.py`
- `tests/test_submission.py`
- `tests/test_training.py`
- `tests/test_training_v2.py`
- `tests/test_training_v5.py`
- `tests/test_transforms_v2.py`
- `tests/test_v2_comparison.py`
