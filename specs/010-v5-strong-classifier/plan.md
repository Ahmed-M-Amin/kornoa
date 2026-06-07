# Implementation Plan: V5 Strong Classifier

**Branch**: `010-v5-strong-classifier` | **Date**: 2026-06-07 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/010-v5-strong-classifier/spec.md`

## Summary

Implement the V5 strong classifier stage after V4.2 hybrid inference showed only a small public-score gain over V2B. The plan keeps the existing classifier pipeline as the foundation, adds a V5 experiment configuration and output contract, trains ConvNeXt-Tiny at 512 image size as the primary candidate, and allows EfficientNet-B2 only as a resource-limit fallback. V5 uses the V2B-compatible train/validation split, validation-only threshold search, analysis-only hard examples by default, probability exports for validation and test images, strict submission generation, and benchmark comparison against V2B. V5 is accepted only if it beats the current V4.2 public score `0.92181` and runs no more than 2x slower than the accepted V2B classifier benchmark. Detector retraining, detector fusion changes, Grad-CAM, dashboard, feature memory bank, ensemble, distillation, public-score tuning, and automatic submission remain out of scope.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Python test environment.

**Primary Dependencies**: Existing `torch`, `torchvision`, `timm`, `numpy`, `pandas`, `scikit-learn`, `Pillow`, `PyYAML`, and `pytest`; existing SPEC-002 dataset loading; SPEC-004 ROI/preprocessing/transforms; SPEC-005/SPEC-007 classifier, losses, metrics, threshold search, hard-example reporting, training, inference, submission, and benchmark behavior.

**Storage**: Filesystem only. Inputs come from configured dataset paths and accepted V2B/V4.2 reference artifacts where needed for split compatibility and runtime comparison. Generated V5 outputs are written under ignored `outputs/kaggle_v5/v5_strong_classifier/models/`, `outputs/kaggle_v5/v5_strong_classifier/reports/`, `outputs/kaggle_v5/v5_strong_classifier/predictions/`, `outputs/kaggle_v5/v5_strong_classifier/submissions/`, and `outputs/kaggle_v5/v5_strong_classifier/benchmarks/`.

**Testing**: Pytest with synthetic classifier datasets, synthetic prediction exports, synthetic threshold reports, and generated-output path checks. Tests should cover V5 config validation, ConvNeXt-Tiny primary candidate selection, EfficientNet-B2 fallback gating, 512 image-size acceptance, V2B-compatible split reuse/failure, hard-example analysis-only default, explicit oversampling opt-in, validation/test prediction schemas, probability bounds, threshold JSON, metrics JSON, strict submission schema, benchmark comparison, generated-output ignore behavior, and no detector-retraining or public-score tuning paths.

**Target Platform**: Local Windows development first, with configuration behavior compatible with Kaggle and Google Colab free-tier GPU execution.

**Project Type**: Single Python computer-vision training, inference, and reporting project.

**Performance Goals**: V5 attempts to beat the current best public score `0.92181` from V4.2. Accepted V5 inference must be no more than 2x slower than the accepted V2B classifier benchmark. If ConvNeXt-Tiny at 512 exceeds memory or speed limits, EfficientNet-B2 is evaluated as the fallback candidate before V5 can be accepted or rejected.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; primary candidate is ConvNeXt-Tiny at 512; fallback is EfficientNet-B2 only when resource limits block the primary; no broad model sweep; reuse the V2B-compatible train/validation split; threshold selection uses validation data only; hard examples are analysis-only by default; hard-example oversampling requires explicit opt-in and split-safety reporting; no test labels, sample-solution labels, manual test inspection, or public-score-derived thresholds; final submission columns must be exactly `image_id,target`; validation prediction core columns must be `image_id,true_label,prob_bad,classifier_prediction,target`; test prediction columns must be `image_id,prob_bad,classifier_prediction,target`; generated outputs are not committed.

**Scale/Scope**: SPEC-010 covers V5 classifier configuration, training candidate selection, validation threshold search, validation/test probability export, classifier metrics reporting, runtime comparison, strict submission generation, and accepted-best decision reporting. It does not retrain or modify the V3 detector, change V4.2 fusion behavior, implement Grad-CAM/dashboard outputs, create a feature memory bank, add ensemble/teacher/distillation behavior, or perform automatic Kaggle submission.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. V5 preserves `0 = Reusable` and `1 = Not Reusable` in validation predictions, test predictions, threshold-derived targets, and strict submissions.
- Accuracy and speed: PASS. V5 targets a public-score improvement over V4.2 `0.92181`, requires validation F1 after threshold search, and caps accepted inference at no more than 2x the V2B classifier benchmark.
- Hybrid ROI-first architecture: PASS. V5 remains a classifier-only ROI-first stage and exports probabilities for later hybrid use; detector fallback and fusion changes are explicitly out of scope.
- Reproducibility and leakage control: PASS. V5 requires saved config, seed, V2B-compatible split reuse, validation predictions, best threshold, metrics, benchmark comparison, and no test/public-score tuning.
- Confidentiality: PASS. Private data, generated model weights, predictions, reports, submissions, and benchmarks stay under ignored local/Kaggle/Colab output directories.
- Explainability and reporting: PASS. V5 reports probabilities, threshold decisions, distributions, confusion metrics, selected model identity, runtime comparison, and public-score decision status; Grad-CAM/dashboard visualization remains deferred.
- Bias and imbalance: PASS. V5 includes weighted/focal objectives, class-balanced sampling, class distribution reporting, FP/FN review, and analysis-only hard-example reporting by default.
- Annotation and model-version strategy: PASS. This is SPEC-010/V5 Strong Classifier based on the latest V4.2 repair decision; detector/segmentation annotations are not required for V5 training.
- Modularity and compatibility: PASS. The plan extends existing config-driven classifier training, transform, threshold, inference, benchmark, and submission modules for local Windows, Kaggle, and Colab.

## Project Structure

### Documentation (this feature)

```text
specs/010-v5-strong-classifier/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- v5-strong-classifier-contract.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
configs/
|-- classifier_v2.yaml
`-- v5_strong_classifier.yaml

src/
|-- data/
|   `-- transforms.py
|-- models/
|   `-- classifier.py
|-- training/
|   |-- train_classifier.py
|   |-- losses.py
|   |-- metrics.py
|   |-- threshold_search.py
|   `-- hard_example_mining.py
`-- inference/
    |-- submission.py
    `-- benchmark.py

tests/
|-- test_classifier_v5.py
|-- test_training_v5.py
|-- test_inference_v5.py
|-- test_transforms_v2.py
|-- test_training_v2.py
`-- test_submission.py

outputs/
`-- kaggle_v5/
    `-- v5_strong_classifier/
        |-- models/
        |-- reports/
        |-- predictions/
        |-- submissions/
        `-- benchmarks/
```

**Structure Decision**: Reuse the existing classifier, transform, training, threshold, inference, submission, benchmark, and hard-example modules instead of creating a separate V5 application. Add V5 behavior through configuration, stricter artifact schemas, and focused tests so V2B/V4.2 behavior remains stable.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/v5-strong-classifier-contract.md](./contracts/v5-strong-classifier-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: Design artifacts preserve the binary label contract and strict submission schema.
- PASS: V5 candidate scope is bounded to ConvNeXt-Tiny at 512 plus EfficientNet-B2 resource fallback.
- PASS: V2B-compatible split reuse, validation-only threshold search, and public-score isolation are modeled explicitly.
- PASS: Hard examples are analysis-only by default; oversampling requires explicit opt-in plus split-safety reporting.
- PASS: Validation and test probability export schemas are explicit and compatible with later hybrid consumption.
- PASS: Runtime acceptance uses the accepted V2B benchmark as baseline and requires <= 2x accepted inference time.
- PASS: Generated V5 artifacts stay under ignored `outputs/kaggle_v5/v5_strong_classifier/` paths.
- PASS: SPEC-010 does not retrain detector candidates, change V4.2 fusion, or implement dashboard/Grad-CAM/memory/ensemble/distillation.
