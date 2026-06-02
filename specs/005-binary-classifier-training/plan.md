# Implementation Plan: Binary Classifier Training

**Branch**: `006-binary-classifier-training` | **Date**: 2026-06-01 | **Spec**: [spec.md](./spec.md)

**Active Spec Directory**: `specs/005-binary-classifier-training`. The branch
name follows the generated Spec Kit branch sequence; no separate
`specs/006-binary-classifier-training` folder is part of this feature.

**Input**: Feature specification from `/specs/005-binary-classifier-training/spec.md`

## Summary

Implement the V1 fast binary classifier training path after SPEC-004 ROI and
preprocessing. The plan adds a lightweight EfficientNet-B0-compatible classifier
module, reproducible 80/20 stratified split handling, weighted binary loss,
F1-driven threshold search, best-checkpoint selection by validation F1, and
generated training artifacts under ignored output directories. Automated
validation uses synthetic fixtures only; private Krones data remains optional
for manual training runs.

## Technical Context

**Language/Version**: Python 3.11.

**Primary Dependencies**: Existing `torch`, `torchvision`, `timm`, `numpy`,
`pandas`, `scikit-learn`, `Pillow`, `PyYAML`, and `pytest` entries in
`requirements.txt`; SPEC-002 dataset helpers and SPEC-004 ROI/preprocessing
helpers.

**Storage**: Filesystem only. Generated artifacts are written under
`outputs/models/`, `outputs/reports/`, and `outputs/predictions/`, all ignored
by default except placeholders.

**Testing**: Pytest with synthetic image fixtures. Tests should cover split
reproducibility, binary-label validation, weighted-loss setup, threshold search,
metrics output, artifact writing, generated-output ignore validation, and
synthetic training smoke runtime under one minute.

**Target Platform**: Local Windows development first, with path/config behavior
compatible with Kaggle and Google Colab.

**Project Type**: Single Python computer-vision training project.

**Performance Goals**: Synthetic training smoke run completes under one minute.
The V1 model path remains lightweight enough for later fast inference
benchmarking.

**Constraints**: Input size is `384x384`; classifier output is one binary
probability for `1 = Not Reusable`; train/validation split is 80/20 stratified
with fixed seed; imbalance handling is weighted binary loss only; tied
best-F1 thresholds choose the lowest threshold; best checkpoint is selected by
validation F1 after threshold search.

**Scale/Scope**: SPEC-005 covers one V1 binary classifier baseline, validation
metrics, threshold search, and saved training artifacts. It does not generate
Kaggle submissions, run test-image inference, train detectors, implement
Grad-CAM, build dashboards, or add memory-bank behavior.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The plan preserves `0 = Reusable` and
  `1 = Not Reusable` labels and trains one binary classifier probability.
- Accuracy and speed: PASS. Validation F1-score drives threshold and checkpoint
  selection; the V1 baseline is lightweight and has a synthetic runtime gate.
- Hybrid ROI-first architecture: PASS. Training consumes SPEC-004 ROI and
  normalized preprocessing outputs; detector fallback and fusion are deferred.
- Reproducibility and leakage control: PASS. Fixed seed, 80/20 stratified split,
  saved threshold, saved metrics, and validation predictions are required; test
  labels and test-image inference are out of scope.
- Confidentiality: PASS. Private data remains local/private; generated model,
  report, and prediction artifacts are ignored by source control by default.
- Explainability and reporting: PASS. Confidence/probability and validation
  prediction facts are saved for later reporting, while Grad-CAM is deferred.
- Bias and imbalance: PASS. Class imbalance is handled with weighted binary
  loss and reported through class/confusion counts.
- Annotation and model-version strategy: PASS. This is SPEC-005 in the V1
  sequence after ROI preprocessing; detector and memory-bank behavior are out of
  scope.
- Modularity and compatibility: PASS. Model, losses, metrics, threshold search,
  and training orchestration remain separate modules and use config-driven
  paths.

## Project Structure

### Documentation (this feature)

```text
specs/005-binary-classifier-training/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- classifier-training-contract.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
configs/
`-- classifier.yaml

src/
|-- models/
|   `-- classifier.py
`-- training/
    |-- train_classifier.py
    |-- losses.py
    |-- metrics.py
    `-- threshold_search.py

tests/
|-- test_classifier.py
|-- test_training_pipeline.py
|-- test_training_metrics.py
`-- fixtures/
    `-- synthetic_dataset/
        |-- train.csv
        |-- train_images/
        |-- train_annotations.json
        `-- classifier_cases/
            |-- classifier_cases.json
            |-- train.csv
            |-- train_images/
            |-- positive/
            `-- negative/

outputs/
|-- models/
|-- reports/
`-- predictions/
```

**Structure Decision**: Keep model definition under `src/models/`, training
orchestration and scoring utilities under `src/training/`, and tests at the
existing root-level pytest layout. This matches the project foundation and keeps
inference/submission code out of SPEC-005.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md),
[contracts/classifier-training-contract.md](./contracts/classifier-training-contract.md),
and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: Design artifacts preserve the binary label contract and save validation
  probabilities without generating test predictions or submissions.
- PASS: Design artifacts select checkpoints and thresholds by validation F1 and
  record confusion counts for imbalance review.
- PASS: Design artifacts consume SPEC-004 normalized ROI preprocessing and do
  not duplicate preprocessing logic in classifier code.
- PASS: Generated outputs stay under ignored `outputs/` subdirectories.
- PASS: Synthetic tests can validate the feature without private Krones data.
- PASS: Detector training, hybrid inference, Grad-CAM, dashboard, memory bank,
  and Kaggle submission generation remain out of scope.
