# Implementation Plan: V1 Evaluation, Submission, Benchmark, and Hard-Example Mining

**Branch**: `006-binary-classifier-training` | **Date**: 2026-06-02 | **Spec**: [spec.md](./spec.md)

**Active Spec Directory**: `specs/006-v1-evaluation-submission-benchmark-hard-example-mining`. The current git branch name predates this spec directory; downstream Spec Kit commands should use `.specify/feature.json` and this active directory rather than inferring the feature from the branch name.

**Input**: Feature specification from `/specs/006-v1-evaluation-submission-benchmark-hard-example-mining/spec.md`

## Summary

Implement the V1 post-training evaluation path after SPEC-005. The plan loads completed V1 artifacts from the concrete run root `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1` by default, uses the saved threshold for test-image prediction, creates a sample-aligned `submission_v1.csv`, records V1 inference speed, and automatically regenerates offline hard-example files from validation predictions. If the parent root `artifacts/kaggle_v1_artifacts/outputs` is supplied, the workflow resolves its `kaggle_v1` child as the concrete run root. Hard-example mining remains an explicit evaluation workflow and does not run during normal inference, submission generation, or benchmarking.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Python test environment.

**Primary Dependencies**: Existing `torch`, `numpy`, `Pillow`, and `pytest` entries in `requirements.txt`; SPEC-004 ROI/preprocessing helpers; SPEC-005 classifier factory and saved artifact contracts.

**Storage**: Filesystem only. Inputs are completed V1 artifacts under the concrete run root `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1` unless overridden. The concrete run root must contain `models/classifier_effnet_b0_best.pth`, `reports/classifier_metrics.json`, `reports/best_threshold.json`, and `predictions/val_classifier_predictions.csv`. The optional parent root `artifacts/kaggle_v1_artifacts/outputs` resolves to its `kaggle_v1` child. Generated outputs are written under ignored `outputs/submissions/`, `outputs/benchmarks/`, `outputs/hard_examples/`, and `outputs/reports/` locations.

**Testing**: Pytest with synthetic image fixtures and a tiny classifier checkpoint path for fast, offline tests. Tests cover threshold loading, checkpoint prediction, sample-aligned submission generation, benchmark report writing, hard-example CSV generation, and no hard-example side effects during inference/submission/benchmark workflows.

**Target Platform**: Local Windows development first, with path/config behavior compatible with Kaggle and Google Colab.

**Project Type**: Single Python computer-vision inference/evaluation project.

**Performance Goals**: Benchmark report records total runtime, image count, average milliseconds per image, and images per second for every run. Normal V1 inference does not invoke detector, Grad-CAM, feature memory bank, or hard-example mining.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; use saved V1 threshold only; preserve sample submission row order and label column contract; mine hard examples from validation predictions only; high-loss proxy ranking uses saved validation probabilities and labels because per-sample BCE loss was not saved; no detector, segmentation, dashboard, Grad-CAM, V2 training, feature memory bank, ensemble, distillation, or hybrid inference behavior.

**Scale/Scope**: SPEC-006 covers V1 test-image prediction, first submission generation, inference benchmark reporting, and offline validation hard-example mining. It does not retrain V1, tune thresholds, infer labels for training/validation data outside the specified artifacts, or implement later roadmap modules.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. All predictions, submissions, and hard-example categories preserve `0 = Reusable` and `1 = Not Reusable`.
- Accuracy and speed: PASS. The saved V1 F1-selected threshold is used, and benchmark timing is a required output.
- Hybrid ROI-first architecture: PASS. SPEC-006 uses the existing ROI/preprocessing plus classifier path only; detector fallback, fusion, and feature memory bank behavior remain out of scope.
- Reproducibility and leakage control: PASS. The workflow consumes saved model, threshold, metrics, and validation predictions; hard examples use validation predictions only and never test labels.
- Confidentiality: PASS. Private-derived V1 artifacts and generated submissions/reports remain local and ignored.
- Explainability and reporting: PASS. Probability, binary target, benchmark report, and hard-example analysis artifacts are saved; Grad-CAM and detector visual explanations are deferred.
- Bias and imbalance: PASS. Hard-example mining exposes false positives, false negatives, uncertain cases, and high-loss proxy samples for later V2 analysis.
- Annotation and model-version strategy: PASS. This is the SPEC-006 V1 post-training checkpoint. The constitution's older future-spec ordering is superseded by the user-provided final implementation plan for this repository sequence.
- Modularity and compatibility: PASS. Prediction, submission, benchmark, and hard-example mining are planned as separate modules with configurable paths for local, Kaggle, and Colab use.

## Project Structure

### Documentation (this feature)

```text
specs/006-v1-evaluation-submission-benchmark-hard-example-mining/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- v1-evaluation-contract.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
src/
|-- inference/
|   |-- predict.py
|   |-- submission.py
|   `-- benchmark.py
`-- training/
    `-- hard_example_mining.py

tests/
|-- test_inference.py
|-- test_submission.py
|-- test_benchmark.py
`-- test_hard_example_mining.py

outputs/
|-- submissions/
|-- benchmarks/
|-- hard_examples/
`-- reports/
```

**Structure Decision**: Keep runtime prediction, submission generation, and benchmarking under `src/inference/`; keep validation-only hard-example mining under `src/training/` because it is an offline training/evaluation artifact for later V2 improvement. Tests stay in the existing root-level pytest layout. No notebooks or old spec folders are changed by this feature.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/v1-evaluation-contract.md](./contracts/v1-evaluation-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: Design artifacts preserve the binary label contract and consume the saved V1 threshold for every decision.
- PASS: Benchmarking is explicitly separated from hard-example mining and normal inference.
- PASS: Hard-example mining uses validation predictions only and defines high-loss proxy ranking from existing saved CSV fields. Existing manual hard-example CSVs under `artifacts/kaggle_v1_artifacts/outputs/hard_examples` may be inspected separately but are not required inputs.
- PASS: Generated outputs stay under ignored output directories.
- PASS: Detector, segmentation, Grad-CAM, dashboard, V2 training, feature memory bank, ensemble, distillation, and hybrid inference remain out of scope.
