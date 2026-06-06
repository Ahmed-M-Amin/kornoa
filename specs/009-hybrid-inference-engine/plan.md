# Implementation Plan: Hybrid Inference Engine

**Branch**: `009-hybrid-inference-engine` | **Date**: 2026-06-06 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/009-hybrid-inference-engine/spec.md`

## Summary

Implement the V4 hybrid inference layer after SPEC-008 detector preparation. The plan keeps the accepted V2B EfficientNet-B1 classifier as the default decision source, applies a configurable uncertainty band around the classifier threshold, and uses SPEC-008 detector evidence only for uncertain images. Detector authority is rejection-only: it can change uncertain images to `1 = Not Reusable` when an always-faulty defect or over-threshold conditional defect is found, but it cannot clear classifier rejections to `0 = Reusable`. Hybrid parameters are tuned on validation overlap only, with selected configurations preferring detector usage at or below 30% of overlap unless a higher-F1 tradeoff is explicitly reported. Grad-CAM, dashboard, feature memory bank, ensemble, distillation, retraining, and final report assets remain out of scope.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Python test environment.

**Primary Dependencies**: Existing `numpy`, `pandas`, `scikit-learn`, `PyYAML`, and `pytest`; existing SPEC-006/SPEC-007 classifier prediction, threshold, benchmark, and submission behavior; SPEC-008 detector prediction/evaluation artifacts and category mapping.

**Storage**: Filesystem only. Inputs come from accepted V2 classifier artifacts, SPEC-008 detector artifacts, and configured dataset paths. Generated hybrid outputs are written under ignored `outputs/hybrid/v4/predictions/`, `outputs/hybrid/v4/reports/`, `outputs/hybrid/v4/submissions/`, and `outputs/hybrid/v4/benchmarks/` when benchmark data is available.

**Testing**: Pytest with synthetic classifier and detector prediction tables. Tests cover column normalization, validation-overlap filtering, rejection-only fusion, confident-classifier preservation, missing detector fallback, per-category conditional-area thresholds with fallback reporting, parameter search tie-breaks, strict submission schema, missing classifier test confidence failure, and leakage safeguards.

**Target Platform**: Local Windows development first, with path/config behavior compatible with Kaggle and Google Colab free-tier inference.

**Project Type**: Single Python computer-vision inference and reporting project.

**Performance Goals**: Preserve classifier-first speed by applying detector evidence only to classifier-uncertain images. Selected validation configuration should use detector review for <= 30% of validation-overlap images unless an explicitly reported higher-F1 tradeoff exceeds the cap. Hybrid average runtime should be benchmarked against classifier-only runtime when timing inputs are available.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; keep V2B classifier as default; do not retrain classifier or detector; detector authority is rejection-only; detector is applied only to uncertain classifier predictions; non-uncertain classifier predictions are never overridden; missing detector evidence falls back to classifier; parameter search uses validation overlap only; no test labels, sample-solution labels, or Kaggle public score tuning; final submission columns must be exactly `image_id,target`; fail loudly when test-time classifier confidence is unavailable.

**Scale/Scope**: SPEC-009 covers hybrid prediction fusion, validation parameter search, overlap/metrics/config reports, strict submission generation, and hybrid benchmark reporting when timing inputs exist. It does not add new model training, detector dataset conversion, Grad-CAM, Streamlit dashboard, memory bank, ensemble teacher models, distillation, final Kaggle notebook, or report figures.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. Hybrid validation predictions and final submissions preserve `0 = Reusable` and `1 = Not Reusable`; diagnostic columns never enter the final submission.
- Accuracy and speed: PASS. Plan defines validation F1 selection, <= 30% preferred detector-usage ceiling, classifier-only comparison, and hybrid benchmark reporting when timing data exists.
- Hybrid ROI-first architecture: PASS. V4 uses the existing ROI/classifier path first, confidence gate second, detector evidence only for uncertain images, and rejection-only fusion logic.
- Reproducibility and leakage control: PASS. Selected thresholds, uncertainty margin, per-category area thresholds, input artifact paths, overlap rows, metrics, and reports are saved; tuning uses validation overlap only and excludes test labels/public score.
- Confidentiality: PASS. Private images, predictions, detector outputs, reports, and submissions remain under local/Kaggle/Colab-safe ignored output paths.
- Explainability and reporting: PASS. Hybrid outputs include classifier score, uncertainty flag, detector-used flag, decision source, defect reason, fallback usage, and timing inputs when available; Grad-CAM/dashboard rendering is deferred.
- Bias and imbalance: PASS. Reports track FP/FN fixes, incorrect-to-correct and correct-to-incorrect changes, detector usage, and conditional defect fallback usage.
- Annotation and model-version strategy: PASS. This is SPEC-009/V4 Hybrid Inference and consumes SPEC-008 detector artifacts without retraining.
- Modularity and compatibility: PASS. Fusion logic, submission orchestration, config, benchmarks, and tests are separate concerns with config-driven paths for local, Kaggle, and Colab.

## Project Structure

### Documentation (this feature)

```text
specs/009-hybrid-inference-engine/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- hybrid-inference-contract.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
configs/
|-- inference.yaml
`-- hybrid_inference.yaml

src/
`-- inference/
    |-- fusion.py
    |-- hybrid_submission.py
    |-- benchmark.py
    |-- predict.py
    `-- submission.py

tests/
|-- test_hybrid_fusion.py
`-- test_hybrid_submission.py

outputs/
`-- hybrid/
    `-- v4/
        |-- predictions/
        |-- reports/
        |-- submissions/
        `-- benchmarks/
```

**Structure Decision**: Extend the existing `src/inference/` package rather than creating a new hybrid application. Keep classifier prediction, detector evidence normalization, fusion, submission export, and benchmark/report writing in inference modules so training and detector-conversion code remain unchanged.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/hybrid-inference-contract.md](./contracts/hybrid-inference-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: Design preserves the binary target contract and strict submission schema.
- PASS: Rejection-only detector authority is encoded in entities, contract behavior, and quickstart acceptance checks.
- PASS: Parameter search uses validation overlap only and reports any detector-usage tradeoff above 30%.
- PASS: Missing test classifier confidence fails before submission generation instead of inferring uncertainty from target-only rows.
- PASS: Per-category conditional-area thresholds and fallback reporting are modeled explicitly.
- PASS: Generated hybrid artifacts stay under ignored `outputs/hybrid/v4/` paths.
- PASS: SPEC-009 does not retrain V2B, retrain detector candidates, implement dashboard/Grad-CAM, or introduce memory/ensemble/distillation.
