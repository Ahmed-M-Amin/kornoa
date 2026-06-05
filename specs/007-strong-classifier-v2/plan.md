# Implementation Plan: Strong Classifier V2

**Branch**: `006-binary-classifier-training` | **Date**: 2026-06-02 | **Spec**: [spec.md](./spec.md)

**Active Spec Directory**: `specs/007-strong-classifier-v2`. The current git branch name predates this spec directory; downstream Spec Kit commands should use `.specify/feature.json` and this active directory rather than inferring the feature from the branch name.

**Input**: Feature specification from `/specs/007-strong-classifier-v2/spec.md`

## Summary

Implement the V2 strong classifier stage after completed V1 training and SPEC-006 evaluation. The plan keeps V2 as a single lightweight classifier by default, improves the classifier recipe through imbalance handling, stronger safe augmentation, optional hard-example oversampling, and controlled backbone/image-size experiments, then selects the best candidate by validation F1 after threshold search. V2 must target at least +0.02 validation F1 over the V1 baseline of 0.91656 while treating 2x V1 inference time as the maximum allowed speed cost, not the target. Close validation F1 means absolute validation F1 difference <= 0.002; candidates within that tolerance select the faster model. Detector, segmentation, Grad-CAM, dashboard, feature memory bank, ensemble default, distillation, and hybrid inference remain out of scope.

## Experiment Results Through V2B-HE

- Current best public model: V2B EfficientNet-B1, 384x384, `hard_example_strategy=analysis_only`, public F1 `0.92121`.
- Valid but slightly worse run: V2A-remake, public F1 `0.92093`.
- Rejected run: V2B-HE oversample. Local validation F1 was `0.974077`, public F1 dropped to `0.90293`, `used_for_oversampling_count=806`, `train_validation_disjoint=true`, and inference speed stayed good at about `25.16` images/sec.
- Diagnosis: oversampling worked technically but overfit and did not generalize. The hard-example source resolved to nested `v1-artifacts` inside the V2B artifact package, not true V2B-generated hard examples.
- Planning consequence: high local F1 is not sufficient evidence of Kaggle/public generalization. Hard-example oversampling must remain disabled by default and must not be retried without fixed source selection plus overfitting controls.
- Next documented experiment: V2C EfficientNet-B2, 384x384, focal loss, weighted sampler, `hard_example_strategy=analysis_only`, with benchmark required. Accept B2 only if public F1 improves meaningfully over V2B `0.92121` and runtime remains acceptable.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Python test environment.

**Primary Dependencies**: Existing `torch`, `torchvision`, `timm`, `numpy`, `pandas`, `scikit-learn`, `Pillow`, `PyYAML`, and `pytest` entries in `requirements.txt`; SPEC-002 dataset loading; SPEC-004 ROI/preprocessing/transforms; SPEC-005 classifier, losses, metrics, threshold search, and training pipeline; SPEC-006 inference, submission, benchmark, and hard-example outputs.

**Storage**: Filesystem only. V1 baseline inputs come from `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1` and SPEC-006 outputs under ignored `outputs/`. V2 generated outputs are written under ignored `outputs/kaggle_v2/models/`, `outputs/kaggle_v2/reports/`, `outputs/kaggle_v2/predictions/`, `outputs/kaggle_v2/submissions/`, and `outputs/kaggle_v2/benchmarks/`.

**Testing**: Pytest with synthetic image fixtures and synthetic hard-example CSV fixtures. Tests should cover V2 configuration validation, focal loss behavior, weighted sampler construction, safe augmentation boundaries, hard-example strategy defaults, oversampling only when explicitly enabled, hard-example validation-split exclusion, split-disjointness reporting, threshold-search selection, V1-vs-V2 comparison reporting, no inference-time hard-example memory, generated-output ignore validation, and explicit V2 hard-example source safety.

**Target Platform**: Local Windows development first, with path/config behavior compatible with Kaggle and Google Colab free-tier GPU execution.

**Project Type**: Single Python computer-vision training and inference project.

**Performance Goals**: Selected V2 targets at least +0.02 validation F1 over V1's 0.91656 and should attempt stronger results toward 0.95+ when speed remains acceptable, but public/Kaggle generalization and runtime decide whether a run is accepted. Current accepted best public V2 is V2B EfficientNet-B1 `analysis_only` with public F1 `0.92121`. V2 inference must be no more than 2x slower than the V1 benchmark. Close F1 means absolute validation F1 difference <= 0.002; candidates within that tolerance prefer the faster model.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; default inference remains one classifier with ROI/preprocess, threshold, and target; 384x384 is the baseline image size; 448x448 is optional only after 384x384 experiments; hard examples are offline training/analysis inputs only; allowed hard-example strategies are `none`, `analysis_only`, and `oversample`; the default strategy is `analysis_only`; oversampling must be explicitly enabled with `hard_example_strategy=oversample`; any hard-example image ID used for oversampling must be excluded from the current V2 validation split; explicit V2 artifact hard-example sources must generate from V2 predictions before existing memory and must not silently resolve nested `v1-artifacts`; test images and Kaggle public feedback cannot be used for training, threshold tuning, or model selection; no detector, segmentation, Grad-CAM, dashboard, feature memory bank, default ensemble, distillation, or hybrid inference.

**Scale/Scope**: SPEC-007 covers V2 classifier training/evaluation, V2 submission generation through the existing classifier-only inference path, V2 benchmark reporting, and mandatory V1-vs-V2 comparison. It does not start later roadmap modules.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. V2 preserves `0 = Reusable` and `1 = Not Reusable` for labels, validation predictions, thresholds, submissions, and comparison reports.
- Accuracy and speed: PASS. V2 has a +0.02 minimum validation-F1 target, a 0.95+ ambition when speed allows, a maximum 2x V1 inference-time ceiling, and a defined close-F1 tie-break tolerance of <= 0.002.
- Hybrid ROI-first architecture: PASS. V2 uses existing ROI preprocessing and classifier inference only; detector fallback, fusion, and feature memory bank behavior are explicitly out of scope.
- Reproducibility and leakage control: PASS. V2 requires saved config, seed, model weights, validation predictions, threshold, metrics, comparison report, validation-only selection, split-disjointness reporting, and exclusion of V2 validation image IDs from hard-example oversampling. Test images and Kaggle public feedback are excluded from training and model selection.
- Confidentiality: PASS. Private data and derived artifacts stay in local/Kaggle/Colab environments and ignored output directories.
- Explainability and reporting: PASS. V2 reports probabilities, thresholds, confusion counts, error groups, speed, and V1-vs-V2 comparison. Visual explainability remains deferred.
- Bias and imbalance: PASS. V2 directly plans focal loss, weighted sampling, class-count reporting, analysis-only hard-example reporting by default, explicitly enabled hard-example oversampling, split-leakage controls, and FP/FN tracking.
- Annotation and model-version strategy: PASS. This is SPEC-007 Strong Classifier V2 in the repository's current final implementation plan; detector, memory bank, ensemble, distillation, and Grad-CAM remain later specs.
- Modularity and compatibility: PASS. Training, losses/sampling, transforms, inference, benchmarking, submission, and comparison reporting remain separate concerns with configurable paths for local, Kaggle, and Colab use.

## Project Structure

### Documentation (this feature)

```text
specs/007-strong-classifier-v2/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- strong-classifier-v2-contract.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
configs/
`-- classifier_v2.yaml

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
    |-- predict.py
    |-- submission.py
    `-- benchmark.py

tests/
|-- test_classifier_v2.py
|-- test_training_v2.py
|-- test_transforms_v2.py
|-- test_v2_comparison.py
`-- test_inference_v2.py

outputs/
`-- kaggle_v2/
    |-- models/
    |-- reports/
    |-- predictions/
    |-- submissions/
    `-- benchmarks/
```

**Structure Decision**: Reuse the existing classifier, training, transform, threshold, inference, submission, and benchmark modules rather than creating a parallel V2 subsystem. Add V2 configuration and targeted V2 behaviors behind explicit config/CLI choices so V1 behavior remains stable and normal inference stays classifier-only.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/strong-classifier-v2-contract.md](./contracts/strong-classifier-v2-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: Design artifacts preserve the binary label contract and use validation F1 after threshold search for model selection.
- PASS: V2 remains a single classifier by default, with detector, segmentation, Grad-CAM, dashboard, feature memory bank, ensemble default, distillation, and hybrid inference excluded.
- PASS: V1-vs-V2 comparison is mandatory and includes accuracy, threshold, FP/FN, uncertain counts, speed, backbone/model size, image size, and Kaggle public score when available.
- PASS: Hard examples are validation-derived analysis inputs by default, training oversampling inputs only when explicitly enabled, excluded from the current V2 validation split before oversampling, and do not run during normal V2 inference.
- PASS: Generated outputs stay under ignored `outputs/kaggle_v2/` paths.
- PASS: Local, Kaggle, and Colab path behavior remains configuration-driven.
