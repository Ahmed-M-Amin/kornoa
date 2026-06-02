# Quickstart: Strong Classifier V2

## Prerequisites

- SPEC-001 through SPEC-006 are complete.
- V1 artifacts are available under `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1`.
- SPEC-006 hard-example files are available under `outputs/hard_examples/` and `outputs/reports/hard_example_summary.json`.
- Dataset paths are configured for local, Kaggle, or Colab execution.

## Validate V1 Baseline Inputs

```powershell
Test-Path artifacts/kaggle_v1_artifacts/outputs/kaggle_v1/reports/classifier_metrics.json
Test-Path artifacts/kaggle_v1_artifacts/outputs/kaggle_v1/reports/best_threshold.json
Test-Path outputs/hard_examples/false_positives.csv
Test-Path outputs/hard_examples/false_negatives.csv
Test-Path outputs/hard_examples/uncertain.csv
Test-Path outputs/hard_examples/high_loss_samples.csv
Test-Path outputs/benchmarks/v1_inference_benchmark.json
```

## Run V2A First

V2A is EfficientNet-B0 with the improved V2 recipe. It should establish whether imbalance handling, safer stronger augmentation, threshold search, and hard-example oversampling improve V1 without increasing backbone size.

```text
python -m src.training.train_classifier --config configs/classifier_v2.yaml --experiment v2a_effnet_b0_recipe
```

## Run Larger Lightweight Candidates

Run B1 and B2 only after V2A is reproducible.

```text
python -m src.training.train_classifier --config configs/classifier_v2.yaml --experiment v2b_effnet_b1
python -m src.training.train_classifier --config configs/classifier_v2.yaml --experiment v2c_effnet_b2
```

ConvNeXt-Tiny is optional and must be speed-checked.

```text
python -m src.training.train_classifier --config configs/classifier_v2.yaml --experiment v2_optional_convnext_tiny
```

## Optional 448x448 Experiment

Only run 448x448 after 384x384 candidates are complete and benchmarked.

```text
python -m src.training.train_classifier --config configs/classifier_v2.yaml --experiment v2_optional_448
```

## Generate V2 Submission

```text
python -m src.inference.submission --artifact-root outputs/kaggle_v2 --output-path outputs/kaggle_v2/submissions/submission_v2.csv
```

## Benchmark V2

```text
python -m src.inference.benchmark --artifact-root outputs/kaggle_v2 --output-path outputs/kaggle_v2/benchmarks/v2_inference_benchmark.json
```

## Required V2 Outputs

```text
outputs/kaggle_v2/models/classifier_best.pth
outputs/kaggle_v2/reports/classifier_metrics.json
outputs/kaggle_v2/reports/best_threshold.json
outputs/kaggle_v2/reports/v1_vs_v2_comparison.json
outputs/kaggle_v2/predictions/val_classifier_predictions.csv
outputs/kaggle_v2/submissions/submission_v2.csv
outputs/kaggle_v2/benchmarks/v2_inference_benchmark.json
```

## Acceptance Checks

- V2 selected validation F1 targets at least `0.93656`.
- V2 comparison reports the exact F1 delta versus V1.
- V2 selected inference time is no more than `2x` V1.
- Close-F1 candidates with absolute validation F1 difference `<= 0.002` prefer the faster model.
- Default hard-example strategy is `analysis_only`.
- Hard-example oversampling changes training composition only when `hard_example_strategy=oversample`.
- Any hard-example image ID used for oversampling is absent from the current V2 validation split.
- V2 training report includes loaded, eligible, validation-excluded, and oversampled hard-example counts plus train/validation disjointness confirmation.
- No test images are used for training, threshold tuning, or model selection.
