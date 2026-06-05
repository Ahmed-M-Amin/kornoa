# Quickstart: Strong Classifier V2

## Prerequisites

- SPEC-001 through SPEC-006 are complete.
- V1 artifacts are available under `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1`.
- SPEC-006 hard-example files are available under `outputs/hard_examples/` and `outputs/reports/hard_example_summary.json`.
- Dataset paths are configured for local, Kaggle, or Colab execution.

## Current Accepted Public Result

- Current best public model is V2B EfficientNet-B1, 384x384, `hard_example_strategy=analysis_only`, public F1 `0.92121`.
- V2A-remake is valid but slightly worse, public F1 `0.92093`.
- V2B-HE oversample is rejected: local validation F1 `0.974077`, public F1 `0.90293`, `used_for_oversampling_count=806`, `train_validation_disjoint=true`, speed about `25.16` images/sec.
- Do not change the accepted best model from V2B `analysis_only`.
- High local F1 alone is not enough to accept a Kaggle competition model.

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

## Completed 384x384 Runs

- V2A-remake: accepted as valid but not selected.
- V2B `analysis_only`: selected as the current best public model.
- V2B-HE oversample: rejected and must not be retried without fixing source selection and overfitting controls.

## Next Planned Experiment

V2C is the next documentation-only planned experiment: EfficientNet-B2, 384x384, focal loss, weighted sampler, `hard_example_strategy=analysis_only`. Benchmarking is required. Accept B2 only if public F1 improves meaningfully over V2B `0.92121` and runtime remains acceptable.

```text
python -m src.training.train_classifier --config configs/classifier_v2.yaml --experiment v2c_effnet_b2
```

## Hard-Example Source Safety

- Default hard-example strategy remains `analysis_only`.
- Heavy or training-composition-changing hard-example oversampling remains disabled by default.
- If an explicit V2 artifact source contains `val_classifier_predictions.csv` and `best_threshold.json`, hard examples must be generated from those V2 predictions first.
- An explicit V2 artifact source must not silently resolve nested `v1-artifacts/outputs/hard_examples`; warn or fail instead.
- Future V2B-HE-like reports must not contain `v1-artifacts` in `hard_example_source_used` when the config explicitly asks for V2B artifacts.

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
- Public F1 and benchmark runtime decide acceptance for new candidates; current accepted public best remains V2B `analysis_only` until a meaningful public improvement is measured.
- Close-F1 candidates with absolute validation F1 difference `<= 0.002` prefer the faster model.
- Default hard-example strategy is `analysis_only`.
- Hard-example oversampling changes training composition only when `hard_example_strategy=oversample`.
- Any hard-example image ID used for oversampling is absent from the current V2 validation split.
- V2 training report includes loaded, eligible, validation-excluded, and oversampled hard-example counts plus train/validation disjointness confirmation.
- No test images are used for training, threshold tuning, or model selection.
