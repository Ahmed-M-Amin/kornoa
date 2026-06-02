# Quickstart: V1 Evaluation, Submission, Benchmark, and Hard-Example Mining

## Automated Validation

Run the SPEC-006 test target once tasks are implemented:

```powershell
pytest tests/test_inference.py tests/test_submission.py tests/test_benchmark.py tests/test_hard_example_mining.py -q
```

Expected behavior:

- V1 threshold loads from the saved threshold JSON.
- V1 checkpoint loads for image prediction.
- Submission rows match `sample_submission.csv` order and binary label contract.
- Benchmark JSON records image count and timing metrics.
- Hard-example mining creates false positive, false negative, uncertain, high-loss proxy CSVs, and a summary JSON from validation predictions only.
- Normal inference, submission, and benchmark workflows do not create hard-example files.

## Generate V1 Submission

Default completed V1 artifact root:

```text
artifacts/kaggle_v1_artifacts/outputs/kaggle_v1
```

The parent artifact root is also accepted:

```text
artifacts/kaggle_v1_artifacts/outputs
```

When the parent root is supplied, the workflow detects `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1` and uses it as the concrete V1 run artifact root.

The concrete V1 run artifact root must contain:

```text
models/classifier_effnet_b0_best.pth
reports/classifier_metrics.json
reports/best_threshold.json
predictions/val_classifier_predictions.csv
```

Example command after implementation:

```powershell
python -m src.inference.submission --dataset-root 1st-krones-vision-ai-challenge --artifact-root artifacts/kaggle_v1_artifacts/outputs/kaggle_v1 --output-path outputs/submissions/submission_v1.csv
```

Expected output:

```text
outputs/submissions/submission_v1.csv
```

## Benchmark V1 Inference

```powershell
python -m src.inference.benchmark --image-dir 1st-krones-vision-ai-challenge/test_images --artifact-root artifacts/kaggle_v1_artifacts/outputs/kaggle_v1 --output-path outputs/benchmarks/v1_inference_benchmark.json
```

Expected output:

```text
outputs/benchmarks/v1_inference_benchmark.json
```

## Mine V1 Hard Examples

```powershell
python -m src.training.hard_example_mining --artifact-root artifacts/kaggle_v1_artifacts/outputs/kaggle_v1 --output-dir outputs/hard_examples --summary-path outputs/reports/hard_example_summary.json
```

Expected outputs:

```text
outputs/hard_examples/false_positives.csv
outputs/hard_examples/false_negatives.csv
outputs/hard_examples/uncertain.csv
outputs/hard_examples/high_loss_samples.csv
outputs/reports/hard_example_summary.json
```

Existing manually created hard-example CSVs may also be present under:

```text
artifacts/kaggle_v1_artifacts/outputs/hard_examples
```

SPEC-006 should be able to regenerate them from:

```text
artifacts/kaggle_v1_artifacts/outputs/kaggle_v1/predictions/val_classifier_predictions.csv
```

## Scope Guard

SPEC-006 should not train V2, tune thresholds from Kaggle score, implement detector/segmentation, run Grad-CAM, build a dashboard, implement feature memory bank, run an ensemble, or add distillation. Hard-example mining is offline only and must not run during normal submission or benchmark workflows.
