# Quickstart: Hybrid Inference Engine

## Prerequisites

- SPEC-001 through SPEC-008 are complete.
- Accepted classifier artifacts are available for V2B EfficientNet-B1 `analysis_only`.
- SPEC-008 detector validation and test prediction artifacts are available.
- Classifier test predictions include confidence or probability scores, not only target labels.
- Generated hybrid outputs can be written under `outputs/hybrid/v4/`.

## Configure Hybrid Paths

Create or update:

```text
configs/hybrid_inference.yaml
```

Expected configuration groups:

```text
classifier:
  model_id: v2b_effnet_b1
  val_predictions: path/to/val_classifier_predictions.csv
  test_predictions: path/to/test_classifier_predictions.csv
  threshold_report: path/to/best_threshold.json

detector:
  model_id: v3_detector
  val_predictions: path/to/val_detector_predictions.csv
  test_predictions: path/to/test_detector_predictions.csv
  category_mapping: path/to/category_mapping.json

hybrid:
  output_dir: outputs/hybrid/v4
  detector_usage_preference: 0.30
  selection_metric: f1
  default_conditional_area_threshold: 0.05
  conditional_area_thresholds:
    example_conditional_category: 0.05
```

## Search Hybrid Parameters

```text
python -m src.inference.hybrid_submission search --config configs/hybrid_inference.yaml
```

Expected outputs:

```text
outputs/hybrid/v4/reports/hybrid_config.json
outputs/hybrid/v4/reports/hybrid_metrics.json
outputs/hybrid/v4/reports/overlap_report.json
outputs/hybrid/v4/predictions/val_hybrid_predictions.csv
```

## Generate Hybrid Submission

```text
python -m src.inference.hybrid_submission submit --config configs/hybrid_inference.yaml
```

Expected output:

```text
outputs/hybrid/v4/submissions/submission_v4_hybrid.csv
```

The submission must contain exactly:

```text
image_id,target
```

## Run Full Hybrid Workflow

```text
python -m src.inference.hybrid_submission run --config configs/hybrid_inference.yaml
```

This runs validation search and submission generation with the same selected hybrid configuration.

## Acceptance Checks

- Confident classifier predictions are unchanged.
- Detector evidence is used only for classifier-uncertain images.
- Detector evidence can only reject uncertain images as `1 = Not Reusable`.
- Detector evidence never clears classifier rejections to `0 = Reusable`.
- Uncertain images without usable detector evidence keep the classifier target.
- Per-category conditional-area thresholds are used when mapped.
- Default conditional-area threshold fallback usage is reported.
- Parameter search uses validation overlap only.
- Detector usage is at or below 30% of validation overlap unless a higher-F1 tradeoff is explicitly reported.
- Final submission has exactly `image_id,target`.
- Workflow fails clearly if classifier test confidence is missing.
- No classifier or detector retraining occurs.
- Existing classifier-only and detector-preparation tests continue to pass.
