# Contract: V5 Strong Classifier

## Configuration Contract

The V5 workflow is driven by a project configuration that identifies dataset paths, split source, model candidate, training choices, threshold search, output root, and benchmark reference.

Required configuration behavior:

- Primary model is `convnext_tiny` at image size `512`.
- Fallback model is `efficientnet_b2`, used only after a primary resource-limit failure or speed/memory rejection.
- Split source resolves to the V2B-compatible train/validation split.
- Hard-example strategy defaults to `analysis_only`.
- Output root defaults to `outputs/kaggle_v5/v5_strong_classifier`.
- Public score is not accepted as an input to training or threshold selection.

## Training Contract

Training a V5 candidate must:

1. Load the configured dataset without hardcoded local-only paths.
2. Reuse the V2B-compatible validation split.
3. Preserve `0 = Reusable` and `1 = Not Reusable`.
4. Apply ROI-preserving preprocessing that does not remove internal dark defects.
5. Train the configured primary or fallback classifier.
6. Select threshold by validation F1 only.
7. Save the best model artifact to:

```text
outputs/kaggle_v5/v5_strong_classifier/models/classifier_best.pth
```

8. Save validation predictions to:

```text
outputs/kaggle_v5/v5_strong_classifier/predictions/val_classifier_predictions.csv
```

9. Save threshold and metrics reports to:

```text
outputs/kaggle_v5/v5_strong_classifier/reports/best_threshold.json
outputs/kaggle_v5/v5_strong_classifier/reports/classifier_metrics.json
```

## Validation Prediction CSV Contract

Required core columns:

```csv
image_id,true_label,prob_bad,classifier_prediction,target
```

Rules:

- `image_id` is unique.
- `true_label`, `classifier_prediction`, and `target` are binary.
- `prob_bad` is a numeric value in `[0, 1]`.
- Rows belong to the V2B-compatible validation split.
- Extra diagnostics must be documented separately and must not replace these core columns.

## Test Prediction CSV Contract

Required columns:

```csv
image_id,prob_bad,classifier_prediction,target
```

Rules:

- `image_id` is unique.
- `classifier_prediction` and `target` are binary.
- `prob_bad` is a numeric value in `[0, 1]`.
- No test label column is allowed.
- The file is saved to:

```text
outputs/kaggle_v5/v5_strong_classifier/predictions/test_classifier_predictions.csv
```

## Submission Contract

Required columns:

```csv
image_id,target
```

Rules:

- No diagnostic, probability, source, threshold, or label columns are allowed.
- `target` is binary integer.
- Row count matches the expected test image count.
- The file is saved to:

```text
outputs/kaggle_v5/v5_strong_classifier/submissions/submission_v5.csv
```

## Metrics and Acceptance Contract

The metrics report must include:

- Validation F1.
- Selected threshold.
- Validation target distribution.
- Validation prediction distribution.
- Test prediction distribution.
- Submission row count.
- Confusion counts.
- V2B benchmark reference.
- V5/V2B inference-time ratio.
- Hard-example strategy.
- Split-safety status.
- Public-score decision status.

Acceptance rules:

- V5 becomes the current best public submission only if public score is greater than `0.92181`.
- Accepted V5 inference must be no more than 2x slower than the accepted V2B classifier benchmark.
- If V5 does not beat V4.2 publicly, V4.2 remains current best and V5 is analysis-only.

## Prohibited Behavior

- Do not retrain V3 detector.
- Do not change V4.2 fusion logic.
- Do not train on test images or test labels.
- Do not use sample-solution labels.
- Do not tune using public leaderboard feedback.
- Do not submit automatically.
- Do not commit generated model weights, predictions, reports, benchmarks, or submissions.
