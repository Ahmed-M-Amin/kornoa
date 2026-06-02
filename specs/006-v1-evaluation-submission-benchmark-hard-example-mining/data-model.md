# Data Model: V1 Evaluation, Submission, Benchmark, and Hard-Example Mining

## V1 Artifact Root

Represents the completed concrete V1 run output tree.

Fields:

- `artifact_root`: Concrete run root path supplied by the caller or defaulted to `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1`.
- `parent_artifact_root`: Optional parent root, `artifacts/kaggle_v1_artifacts/outputs`, which resolves to its `kaggle_v1` child when supplied.
- `model_path`: Path to `models/classifier_effnet_b0_best.pth`.
- `metrics_path`: Path to `reports/classifier_metrics.json`.
- `threshold_path`: Path to `reports/best_threshold.json`.
- `validation_predictions_path`: Path to `predictions/val_classifier_predictions.csv`.

Validation rules:

- Required concrete run files must exist before V1 evaluation workflows run: `models/classifier_effnet_b0_best.pth`, `reports/classifier_metrics.json`, `reports/best_threshold.json`, and `predictions/val_classifier_predictions.csv`.
- If the supplied root contains a `kaggle_v1` child with the required concrete run files, that child becomes the concrete V1 run artifact root.
- Paths must be configurable and must not require the local Windows dataset path.
- Private-derived artifacts remain ignored and local.

## Threshold Record

Represents the saved V1 decision threshold.

Fields:

- `threshold`: Numeric value between `0.0` and `1.0`.
- `f1_score`: Optional validation F1-score from threshold search.
- `tie_break`: Optional threshold-search tie-break policy.
- `candidate_count`: Optional number of threshold candidates evaluated.

Validation rules:

- `threshold` is required and is the only decision threshold for SPEC-006.
- Missing or non-numeric threshold values fail before predictions are written.

## Image Prediction

Represents one V1 inference result for a single image.

Fields:

- `image_id`: Sample submission image identifier.
- `image_path`: Resolved image path under the test image directory.
- `probability`: Probability for `1 = Not Reusable`.
- `threshold`: Saved threshold applied to the probability.
- `target`: Binary output, `0` or `1`.
- `inference_seconds`: Optional per-image or per-batch timing contribution.

Validation rules:

- Probability must be between `0.0` and `1.0`.
- Target must equal `1` when `probability >= threshold`, otherwise `0`.
- Test image rows must not include true labels.

## Submission File

Represents the first V1 Kaggle submission.

Fields:

- `image_id`: Copied from `sample_submission.csv`.
- `target` or `label`: Binary target column following the sample submission contract.

Validation rules:

- Must contain exactly one output row for every sample submission row.
- Must preserve sample submission row order.
- Must contain binary values only.

## Benchmark Report

Represents V1 inference speed results.

Fields:

- `image_count`: Number of images processed.
- `total_seconds`: Total measured inference runtime.
- `average_ms_per_image`: Average runtime per image.
- `images_per_second`: Throughput.
- `artifact_paths`: Model and threshold paths used by the run.

Validation rules:

- Image count must be positive.
- Timing values must be non-negative.
- Benchmarking must not generate hard-example files.

## Hard Example Set

Represents offline validation-derived error-analysis artifacts.

Fields:

- `false_positives_path`: Reusable bottles predicted as not reusable.
- `false_negatives_path`: Not reusable bottles predicted as reusable.
- `uncertain_path`: Rows where probability is near the saved threshold.
- `high_loss_samples_path`: High-loss proxy rows ranked from saved probability severity.
- `summary_path`: JSON summary with counts, thresholds, source paths, and generated artifact paths.
- `manual_parent_outputs`: Optional existing hard-example CSVs may be present under `artifacts/kaggle_v1_artifacts/outputs/hard_examples`.

Validation rules:

- Source rows must come from V1 validation predictions only.
- Required source columns are `image_id`, `true_label`, `probability`, `threshold` or external saved threshold, and `predicted_label` or enough fields to recompute it.
- High-loss proxy ranking uses false positives with high probability, false negatives with low probability, then near-threshold uncertain cases.
- Mining must not read test images or write submissions.
- Existing manual hard-example CSVs are not required inputs; the source of truth is `predictions/val_classifier_predictions.csv` under the concrete V1 run artifact root.
