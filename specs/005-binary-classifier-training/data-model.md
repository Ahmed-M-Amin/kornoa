# Data Model: Binary Classifier Training

## Classifier Training Run

Represents one reproducible V1 classifier training execution.

Fields:

- `run_id`: Stable run identifier or generated timestamp.
- `config_path`: Classifier configuration used for the run.
- `dataset_root`: Configured dataset root.
- `seed`: Fixed seed used for split and training reproducibility.
- `split_policy`: Must be `80/20 stratified`.
- `model_name`: V1 baseline model name.
- `image_size`: Must be `384`.
- `class_counts`: Count of `0 = Reusable` and `1 = Not Reusable` examples.
- `best_epoch`: Epoch selected by validation F1 after threshold search.
- `best_f1`: Best validation F1-score.
- `best_threshold`: Threshold selected by validation F1.
- `artifact_paths`: Saved model, metrics, threshold, and validation prediction
  paths.

Validation rules:

- Must include both binary classes.
- Must not include test-image labels.
- Must write artifacts only to ignored output directories.

## Training Example

Represents one labeled image after dataset validation and SPEC-004
preprocessing.

Fields:

- `image_id`: Stable identifier from `train.csv`.
- `image_path`: Resolved training image path.
- `label`: Binary target, `0` or `1`.
- `split`: `train` or `validation`.
- `preprocessed_input`: Normalized `3x384x384` model input from SPEC-004.
- `roi_method`: Annotation ROI, square-center fallback, or full-resize fallback
  metadata when available.

Validation rules:

- `label` must be `0` or `1`.
- `image_path` must be readable.
- `preprocessed_input` must have normalized channel-first shape.
- Validation examples must not receive training-only random augmentation.

## Validation Prediction

Represents one validation prediction row.

Fields:

- `image_id`: Validation image identifier.
- `true_label`: Binary target.
- `probability`: Predicted probability for `1 = Not Reusable`.
- `threshold`: Selected decision threshold.
- `predicted_label`: Binary label after thresholding.
- `split`: Must be `validation`.

Validation rules:

- There must be exactly one row per validation example.
- `probability` must be between `0.0` and `1.0`.
- `predicted_label` must be derived from `probability >= threshold`.

## Classifier Metrics Report

Represents saved validation metrics for the selected checkpoint.

Fields:

- `run_id`: Matching training run identifier.
- `f1_score`: Validation F1-score at selected threshold.
- `threshold`: Best threshold.
- `class_counts`: Train and validation counts by binary class.
- `confusion_counts`: True positives, false positives, true negatives, false
  negatives.
- `best_epoch`: Selected epoch.
- `runtime_seconds`: Training runtime for smoke/private runs.
- `artifact_paths`: Paths to saved generated outputs.

Validation rules:

- Must include F1-score and confusion counts.
- Must include threshold tie-break policy metadata.
- Must not include test-label metrics.

## Best Threshold Record

Represents the threshold selection result consumed by later inference specs.

Fields:

- `threshold`: Selected decision threshold.
- `f1_score`: Validation F1-score at selected threshold.
- `tie_break`: Must be `lowest_threshold`.
- `candidate_count`: Number of thresholds evaluated.
- `selected_at`: Run timestamp or run identifier.

Validation rules:

- If multiple thresholds tie on best F1, select the lowest threshold.
- Threshold must be between `0.0` and `1.0`.
