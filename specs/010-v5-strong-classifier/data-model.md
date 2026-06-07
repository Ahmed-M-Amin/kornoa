# Data Model: V5 Strong Classifier

## V5 Experiment Configuration

Represents the reproducible settings for one V5 classifier run.

**Fields**

- `experiment_name`: Stable name for the run, expected default `v5_strong_classifier`.
- `model_name`: Primary `convnext_tiny`; fallback `efficientnet_b2` only when resource limits block the primary.
- `image_size`: `512` for the primary V5 candidate.
- `pretrained`: Whether pretrained weights are used.
- `seed`: Fixed random seed for reproducibility.
- `split_source`: Reference to the V2B-compatible train/validation split.
- `loss`: Weighted BCE or focal loss choice.
- `sampler`: Class-balanced sampler setting.
- `hard_example_strategy`: `analysis_only` by default; `oversample` only with explicit opt-in.
- `output_root`: `outputs/kaggle_v5/v5_strong_classifier`.

**Validation Rules**

- `model_name` must not introduce a broad model sweep.
- `hard_example_strategy=oversample` must require split-safety reporting.
- Missing or unreconstructable V2B-compatible split is a blocking error.

## V5 Classifier Candidate

Represents a trained classifier candidate and its selection status.

**Fields**

- `candidate_id`: Stable identifier for the candidate.
- `model_name`: Candidate backbone identity.
- `image_size`: Candidate input size.
- `artifact_path`: Saved model artifact path.
- `validation_f1`: F1 after threshold search.
- `selected_threshold`: Validation-selected decision threshold.
- `validation_accuracy`, `precision`, `recall`: Supporting metrics.
- `tp`, `fp`, `tn`, `fn`: Confusion counts.
- `inference_images_per_second`: Measured inference throughput.
- `v2b_speed_ratio`: Candidate inference time relative to V2B.
- `acceptance_status`: `candidate`, `accepted`, `rejected_public_score`, `rejected_speed`, or `analysis_only`.

**Validation Rules**

- Accepted candidates must beat public score `0.92181` after manual review.
- Accepted candidates must be no more than 2x slower than V2B.
- Public score must not affect threshold or training settings.

## Validation Prediction Export

Image-level validation probabilities and threshold-derived targets.

**Required Core Columns**

- `image_id`
- `true_label`
- `prob_bad`
- `classifier_prediction`
- `target`

**Validation Rules**

- `image_id` values must be unique.
- `true_label`, `classifier_prediction`, and `target` must be binary.
- `prob_bad` must be within `[0, 1]`.
- Rows must come only from the V2B-compatible validation split.

## Test Prediction Export

Image-level test probabilities used for submission generation and future hybrid consumption.

**Required Columns**

- `image_id`
- `prob_bad`
- `classifier_prediction`
- `target`

**Validation Rules**

- `image_id` values must be unique and match expected test images.
- `target` and `classifier_prediction` must be binary.
- `prob_bad` must be within `[0, 1]`.
- No label column is allowed.

## Threshold Report

Records the validation-only threshold selection result.

**Fields**

- `selected_threshold`
- `validation_f1`
- `validation_accuracy`
- `precision`
- `recall`
- `tp`, `fp`, `tn`, `fn`
- `split_source`
- `selection_source`: Must state validation-only selection.

**Validation Rules**

- Threshold must be selected without test labels, sample-solution labels, manual test inspection, or public-score tuning.

## Classifier Metrics Report

Summarizes V5 candidate quality and acceptance readiness.

**Fields**

- `experiment_name`
- `candidate_id`
- `model_name`
- `image_size`
- `validation_target_distribution`
- `validation_prediction_distribution`
- `test_prediction_distribution`
- `selected_threshold`
- `validation_f1`
- `submission_row_count`
- `v2b_benchmark_reference`
- `v2b_speed_ratio`
- `hard_example_strategy`
- `train_validation_split_status`
- `public_score_decision`

**Validation Rules**

- Report must identify V4.2 public score `0.92181` as the acceptance baseline.
- Report must identify V3 detector-only score `0.74169` as weak and out of scope for retraining.

## V5 Submission

Strict competition submission generated from V5 test probabilities and selected threshold.

**Required Columns**

- `image_id`
- `target`

**Validation Rules**

- No diagnostic columns are allowed.
- `target` must be binary integer values.
- Row count must match expected test image count.

## State Transitions

```text
configured
  -> training_started
  -> trained_candidate
  -> validation_threshold_selected
  -> test_probabilities_exported
  -> submission_generated
  -> reviewed_for_public_submission
  -> accepted_best_public_submission | analysis_only
```

Invalid transitions:

- `training_started` from test labels or public-score-derived settings.
- `accepted_best_public_submission` without public score greater than `0.92181`.
- `accepted_best_public_submission` when runtime exceeds 2x V2B.
