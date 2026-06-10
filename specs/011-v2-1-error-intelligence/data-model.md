# Data Model: V2.1 Error Intelligence

## Baseline Reference

Represents the locked original V2B baseline used as the comparison authority.

**Fields**

- `model_name`: canonical baseline name, expected to identify original V2B.
- `checkpoint_path`: path or identifier for the V2B checkpoint.
- `validation_prediction_path`: source for V2B validation probabilities.
- `validation_label_path`: source for validation labels.
- `threshold_path`: source for the locked V2B threshold.
- `best_threshold`: selected V2B validation threshold.
- `f1`: locked validation F1 for `target = 1`.
- `precision`: locked validation precision for `target = 1`.
- `recall`: locked validation recall for `target = 1`.
- `tp`, `fp`, `fn`, `tn`: locked confusion counts.
- `validation_target_distribution`: target distribution in validation labels.
- `prediction_target_distribution`: target distribution after applying the locked threshold.
- `test_submission_distribution`: optional reference distribution from original V2B submission.
- `runtime_per_image`: optional baseline speed value.

**Validation Rules**

- Confusion counts must be non-negative integers.
- Metrics must be numeric and within `[0, 1]`.
- `best_threshold` must be numeric and within `[0, 1]`.
- Baseline reference must not include test labels or public leaderboard threshold choices.

## Validation Audit Row

Represents one validation image after joining labels, V2B probability, V2B prediction, and optional evidence.

**Fields**

- `image_id`: normalized validation image identifier.
- `true_label`: validation label, `0` or `1`.
- `v2b_probability`: probability or score for `target = 1`.
- `v2b_prediction`: binary prediction after applying the locked threshold.
- `error_type`: one of `TP`, `TN`, `FP`, or `FN`.
- `threshold`: locked V2B threshold used for the row.
- `threshold_distance`: `abs(v2b_probability - threshold)`.
- `is_near_threshold`: true when `threshold_distance <= 0.05`.
- `is_high_confidence`: true when `threshold_distance >= 0.30`.
- `is_over_rejected_reusable`: true when `true_label = 0` and `v2b_prediction = 1`.
- `detector_evidence_status`: `present` or `missing`.
- `image_quality_status`: `present` or `missing`.
- `source_prediction_path`: source identifier for traceability.

**Validation Rules**

- Each `image_id` appears exactly once.
- `true_label` and `v2b_prediction` must be binary.
- `v2b_probability` must be within `[0, 1]`.
- `error_type` must match `true_label` and `v2b_prediction`.
- Missing optional evidence must not remove the row.

## Detector Evidence Record

Represents optional category-level detector evidence for an image.

**Fields**

- `image_id`: normalized image identifier.
- `category`: detector category or defect label.
- `confidence`: detector confidence.
- `bbox`: bounding box evidence when available.
- `area`: detected area when available.
- `evidence_source_path`: detector evidence source.

**Validation Rules**

- `confidence` must be numeric when present.
- Rows may be one-to-many per image; audit outputs should preserve the strongest or summarized evidence while retaining category visibility.
- Missing detector evidence is reported as missing, not treated as negative evidence.

## Image Quality Record

Represents optional non-label diagnostic evidence for an image.

**Fields**

- `image_id`: normalized image identifier.
- `brightness`: image brightness diagnostic when available.
- `blur`: blur diagnostic when available.
- `crop_size`: crop size or crop-area diagnostic when available.
- `crop_confidence`: crop confidence when available.
- `quality_source_path`: source identifier.

**Validation Rules**

- Values may be absent; absence must be reported.
- Quality diagnostics are analysis features only and must not change labels or predictions.

## Error Review Group

Represents a named subset of validation audit rows for manual review.

**Fields**

- `group_name`: one of `high_confidence_false_positive`, `high_confidence_false_negative`, `near_threshold_false_positive`, `near_threshold_false_negative`, or `over_rejected_reusable`.
- `selection_rule`: human-readable rule used to create the group.
- `row_count`: number of audit rows in the group.
- `output_path`: path to the group CSV.

**Validation Rules**

- Fixed probability-distance bands are mandatory.
- False-positive groups must have `true_label = 0` and `v2b_prediction = 1`.
- False-negative groups must have `true_label = 1` and `v2b_prediction = 0`.

## Audit Summary

Represents aggregate evidence proving the audit is aligned with the locked baseline.

**Fields**

- `baseline_match`: boolean result of metric and count comparison.
- `computed_metrics`: recomputed F1, precision, recall, and confusion counts.
- `baseline_metrics`: locked baseline values used for comparison.
- `target_distribution`: validation label distribution.
- `prediction_distribution`: V2B prediction distribution.
- `review_group_counts`: counts for all required review groups.
- `missing_optional_evidence_counts`: missing detector and image-quality counts.
- `provenance`: input source identifiers and output paths.
- `safety_flags`: no-test-labels, no-sample-submission-labels, no-public-leaderboard-tuning, no-training, no-submission.

**Validation Rules**

- `baseline_match` must be true before the audit is considered accepted.
- Safety flags must all confirm the feature stayed analysis-only and validation-only.
