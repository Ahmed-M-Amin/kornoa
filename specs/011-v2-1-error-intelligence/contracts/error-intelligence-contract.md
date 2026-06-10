# Contract: V2.1 Error Intelligence

## Purpose

Define the expected config, command behavior, outputs, and safety checks for the V2B validation error audit.

## Command Contract

```text
python -m src.analysis.v2_1_error_intelligence --config configs/v2_1_error_intelligence.yaml
```

## Required Inputs

The config must provide:

- `baseline.validation_predictions`: V2B validation probabilities.
- `baseline.validation_labels`: validation labels for the validation split.
- `baseline.threshold`: locked V2B threshold record.
- `baseline.metrics`: locked V2B validation metrics and confusion counts.
- `output.root`: output root for audit artifacts.

## Optional Inputs

The config may provide:

- `baseline.checkpoint`: V2B checkpoint path or identifier.
- `baseline.submission`: original V2B submission for distribution reference only.
- `evidence.detector`: category-level detector evidence with `image_id`, category, confidence, bounding box, and area when available.
- `evidence.image_quality`: image-quality diagnostics with brightness, blur, crop size, and crop confidence when available.

## Required Configuration Values

- `analysis.positive_class`: must be `1`.
- `analysis.near_threshold_distance`: must default to `0.05`.
- `analysis.high_confidence_distance`: must default to `0.30`.
- `analysis.allow_test_labels`: must be absent or false.
- `analysis.generate_submission`: must be absent or false.
- `analysis.train_model`: must be absent or false.

## Output Contract

The command must write:

- `audit/v2b_validation_error_audit.csv`
- `review_groups/high_confidence_false_positives.csv`
- `review_groups/high_confidence_false_negatives.csv`
- `review_groups/near_threshold_false_positives.csv`
- `review_groups/near_threshold_false_negatives.csv`
- `review_groups/over_rejected_reusable.csv`
- `reports/v2b_error_intelligence_summary.json`
- `reports/v2b_error_group_counts.csv`
- `reports/v2b_target_distribution_report.json`
- `reports/v2b_optional_evidence_report.json`
- `reports/v2b_audit_provenance.json`

## Audit CSV Required Columns

- `image_id`
- `true_label`
- `v2b_probability`
- `v2b_prediction`
- `error_type`
- `threshold`
- `threshold_distance`
- `is_near_threshold`
- `is_high_confidence`
- `is_over_rejected_reusable`
- `detector_category`
- `detector_confidence`
- `detector_bbox`
- `detector_area`
- `detector_evidence_status`
- `brightness`
- `blur`
- `crop_size`
- `crop_confidence`
- `image_quality_status`

## Summary JSON Required Fields

- `baseline_match`
- `computed_metrics`
- `baseline_metrics`
- `target_distribution`
- `prediction_distribution`
- `review_group_counts`
- `missing_optional_evidence_counts`
- `provenance`
- `safety_flags`

## Safety Requirements

- Must fail clearly when required baseline inputs are missing.
- Must fail clearly when labels or predictions cannot be aligned by `image_id`.
- Must fail clearly when computed metrics do not match the locked baseline within configured tolerance.
- Must preserve all validation rows when optional detector or image-quality evidence is missing.
- Must record missing optional evidence counts.
- Must not read test labels.
- Must not read sample-solution labels.
- Must not use public leaderboard score.
- Must not train.
- Must not generate a Kaggle submission.
