# Data Model: Hybrid Inference Engine

## ClassifierPrediction

Represents one image-level classifier prediction.

**Fields**:
- `image_id`: Stable image identifier normalized from filename, path, or source ID.
- `classifier_score`: Probability or score for `1 = Not Reusable`.
- `classifier_threshold`: Threshold used to derive the binary classifier target.
- `classifier_target`: Binary target derived from `classifier_score >= classifier_threshold` when not supplied.
- `y_true`: Optional validation label; required for validation parameter search.
- `classifier_uncertain`: Whether the absolute distance from threshold is inside the uncertainty margin.

**Validation Rules**:
- `image_id` must be present and non-empty after normalization.
- `classifier_score` must be numeric and within the accepted score range for threshold comparison.
- Validation search requires `y_true`; test submission generation must not load or require `y_true`.
- When `classifier_target` is missing, it is computed from the selected classifier threshold.

## DetectorEvidence

Represents image-level detector evidence after aggregating raw detections.

**Fields**:
- `image_id`
- `detector_confidence`: Highest usable confidence for the image or defect rule.
- `defect_category`: Detector category associated with the selected evidence when available.
- `defect_area`: Estimated defect area when available.
- `detector_target`: Whether detector evidence is strong enough to reject the bottle.
- `evidence_available`: Whether usable detector evidence exists for the image.

**Validation Rules**:
- Multiple detections per image are aggregated deterministically, preserving the strongest usable rejection evidence.
- Missing detector evidence never changes the classifier decision.
- Detector evidence for confident classifier predictions is retained for reporting only and must not override the classifier.
- Detector evidence below configured confidence does not trigger rejection.

## HybridRuleConfiguration

Represents the selected fusion parameters.

**Fields**:
- `version`: Hybrid version identifier, expected to represent V4/SPEC-009.
- `classifier_model_id`
- `detector_model_id`
- `classifier_threshold`
- `uncertainty_margin`
- `detector_conf_threshold`
- `conditional_area_thresholds`: Per-category area thresholds for conditional defects.
- `default_conditional_area_threshold`: Fallback threshold for unmapped conditional categories.
- `detector_usage_preference`: Preferred maximum detector usage ratio, set to `0.30`.
- `selection_metric`: Primary validation metric, expected to be F1.
- `tie_break_policy`: Ordered tie-breaks for close candidates.
- `input_paths`
- `uses_test_labels`: Must be false.
- `retrained_models`: Must be false.

**Validation Rules**:
- Threshold and margin values must be bounded and reproducible.
- Detector usage preference must be reported even when a higher-F1 candidate exceeds it.
- Fallback conditional-area threshold must be present if conditional categories are enabled.
- Input paths must identify classifier and detector artifacts without modifying them.

## ValidationOverlapSet

Represents validation rows eligible for hybrid parameter search.

**Fields**:
- `classifier_val_count`
- `detector_val_count`
- `overlap_image_ids`
- `overlap_count`
- `classifier_only_count`
- `detector_only_count`
- `overlap_ratio_classifier`
- `overlap_ratio_detector`
- `used_for_parameter_search`

**Validation Rules**:
- Only image IDs present in classifier predictions, detector predictions, and labels are eligible for search.
- Overlap rows must be sorted deterministically by `image_id` before reporting.
- Empty overlap is a hard failure for parameter search.
- Non-overlap rows are counted and reported, not silently ignored.

## HybridPrediction

Represents one fused validation or test prediction.

**Fields**:
- `image_id`
- `y_true`: Validation only.
- `classifier_score`
- `classifier_target`
- `classifier_uncertain`
- `detector_confidence`
- `defect_category`
- `defect_area`
- `detector_rejection_reason`
- `detector_used`
- `hybrid_target`
- `decision_source`
- `timing_ms`: Optional per-image timing when available.

**Validation Rules**:
- Confident classifier rows must have `hybrid_target == classifier_target` and `detector_used == false`.
- Uncertain rows without usable detector rejection evidence must keep the classifier target.
- Detector-used rows can only set `hybrid_target` to `1 = Not Reusable`.
- Detector evidence must never clear an uncertain classifier rejection to `0 = Reusable`.
- Final submission exports only `image_id` and `target`, where `target` is the integer `hybrid_target`.

## HybridMetricsReport

Represents validation comparison and usage metrics.

**Fields**:
- `overlap_count`
- `classifier_baseline`
- `detector_baseline`
- `hybrid`
- `detector_usage_count`
- `detector_usage_ratio`
- `changed_predictions`
- `changed_incorrect_to_correct`
- `changed_correct_to_incorrect`
- `changed_fn_to_tp`
- `changed_fp_to_tn`
- `fallback_area_threshold_uses`
- `usage_tradeoff_reported`

**Validation Rules**:
- Classifier, detector, and hybrid metrics are computed on the same validation overlap set.
- Metrics include accuracy, precision, recall, F1, TP, FP, TN, FN, positive support, and negative support.
- Detector-only baseline is marked unavailable when detector confidence or binary evidence is insufficient.
- Any selected configuration above 30% detector usage must explicitly record the F1 tradeoff that justified it.

## HybridSubmission

Represents competition-ready output.

**Fields**:
- `image_id`
- `target`

**Validation Rules**:
- Columns must be exactly `image_id,target`.
- `target` values must be integer binary labels.
- No confidence, decision source, debug, or label columns are allowed.
- Submission generation must fail if classifier test confidence is unavailable.
