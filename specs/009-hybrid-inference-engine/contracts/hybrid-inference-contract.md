# Contract: Hybrid Inference Engine

## Scope

This contract defines SPEC-009 user-facing CLI workflows and artifact contracts. SPEC-009 is inference-only. It consumes accepted V2 classifier prediction artifacts and SPEC-008 detector artifacts, then writes hybrid validation reports and a strict Kaggle submission. It must not retrain any model or implement dashboard, Grad-CAM, memory bank, ensemble, distillation, or final report assets.

## Hybrid Search Contract

**Command shape**:

```text
python -m src.inference.hybrid_submission search --config configs/hybrid_inference.yaml
```

**Required inputs**:
- V2 classifier validation predictions with image IDs, classifier scores, and labels.
- V2 classifier best-threshold report.
- SPEC-008 detector validation predictions or evaluation outputs with image IDs and usable detector confidence.
- SPEC-008 detector category mapping or defect-rule mapping when category-specific rules are enabled.

**Required behavior**:
- Normalize classifier and detector prediction columns into canonical fields.
- Build the validation overlap set from rows with classifier prediction, detector prediction, and label.
- Search classifier threshold candidates, uncertainty margins, detector confidence thresholds, and per-category conditional-area thresholds using validation overlap only.
- Apply rejection-only fusion during search.
- Prefer configurations at or below 30% detector usage when F1 remains competitive.
- Report any selected higher-F1 candidate that exceeds 30% detector usage as a tradeoff.
- Save selected configuration, overlap report, validation hybrid predictions, and metrics.
- Never load test labels, sample-solution labels, or public-score-derived labels.

**Required outputs**:
- `outputs/hybrid/v4/reports/hybrid_config.json`
- `outputs/hybrid/v4/reports/hybrid_metrics.json`
- `outputs/hybrid/v4/reports/overlap_report.json`
- `outputs/hybrid/v4/predictions/val_hybrid_predictions.csv`

## Hybrid Submission Contract

**Command shape**:

```text
python -m src.inference.hybrid_submission submit --config configs/hybrid_inference.yaml
```

**Required inputs**:
- Selected hybrid configuration.
- Classifier test predictions containing classifier confidence or probability.
- Detector test predictions containing detector confidence and category/area evidence when available.

**Required behavior**:
- Fail before writing submission if classifier test confidence is unavailable.
- Apply the same classifier threshold, uncertainty margin, detector confidence threshold, conditional-area thresholds, and rejection-only fusion rules selected from validation.
- Keep confident classifier predictions unchanged.
- Keep uncertain classifier predictions unchanged when detector evidence is missing or does not trigger rejection.
- Allow detector evidence to change only uncertain images to `1 = Not Reusable`.
- Validate final submission schema exactly.

**Required output**:
- `outputs/hybrid/v4/submissions/submission_v4_hybrid.csv`

**Submission schema**:

```text
image_id,target
```

No additional columns are allowed.

## Full Pipeline Contract

**Command shape**:

```text
python -m src.inference.hybrid_submission run --config configs/hybrid_inference.yaml
```

**Required behavior**:
- Execute search, save reports, then generate final submission.
- Reuse selected validation parameters for test inference.
- Save all generated artifacts under `outputs/hybrid/v4/`.
- Preserve accepted V2 classifier and SPEC-008 detector artifacts unchanged.

## Report Contracts

### `hybrid_config.json`

Required fields:
- `version`
- `classifier_model_id`
- `detector_model_id`
- `selection_rule`
- `classifier_threshold`
- `uncertainty_margin`
- `detector_conf_threshold`
- `conditional_area_thresholds`
- `default_conditional_area_threshold`
- `detector_usage_preference`
- `optimization_metric`
- `tie_break_policy`
- `tuned_on`
- `uses_test_labels`
- `retrained_models`
- `input_paths`

### `hybrid_metrics.json`

Required fields:
- `overlap_count`
- `classifier_baseline`
- `detector_baseline`
- `hybrid`
- `detector_usage`
- `fallback_area_threshold_uses`
- `usage_tradeoff_reported`

Each metric block must include accuracy, precision, recall, F1, TP, FP, TN, FN, positive support, and negative support when labels are available.

### `overlap_report.json`

Required fields:
- `classifier_val_count`
- `detector_val_count`
- `overlap_count`
- `classifier_only_count`
- `detector_only_count`
- `overlap_ratio_classifier`
- `overlap_ratio_detector`
- `used_for_parameter_search`

### `val_hybrid_predictions.csv`

Required columns:

```text
image_id
y_true
classifier_score
classifier_target
classifier_uncertain
detector_confidence
defect_category
defect_area
detector_rejection_reason
detector_used
hybrid_target
decision_source
```

Timing columns may be included when timing inputs exist.

## Error Contract

The workflow must fail clearly for these cases:
- Missing classifier validation scores.
- Missing validation labels during parameter search.
- Empty validation overlap set.
- Missing detector confidence-like evidence for detector threshold search.
- Missing classifier test confidence for submission generation.
- Final submission contains columns other than `image_id,target`.
- Any attempt to load test labels or sample-solution labels.
