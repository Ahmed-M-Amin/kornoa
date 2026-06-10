# V2.1 FiftyOne Review Workflow

## Purpose

This workflow builds a human-in-the-loop review dataset for V2B validation mistakes. It does not train, submit, use test labels, or change model predictions.

Suggested tags are review hints only. They are not ground truth.

## Inputs

- `outputs/analysis/v2_1_error_intelligence/manual_review/manual_review_prefilled.csv`
- `1st-krones-vision-ai-challenge/train_images`

## Build

```bash
python -m src.analysis.v2_1_fiftyone_review build --config configs/v2_1_fiftyone_review.yaml
```

Outputs:

- `outputs/analysis/v2_1_error_intelligence/fiftyone_review/fiftyone_dataset_summary.json`
- `outputs/analysis/v2_1_error_intelligence/fiftyone_review/fiftyone_dataset_records.csv`

## Launch

```bash
python -m src.analysis.v2_1_fiftyone_review launch --config configs/v2_1_fiftyone_review.yaml
```

Filter by `review_group`:

- `high_confidence_false_positives`
- `near_threshold_false_positives`
- `near_threshold_false_negatives`
- `over_rejected_reusable`

Write final decisions into:

- `final_visual_tag`
- `final_recommended_action`
- `reviewer_note`
- `review_status`

If `fiftyone` is not installed locally, build and export still work from saved artifacts, but launch requires installing the package first.

## Export

```bash
python -m src.analysis.v2_1_fiftyone_review export --config configs/v2_1_fiftyone_review.yaml
```

Outputs:

- `outputs/analysis/v2_1_error_intelligence/fiftyone_review/reviewed_manual_tags.csv`
- `outputs/analysis/v2_1_error_intelligence/fiftyone_review/reviewed_tag_summary.json`
- `outputs/analysis/v2_1_error_intelligence/fiftyone_review/reviewed_hard_negatives.csv`
- `outputs/analysis/v2_1_error_intelligence/fiftyone_review/reviewed_hard_positives.csv`
- `outputs/analysis/v2_1_error_intelligence/fiftyone_review/reviewed_uncertain_examples.csv`
