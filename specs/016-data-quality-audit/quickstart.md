# Quickstart: Controlled Data Quality Audit Pipeline

## Goal

Run one conservative audit over the full training set, merge existing V2B/OOF and Spec 014 evidence, split rows into controlled buckets, and produce a cleaned training manifest plus manual-review artifacts without starting any model training.

## Inputs

Confirm that the workflow can access:

- the training label file
- the training image directory
- the data quality audit config
- optional V2B validation predictions
- optional precomputed OOF predictions
- optional Spec 014 allowed and blocked hard-row outputs

Representative inputs live under:

```text
configs/data_quality_audit.yaml
1st-krones-vision-ai-challenge/
artifacts/kaggle_v2b_artifacts/
outputs/analysis/hard_row_visual_review/reports/
```

## Run The Audit

```bash
python -m src.analysis.data_quality_audit run --config configs/data_quality_audit.yaml
```

## Build Optional Contact Sheets

```bash
python -m src.analysis.data_quality_audit build-contact-sheets --config configs/data_quality_audit.yaml
```

## Expected Outputs

```text
outputs/analysis/data_quality_audit/
|-- data_quality_master.csv
|-- clean_train_rows.csv
|-- hard_valid_train_rows.csv
|-- exclude_from_training_rows.csv
|-- manual_review_required_rows.csv
|-- cleaned_training_manifest.csv
|-- review_decision_template.csv
|-- reports/
|   |-- data_quality_summary.json
|   |-- bucket_counts.csv
|   |-- exclusion_reason_counts.csv
|   |-- top_suspicious_rows.csv
|   |-- top_label_issue_rows.csv
|   |-- top_confident_wrong_rows.csv
|   |-- top_anomaly_rows.csv
|   |-- duplicate_conflicts.csv
|   |-- roi_quality_issues.csv
|   |-- manual_review_plan.json
|   |-- input_file_inventory.csv
|   `-- validation_alignment_summary.json
`-- contact_sheets/
```

## Acceptance Checks

The workflow is accepted only when:

- every training row appears exactly once in `data_quality_master.csv`
- missing optional evidence is reported rather than silently ignored
- all suspected-mislabel blocked rows stay out of the cleaned manifest
- all `manual_review_required` rows stay out of the cleaned manifest
- duplicate label-conflict rows and missing/unreadable image rows stay out of the cleaned manifest
- summary reports confirm no training, no submission creation, no test-label use, no leaderboard tuning, and no original label modification

## Verification Commands

```bash
python -m pytest tests/test_data_quality_audit.py -q
python -m py_compile src/analysis/data_quality_audit.py
```

## Out Of Scope

This quickstart does not generate fresh OOF predictions, relabel training data, approve manual-review rows for training, start a training run, generate a Kaggle submission, or redesign the hybrid inference pipeline.
