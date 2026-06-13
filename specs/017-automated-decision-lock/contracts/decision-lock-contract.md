# Contract: Spec 017 Decision Lock Workflow

## User-Facing Command

The feature exposes one analysis workflow command:

```powershell
python -m src.analysis.data_quality_decision_lock run --config configs/data_quality_decision_lock.yaml
```

The command must:

- Validate required Spec 016 and Spec 014 inputs.
- Load original training labels and verify approved image availability.
- Assign exactly one decision to every row in the Spec 016 master audit table.
- Write all required CSV, JSON, and contact-sheet outputs.
- Exit with a non-zero status when required inputs are missing or safety gates fail.
- Exit successfully when all required outputs are written and safety gates pass.

## Required Config Contract

Required path groups:

- `inputs.spec016_master_path`
- `inputs.spec016_cleaned_manifest_path`
- `inputs.spec016_manual_review_required_path`
- `inputs.spec016_review_template_path`
- `inputs.spec016_summary_path`
- `inputs.spec016_exclusion_reasons_path`
- `inputs.spec014_allowed_hard_rows_path`
- `inputs.spec014_blocked_roi_pipeline_bug_rows_path`
- `inputs.spec014_blocked_suspected_mislabel_rows_path`
- `inputs.spec014_blocked_other_high_risk_rows_path`
- `dataset.train_csv_path`
- `dataset.train_images_dir`
- `output.analysis_root`

Optional path groups:

- `optional_evidence.oof_predictions_path`
- `optional_evidence.v2b_predictions_path`
- `optional_evidence.image_embeddings_path`
- `optional_evidence.cleanlab_scores_path`
- `optional_evidence.cluster_assignments_path`

Safety flags:

- `safety.train_model` must be false.
- `safety.generate_submission` must be false.
- `safety.allow_test_labels` must be false.
- `safety.public_leaderboard_input` must be false.
- `safety.apply_relabels` must be false.

## Required Output Contract

The command must write:

- `outputs/analysis/data_quality_decision_lock/decision_master.csv`
- `outputs/analysis/data_quality_decision_lock/approved_cleaned_training_manifest.csv`
- `outputs/analysis/data_quality_decision_lock/auto_keep_rows.csv`
- `outputs/analysis/data_quality_decision_lock/auto_exclude_rows.csv`
- `outputs/analysis/data_quality_decision_lock/needs_adjudication_rows.csv`
- `outputs/analysis/data_quality_decision_lock/deferred_uncertain_rows.csv`
- `outputs/analysis/data_quality_decision_lock/adjudication_queue.csv`
- `outputs/analysis/data_quality_decision_lock/reports/decision_lock_summary.json`
- `outputs/analysis/data_quality_decision_lock/reports/decision_rule_audit.csv`
- `outputs/analysis/data_quality_decision_lock/reports/roi_recalibration_summary.csv`
- `outputs/analysis/data_quality_decision_lock/reports/evidence_inventory.json`
- `outputs/analysis/data_quality_decision_lock/contact_sheets/needs_adjudication_top_risk.jpg`
- `outputs/analysis/data_quality_decision_lock/contact_sheets/auto_exclude_representatives.jpg`
- `outputs/analysis/data_quality_decision_lock/contact_sheets/roi_group_representatives.jpg`

## Required Summary Fields

`decision_lock_summary.json` must include:

- `total_rows`
- `auto_keep_count`
- `auto_exclude_count`
- `needs_adjudication_count`
- `defer_count`
- `approved_manifest_count`
- `blocked_overlap_count`
- `duplicate_conflict_approved_count`
- `unresolved_review_approved_count`
- `missing_approved_image_count`
- `original_labels_unchanged`
- `no_training_started`
- `no_submission_created`
- `no_test_labels_used`
- `no_leaderboard_tuning`
- `warnings`

## Safety Contract

The workflow is invalid if:

- A blocked Spec 014 image_id appears in the approved manifest.
- A row with decision other than `auto_keep` appears in the approved manifest.
- A duplicate-conflict row appears in the approved manifest without explicit resolution.
- A manual-review-required row appears in the approved manifest without explicit keep decision.
- Any safety flag permits training, submission generation, test labels, leaderboard input, or relabeling.
