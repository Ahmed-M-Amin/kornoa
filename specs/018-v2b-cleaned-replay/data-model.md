# Data Model: Spec 018 V2B Cleaned Replay

## CleanedReplayConfig

Represents the controlled training configuration for Spec 018.

Fields:

- `experiment_name`: Stable run name for the cleaned replay.
- `seed`: Reproducibility seed.
- `dataset_root`: Root containing training labels and training images.
- `approved_manifest_path`: Spec 017 approved manifest path.
- `auto_exclude_rows_path`: Spec 017 excluded rows path.
- `needs_adjudication_rows_path`: Spec 017 adjudication rows path.
- `deferred_uncertain_rows_path`: Spec 017 deferred rows path.
- `decision_lock_summary_path`: Spec 017 decision summary path.
- `baseline_checkpoint_path`: Locked V2B checkpoint.
- `baseline_predictions_path`: Locked V2B validation predictions.
- `baseline_threshold_path`: Locked V2B threshold report.
- `baseline_metrics_path`: Locked V2B metrics report.
- `baseline_runtime_report_path`: Locked V2B runtime report.
- `output_root`: Cleaned replay output root.
- `safety_flags`: Booleans controlling training, submission, test labels, leaderboard input, and relabeling.

Validation rules:

- Safety flags must forbid test labels, leaderboard input, submission generation during dry-run, and relabeling.
- All required Spec 017 and V2B baseline paths must exist before training.
- `approved_manifest_path` must be the only training manifest.

## DryRunValidationReport

Represents the pre-training safety validation result.

Fields:

- `approved_manifest_count`
- `auto_exclude_overlap_count`
- `needs_adjudication_overlap_count`
- `deferred_overlap_count`
- `missing_train_label_count`
- `missing_train_image_count`
- `baseline_artifacts_available`
- `validation_split_status`
- `no_test_labels_used`
- `no_leaderboard_tuning`
- `no_submission_created`
- `training_started`
- `warnings`

Validation rules:

- All overlap counts must be 0.
- Missing label and image counts must be 0.
- `training_started` must be false.
- Baseline artifacts must be available.

## CleanedReplayRun

Represents one completed controlled training run.

Fields:

- `run_name`
- `approved_manifest_path`
- `approved_manifest_count`
- `train_row_count`
- `validation_row_count`
- `seed`
- `split_source`
- `model_checkpoint_path`
- `validation_predictions_path`
- `best_threshold_path`
- `metrics_path`
- `config_snapshot_path`
- `manifest_snapshot_path`
- `runtime_metadata_path`

Validation rules:

- Training can only start after a passing dry-run report.
- All listed artifact paths must be written after a completed run.
- Validation predictions and metrics must use validation labels only.

## BaselineComparisonReport

Represents the direct cleaned replay versus locked V2B comparison.

Fields:

- `baseline_name`
- `baseline_validation_f1`
- `cleaned_replay_validation_f1`
- `f1_delta`
- `baseline_threshold`
- `cleaned_replay_threshold`
- `validation_row_count`
- `decision`
- `decision_reason`
- `safety_gate_status`
- `artifact_paths`

Validation rules:

- `decision` must be `accepted`, `rejected`, or `inconclusive`.
- A run cannot be accepted if any safety gate fails.
- A run cannot be accepted if validation F1 does not beat the locked V2B baseline.

## State Transitions

```text
Spec 017 approved manifest
  -> dry-run validation
  -> cleaned replay training
  -> artifact completeness validation
  -> baseline comparison
  -> accepted | rejected | inconclusive
```

Blocked transitions:

- Training before dry-run validation passes.
- Submission generation during dry-run validation.
- Use of auto-excluded, deferred, or needs-adjudication rows in training.
- Acceptance without validation improvement over locked V2B.
